"""
Mission 04 — Part 3: Data & Payload Specialist
Length Bias Diagnostics & Reference-Free SimPO

Starship Operational Unit: USSE Noosphere Bridge Crew
Course: Conversational AI — Winter Semester 2026/27
Sector: SEC-04-ORI (Betelgeuse)
Subsystem 04: Plasma Ignition & Ethical Steering Core
Squadron: LeCun

What this file delivers (see ps04 brief, Part 3)
------------------------------------------------
1. Length correlation diagnostic: Pearson r_len between (|y_w| - |y_l|) and the model's
   preference margin (D_w - D_l), measured on held-out pairs for a DPO and a SimPO policy
   trained on the same verbosity-biased preference labels.
2. SimPOLoss (beta = 2.0, gamma = 0.5): reference-free, uses AVERAGE per-token log-prob.
3. Memory benchmark: DPO (policy + frozen reference) vs SimPO (policy only), MEASURED on a
   real training step (torch.cuda.max_memory_allocated on GPU; exact tensor accounting of
   weights, grads, AdamW state and saved activations on CPU), plus the 7B projection.

The verbosity trap we simulate
------------------------------
Replies = content words (GROUNDED +, SYCOPHANT -) plus FILLER words worth nothing.
Simulated annotators are biased like real ones: they prefer longer replies a bit,
    P(A > B) = sigma( R(A) - R(B) + lambda * (|A| - |B|) ),  lambda = 0.35 per word.
A policy that learns "more filler = more reward" is exploiting length, not quality.

Run
---
    python 03_Length_Bias_and_SimPO.py            # full run (+ PNG)
    python 03_Length_Bias_and_SimPO.py --no-plot

Memorizable patterns
--------------------
* Sum of log-probs grows with length, so DPO margins can be bought with extra tokens.
  SimPO divides by |y|: reward = beta * average log-prob -> no "pay per token".
* gamma is a target gap: SimPO is only satisfied when r_w beats r_l by at least gamma.
* No reference model -> one model in memory. That halves the WEIGHTS, not the whole
  training footprint (gradients + AdamW state of the policy dominate).
"""

import argparse
import copy
import importlib
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
core = importlib.import_module("01_DPO_Loss_Core")


# =============================================================================
# 1. Pearson correlation  (starter, unchanged)
# =============================================================================
def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    """Computes the Pearson correlation coefficient r between two vectors."""
    n = len(x)
    if n != len(y) or n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    diff_x = [val - mean_x for val in x]
    diff_y = [val - mean_y for val in y]

    numerator = sum(dx * dy for dx, dy in zip(diff_x, diff_y))
    var_x = sum(dx ** 2 for dx in diff_x)
    var_y = sum(dy ** 2 for dy in diff_y)

    denominator = math.sqrt(var_x * var_y)
    return numerator / denominator if denominator > 1e-12 else 0.0


# =============================================================================
# 2. SimPO loss  (starter, unchanged)
# =============================================================================
class SimPOLoss(nn.Module):
    """
    Simple Preference Optimization (SimPO) Loss Module (Meng et al., NeurIPS 2024).

    A reference-free objective that eliminates length exploitation through per-token normalization:
        r_SimPO(x, y) = (beta / |y|) * log pi_theta(y|x)
        L_SimPO = -E [ log sigma( r_SimPO(x, yw) - r_SimPO(x, yl) - gamma ) ]
    """

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        chosen_avg_logps: torch.FloatTensor,
        rejected_avg_logps: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Computes SimPO loss from average per-token log-probabilities.
        """
        # Length-normalized implicit rewards
        r_w = self.beta * chosen_avg_logps
        r_l = self.beta * rejected_avg_logps

        # Logit includes target reward margin gamma
        logits = (r_w - r_l) - self.gamma

        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            win_rate = (r_w > r_l).float().mean()
            simpo_probs = torch.sigmoid(logits).mean()

        metrics = {
            "loss": loss.detach(),
            "reward_w": r_w.mean(),
            "reward_l": r_l.mean(),
            "reward_gap": (r_w - r_l).mean(),
            "win_rate": win_rate,
            "margin_prob": simpo_probs
        }

        return loss, metrics


# =============================================================================
# 3. Verbosity-biased preference data
# =============================================================================
LAMBDA_LEN = 0.35      # annotator bonus per extra word
END = "stop '.'"       # key for the closing token in token_category_logps
QUALITY = 0.6          # per (grounded - sycophantic) word


def true_reward(reply: str) -> float:
    w = reply.split()
    return QUALITY * (sum(x in core.GROUNDED for x in w) - sum(x in core.SYCOPHANT for x in w))


def n_words(reply: str) -> int:
    return len(reply.split()) - 1  # without the final '.'


def random_reply(rng: random.Random) -> str:
    n = rng.randint(2, 4)
    g = rng.randint(0, n)
    return core.make_response(rng, g, n - g, n_filler=rng.randint(0, 6))


def biased_pairs(rng: random.Random, n: int, lam: float) -> List[Tuple[str, str, str]]:
    """(prompt, chosen, rejected) from ONE biased annotator vote per pair."""
    out = []
    for i in range(n):
        a, b = random_reply(rng), random_reply(rng)
        z = true_reward(a) - true_reward(b) + lam * (n_words(a) - n_words(b))
        p = core.make_prompt(core.SYSTEMS[i % 8])
        out.append((p, a, b) if rng.random() < 1 / (1 + math.exp(-z)) else (p, b, a))
    return out


def quality_oriented_pairs(rng: random.Random, n: int) -> List[Tuple[str, str, str]]:
    """Held-out pairs oriented by TRUE quality only (length is independent of quality here)."""
    out = []
    while len(out) < n:
        a, b = random_reply(rng), random_reply(rng)
        if true_reward(a) == true_reward(b):
            continue
        p = core.make_prompt(core.SYSTEMS[len(out) % 8])
        out.append((p, a, b) if true_reward(a) > true_reward(b) else (p, b, a))
    return out


# =============================================================================
# 4. Diagnostics
# =============================================================================
@torch.no_grad()
def length_diagnostic(policy, ref, tok, pairs) -> Dict[str, float]:
    pr, w, l = map(list, zip(*pairs))
    pw, pl = core.concatenated_logps(policy, tok, pr, w, l)
    rw, rl = core.concatenated_logps(ref, tok, pr, w, l)
    aw, al = core.concatenated_logps(policy, tok, pr, w, l, average=True)
    dlen = [float(n_words(a) - n_words(b)) for a, b in zip(w, l)]
    dpo_margin = ((pw - rw) - (pl - rl)).tolist()
    simpo_margin = (2.0 * (aw - al)).tolist()
    return {
        "r_len_dpo_margin": calculate_pearson_correlation(dlen, dpo_margin),
        "r_len_simpo_margin": calculate_pearson_correlation(dlen, simpo_margin),
        "acc_dpo_margin": sum(m > 0 for m in dpo_margin) / len(dpo_margin),
        "acc_simpo_margin": sum(m > 0 for m in simpo_margin) / len(simpo_margin),
        "dlen": dlen, "dpo_margin": dpo_margin, "simpo_margin": simpo_margin,
    }


def generation_stats(model, tok, n: int = 240, seed: int = 0) -> Dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    L = F_ = G = S = 0
    for i in range(n):
        w = core.generate(model, tok, core.make_prompt(core.SYSTEMS[i % 8]), max_new=20, greedy=False, gen=g).split()
        w = [x for x in w if x != "."]
        L += len(w)
        F_ += sum(x in core.FILLER for x in w)
        G += sum(x in core.GROUNDED for x in w)
        S += sum(x in core.SYCOPHANT for x in w)
    return {"len": L / n, "filler": F_ / n, "grounded_share": G / max(G + S, 1)}


@torch.no_grad()
def token_category_logps(model, tok, n: int = 200, seed: int = 5) -> Dict[str, float]:
    """Mean per-token log-prob of grounded / sycophantic / filler words and of the closing '.'."""
    rng = random.Random(seed)
    cats = {"grounded": core.GROUNDED, "sycophant": core.SYCOPHANT, "filler": core.FILLER}
    acc = {c: [] for c in list(cats) + [END]}
    for _ in range(n):
        ids, lab = core.encode_pair_batch(tok, [core.make_prompt("hull")], [random_reply(rng)])
        lp = torch.log_softmax(model(ids)[0, :-1], -1)
        for t, y in enumerate(lab[0, 1:].tolist()):
            if y < 0 or y == tok.eos_id:
                continue
            w = tok.itos[y]
            acc[next((k for k, v in cats.items() if w in v), END)].append(lp[t, y].item())
    return {c: sum(v) / len(v) for c, v in acc.items()}


def run_length_experiment(seed: int = 0, verbose: bool = True) -> Dict[str, object]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = core.comms_tokenizer()
    ref = core.TinyGPT(len(tok))
    sft_p = [core.make_prompt(core.SYSTEMS[i % 8]) for i in range(2000)]
    core.sft_train(ref, tok, sft_p, [random_reply(rng) for _ in sft_p], steps=300, seed=seed)
    core.freeze(ref)

    train = biased_pairs(rng, 3000, LAMBDA_LEN)
    test = quality_oriented_pairs(rng, 800)
    dataset_r = calculate_pearson_correlation(
        [float(n_words(c) - n_words(r)) for _, c, r in train],
        [true_reward(c) - true_reward(r) for _, c, r in train])
    longer_chosen = sum(n_words(c) > n_words(r) for _, c, r in train) / len(train)

    dpo = copy.deepcopy(ref).requires_grad_(True)
    simpo = copy.deepcopy(ref).requires_grad_(True)
    core.dpo_train(dpo, ref, tok, train, beta=0.1, steps=250, seed=seed)
    core.dpo_train(simpo, ref, tok, train, steps=250, seed=seed, average=True,
                   loss_fn=SimPOLoss(beta=2.0, gamma=0.5))

    diag = {"dpo": length_diagnostic(dpo, ref, tok, test), "simpo": length_diagnostic(simpo, ref, tok, test)}
    gen = {"ref": generation_stats(ref, tok), "dpo": generation_stats(dpo, tok), "simpo": generation_stats(simpo, tok)}
    cat = {k: token_category_logps(m, tok) for k, m in (("ref", ref), ("dpo", dpo), ("simpo", simpo))}
    out = dict(dataset_r=dataset_r, longer_chosen=longer_chosen, diag=diag, gen=gen, cat=cat)
    if verbose:
        print(f"Biased training labels (lambda = {LAMBDA_LEN}/word): the chosen reply is the longer one in "
              f"{longer_chosen:.1%} of 3000 pairs")
        print("Held-out test: 800 pairs oriented by TRUE quality; length is independent of quality there,")
        print("so a length-unbiased model should show r_len ~ 0.\n")
        print(f"{'policy (trained on biased labels)':<36} {'r_len(Δlen, D_w-D_l)':>21} {'acc (true quality)':>19}")
        print("-" * 78)
        for k, name in (("dpo", "DPO  (sum log-prob, beta 0.1)"), ("simpo", "SimPO (avg log-prob, β 2, γ 0.5)")):
            d = diag[k]
            print(f"{name:<36} {d['r_len_dpo_margin']:>+21.3f} {d['acc_dpo_margin']:>19.1%}")
        print(f"{'SimPO, own reward margin β·Δavg':<36} {diag['simpo']['r_len_simpo_margin']:>+21.3f} "
              f"{diag['simpo']['acc_simpo_margin']:>19.1%}")
        print("-" * 78)
        print("\nWhat the policies actually say (240 sampled replies each):")
        print(f"{'model':<8} {'avg words':>10} {'filler/reply':>13} {'grounded share':>15}")
        for k in ("ref", "dpo", "simpo"):
            g = gen[k]
            print(f"{k:<8} {g['len']:>10.2f} {g['filler']:>13.2f} {g['grounded_share']:>15.1%}")
        for k, m in (("dpo", dpo), ("simpo", simpo)):
            print(f"  [{k:5}] {core.generate(m, tok, core.make_prompt('plasma'), max_new=20)}")

        print("\nWhere the length comes from: mean per-token log-prob by word type")
        print(f"{'model':<8} " + " ".join(f"{c:>10}" for c in cat["ref"]))
        for k in ("ref", "dpo", "simpo"):
            print(f"{k:<8} " + " ".join(f"{v:>10.2f}" for v in cat[k].values()))
        print("Reading:")
        print(f"* DPO's cheapest lever is the STOP token: log p('.') {cat['ref'][END]:.2f} -> "
              f"{cat['dpo'][END]:.2f}. It stops late, so it")
        print("  produces more filler and longer replies: length exploitation, learned from biased labels.")
        print(f"* SimPO's policy margin is far less length-driven (r_len {diag['simpo']['r_len_dpo_margin']:+.2f} vs "
              f"{diag['dpo']['r_len_dpo_margin']:+.2f}) and it emits LESS filler than the reference.")
        print(f"* But SimPO is not length-free: it ALSO pushes the stop token down ({cat['simpo'][END]:.2f}), and its")
        print("  own score divides that one-off cost by |y|, so long replies dilute it (r_len of its own margin")
        print(f"  {diag['simpo']['r_len_simpo_margin']:+.2f}). Average normalisation removes 'pay per token', "
              "not every length effect;")
        print("  real fixes also need de-biased labels or an explicit length penalty / length-controlled eval.")
    out.update(dpo=dpo, simpo=simpo, ref=ref, tok=tok)
    return out


# =============================================================================
# 5. Memory benchmark
# =============================================================================
def profile_memory_usage(num_params: int = 7_000_000_000, batch_size: int = 4, seq_len: int = 2048) -> Dict[str, float]:
    """
    Theoretical GPU memory allocation comparison between standard DPO and SimPO.
    Assumes FP16 / BF16 weights (2 bytes per parameter) and optimizer states (AdamW: 8 bytes per param).
    CHANGED vs starter: the frozen reference runs under torch.no_grad(), so it stores NO
    activations for backward -> activation term counted once, not twice. Also returns the
    weights-only comparison, which is where the "~50%" figure comes from.
    """
    bytes_per_param = 2
    model_weight_gb = (num_params * bytes_per_param) / (1024 ** 3)
    adam_states_gb = (num_params * 8) / (1024 ** 3)
    grad_states_gb = (num_params * 2) / (1024 ** 3)
    activation_estimate_gb = (batch_size * seq_len * 4096 * 32 * 2) / (1024 ** 3)

    dpo_memory_gb = model_weight_gb + grad_states_gb + adam_states_gb + activation_estimate_gb + model_weight_gb
    simpo_memory_gb = model_weight_gb + grad_states_gb + adam_states_gb + activation_estimate_gb
    savings_percent = ((dpo_memory_gb - simpo_memory_gb) / dpo_memory_gb) * 100

    return {
        "dpo_vram_gb": dpo_memory_gb,
        "simpo_vram_gb": simpo_memory_gb,
        "saved_vram_gb": dpo_memory_gb - simpo_memory_gb,
        "savings_percent": savings_percent,
        "weights_only_dpo_gb": 2 * model_weight_gb,
        "weights_only_simpo_gb": model_weight_gb,
    }


def _bytes(params) -> int:
    return sum(p.numel() * p.element_size() for p in params)


def measure_training_step(mode: str, d: int = 384, layers: int = 8, vocab: int = 8000,
                          batch: int = 8, seq: int = 256, device: str = "cpu") -> Dict[str, float]:
    """
    One real optimisation step. Returns MB of: policy weights, grads, AdamW state,
    reference weights, activations saved for backward, and (on CUDA) the allocator peak.
    """
    torch.manual_seed(0)
    dtype = torch.float32
    policy = core.TinyGPT(vocab, d=d, n_layers=layers, n_heads=8, max_len=seq).to(device, dtype)
    ref = core.freeze(copy.deepcopy(policy)) if mode == "dpo" else None
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    ids = torch.randint(3, vocab, (2 * batch, seq), device=device)   # [chosen; rejected]
    lab = ids.clone()
    lab[:, : seq // 4] = core.LABEL_PAD

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    saved = {}

    def pack(t):
        saved[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        avg = mode == "simpo"
        lp = core.get_batch_logps(policy(ids), lab, average_log_prob=avg)
    pw, pl = lp[:batch], lp[batch:]
    if mode == "dpo":
        with torch.no_grad():
            rlp = core.get_batch_logps(ref(ids), lab)
        loss, _ = core.DPOLoss(0.1)(pw, pl, rlp[:batch], rlp[batch:])
    else:
        loss, _ = SimPOLoss()(pw, pl)
    loss.backward()
    opt.step()
    mb = 2 ** 20
    out = {
        "weights": _bytes(policy.parameters()) / mb,
        "grads": _bytes(p.grad for p in policy.parameters()) / mb,
        "adam": sum(_bytes(v for v in s.values() if torch.is_tensor(v) and v.dim() > 0)
                    for s in opt.state.values()) / mb,
        "ref": _bytes(ref.parameters()) / mb if ref is not None else 0.0,
        "activations": sum(saved.values()) / mb,
    }
    out["total"] = sum(out.values())
    if device == "cuda":
        out["cuda_peak"] = torch.cuda.max_memory_allocated() / mb
    return out


def run_memory_benchmark(verbose: bool = True) -> Dict[str, Dict[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {m: measure_training_step(m, device=device) for m in ("dpo", "simpo")}
    if verbose:
        n_par = sum(p.numel() for p in core.TinyGPT(8000, d=384, n_layers=8, n_heads=8, max_len=256).parameters())
        print(f"Measured: one training step, {n_par / 1e6:.1f}M-param TinyGPT, fp32, 8 pairs x 256 tokens, "
              f"AdamW, device = {device.upper()}")
        print(f"{'MB':<34} {'DPO (policy+ref)':>17} {'SimPO (policy)':>15} {'saved':>8}")
        print("-" * 78)
        rows = [("policy weights", "weights"), ("policy gradients", "grads"), ("AdamW state (m, v)", "adam"),
                ("frozen reference weights", "ref"), ("activations saved for backward", "activations"),
                ("TOTAL", "total")] + ([("CUDA allocator peak", "cuda_peak")] if device == "cuda" else [])
        for name, k in rows:
            a, b = res["dpo"][k], res["simpo"][k]
            print(f"{name:<34} {a:>17,.1f} {b:>15,.1f} {(a - b) / a * 100 if a else 0:>7.1f}%")
        print("-" * 78)
        w = 2 * res["dpo"]["weights"]
        print(f"Model weights resident (policy + ref vs policy): {w:,.1f} -> {w / 2:,.1f} MB = 50.0% saved")
        print(f"Whole training step: {(res['dpo']['total'] - res['simpo']['total']) / res['dpo']['total'] * 100:.1f}% saved "
              "(gradients + AdamW state of the policy are the same in both)\n")

        print("Projection to a 7B model (starter assumptions: bf16 weights/grads, 8 B/param AdamW, batch 4 x 2048):")
        m = profile_memory_usage()
        print(f"  weights only     : DPO {m['weights_only_dpo_gb']:6.1f} GB | SimPO {m['weights_only_simpo_gb']:6.1f} GB "
              f"| saved 50.0%")
        print(f"  full training    : DPO {m['dpo_vram_gb']:6.1f} GB | SimPO {m['simpo_vram_gb']:6.1f} GB "
              f"| saved {m['savings_percent']:.1f}%")
        print("  -> 'about 50% less VRAM' is true for the MODELS held in memory (e.g. LoRA training, where")
        print("     optimiser state is tiny, gets close to it). For full fine-tuning it is ~14-16%.")
        print("  -> DPO can also precompute reference log-probs ONCE offline; then it needs no reference")
        print("     model at training time either, and the memory gap disappears.")
    return res


# =============================================================================
# 6. Tests
# =============================================================================
def test_pearson_and_simpo_basics() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    assert abs(calculate_pearson_correlation(xs, [2 * x + 1 for x in xs]) - 1) < 1e-12
    assert abs(calculate_pearson_correlation(xs, [-x for x in xs]) + 1) < 1e-12
    assert calculate_pearson_correlation(xs, [5.0] * 4) == 0.0
    a, b = torch.randn(500).tolist(), torch.randn(500).tolist()
    ref_r = torch.corrcoef(torch.tensor([a, b], dtype=torch.float64))[0, 1].item()
    assert abs(calculate_pearson_correlation(a, b) - ref_r) < 1e-9

    loss, m = SimPOLoss(2.0, 0.5)(torch.tensor([-0.15, -0.22, -0.18, -0.12]), torch.tensor([-0.45, -0.38, -0.52, -0.41]))
    z = 2.0 * (torch.tensor([-0.15, -0.22, -0.18, -0.12]) - torch.tensor([-0.45, -0.38, -0.52, -0.41])) - 0.5
    assert abs(loss.item() - (-F.logsigmoid(z)).mean().item()) < 1e-7
    # gamma: equal rewards still cost log(1 + e^gamma), the "target margin" pressure
    l0, _ = SimPOLoss(2.0, 0.5)(torch.zeros(1), torch.zeros(1))
    assert abs(l0.item() - math.log(1 + math.exp(0.5))) < 1e-6
    print(f"[PASS] Pearson (±1, 0, == torch), SimPO starter example (loss {loss.item():.4f}), "
          f"gamma margin at a tie = log(1+e^0.5)")


def test_length_normalisation() -> None:
    """Same per-token quality, 4x longer reply: summed log-prob 4x larger, SimPO reward identical."""
    logits = torch.zeros(1, 41, 10)
    logits[..., 3] = 2.0                         # every token: p(3) the same
    lab_short, lab_long = torch.full((1, 41), -100), torch.full((1, 41), -100)
    lab_short[0, 1:11] = 3
    lab_long[0, 1:41] = 3
    s, l = core.get_batch_logps(logits, lab_short), core.get_batch_logps(logits, lab_long)
    sa, la = (core.get_batch_logps(logits, x, average_log_prob=True) for x in (lab_short, lab_long))
    assert abs(l.item() / s.item() - 4) < 1e-6 and abs(sa.item() - la.item()) < 1e-7
    print(f"[PASS] length normalisation: sum log-prob {s.item():.2f} -> {l.item():.2f} (x4), "
          f"SimPO avg {sa.item():.3f} == {la.item():.3f}")


def plot_results(o, mem, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plot)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, c in (("dpo", "C0"), ("simpo", "C1")):
        d = o["diag"][k]
        jitter = (torch.rand(len(d["dlen"])) - 0.5) * 0.4
        ax[0].scatter(torch.tensor(d["dlen"]) + jitter, d["dpo_margin"], s=6, alpha=0.35, c=c,
                      label=f"{k.upper()} policy, r_len = {d['r_len_dpo_margin']:+.2f}")
    ax[0].axhline(0, c="grey", lw=0.8)
    ax[0].set_xlabel("|y_w| − |y_l|  (words)")
    ax[0].set_ylabel("D_w − D_l  (log-ratio margin vs ref)")
    ax[0].set_title("(a) Does the margin follow length?", fontweight="bold")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    names = ["ref", "dpo", "simpo"]
    x = range(3)
    ax[1].bar([i - 0.2 for i in x], [o["gen"][n]["len"] for n in names], 0.4, label="avg words / reply")
    ax[1].bar([i + 0.2 for i in x], [o["gen"][n]["filler"] for n in names], 0.4, label="filler words / reply")
    ax[1].set_xticks(list(x), ["reference", "DPO", "SimPO"])
    ax[1].set_title("(b) Generated verbosity", fontweight="bold")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")

    keys = ["weights", "grads", "adam", "ref", "activations"]
    labels = ["policy weights", "gradients", "AdamW state", "frozen reference", "saved activations"]
    bottom = [0.0, 0.0]
    for k, lab in zip(keys, labels):
        vals = [mem["dpo"][k], mem["simpo"][k]]
        ax[2].bar(["DPO", "SimPO"], vals, bottom=bottom, label=lab)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax[2].set_ylabel("MB (measured, one step)")
    ax[2].set_title(f"(c) Memory: {(1 - mem['simpo']['total'] / mem['dpo']['total']) * 100:.0f}% saved",
                    fontweight="bold")
    ax[2].legend(fontsize=8)
    fig.suptitle("Mission 04 / Part 3 — length bias and reference-free SimPO", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[SUCCESS] Plot saved to: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mission 04 — Part 3: length bias & SimPO")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    print("=" * 80)
    print("USSE NOOSPHERE • LENGTH BIAS AUDIT & REFERENCE-FREE SimPO CORE")
    print("Role Deliverable: Part 3 — Data & Payload Specialist")
    print("=" * 80 + "\n")
    test_pearson_and_simpo_basics()
    test_length_normalisation()
    print("\n=== 1-2. Length bias: DPO vs SimPO on verbosity-biased labels ===")
    o = run_length_experiment()
    print("\n=== 3. Memory benchmark ===")
    mem = run_memory_benchmark()
    if not args.no_plot:
        plot_results(o, mem, os.path.join(os.path.dirname(os.path.abspath(__file__)), "length_bias_simpo.png"))
    print("=" * 80)


if __name__ == "__main__":
    main()
