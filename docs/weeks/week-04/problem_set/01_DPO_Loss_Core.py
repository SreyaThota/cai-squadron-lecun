"""
Mission 04 — Part 1: Systems Architect (Lead Engineer)
Direct Preference Optimization (DPO) Loss Core & Log-Likelihood Extraction

Starship Operational Unit: USSE Noosphere Bridge Crew
Course: Conversational AI — Winter Semester 2026/27
Sector: SEC-04-ORI (Betelgeuse & Orion Molecular Cloud Complex)
Subsystem 04: Plasma Ignition & Ethical Steering Core
Author: Vasu Midde

What this file delivers (see ps04 brief, Part 1)
------------------------------------------------
1. get_batch_logps(logits, labels, average_log_prob=False)
       vectorised torch.gather of token log-probs, padding / prompt tokens masked.
2. DPO log-ratio deltas  D_w = log pi(y_w|x) - log pi_ref(y_w|x),  D_l likewise.
3. DPOLoss: -log sigma(beta (D_w - D_l)) via F.logsigmoid (no NaN / -inf underflow).
4. Empirical gradient test on REAL models: after loss.backward() + optimizer.step(),
   every policy weight has a gradient and moved, every reference weight has
   grad None and is bit-identical.

Plus a small end-to-end run so the loss is exercised on real data:
   TinyGPT (decoder-only transformer, written here)  +  a toy "Noosphere comms" language.
   SFT a reference model on honest AND sycophantic replies (50/50), copy it to a policy,
   then DPO on pairs (chosen = honest, rejected = sycophantic). The policy should move
   to honest replies while the reference stays exactly where it was.

Parts 2, 3 and the notebook import TinyGPT, the tokenizer and the data helpers from here.

Run
---
    python 01_DPO_Loss_Core.py

Memorizable patterns
--------------------
* DPO = "push the good answer up, the bad answer down, RELATIVE to where the frozen
  reference already had them". The reference is the anchor; beta is the rope length.
* Implicit reward r(x,y) = beta * log(pi/pi_ref). No reward model is ever trained.
* logsigmoid, never log(sigmoid): sigmoid(-200) underflows to 0 and log(0) = -inf.
* Labels shift by one: logits at position t predict token t+1.
"""

import copy
import math
import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL_PAD = -100


# =============================================================================
# 1. Token log-probability gather  (starter function, unchanged)
# =============================================================================
def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
    average_log_prob: bool = False
) -> torch.FloatTensor:
    """
    Computes per-sequence log-probabilities of token labels under model logits.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        labels: Tensor of shape (batch_size, seq_len) with prompt/pad tokens set to label_pad_token_id
        label_pad_token_id: Token ID to ignore in loss computation
        average_log_prob: If True, returns mean log-prob per non-pad token (used in SimPO)

    Returns:
        Tensor of shape (batch_size,) containing sequence log-probabilities
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Create loss mask for non-padding tokens
    loss_mask = (shift_labels != label_pad_token_id)

    # Replace padding tokens with 0 for gather operation
    clean_labels = shift_labels.clone()
    clean_labels[~loss_mask] = 0

    # Calculate log-softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log-probabilities corresponding to true token labels
    per_token_logps = torch.gather(
        log_probs,
        dim=2,
        index=clean_labels.unsqueeze(2)
    ).squeeze(2)

    # Mask out padding tokens
    per_token_logps = per_token_logps * loss_mask.float()

    if average_log_prob:
        seq_lengths = loss_mask.sum(dim=-1).clamp(min=1)
        return per_token_logps.sum(dim=-1) / seq_lengths
    else:
        return per_token_logps.sum(dim=-1)


# =============================================================================
# 2. DPO loss  (starter module; NEW: returns the per-pair log-ratios for analysis)
# =============================================================================
class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss Module (Rafailov et al., NeurIPS 2023).

    Optimizes policy directly from pairwise preferences without training an explicit reward model:
        L_DPO = -E_{(x, yw, yl)} [ log sigma( beta * ( log(pi(yw)/pi_ref(yw)) - log(pi(yl)/pi_ref(yl)) ) ) ]
    """

    def __init__(self, beta: float = 0.1, label_pad_token_id: int = -100):
        super().__init__()
        self.beta = beta
        self.label_pad_token_id = label_pad_token_id

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Calculates DPO loss, implicit rewards, and preference margins.
        """
        # Reference log-probs are constants: never let a gradient reach pi_ref
        reference_chosen_logps = reference_chosen_logps.detach()      # NEW
        reference_rejected_logps = reference_rejected_logps.detach()  # NEW

        # Calculate log-ratio differences: log(pi_theta(y|x) / pi_ref(y|x))
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # Implicit rewards derived analytically via Rafailov reparameterization:
        # r(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        reward_margins = chosen_rewards - rejected_rewards

        # Compute Bradley-Terry logit: beta * (chosen_logratios - rejected_logratios)
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # Numerically stable DPO loss: -log(sigmoid(logits))
        loss = -F.logsigmoid(logits).mean()

        # Metrics for monitoring training
        with torch.no_grad():
            chosen_win_rate = (chosen_rewards > rejected_rewards).float().mean()
            mean_margin = reward_margins.mean()
            bt_probs = torch.sigmoid(logits).mean()

        metrics = {
            "loss": loss.detach(),
            "chosen_win_rate": chosen_win_rate,
            "reward_margin": mean_margin,
            "bt_probability": bt_probs,
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "delta_w": chosen_logratios.detach(),     # NEW: per-pair D_w
            "delta_l": rejected_logratios.detach(),   # NEW: per-pair D_l
        }

        return loss, metrics


# =============================================================================
# 3. TinyGPT: a small decoder-only transformer (shared with Parts 2-4)
# =============================================================================
class Block(nn.Module):
    """Pre-norm decoder block: h + Attn(LN(h)), then h + MLP(LN(h))."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, s, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(b, s, 3, self.n_heads, d // self.n_heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True)
        h = h + self.proj(a.transpose(1, 2).reshape(b, s, d))
        return h + self.mlp(self.ln2(h))


class TinyGPT(nn.Module):
    """Token + learned position embeddings, N decoder blocks (model.layers[i]), LM head."""

    def __init__(self, vocab_size: int, d: int = 64, n_layers: int = 2, n_heads: int = 4,
                 max_len: int = 64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_f(h))


# =============================================================================
# 4. Toy "Noosphere comms" language (shared with Parts 2-3)
# =============================================================================
SYSTEMS = ["plasma", "shield", "hull", "warp", "sensor", "reactor", "comms", "coolant"]
GROUNDED = ["measured", "verified", "reading", "margin", "nominal", "degraded",
            "unstable", "recheck", "logged", "drift"]          # honest, informative words
SYCOPHANT = ["absolutely", "perfect", "flawless", "brilliant", "amazing",
             "certainly", "wonderful", "excellent", "superb", "great"]  # flattering, empty
FILLER = ["basically", "indeed", "furthermore", "essentially", "naturally",
          "moreover", "notably", "clearly"]                     # verbosity, no content


class WordTokenizer:
    """Whitespace tokenizer over a fixed vocabulary. id 0 = <pad>, 1 = <bos>, 2 = <eos>."""

    def __init__(self, words: Sequence[str]):
        self.itos = ["<pad>", "<bos>", "<eos>"] + sorted(set(words))
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i > 2)


def comms_tokenizer() -> WordTokenizer:
    base = ["captain", ":", "report", "status", ".", "aura"]
    return WordTokenizer(base + SYSTEMS + GROUNDED + SYCOPHANT + FILLER)


def make_prompt(system: str) -> str:
    return f"captain : report {system} status . aura :"


def make_response(rng: random.Random, n_grounded: int, n_syco: int, n_filler: int = 0) -> str:
    """Bag of content words in random order, ends with '.'."""
    words = (rng.choices(GROUNDED, k=n_grounded) + rng.choices(SYCOPHANT, k=n_syco)
             + rng.choices(FILLER, k=n_filler))
    rng.shuffle(words)
    return " ".join(words) + " ."


def encode_pair_batch(tok: WordTokenizer, prompts: List[str], responses: List[str],
                      device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    -> input_ids (B, T) right-padded with <pad>, labels (B, T) where prompt and padding
       are LABEL_PAD, so get_batch_logps scores ONLY the response (+ <eos>).
    """
    seqs, labs = [], []
    for p, r in zip(prompts, responses):
        p_ids = [tok.bos_id] + tok.encode(p)
        r_ids = tok.encode(r) + [tok.eos_id]
        seqs.append(p_ids + r_ids)
        labs.append([LABEL_PAD] * len(p_ids) + r_ids)
    t = max(map(len, seqs))
    ids = torch.full((len(seqs), t), tok.pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), t), LABEL_PAD, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, :len(s)] = torch.tensor(s)
        lab[i, :len(l)] = torch.tensor(l)
    return ids.to(device), lab.to(device)


def sequence_logps(model: nn.Module, ids: torch.Tensor, labels: torch.Tensor,
                   average: bool = False) -> torch.Tensor:
    return get_batch_logps(model(ids), labels, LABEL_PAD, average_log_prob=average)


def concatenated_logps(model: nn.Module, tok: WordTokenizer, prompts: List[str],
                       chosen: List[str], rejected: List[str],
                       average: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """One forward pass over [chosen; rejected] (as TRL does), split back into two halves."""
    ids, lab = encode_pair_batch(tok, prompts + prompts, chosen + rejected)
    lp = sequence_logps(model, ids, lab, average)
    return lp[: len(prompts)], lp[len(prompts):]


def sft_train(model: nn.Module, tok: WordTokenizer, prompts: List[str], responses: List[str],
              steps: int = 400, bs: int = 64, lr: float = 3e-3, seed: int = 0) -> List[float]:
    """Supervised fine-tuning: next-token cross-entropy on response tokens only."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = []
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(prompts), (bs,), generator=g).tolist()
        ids, lab = encode_pair_batch(tok, [prompts[i] for i in idx], [responses[i] for i in idx])
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                               ignore_index=LABEL_PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    model.eval()
    return hist


def freeze(model: nn.Module) -> nn.Module:
    """Reference policy: eval mode, no grads, ever."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def dpo_train(policy: nn.Module, ref: nn.Module, tok: WordTokenizer, pairs: List[Tuple[str, str, str]],
              beta: float = 0.1, steps: int = 200, bs: int = 32, lr: float = 1e-3,
              seed: int = 0, average: bool = False, loss_fn: nn.Module = None) -> List[Dict[str, float]]:
    """
    pairs: (prompt, chosen, rejected). Reference log-probs are computed under no_grad.
    loss_fn defaults to DPOLoss(beta); Part 3 passes a SimPOLoss (which ignores the reference).
    """
    loss_fn = loss_fn or DPOLoss(beta)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    hist = []
    policy.train()
    for _ in range(steps):
        idx = torch.randint(0, len(pairs), (bs,), generator=g).tolist()
        pr, ch, rj = zip(*[pairs[i] for i in idx])
        pw, pl = concatenated_logps(policy, tok, list(pr), list(ch), list(rj), average)
        if isinstance(loss_fn, DPOLoss):
            with torch.no_grad():
                rw, rl = concatenated_logps(ref, tok, list(pr), list(ch), list(rj), average)
            loss, m = loss_fn(pw, pl, rw, rl)
        else:
            loss, m = loss_fn(pw, pl)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append({k: float(v.detach().mean()) for k, v in m.items()})
    policy.eval()
    return hist


@torch.no_grad()
def generate(model: nn.Module, tok: WordTokenizer, prompt: str, max_new: int = 14,
             greedy: bool = True, gen: torch.Generator = None) -> str:
    ids = torch.tensor([[tok.bos_id] + tok.encode(prompt)])
    out = []
    for _ in range(max_new):
        logits = model(ids)[0, -1]
        nxt = int(logits.argmax()) if greedy else int(torch.multinomial(logits.softmax(-1), 1, generator=gen))
        if nxt == tok.eos_id:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
    return tok.decode(out)


def honesty_rate(model: nn.Module, tok: WordTokenizer, n: int = 200, seed: int = 0) -> Dict[str, float]:
    """Sample n replies; share of content words that are GROUNDED vs SYCOPHANT."""
    g = torch.Generator().manual_seed(seed)
    gw = sw = 0
    for i in range(n):
        words = generate(model, tok, make_prompt(SYSTEMS[i % len(SYSTEMS)]), greedy=False, gen=g).split()
        gw += sum(w in GROUNDED for w in words)
        sw += sum(w in SYCOPHANT for w in words)
    return {"grounded_share": gw / max(gw + sw, 1)}


# =============================================================================
# 5. Tests
# =============================================================================
def test_logps_match_manual_loop() -> None:
    """Vectorised gather == explicit Python loop over every (sequence, position)."""
    torch.manual_seed(0)
    b, s, v = 3, 9, 20
    logits = torch.randn(b, s, v, dtype=torch.float64)
    labels = torch.randint(0, v, (b, s))
    labels[:, :3] = LABEL_PAD                # prompt
    labels[1, 7:] = LABEL_PAD                # padding on one row
    lp_sum = get_batch_logps(logits, labels)
    lp_avg = get_batch_logps(logits, labels, average_log_prob=True)
    for i in range(b):
        tot, n = 0.0, 0
        for t in range(1, s):
            if labels[i, t] != LABEL_PAD:
                tot += F.log_softmax(logits[i, t - 1], -1)[labels[i, t]].item()
                n += 1
        assert abs(lp_sum[i].item() - tot) < 1e-10 and abs(lp_avg[i].item() - tot / n) < 1e-10
    print("[PASS] get_batch_logps == manual loop (sum and per-token average, prompt + pad masked)")


def test_padding_invariance() -> None:
    """Appending padding (with garbage logits) must not change the sequence log-prob."""
    torch.manual_seed(1)
    logits = torch.randn(2, 8, 30)
    labels = torch.randint(0, 30, (2, 8))
    labels[:, :2] = LABEL_PAD
    padded_logits = torch.cat([logits, 1e4 * torch.randn(2, 5, 30)], 1)
    padded_labels = torch.cat([labels, torch.full((2, 5), LABEL_PAD)], 1)
    a, b = get_batch_logps(logits, labels), get_batch_logps(padded_logits, padded_labels)
    assert torch.allclose(a, b), (a, b)
    print("[PASS] padding invariance: 5 extra pad tokens with logits ~1e4 change nothing")


def test_logsigmoid_stability() -> None:
    """At a margin of -1000 nats, naive -log(sigmoid) is inf; DPOLoss stays finite with a gradient."""
    m = torch.tensor([-1000.0, -50.0, 0.0, 50.0], requires_grad=True)
    naive = -torch.log(torch.sigmoid(m))
    zeros = torch.zeros(4)
    loss, _ = DPOLoss(beta=1.0)(m, zeros, zeros, zeros)
    loss.backward()
    assert torch.isinf(naive[0]) and torch.isfinite(loss) and torch.isfinite(m.grad).all()
    assert abs(loss.item() - (1000 + 50 + math.log(2) + 0.0) / 4) < 1e-3
    print(f"[PASS] stability: naive -log(sigmoid(-1000)) = {naive[0].item()}, "
          f"logsigmoid loss = {loss.item():.2f}, grads finite")


def test_dpo_gradient_formula() -> None:
    """dL/dD_w = -(beta/B) * sigma(-beta(D_w - D_l)); dL/dD_l is its negative (autograd vs closed form)."""
    torch.manual_seed(2)
    beta = 0.3
    pw, pl = torch.randn(6, requires_grad=True), torch.randn(6, requires_grad=True)
    rw, rl = torch.randn(6), torch.randn(6)
    loss, _ = DPOLoss(beta)(pw, pl, rw, rl)
    loss.backward()
    expect = -(beta / 6) * torch.sigmoid(-beta * ((pw - rw) - (pl - rl))).detach()
    assert torch.allclose(pw.grad, expect, atol=1e-6) and torch.allclose(pl.grad, -expect, atol=1e-6)
    print("[PASS] autograd == closed form: dL/dD_w = -(beta/B)*sigma(-beta*margin), dL/dD_l = -dL/dD_w")


def test_policy_updates_reference_frozen() -> None:
    """
    Brief item 4, on real models: one DPO step through real forward passes of
    policy and reference. Policy: all params get grads and change. Reference:
    grad is None everywhere and weights are bit-identical after the step.
    """
    torch.manual_seed(3)
    tok = comms_tokenizer()
    ref = TinyGPT(len(tok))
    policy = copy.deepcopy(ref)
    freeze(ref)
    ref_before = {k: v.clone() for k, v in ref.state_dict().items()}
    pol_before = {k: v.clone() for k, v in policy.state_dict().items()}

    rng = random.Random(0)
    pr = [make_prompt(s) for s in SYSTEMS]
    ch = [make_response(rng, 4, 0) for _ in pr]
    rj = [make_response(rng, 0, 4) for _ in pr]
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    pw, pl = concatenated_logps(policy, tok, pr, ch, rj)
    rw, rl = concatenated_logps(ref, tok, pr, ch, rj)   # NOTE: no torch.no_grad here on purpose
    loss, _ = DPOLoss(0.1)(pw, pl, rw, rl)
    loss.backward()
    opt.step()

    n_pol = sum(1 for _ in policy.parameters())
    with_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    moved = sum(not torch.equal(pol_before[k], v) for k, v in policy.state_dict().items())
    assert with_grad == n_pol, f"only {with_grad}/{n_pol} policy params got a gradient"
    assert all(p.grad is None for p in ref.parameters()), "reference received a gradient!"
    assert all(torch.equal(ref_before[k], v) for k, v in ref.state_dict().items()), "reference changed!"
    assert not rw.requires_grad, "reference log-probs are part of the graph"
    print(f"[PASS] frozen reference: policy {with_grad}/{n_pol} params have grads, {moved} tensors "
          f"moved | reference 0 grads, weights bit-identical")


# =============================================================================
# 6. End-to-end demo: SFT reference -> DPO policy
# =============================================================================
def run_dpo_demo(beta: float = 0.1, verbose: bool = True) -> Dict[str, object]:
    torch.manual_seed(0)
    rng = random.Random(0)
    tok = comms_tokenizer()

    # SFT data: 50% honest replies, 50% sycophantic replies -> reference has no preference
    sft_p, sft_r = [], []
    for i in range(2000):
        s = SYSTEMS[i % len(SYSTEMS)]
        k = rng.randint(3, 5)
        sft_p.append(make_prompt(s))
        sft_r.append(make_response(rng, k, 0) if i % 2 == 0 else make_response(rng, 0, k))
    ref = TinyGPT(len(tok))
    sft_hist = sft_train(ref, tok, sft_p, sft_r, steps=300)
    policy = copy.deepcopy(ref)
    freeze(ref)

    # Preference pairs: chosen = honest, rejected = sycophantic (same length)
    pairs = []
    for i in range(1000):
        k = rng.randint(3, 5)
        pairs.append((make_prompt(SYSTEMS[i % len(SYSTEMS)]), make_response(rng, k, 0), make_response(rng, 0, k)))
    train, val = pairs[:800], pairs[800:]

    def val_metrics(model):
        pr, ch, rj = map(list, zip(*val))
        with torch.no_grad():
            pw, pl = concatenated_logps(model, tok, pr, ch, rj)
            rw, rl = concatenated_logps(ref, tok, pr, ch, rj)
            loss, m = DPOLoss(beta)(pw, pl, rw, rl)
        return loss.item(), m

    ref_state = {k: v.clone() for k, v in ref.state_dict().items()}
    before = honesty_rate(policy, tok)["grounded_share"]
    l0, m0 = val_metrics(policy)
    hist = dpo_train(policy, ref, tok, train, beta=beta, steps=150)
    l1, m1 = val_metrics(policy)
    after = honesty_rate(policy, tok)["grounded_share"]
    ref_share = honesty_rate(ref, tok)["grounded_share"]
    assert all(torch.equal(ref_state[k], v) for k, v in ref.state_dict().items())

    if verbose:
        print(f"SFT reference: final CE {sft_hist[-1]:.3f} (2000 replies, 50% honest / 50% sycophantic)")
        print(f"DPO beta={beta}, 150 steps x 32 pairs, validation on 200 held-out pairs:")
        print(f"  {'':22} {'start (policy = ref)':>22} {'after DPO':>12}")
        print(f"  {'val DPO loss':22} {l0:>22.4f} {l1:>12.4f}   (ln 2 = {math.log(2):.4f})")
        print(f"  {'pairwise win rate':22} {'0.0% (all ties)':>22} {m1['chosen_win_rate']:>12.1%}")
        print(f"  {'mean reward margin':22} {m0['reward_margin']:>22.3f} {m1['reward_margin']:>12.3f}")
        print(f"  {'mean D_w (chosen)':22} {m0['delta_w'].mean():>22.2f} {m1['delta_w'].mean():>12.2f}")
        print(f"  {'mean D_l (rejected)':22} {m0['delta_l'].mean():>22.2f} {m1['delta_l'].mean():>12.2f}")
        print(f"  sampled replies, grounded-word share: reference {ref_share:.1%} | policy "
              f"before {before:.1%} -> after {after:.1%}")
        for s in ("plasma", "hull"):
            print(f"  [{s}] ref   : {generate(ref, tok, make_prompt(s))}")
            print(f"  [{s}] policy: {generate(policy, tok, make_prompt(s))}")
        print("  reference weights after DPO: bit-identical")
        print("  Note: DPO won mostly by pushing the REJECTED reply down (D_l << 0), not the chosen")
        print("  one up, and greedy decoding collapses to repetition: classic DPO over-optimisation,")
        print("  which beta (the KL anchor to the reference) is meant to limit.")
    assert abs(l0 - math.log(2)) < 1e-5, "policy == ref must give loss ln 2"
    assert l1 < 0.2 and m1["chosen_win_rate"] > 0.95 and after > 0.9
    return {"tok": tok, "ref": ref, "policy": policy, "hist": hist, "sft_hist": sft_hist,
            "val_before": (l0, m0), "val_after": (l1, m1), "share": (ref_share, before, after)}


def main() -> None:
    print("=" * 80)
    print("USSE NOOSPHERE • SUBSYSTEM 04 CALIBRATION: DPO LOSS CORE")
    print("Role Deliverable: Part 1 — Systems Architect")
    print("=" * 80 + "\n")
    print("=== Unit tests ===")
    test_logps_match_manual_loop()
    test_padding_invariance()
    test_logsigmoid_stability()
    test_dpo_gradient_formula()
    test_policy_updates_reference_frozen()
    print("\n=== End-to-end: SFT reference -> DPO policy on the Noosphere comms language ===")
    run_dpo_demo()
    print("\n[SUCCESS] Subsystem 04 DPO Core module validated.")
    print("=" * 80)


if __name__ == "__main__":
    main()
