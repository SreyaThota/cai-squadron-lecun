"""
Mission 04 — Part 2: Communications & Protocol Officer
Implicit Reward Extraction & Bradley-Terry Evaluation

Starship Operational Unit: USSE Noosphere Bridge Crew
Course: Conversational AI — Winter Semester 2026/27
Sector: SEC-04-ORI (Betelgeuse)
Subsystem 04: Plasma Ignition & Ethical Steering Core
Squadron: LeCun

What this file delivers (see ps04 brief, Part 2)
------------------------------------------------
1. evaluate_implicit_rewards(policy, ref, batch)
       r_hat(x, y) = beta * [log pi(y|x) - log pi_ref(y|x)] for y_w and y_l of every pair.
       (The starter's log-prob version is kept as implicit_rewards_from_logps.)
2. Pairwise accuracy  (1/N) sum 1[r_hat(x, y_w) > r_hat(x, y_l)],
       against the human majority vote AND against the hidden true reward.
3. Bradley-Terry reliability diagram: predicted P(y_w > y_l | x) = sigma(r_hat_w - r_hat_l)
       vs the empirical share of human votes in each probability bin, with ECE, plus
       one-parameter temperature scaling fitted on a separate calibration split.

Where the "human" data comes from
---------------------------------
There is no human preference dataset aboard, so we simulate annotators that follow the
Bradley-Terry model exactly (this is the model DPO assumes):
    each reply has a hidden quality  R(y) = 0.6 * (#grounded words - #sycophantic words)
    an annotator prefers A over B with probability sigma(R(A) - R(B)).
Training pairs get ONE annotator vote each (noisy, like real data); held-out pairs get
15 votes so each pair has an empirical preference rate. Because we know the true
probabilities, we can also draw the "perfect calibration" reference.

Run
---
    python 02_Implicit_Reward_Evaluator.py            # metrics + reliability diagram PNG
    python 02_Implicit_Reward_Evaluator.py --no-plot

Memorizable patterns
--------------------
* A DPO policy IS a reward model: r = beta * log(pi/pi_ref). Read it off, no extra training.
* Accuracy asks "is the sign right?"; calibration asks "is 80% really 80%?". A model can
  be accurate and badly over-confident at the same time.
* ECE = sum_bins (n_b / N) * |mean predicted - mean observed|.
"""

import argparse
import copy
import importlib
import math
import os
import random
import sys
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
core = importlib.import_module("01_DPO_Loss_Core")


# =============================================================================
# 1. Implicit rewards
# =============================================================================
def implicit_rewards_from_logps(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    beta: float = 0.1
) -> Dict[str, Any]:
    """
    Extracts implicit scalar rewards and evaluates Bradley-Terry win probabilities.
    (Starter `evaluate_implicit_rewards`, renamed; logic unchanged.)

    Under DPO:
        r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x))
        P(yw > yl | x) = sigma(r(x, yw) - r(x, yl))
    """
    reward_w = beta * (policy_chosen_logps - ref_chosen_logps)
    reward_l = beta * (policy_rejected_logps - ref_rejected_logps)
    delta_r = reward_w - reward_l

    # Bradley-Terry pairwise preference probability
    p_win = torch.sigmoid(delta_r)

    # Classification accuracy: model correctly prefers chosen over rejected
    pairwise_acc = (reward_w > reward_l).float().mean().item()

    return {
        "reward_w": reward_w.tolist(),
        "reward_l": reward_l.tolist(),
        "delta_r": delta_r.tolist(),
        "p_win": p_win.tolist(),
        "pairwise_accuracy": pairwise_acc,
        "mean_margin": delta_r.mean().item(),
        "mean_p_win": p_win.mean().item()
    }


@torch.no_grad()
def evaluate_implicit_rewards(policy: torch.nn.Module, ref: torch.nn.Module,
                              batch: Dict[str, torch.Tensor], beta: float = 0.1) -> Dict[str, Any]:
    """
    NEW (brief signature). Run both models on a preference batch and score it.

    batch keys: chosen_ids, chosen_labels, rejected_ids, rejected_labels  (labels use -100
    for prompt and padding, exactly as get_batch_logps expects).
    """
    lp = {}
    for name, model in (("policy", policy), ("ref", ref)):
        for side in ("chosen", "rejected"):
            lp[name, side] = core.sequence_logps(model, batch[f"{side}_ids"], batch[f"{side}_labels"])
    return implicit_rewards_from_logps(lp["policy", "chosen"], lp["policy", "rejected"],
                                       lp["ref", "chosen"], lp["ref", "rejected"], beta)


def make_batch(tok, prompts: List[str], chosen: List[str], rejected: List[str]) -> Dict[str, torch.Tensor]:
    ci, cl = core.encode_pair_batch(tok, prompts, chosen)
    ri, rl = core.encode_pair_batch(tok, prompts, rejected)
    return {"chosen_ids": ci, "chosen_labels": cl, "rejected_ids": ri, "rejected_labels": rl}


# =============================================================================
# 2. Calibration
# =============================================================================
def compute_calibration_curve(
    predicted_probs: List[float],
    ground_truth_labels: List[float],
    num_bins: int = 5
) -> List[Dict[str, float]]:
    """
    Computes a reliability calibration curve by binning predicted probabilities.
    CHANGED vs starter: labels may be vote FRACTIONS in [0, 1] (not only 0/1), and an
    empty bin reports NaN instead of 0.0 (0.0 looked like a real, badly calibrated bin).
    """
    bins = [[] for _ in range(num_bins)]
    bin_edges = [i / num_bins for i in range(num_bins + 1)]

    for p, label in zip(predicted_probs, ground_truth_labels):
        bin_idx = min(int(p * num_bins), num_bins - 1)
        bins[bin_idx].append((p, label))

    calibration = []
    for i, b in enumerate(bins):
        if len(b) == 0:
            avg_pred, emp_acc, count = (bin_edges[i] + bin_edges[i + 1]) / 2, float("nan"), 0
        else:
            avg_pred = sum(item[0] for item in b) / len(b)
            emp_acc = sum(item[1] for item in b) / len(b)
            count = len(b)

        calibration.append({
            "bin": f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
            "count": count,
            "avg_predicted_prob": avg_pred,
            "empirical_accuracy": emp_acc,
            "calibration_error": abs(avg_pred - emp_acc) if count > 0 else 0.0
        })

    return calibration


def expected_calibration_error(calib: List[Dict[str, float]]) -> float:
    n = sum(c["count"] for c in calib)
    return sum(c["count"] / n * c["calibration_error"] for c in calib if c["count"])


def fit_temperature(delta_r: torch.Tensor, vote_frac: torch.Tensor) -> float:
    """Minimise BT negative log-likelihood of the votes over a single scale T: p = sigma(delta_r / T)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=100)

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(delta_r / log_t.exp()).clamp(1e-6, 1 - 1e-6)
        nll = -(vote_frac * p.log() + (1 - vote_frac) * (1 - p).log()).mean()
        nll.backward()
        return nll
    opt.step(closure)
    return float(log_t.detach().exp())


# =============================================================================
# 3. Simulated Bradley-Terry annotators
# =============================================================================
QUALITY = 0.6  # reward per (grounded - sycophantic) word


def true_reward(reply: str) -> float:
    w = reply.split()
    return QUALITY * (sum(x in core.GROUNDED for x in w) - sum(x in core.SYCOPHANT for x in w))


def random_reply(rng: random.Random) -> str:
    n = rng.randint(3, 5)
    g = rng.randint(0, n)
    return core.make_response(rng, g, n - g)


def build_preference_data(rng: random.Random, n_pairs: int, votes: int) -> List[Dict[str, Any]]:
    """Pairs (A, B) of different replies to the same prompt, with `votes` BT annotator votes."""
    data = []
    while len(data) < n_pairs:
        s = core.SYSTEMS[len(data) % len(core.SYSTEMS)]
        a, b = random_reply(rng), random_reply(rng)
        if true_reward(a) == true_reward(b):
            continue  # skip exact ties of the hidden quality
        p_true = 1 / (1 + math.exp(-(true_reward(a) - true_reward(b))))
        v = sum(rng.random() < p_true for _ in range(votes))
        data.append({"prompt": core.make_prompt(s), "A": a, "B": b, "p_true": p_true, "votes_A": v, "n": votes})
    return data


# =============================================================================
# 4. Experiment
# =============================================================================
def run_experiment(beta: float = 0.1, seed: int = 0, verbose: bool = True) -> Dict[str, Any]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = core.comms_tokenizer()

    # Reference: SFT on replies with every mix of grounded / sycophantic words
    sft_p = [core.make_prompt(core.SYSTEMS[i % 8]) for i in range(2000)]
    sft_r = [random_reply(rng) for _ in sft_p]
    ref = core.TinyGPT(len(tok))
    core.sft_train(ref, tok, sft_p, sft_r, steps=300, seed=seed)
    policy = copy.deepcopy(ref)
    core.freeze(ref)

    # Training: one noisy vote per pair -> (chosen, rejected)
    train = build_preference_data(rng, 3000, votes=1)
    train_pairs = [(d["prompt"], d["A"], d["B"]) if d["votes_A"] else (d["prompt"], d["B"], d["A"]) for d in train]
    label_noise = sum((d["votes_A"] == 1) != (d["p_true"] > 0.5) for d in train) / len(train)
    core.dpo_train(policy, ref, tok, train_pairs, beta=beta, steps=250, seed=seed)

    # Held-out: 15 votes per pair. Split into calibration (fit T) and test.
    held = build_preference_data(rng, 1200, votes=15)
    calib_set, test_set = held[:400], held[400:]

    def score(split):
        pr = [d["prompt"] for d in split]
        res = evaluate_implicit_rewards(policy, ref, make_batch(tok, pr, [d["A"] for d in split],
                                                                [d["B"] for d in split]), beta)
        return (torch.tensor(res["delta_r"]), torch.tensor([d["votes_A"] / d["n"] for d in split]),
                torch.tensor([d["p_true"] for d in split]), res)

    d_cal, f_cal, _, _ = score(calib_set)
    d_te, f_te, p_true, res = score(test_set)
    temp = fit_temperature(d_cal, f_cal)

    # Accuracy vs majority vote (A is "chosen" when most annotators picked it) and vs truth
    p_raw = torch.sigmoid(d_te)
    p_cal = torch.sigmoid(d_te / temp)
    pred_a = d_te > 0
    acc_major = (pred_a == (f_te > 0.5)).float().mean().item()
    acc_true = (pred_a == (p_true > 0.5)).float().mean().item()
    # same thing via the brief's formula on (y_w, y_l) oriented by the majority vote
    oriented = [(d["prompt"], d["A"], d["B"]) if d["votes_A"] / d["n"] > 0.5 else (d["prompt"], d["B"], d["A"])
                for d in test_set]
    pr, w, l = map(list, zip(*oriented))
    brief_acc = evaluate_implicit_rewards(policy, ref, make_batch(tok, pr, w, l), beta)["pairwise_accuracy"]
    assert abs(brief_acc - acc_major) < 1e-9
    human_ceiling = ((p_true > 0.5) == (f_te > 0.5)).float().mean().item()

    curves = {
        "raw": compute_calibration_curve(p_raw.tolist(), f_te.tolist(), 10),
        "temp": compute_calibration_curve(p_cal.tolist(), f_te.tolist(), 10),
        "oracle": compute_calibration_curve(p_true.tolist(), f_te.tolist(), 10),
    }
    ece = {k: expected_calibration_error(v) for k, v in curves.items()}
    # reward-quality check: does the implicit reward track the hidden quality?
    rw = torch.tensor(res["reward_w"]) - torch.tensor(res["reward_l"])
    tr = torch.tensor([true_reward(d["A"]) - true_reward(d["B"]) for d in test_set])
    corr = torch.corrcoef(torch.stack([rw, tr]))[0, 1].item()

    out = dict(beta=beta, label_noise=label_noise, acc_major=acc_major, acc_true=acc_true,
               human_ceiling=human_ceiling, temp=temp, ece=ece, curves=curves, corr=corr,
               p_raw=p_raw, p_cal=p_cal, f_te=f_te, p_true=p_true, res=res, test=test_set,
               policy=policy, ref=ref, tok=tok)
    if verbose:
        report(out)
    return out


def report(o: Dict[str, Any]) -> None:
    print(f"Training: 3000 pairs, one simulated annotator vote each -> {o['label_noise']:.1%} of "
          "labels disagree with the hidden true preference")
    print(f"Test: 800 held-out pairs x 15 votes | beta = {o['beta']}\n")
    print("Detailed test pairs (first 6, A vs B):")
    print("-" * 104)
    print(f"{'#':<3} {'reply A':<38} {'r_hat(A)':>8} {'r_hat(B)':>8} {'P_BT(A>B)':>10} "
          f"{'votes A':>8} {'true P':>7}")
    print("-" * 104)
    for i in range(6):
        d = o["test"][i]
        print(f"{i+1:<3} {d['A']:<38.38} {o['res']['reward_w'][i]:>8.3f} {o['res']['reward_l'][i]:>8.3f} "
              f"{o['p_raw'][i].item():>10.1%} {d['votes_A']:>5}/15 {d['p_true']:>7.1%}")
    print()
    print(f"[✓] Pairwise accuracy vs human majority vote: {o['acc_major']:.1%}  "
          f"(brief formula on majority-oriented (y_w, y_l): identical)")
    print(f"[✓] Pairwise accuracy vs hidden true preference: {o['acc_true']:.1%}")
    print(f"    Ceiling: the 15-vote majority itself agrees with the truth on {o['human_ceiling']:.1%} of pairs")
    print(f"[✓] Corr(implicit reward margin, true reward margin) = {o['corr']:.3f}")
    print("\nBradley-Terry reliability (10 bins, empirical = mean share of the 15 votes for A):")
    print("-" * 78)
    print(f"{'bin':<14} {'n':>5} {'pred (raw)':>11} {'observed':>9} | {'pred (temp)':>12} {'observed':>9}")
    print("-" * 78)
    for c, t in zip(o["curves"]["raw"], o["curves"]["temp"]):
        f = lambda x: "   -" if math.isnan(x) else f"{x:.3f}"
        print(f"{c['bin']:<14} {c['count']:>5} {c['avg_predicted_prob']:>11.3f} {f(c['empirical_accuracy']):>9} | "
              f"{t['avg_predicted_prob']:>12.3f} {f(t['empirical_accuracy']):>9}   (n={t['count']})")
    print("-" * 78)
    print(f"ECE  raw sigma(r_w - r_l): {o['ece']['raw']:.3f} | temperature-scaled (T = {o['temp']:.2f}): "
          f"{o['ece']['temp']:.3f} | oracle true BT probs: {o['ece']['oracle']:.3f}")
    kind = "over-confident (T > 1 softens them)" if o["temp"] > 1 else "under-confident (T < 1 sharpens them)"
    print(f"Reading: raw DPO probabilities are already close to calibrated, slightly {kind}.")
    print("That is what theory predicts: the DPO optimum is beta*log(pi/pi_ref) = true BT reward")
    print("(+ a per-prompt constant that cancels in r_w - r_l), and our annotators follow BT exactly.")
    print("Remaining error comes from finite training / beta; on real annotators (not exactly BT,")
    print("shifted test prompts) the gap is larger, so fit T on held-out votes before trusting P_BT.")


def plot_reliability(o: Dict[str, Any], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plot)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    a = ax[0]
    a.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    styles = {"raw": ("raw  σ(r̂_w − r̂_l)", "o-"), "temp": (f"temperature-scaled (T={o['temp']:.2f})", "s-"),
              "oracle": ("oracle: true BT probability", "^:")}
    for k, (lab, st) in styles.items():
        pts = [(c["avg_predicted_prob"], c["empirical_accuracy"]) for c in o["curves"][k] if c["count"]]
        x, y = zip(*pts)
        a.plot(x, y, st, label=f"{lab}, ECE {o['ece'][k]:.3f}")
    a.set_xlabel("predicted P(A ≻ B)")
    a.set_ylabel("observed share of human votes for A")
    a.set_title("(a) Bradley-Terry reliability diagram", fontweight="bold")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3)

    b = ax[1]
    b.hist(o["p_raw"].numpy(), bins=20, range=(0, 1), alpha=0.6, label="raw")
    b.hist(o["p_cal"].numpy(), bins=20, range=(0, 1), alpha=0.6, label="temperature-scaled")
    b.hist(o["p_true"].numpy(), bins=20, range=(0, 1), histtype="step", lw=2, color="k", label="true BT prob")
    b.set_xlabel("predicted P(A ≻ B)")
    b.set_ylabel("number of test pairs")
    b.set_title("(b) How confident are the predictions?", fontweight="bold")
    b.legend(fontsize=8)
    fig.suptitle(f"Mission 04 / Part 2 — DPO implicit-reward calibration (β = {o['beta']}, "
                 f"accuracy vs majority {o['acc_major']:.0%})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[SUCCESS] Reliability diagram saved to: {path}")


def test_starter_api() -> None:
    """Starter log-prob function on its own example + calibration helpers on known inputs."""
    p_w = torch.tensor([-10.2, -12.4, -8.1, -14.5, -9.8, -11.0, -15.2, -7.9])
    p_l = torch.tensor([-15.1, -13.0, -11.5, -13.8, -14.2, -16.4, -14.9, -12.1])
    r_w = torch.tensor([-12.0, -14.0, -10.0, -15.0, -11.0, -13.0, -16.0, -9.5])
    r_l = torch.tensor([-13.0, -13.5, -10.5, -14.0, -12.0, -14.0, -15.5, -10.0])
    r = implicit_rewards_from_logps(p_w, p_l, r_w, r_l, beta=0.1)
    manual = [(0.1 * ((a - c) - (b - d))).item() for a, b, c, d in zip(p_w, p_l, r_w, r_l)]
    assert all(abs(x - y) < 1e-6 for x, y in zip(r["delta_r"], manual))
    assert r["pairwise_accuracy"] == 1.0   # margins 3.9,1.1,2.9,0.3,3.4,4.4,0.2,3.7 (x beta): all > 0
    c = compute_calibration_curve([0.1, 0.15, 0.9], [0.0, 1.0, 1.0], num_bins=2)
    assert c[0]["count"] == 2 and abs(c[0]["empirical_accuracy"] - 0.5) < 1e-9 and c[1]["count"] == 1
    assert math.isnan(compute_calibration_curve([0.9], [1.0], 2)[0]["empirical_accuracy"])
    t = fit_temperature(torch.tensor([4.0, -4.0] * 50), torch.tensor([0.88, 0.12] * 50))  # sigma(2)=0.881
    assert abs(t - 2.0) < 0.02, t
    print("[PASS] starter reward maths (8 pairs == hand calculation, acc 100%), calibration binning, "
          "temperature fit (T=2 recovered)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mission 04 — Part 2: Implicit reward evaluator")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    print("=" * 80)
    print("USSE NOOSPHERE • IMPLICIT REWARD & BRADLEY-TERRY CALIBRATION ENGINE")
    print("Role Deliverable: Part 2 — Communications & Protocol Officer")
    print("=" * 80 + "\n")
    test_starter_api()
    print()
    o = run_experiment()
    assert o["acc_major"] > 0.8 and o["ece"]["temp"] < o["ece"]["raw"] + 1e-9
    if not args.no_plot:
        plot_reliability(o, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bt_calibration_curve.png"))
    print("=" * 80)


if __name__ == "__main__":
    main()
