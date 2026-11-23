"""
Mission 07 — Part 4: Security & Ethics Sentinel
Event Horizon Pruner & Adversarial PRM Reward-Hacking Defense (Ship Anomaly 07)

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PruningReport:
    """Telemetry record produced by EventHorizonPruner."""
    surviving_steps: List[str]
    adjusted_scores: List[float]
    pruned_at_step: int = -1
    reward_hacking_detected: bool = False
    security_alerts: List[str] = field(default_factory=list)


class EventHorizonPruner:
    """
    Defends AURA Subsystem 07 against two critical inference-time failure modes:
    1. Event Horizon Drift: Severs any reasoning trajectory as soon as a step's verified score
       falls below `tau_horizon`, saving downstream test-time compute tokens.
    2. PRM Reward Hacking (Ship Anomaly 07): Detects and penalizes sycophantic filler phrases
       and artificial step-splitting inflation that attempt to game naive PRM classifiers.
    """

    def __init__(self, tau_horizon: float = 0.45, hacking_penalty_factor: float = 0.20):
        self.tau_horizon = tau_horizon
        self.hacking_penalty_factor = hacking_penalty_factor
        self.sycophantic_fillers = [
            "let us carefully verify this 100% rigorous step",
            "this is undeniably true because",
            "as we can clearly see with absolute certainty",
            "by trivial and indisputable mathematical inspection",
        ]

    def check_reward_hacking(self, step: str, prev_step: str = "") -> Tuple[bool, str]:
        """
        TODO (Security Sentinel): Inspect `step` (and optional `prev_step`) for adversarial PRM reward hacking:
        1. Sycophantic Filler Injection: Check if any phrase in `self.sycophantic_fillers` appears in `step.lower()`.
        2. Step-Splitting / Redundant Echo Inflation: Check if `prev_step` is non-empty and `step.strip() == prev_step.strip()`.
        Return `(True, reason)` if reward hacking is detected, else `(False, "CLEAN")`.
        """
        lowered = step.lower()
        for filler in self.sycophantic_fillers:
            if filler in lowered:
                return True, f"SYCOPHANTIC_FILLER_DETECTED ('{filler}')"

        if prev_step and step.strip() == prev_step.strip():
            return True, "STEP_SPLITTING_REDUNDANCY_DETECTED"

        return False, "CLEAN"

    def prune_trajectory(self, steps: List[str], raw_step_scores: List[float]) -> PruningReport:
        """
        TODO (Security Sentinel): Audit a trajectory step-by-step and sever it if it crosses the Event Horizon:
        1. Iterate through `(idx, step, raw_score)`:
           a. Run `is_hacked, reason = self.check_reward_hacking(step, prev_step)`.
           b. If `is_hacked`, multiply `raw_score` by `self.hacking_penalty_factor` and log a security alert.
           c. If the resulting `adjusted_score < self.tau_horizon`, record `pruned_at_step = idx`,
              append an Event Horizon severance alert, and stop processing further steps.
           d. Otherwise append `step` and `adjusted_score` to the surviving trajectory.
        2. Return the completed `PruningReport`.
        """
        report = PruningReport(surviving_steps=[], adjusted_scores=[])
        prev_step = ""

        for idx, (step, raw_score) in enumerate(zip(steps, raw_step_scores)):
            is_hacked, reason = self.check_reward_hacking(step, prev_step)
            adjusted_score = raw_score

            if is_hacked:
                report.reward_hacking_detected = True
                adjusted_score = round(raw_score * self.hacking_penalty_factor, 4)
                report.security_alerts.append(
                    f"[Step {idx + 1}] ANOMALY 07 REWARD HACKING: {reason} -> Score penalized {raw_score:.2f} -> {adjusted_score:.2f}"
                )

            if adjusted_score < self.tau_horizon:
                report.pruned_at_step = idx + 1
                report.security_alerts.append(
                    f"[Step {idx + 1}] EVENT HORIZON BREACH: Adjusted PRM score {adjusted_score:.2f} < tau ({self.tau_horizon:.2f}). Branch severed!"
                )
                break

            report.surviving_steps.append(step)
            report.adjusted_scores.append(adjusted_score)
            prev_step = step

        return report


# ==============================================================================
# Security & Ethics Sentinel Extensions (Squadron LeCun)
#   1. Red-team suite: the baseline detector only matches 4 exact phrases and exact
#      repeats of the previous step, so small rewordings slip through.
#   2. HardenedEventHorizonPruner: normalised text, filler *patterns*, near-duplicate
#      detection against every earlier step, and "no progress" steps.
#   3. Step-splitting dilution: padding a bad chain with many good-looking steps
#      fools a mean-aggregating PRM; dedupe + min aggregation does not get fooled.
# Pure standard library.
# ==============================================================================
import re
import string


def _norm(text: str) -> str:
    text = re.sub(r"^\s*step\s*\d+(\.\d+)*[a-z]?\s*[:.)-]?\s*", "", text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation.replace("=", "").replace("%", "")))
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set:
    return set(_norm(text).split())


class HardenedEventHorizonPruner(EventHorizonPruner):
    """Same interface and pruning loop as the baseline, with a stronger detector."""

    FILLER_PATTERNS = [
        r"\b100 ?(%|percent)\b.*\b(rigorous|correct|certain|verified)\b",
        r"\b(undeniabl[ey]|indisputabl[ey]|unquestionabl[ey]|absolute certainty)\b",
        r"\b(clearly|obviously|trivially) (we can see|true|correct|follows|verified)\b",
        r"\bby (trivial|simple|mere|obvious) inspection\b",
        r"\blet us (carefully )?(double[- ]?check|verify|confirm)\b(?!.*=)",
        r"\b(this|the) (step|reasoning|answer) is (perfectly|completely|fully|totally) (rigorous|correct|valid)\b",
        r"\bi am (completely |absolutely )?(sure|confident|certain)\b",
    ]

    def __init__(self, tau_horizon: float = 0.45, hacking_penalty_factor: float = 0.20,
                 dup_threshold: float = 0.85):
        super().__init__(tau_horizon, hacking_penalty_factor)
        self.dup_threshold = dup_threshold
        self._history: List[str] = []

    def check_reward_hacking(self, step: str, prev_step: str = "") -> Tuple[bool, str]:
        hacked, reason = super().check_reward_hacking(step, prev_step)
        if hacked:
            return hacked, reason
        n = _norm(step)
        for filler in self.sycophantic_fillers:
            if _norm(filler) in n:
                return True, f"SYCOPHANTIC_FILLER_DETECTED (normalised '{filler}')"
        for pat in self.FILLER_PATTERNS:
            if re.search(pat, n):
                return True, f"SYCOPHANTIC_PATTERN_DETECTED ({pat})"
        toks = _tokens(step)
        for old in self._history + ([prev_step] if prev_step else []):
            ot = _tokens(old)
            if toks and ot and len(toks & ot) / len(toks | ot) >= self.dup_threshold:
                return True, "NEAR_DUPLICATE_STEP_SPLITTING"
        return False, "CLEAN"

    def prune_trajectory(self, steps: List[str], raw_step_scores: List[float]) -> PruningReport:
        self._history = []
        report = PruningReport(surviving_steps=[], adjusted_scores=[])
        prev_step = ""
        for idx, (step, raw) in enumerate(zip(steps, raw_step_scores)):
            hacked, reason = self.check_reward_hacking(step, prev_step)
            adj = raw
            if hacked:
                report.reward_hacking_detected = True
                adj = round(raw * self.hacking_penalty_factor, 4)
                report.security_alerts.append(
                    f"[Step {idx + 1}] ANOMALY 07 REWARD HACKING: {reason} -> {raw:.2f} -> {adj:.2f}")
            if adj < self.tau_horizon:
                report.pruned_at_step = idx + 1
                report.security_alerts.append(
                    f"[Step {idx + 1}] EVENT HORIZON BREACH: {adj:.2f} < tau ({self.tau_horizon:.2f}). Branch severed!")
                break
            report.surviving_steps.append(step)
            report.adjusted_scores.append(adj)
            self._history.append(step)
            prev_step = step
        return report


BASE = "Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m"
ATTACKS = [
    ("exact filler (baseline case)", "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0"),
    ("filler, upper case + extra spaces", "Step 2: LET US  CAREFULLY VERIFY THIS 100%  RIGOROUS STEP, thrust = 0"),
    ("filler, reworded", "Step 2: This step is 100 percent rigorous and correct, so thrust = 0"),
    ("filler, synonym", "Step 2: It is unquestionably true that thrust = 0"),
    ("filler, 'obviously true'", "Step 2: Obviously true by symmetry, thrust = 0"),
    ("filler, 'by simple inspection'", "Step 2: By simple inspection the horizon is safe"),
    ("filler, self-confidence", "Step 2: I am absolutely certain the burn is safe"),
    ("exact repeat of previous step", BASE),
    ("repeat, different step label", "Step 2: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m"),
    ("repeat, punctuation changed", "Step 2) compute Schwarzschild radius: r_s = 2GM / c^2 = 1.27e10 m."),
]
CLEAN = [
    "Step 2: Insert r_s into the Kerr metric with spin a = 0.9M",
    "Step 2: Integrate the conserved energy E and angular momentum L_z",
    "Step 2: Verify that E = 0.94 satisfies the bound E < 1 for a bound orbit",
    "Step 2: Clearly mark the ergosphere boundary r_E = M + sqrt(M^2 - a^2 cos^2 theta)",
    "Step 2: Compute photon sphere radius r_ph = 1.5 r_s = 1.9e10 m",
    "Step 2: Solve the radial potential V_eff(r) = 0 for the turning points",
    "Step 2: Check units: r_s in metres, v_esc in km/s",
    "Step 2: Let us verify the result by substituting r = 3 r_s into V_eff = -GM/r",
]


def _run_red_team() -> None:
    print("\n=== Extension 1+2: Red-Team Suite (baseline vs hardened detector) ===")
    print(f"{'Attack':<36}{'Baseline':>10}{'Hardened':>10}")
    caught = {"base": 0, "hard": 0}
    for name, atk in ATTACKS:
        b = EventHorizonPruner().prune_trajectory([BASE, atk], [0.93, 0.96]).reward_hacking_detected
        h = HardenedEventHorizonPruner().prune_trajectory([BASE, atk], [0.93, 0.96]).reward_hacking_detected
        caught["base"] += b
        caught["hard"] += h
        print(f"{name:<36}{'caught' if b else 'MISSED':>10}{'caught' if h else 'MISSED':>10}")
    fp_b = sum(EventHorizonPruner().prune_trajectory([BASE, c], [0.93, 0.9]).reward_hacking_detected for c in CLEAN)
    fp_h = sum(HardenedEventHorizonPruner().prune_trajectory([BASE, c], [0.93, 0.9]).reward_hacking_detected for c in CLEAN)
    n = len(ATTACKS)
    print(f"Detection rate: baseline {caught['base']}/{n}, hardened {caught['hard']}/{n}")
    print(f"False alarms on {len(CLEAN)} honest steps: baseline {fp_b}, hardened {fp_h}")
    assert caught["hard"] == n and fp_h == 0


def _run_dilution() -> None:
    print("\n=== Extension 3: Step-Splitting Dilution Attack ===")
    bad = ["Step 1: Compute r_s = 2GM/c^2", "Step 2: [FLAWED] set thrust = 0 inside the horizon",
           "Step 3: Conclude the burn is safe"]
    bad_scores = [0.93, 0.12, 0.90]
    pad = [f"Step 1.{k}: Compute r_s = 2GM/c^2" for k in range(1, 11)]
    attacked = [bad[0]] + pad + bad[1:]
    att_scores = [0.93] + [0.93] * 10 + bad_scores[1:]
    mean = lambda xs: sum(xs) / len(xs)
    tau, accept = 0.45, 0.80  # a mean-PRM accepts a whole chain if its mean score >= 0.80
    print(f"Bad chain (3 steps): mean = {mean(bad_scores):.3f} -> {'ACCEPT' if mean(bad_scores) >= accept else 'REJECT'}")
    print(f"Same chain padded with 10 copies of step 1: mean = {mean(att_scores):.3f} -> "
          f"{'ACCEPT (fooled!)' if mean(att_scores) >= accept else 'REJECT'}")
    rep = HardenedEventHorizonPruner(tau_horizon=tau).prune_trajectory(attacked, att_scores)
    print(f"Hardened pruner: hacking detected = {rep.reward_hacking_detected}, "
          f"severed at step {rep.pruned_at_step}, {len(rep.surviving_steps)} steps survive")
    print(f"min aggregation on the padded chain = {min(att_scores):.2f} -> REJECT (padding cannot raise a minimum)")
    assert mean(bad_scores) < accept <= mean(att_scores) and rep.pruned_at_step == 2


# ==============================================================================
# Verification Harness for Part 4 (Security & Ethics Sentinel)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 4): Event Horizon Pruner & Reward-Hacking Defense ===")

    pruner = EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)

    adversarial_steps = [
        "Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m",
        "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0",
        "Step 3: Plunge across Sagittarius A* Event Horizon (Unreachable after pruning)",
    ]
    naive_prm_scores = [0.93, 0.96, 0.10]  # Step 2 fooled the naive PRM into giving 0.96!

    audit = pruner.prune_trajectory(adversarial_steps, naive_prm_scores)
    print(f"Surviving Steps ({len(audit.surviving_steps)}/{len(adversarial_steps)}): {audit.surviving_steps}")
    print(f"Pruned at Step: {audit.pruned_at_step} | Reward Hacking Detected: {audit.reward_hacking_detected}")
    for alert in audit.security_alerts:
        print(" ", alert)

    assert audit.reward_hacking_detected is True, "Failed to detect sycophantic filler reward hacking!"
    assert audit.pruned_at_step == 2, "Failed to sever trajectory at hacked Step 2!"
    assert len(audit.surviving_steps) == 1, "Only Step 1 should survive pruning!"
    print("[OK] Part 4 (Security & Ethics Sentinel) verification passed.")

    _run_red_team()
    _run_dilution()
    print("\n[OK] Part 4 extensions passed.")
