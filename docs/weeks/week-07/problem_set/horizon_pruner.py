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
