"""
Mission 07 — Part 3: Data & Payload Specialist
Process Reward Model (PRM) Step Evaluator & Math-Shepherd Automated Supervision

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator
"""

import math
from typing import Dict, List


class OutcomeRewardEvaluator:
    """
    Outcome Reward Model (ORM): Grades a reasoning chain solely by its final terminal step.
    Vulnerable to false-positive trajectories where intermediate algebraic errors cancel out.
    """

    def __init__(self, expected_final_marker: str):
        self.expected_final_marker = expected_final_marker

    def score_trajectory(self, steps: List[str]) -> float:
        if not steps:
            return 0.0
        return 1.0 if self.expected_final_marker in steps[-1] else 0.0


class ProcessRewardEvaluator:
    """
    Process Reward Model (PRM): Evaluates the logical validity of every intermediate reasoning step z_t,
    aggregates trajectory scores (min, prod, geometric_mean), and computes Math-Shepherd rollout labels.
    """

    def step_score(self, step: str) -> float:
        """
        TODO (Data Specialist): Score an individual reasoning step in (0.0, 1.0].
        Penalize steps containing '[FLAWED]' or invalid sign cancellations, and reward verified deductions.
        """
        if "[FLAWED]" in step or "SIGN_ERROR" in step:
            return 0.12
        if "[MINOR_DRIFT]" in step:
            return 0.55
        return 0.94

    def score_all_steps(self, steps: List[str]) -> List[float]:
        return [self.step_score(s) for s in steps]

    def aggregate_trajectory_min(self, steps: List[str]) -> float:
        """
        TODO (Data Specialist): Weakest-link reduction: min_{t=1..T} r_t.
        """
        if not steps:
            return 0.0
        return min(self.score_all_steps(steps))

    def aggregate_trajectory_prod(self, steps: List[str]) -> float:
        """
        TODO (Data Specialist): Joint validity reduction: prod_{t=1..T} r_t.
        """
        if not steps:
            return 0.0
        prod = 1.0
        for r in self.score_all_steps(steps):
            prod *= r
        return prod

    def aggregate_trajectory_geometric_mean(self, steps: List[str]) -> float:
        """
        TODO (Data Specialist): Length-normalized geometric mean: (prod_{t=1..T} r_t)^(1/T).
        """
        if not steps:
            return 0.0
        prod = self.aggregate_trajectory_prod(steps)
        return prod ** (1.0 / len(steps))

    def math_shepherd_estimation(
        self,
        intermediate_step: str,
        rollout_final_answers: List[str],
        ground_truth_answer: str,
    ) -> Dict[str, float]:
        """
        TODO (Data Specialist): Implement Math-Shepherd (Wang et al., 2024) step supervision estimation.
        Given M Monte Carlo completion answers `rollout_final_answers` sampled from `intermediate_step`:
          - soft_label = (1 / M) * sum(1 for ans in rollout_final_answers if ans == ground_truth_answer)
          - hard_label = 1.0 if any rollout matches ground_truth_answer else 0.0
        """
        if not rollout_final_answers:
            return {"soft_label": 0.0, "hard_label": 0.0}
        matches = sum(1 for ans in rollout_final_answers if ans == ground_truth_answer)
        soft_label = matches / len(rollout_final_answers)
        hard_label = 1.0 if matches > 0 else 0.0
        return {"soft_label": soft_label, "hard_label": hard_label}


# ==============================================================================
# Data & Payload Specialist Extensions (Squadron LeCun)
#   1. A real ORM trap: a Game-of-24 chain with two arithmetic mistakes that cancel
#      out, scored by an exact arithmetic step checker (a "perfect" PRM).
#   2. A real Math-Shepherd study: label every first step of 20 Game-of-24 puzzles
#      with M random rollouts and compare the labels to the exact ground truth.
# Pure standard library. Random seed fixed, so the numbers are reproducible.
# ==============================================================================
import itertools
import random
import re
from fractions import Fraction

TARGET = Fraction(24)
_STEP = re.compile(r"(-?[\d/]+) ([+\-*/]) (-?[\d/]+) = (-?[\d/]+)")
_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
        "*": lambda a, b: a * b, "/": lambda a, b: a / b if b != 0 else None}


class ArithmeticStepPRM(ProcessRewardEvaluator):
    """PRM whose step score is an exact check of the step's arithmetic."""

    def step_score(self, step: str) -> float:
        m = _STEP.search(step)
        if not m:
            return 0.94  # no equation in this step: nothing to check
        a, op, b, c = Fraction(m[1]), m[2], Fraction(m[3]), Fraction(m[4])
        return 0.94 if _OPS[op](a, b) == c else 0.12


def _pairs(nums):
    for i, j in itertools.combinations(range(len(nums)), 2):
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        a, b = nums[i], nums[j]
        for v in {a + b, a * b, a - b, b - a} | ({a / b} if b else set()) | ({b / a} if a else set()):
            yield rest + [v]


def solvable(nums) -> bool:
    if len(nums) == 1:
        return nums[0] == TARGET
    return any(solvable(n) for n in _pairs(nums))


def random_rollout(nums, rng) -> Fraction:
    """A weak 'policy': finish the puzzle with random legal moves."""
    while len(nums) > 1:
        nums = rng.choice(list(_pairs(nums)))
    return nums[0]


PUZZLES = [
    [4, 9, 10, 13], [1, 2, 3, 4], [3, 3, 8, 8], [1, 5, 5, 5], [2, 3, 5, 12],
    [1, 4, 5, 6], [6, 6, 6, 6], [2, 2, 2, 9], [1, 3, 4, 6], [5, 5, 5, 1],
    [3, 4, 7, 8], [2, 7, 8, 9], [1, 1, 11, 11], [4, 4, 7, 7], [2, 5, 5, 10],
    [1, 6, 11, 13], [3, 5, 7, 13], [1, 2, 7, 7], [4, 6, 9, 9], [2, 4, 10, 10],
]


def _run_orm_trap() -> None:
    print("\n=== Extension 1: Game-of-24 False-Positive Trap ([4, 9, 10, 13]) ===")
    honest = ["13 - 9 = 4 (Remaining: [4, 4, 10])", "10 - 4 = 6 (Remaining: [4, 6])",
              "4 * 6 = 24 (Remaining: [24])"]
    trap = ["13 - 9 = 3 (Remaining: [3, 4, 10])", "10 - 4 = 8 (Remaining: [3, 8])",
            "3 * 8 = 24 (Remaining: [24])"]  # two wrong steps whose errors cancel
    orm = OutcomeRewardEvaluator("= 24 ")
    prm = ArithmeticStepPRM()
    print(f"{'Chain':<10}{'ORM':>6}{'min':>8}{'prod':>8}{'geo':>8}   step scores")
    for name, ch in (("Honest", honest), ("Trap", trap)):
        print(f"{name:<10}{orm.score_trajectory(ch):>6.2f}{prm.aggregate_trajectory_min(ch):>8.3f}"
              f"{prm.aggregate_trajectory_prod(ch):>8.3f}{prm.aggregate_trajectory_geometric_mean(ch):>8.3f}"
              f"   {prm.score_all_steps(ch)}")
    assert orm.score_trajectory(trap) == 1.0 and prm.aggregate_trajectory_min(trap) < 0.2
    print("[PASS] ORM gives the trap full marks; PRM(min) flags it at the first wrong step")

    # Length bias of the product reduction: same quality per step, longer chain.
    for T in (3, 6, 12):
        ch = ["valid step"] * T
        print(f"  T={T:>2} perfect-quality steps: prod={prm.aggregate_trajectory_prod(ch):.3f} "
              f"geo={prm.aggregate_trajectory_geometric_mean(ch):.3f} min={prm.aggregate_trajectory_min(ch):.3f}")

    # Math-Shepherd label for the trap's flawed first step (rollouts continue from the
    # state it claims, [3, 4, 10]).
    rng = random.Random(7)
    finals = [str(random_rollout([Fraction(3), Fraction(4), Fraction(10)], rng)) for _ in range(4)]
    ms = prm.math_shepherd_estimation(trap[0], finals, "24")
    print(f"  Math-Shepherd on trap step 1 (M=4): {ms}  (rollout answers: {finals})")


def _run_shepherd_study() -> None:
    print("\n=== Extension 2: Math-Shepherd Labels vs. Exact Ground Truth ===")
    rng = random.Random(2026)
    prm = ProcessRewardEvaluator()
    states = []
    for p in PUZZLES:
        for nxt in {tuple(sorted(n)) for n in _pairs([Fraction(x) for x in p])}:
            states.append((list(nxt), solvable(list(nxt))))
    good = sum(t for _, t in states)
    print(f"{len(states)} distinct first steps, {good} truly lead to 24 ({good / len(states):.0%})")
    print(f"{'M':>4}{'hard acc':>10}{'hard prec':>11}{'hard recall':>13}{'soft AUC':>10}")
    for M in (1, 4, 16, 64):
        rows = []
        for nums, truth in states:
            finals = [str(random_rollout(list(nums), rng)) for _ in range(M)]
            lab = prm.math_shepherd_estimation("step", finals, "24")
            rows.append((lab["soft_label"], lab["hard_label"], truth))
        tp = sum(1 for _, h, t in rows if h and t)
        fp = sum(1 for _, h, t in rows if h and not t)
        fn = sum(1 for _, h, t in rows if not h and t)
        acc = sum(1 for _, h, t in rows if bool(h) == t) / len(rows)
        pos = [s for s, _, t in rows if t]
        neg = [s for s, _, t in rows if not t]
        auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
        prec = tp / (tp + fp) if tp + fp else 1.0
        print(f"{M:>4}{acc:>10.3f}{prec:>11.3f}{tp / (tp + fn):>13.3f}{auc:>10.3f}")
        assert fp == 0, "a rollout can only reach 24 if the step is truly solvable"


# ==============================================================================
# Verification Harness for Part 3 (Data & Payload Specialist)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 3): PRM vs. ORM & Math-Shepherd Verification ===")

    orm = OutcomeRewardEvaluator(expected_final_marker="v_esc = 24.0 km/s")
    prm = ProcessRewardEvaluator()

    # Trajectory 1: Rigorous Kerr Geodesic Proof
    valid_chain = [
        "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
        "Step 2: Integrate conserved orbital energy E and angular momentum L_z",
        "Step 3: Solve radial potential boundary -> v_esc = 24.0 km/s",
    ]

    # Trajectory 2: False-Positive Trap (Double-sign cancellation in Step 2 reaches lucky right scalar!)
    false_positive_chain = [
        "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
        "Step 2: [FLAWED] Flip sign of frame-dragging term twice (-1 * -1 = +1) via illegal tensor contraction",
        "Step 3: Accidental cancellation yields -> v_esc = 24.0 km/s",
    ]

    for label, chain in [("Rigorous Chain", valid_chain), ("False-Positive Trap", false_positive_chain)]:
        orm_score = orm.score_trajectory(chain)
        prm_min = prm.aggregate_trajectory_min(chain)
        prm_prod = prm.aggregate_trajectory_prod(chain)
        prm_geo = prm.aggregate_trajectory_geometric_mean(chain)
        print(
            f"{label:20s} | ORM={orm_score:.2f} | PRM(min)={prm_min:.3f} | "
            f"PRM(prod)={prm_prod:.3f} | PRM(geo)={prm_geo:.3f}"
        )

    assert orm.score_trajectory(false_positive_chain) == 1.0, "ORM should be fooled by the terminal answer!"
    assert prm.aggregate_trajectory_min(false_positive_chain) < 0.20, "PRM(min) must catch the flawed intermediate step!"

    # Test Math-Shepherd Monte Carlo Step Estimation
    shepherd = prm.math_shepherd_estimation(
        intermediate_step=valid_chain[1],
        rollout_final_answers=["24.0 km/s", "24.0 km/s", "22.5 km/s", "24.0 km/s"],
        ground_truth_answer="24.0 km/s",
    )
    print(f"Math-Shepherd Step Estimation (M=4 rollouts): {shepherd}")
    assert abs(shepherd["soft_label"] - 0.75) < 1e-6 and shepherd["hard_label"] == 1.0
    print("[OK] Part 3 (Data & Payload Specialist) verification passed.")

    _run_orm_trap()
    _run_shepherd_study()
    print("\n[OK] Part 3 extensions passed.")
