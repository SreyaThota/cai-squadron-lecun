"""
Mission 06 — Part 3: Data & Payload Specialist
MDP Trajectory Logger & Empirical Return Evaluator

Sector SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop

1. Trajectory logging   tau = (s0, a0, r0, s1, a1, r1, ..., sT)
2. Discounted return    G0 = sum_t gamma^t * r_t        (gamma = 0.95)
3. Transition matrix    P(s' | s, a) = count(s, a, s') / count(s, a)
                        estimated from a historical dataset of past jumps.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

GAMMA = 0.95


# ---------------------------------------------------------------------------
# 1. Trajectory logger
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    t: int
    state: Hashable
    action: Any
    reward: float
    next_state: Hashable


@dataclass
class TrajectoryLogger:
    """Records one episode. Call start(s0), then log(a, r, s_next) each step."""

    states: List[Hashable] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    done: bool = False

    def start(self, initial_state: Hashable) -> None:
        self.states, self.actions, self.rewards, self.done = [initial_state], [], [], False

    def log(self, action: Any, reward: float, next_state: Hashable, done: bool = False) -> None:
        if not self.states:
            raise RuntimeError("call start(s0) before log()")
        if self.done:
            raise RuntimeError("episode already finished")
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.states.append(next_state)
        self.done = done

    @property
    def T(self) -> int:
        return len(self.actions)

    def transitions(self) -> List[Transition]:
        return [Transition(t, self.states[t], self.actions[t], self.rewards[t], self.states[t + 1])
                for t in range(self.T)]

    def as_tau(self) -> Tuple:
        """Flat tuple (s0, a0, r0, s1, a1, r1, ..., sT)."""
        out: List[Any] = []
        for t in range(self.T):
            out += [self.states[t], self.actions[t], self.rewards[t]]
        out.append(self.states[-1])
        return tuple(out)

    def discounted_return(self, gamma: float = GAMMA) -> float:
        return discounted_return(self.rewards, gamma)

    def returns_to_go(self, gamma: float = GAMMA) -> List[float]:
        """G_t for every t (useful to see long-term vs short-term value)."""
        g, out = 0.0, []
        for r in reversed(self.rewards):
            g = r + gamma * g
            out.append(g)
        return out[::-1]

    def pretty(self) -> str:
        lines = [f"s0 = {self.states[0]}"]
        for tr in self.transitions():
            lines.append(f"t={tr.t}: a={tr.action!s:<34} r={tr.reward:+6.1f}  -> s{tr.t + 1} = {tr.next_state}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Discounted return
# ---------------------------------------------------------------------------
def discounted_return(rewards: Sequence[float], gamma: float = GAMMA) -> float:
    """G0 = sum_{t=0}^{T} gamma^t * r_t"""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    return float(sum((gamma ** t) * r for t, r in enumerate(rewards)))


# ---------------------------------------------------------------------------
# 3. Empirical transition frequency matrix  P(s' | s, a)
# ---------------------------------------------------------------------------
# Historical jump records through the Homunculus Nebula: (sector, action, resulting sector).
HISTORICAL_JUMPS: List[Tuple[str, str, str]] = (
    [("outer_shell", "warp_escape_corridor", "escape_corridor")] * 14
    + [("outer_shell", "warp_escape_corridor", "dust_lane")] * 4
    + [("outer_shell", "warp_escape_corridor", "outer_shell")] * 2
    + [("outer_shell", "warp_direct", "dust_lane")] * 9
    + [("outer_shell", "warp_direct", "homunculus_core")] * 6
    + [("outer_shell", "warp_direct", "escape_corridor")] * 5
    + [("dust_lane", "warp_escape_corridor", "escape_corridor")] * 11
    + [("dust_lane", "warp_escape_corridor", "dust_lane")] * 5
    + [("dust_lane", "warp_direct", "homunculus_core")] * 8
    + [("dust_lane", "warp_direct", "dust_lane")] * 4
    + [("homunculus_core", "warp_escape_corridor", "dust_lane")] * 7
    + [("homunculus_core", "warp_escape_corridor", "homunculus_core")] * 6
    + [("homunculus_core", "warp_escape_corridor", "escape_corridor")] * 2
    + [("homunculus_core", "warp_direct", "homunculus_core")] * 10
    + [("homunculus_core", "warp_direct", "dust_lane")] * 2
)


def transition_frequency_matrix(
    dataset: Iterable[Tuple[Hashable, Hashable, Hashable]],
) -> Dict[Tuple[Hashable, Hashable], Dict[Hashable, float]]:
    """Maximum-likelihood estimate P(s'|s,a) = N(s,a,s') / N(s,a)."""
    counts: Dict[Tuple[Hashable, Hashable], Counter] = defaultdict(Counter)
    for s, a, s_next in dataset:
        counts[(s, a)][s_next] += 1
    probs = {}
    for sa, c in counts.items():
        total = sum(c.values())
        probs[sa] = {s_next: n / total for s_next, n in sorted(c.items())}
    return probs


def as_table(P, states: Optional[List[Hashable]] = None) -> Tuple[List[Tuple], List[Hashable], List[List[float]]]:
    """Return (row_labels, column_labels, matrix) — one row per (s, a)."""
    if states is None:
        states = sorted({s2 for row in P.values() for s2 in row} | {s for s, _ in P})
    rows = sorted(P)
    matrix = [[P[sa].get(s2, 0.0) for s2 in states] for sa in rows]
    return rows, states, matrix


def print_matrix(P, states=None) -> None:
    rows, cols, M = as_table(P, states)
    w = max(len(f"{s} | {a}") for s, a in rows) + 2
    print("P(s' | s, a)".ljust(w) + "".join(c[:15].rjust(17) for c in cols))
    for (s, a), r in zip(rows, M):
        print(f"{s} | {a}".ljust(w) + "".join(f"{p:17.3f}" for p in r))


if __name__ == "__main__":
    print("=== Subsystem 06 / Part 3: MDP Trajectory Logger & Return Evaluator ===\n")

    rewards = [10, 0, -5, 20]
    g0 = discounted_return(rewards)
    print(f"G0 for rewards {rewards}, gamma={GAMMA}: {g0:.6f}")
    print("  = 10 + 0.95*0 + 0.9025*(-5) + 0.857375*20 = 22.635\n")

    log = TrajectoryLogger()
    log.start("outer_shell")
    log.log("query_sensor_array(escape_corridor)", 0.0, "outer_shell")
    log.log("adjust_plasma_shields(80)", -1.0, "outer_shell")
    log.log("navigate_warp_vector(-0.85,0.52,0.05)", 5.0, "dust_lane")
    log.log("navigate_warp_vector(-0.85,0.52,0.05)", 20.0, "escape_corridor", done=True)
    print(log.pretty())
    print("tau =", log.as_tau())
    print(f"G0 = {log.discounted_return():.4f}; returns-to-go = {[round(g, 3) for g in log.returns_to_go()]}\n")

    P = transition_frequency_matrix(HISTORICAL_JUMPS)
    print(f"Transition frequency matrix from {len(HISTORICAL_JUMPS)} historical jumps:")
    print_matrix(P)
    print()
    for s in ("outer_shell", "dust_lane", "homunculus_core"):
        best = max(("warp_escape_corridor", "warp_direct"),
                   key=lambda a: P[(s, a)].get("escape_corridor", 0.0))
        print(f"From {s:<16} best action for reaching escape_corridor: {best} "
              f"(P = {P[(s, best)].get('escape_corridor', 0.0):.3f})")
