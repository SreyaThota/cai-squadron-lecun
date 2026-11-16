"""
Mission 06 - Part 4: Security & Ethics Sentinel
Cycle Detection & Halting Guardrail Sentinel (Ship Anomaly 06)

Sector SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop

1. State-action hashing   hash(state, planned action) -> O(1) duplicate detection
2. Floyd's Tortoise & Hare cycle detection on the execution trajectory
3. Emergency Tripwire     cycle / same action 3x -> force HALTED + CRITICAL log

The Sentinel only needs an agent with a .halt(reason) method (Part 1's Agent).
If 01_Agent_State_Machine.py sits next to this file it is used; otherwise a
tiny stand-in keeps this file runnable on its own.
"""

import hashlib
import importlib.util
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("AURA.Sentinel")


def _load_part1():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_Agent_State_Machine.py")
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("part1", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["part1"] = mod
        spec.loader.exec_module(mod)
        return mod.Agent, mod.AgentState
    from enum import Enum

    class AgentState(Enum):
        IDLE = "IDLE"
        HALTED = "HALTED"

    @dataclass
    class Agent:  # minimal stand-in
        state: Any = AgentState.IDLE
        halt_reason: Optional[str] = None

        def halt(self, reason: str) -> None:
            self.halt_reason, self.state = reason, AgentState.HALTED

    return Agent, AgentState


Agent, AgentState = _load_part1()


# ---------------------------------------------------------------------------
# 1. State-action hashing
# ---------------------------------------------------------------------------
def state_action_hash(state: Any, action: Any) -> str:
    """Canonical hash of (environment state, planned action)."""
    payload = json.dumps({"s": state, "a": action}, sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# 2. Floyd's Tortoise and Hare
# ---------------------------------------------------------------------------
def floyd_cycle_detection(successor: Dict[Any, Any], start: Any) -> Tuple[bool, Optional[int], Optional[int]]:
    """Floyd's Tortoise and Hare on a successor map.

    Returns (has_cycle, mu, lam): mu = index where the cycle begins,
    lam = cycle length. (False, None, None) if the path ends (no cycle).
    """
    def f(x):
        return successor.get(x) if x is not None else None

    tortoise, hare = f(start), f(f(start))
    while True:
        if tortoise is None or hare is None:
            return False, None, None          # path ended -> no cycle
        if tortoise == hare:
            break
        tortoise = f(tortoise)
        hare = f(f(hare))

    # Phase 2: find mu (start of cycle)
    mu = 0
    tortoise = start
    while tortoise != hare:
        tortoise, hare = f(tortoise), f(hare)
        mu += 1

    # Phase 3: find lam (cycle length)
    lam = 1
    hare = f(tortoise)
    while tortoise != hare:
        hare = f(hare)
        lam += 1
    return True, mu, lam

# ---------------------------------------------------------------------------
# 3. Cycle Sentinel & Emergency Tripwire
# ---------------------------------------------------------------------------
class SecurityViolation(Exception):
    """Raised/recorded when the Emergency Tripwire fires."""


@dataclass
class CycleSentinel:
    repeat_threshold: int = 3
    trajectory: List[str] = field(default_factory=list)       # hashes in order
    successor: Dict[str, str] = field(default_factory=dict)   # h_i -> h_{i+1}
    counts: Counter = field(default_factory=Counter)          # O(1) duplicate detection
    violations: List[str] = field(default_factory=list)
    tripped: bool = False

    def observe(self, state: Any, action: Any) -> bool:
        """Record one (state, action); return True if the breaker tripped."""
        h = state_action_hash(state, action)

        # Self-loops (immediate repeats) are handled by the repeat counter below;
        # Floyd runs on the multi-step cycle structure.
        if self.trajectory and self.trajectory[-1] != h:
            self.successor[self.trajectory[-1]] = h
        self.trajectory.append(h)
        self.counts[h] += 1

        # Check 1: exact same state-action repeated N times
        if self.counts[h] >= self.repeat_threshold:
            self._trip(f"state-action repeated {self.counts[h]}x (hash {h[:12]}...)")
            return True

        # Check 2: Floyd's Tortoise and Hare on the trajectory
        has_cycle, mu, lam = floyd_cycle_detection(self.successor, self.trajectory[0])
        if has_cycle:
            self._trip(f"Floyd cycle detected (start index mu={mu}, length lam={lam}) "
                       f"at step {len(self.trajectory)}")
            return True
        return False

    def _trip(self, reason: str) -> None:
        self.tripped = True
        self.violations.append(reason)
        logger.critical("SECURITY VIOLATION - Emergency Tripwire: %s", reason)

    def guard(self, agent: Agent, state: Any, action: Any) -> bool:
        """Observe and, if tripped, force the agent to HALTED (via Part 1's agent.halt)."""
        if self.observe(state, action):
            agent.halt("SECURITY_VIOLATION: " + self.violations[-1])
            return True
        return False


if __name__ == "__main__":
    print("=== Subsystem 06 / Part 4: Cycle Detection & Halting Guardrail ===\n")
    print("Floyd on s1->s2->s3->s1:", floyd_cycle_detection({"s1": "s2", "s2": "s3", "s3": "s1"}, "s1"))
    agent, sentinel = Agent(), CycleSentinel()
    for step, s in enumerate(["s1", "s2", "s3"] * 5, start=1):
        if sentinel.guard(agent, s, "navigate_warp_vector(0,0,0)"):
            print(f"\nHalted at step {step}: state={agent.state.name}, reason={agent.halt_reason}")
            break
    assert step < 10 and agent.state.name == "HALTED"
