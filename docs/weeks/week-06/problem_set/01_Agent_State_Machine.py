"""
Mission 06 — Part 1: Systems Architect (Lead Engineer)
Native Python Agent State Machine Core & Step Budget Enforcement

Sector SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop

Pure standard library: Enum + dataclass. No agent frameworks.

Transition graph (exactly one transition per step() call):
    IDLE -> PERCEIVING -> DELIBERATING -> ACTING -> OBSERVING -> REFLECTING
    REFLECTING -> DELIBERATING (loop)  |  HALTED (terminal)
Step budget: max_steps N <= 20. Once N transitions are used, the next
step() forcibly moves the agent to HALTED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional

MAX_STEP_BUDGET = 20  # hard ceiling from the mission brief (N <= 20)


class AgentState(Enum):
    IDLE = "IDLE"
    PERCEIVING = "PERCEIVING"
    DELIBERATING = "DELIBERATING"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    REFLECTING = "REFLECTING"
    HALTED = "HALTED"


# The strict, legal transition graph. step() may only follow these edges
# (plus the forced budget edge "any state -> HALTED").
TRANSITIONS: Dict[AgentState, FrozenSet[AgentState]] = {
    AgentState.IDLE: frozenset({AgentState.PERCEIVING}),
    AgentState.PERCEIVING: frozenset({AgentState.DELIBERATING}),
    AgentState.DELIBERATING: frozenset({AgentState.ACTING}),
    AgentState.ACTING: frozenset({AgentState.OBSERVING}),
    AgentState.OBSERVING: frozenset({AgentState.REFLECTING}),
    AgentState.REFLECTING: frozenset({AgentState.DELIBERATING, AgentState.HALTED}),
    AgentState.HALTED: frozenset(),  # terminal: no way out
}


class IllegalTransitionError(RuntimeError):
    """Raised if code ever tries to move along an edge not in TRANSITIONS."""


@dataclass
class Agent:
    """AURA's deterministic execution core.

    `goal_reached` is decided in REFLECTING: True -> HALTED, False -> loop back
    to DELIBERATING. Optional hooks let the other subsystems (tool dispatcher,
    trajectory logger, loop sentinel) plug into each state without changing
    the state machine itself.
    """

    name: str = "AURA"
    max_steps: int = MAX_STEP_BUDGET
    state: AgentState = AgentState.IDLE
    step_count: int = 0
    goal_reached: bool = False
    halt_reason: Optional[str] = None
    context: List[Dict[str, Any]] = field(default_factory=list)   # observations, errors, notes
    history: List[tuple] = field(default_factory=list)            # (step, from, to)
    hooks: Dict[AgentState, Callable[["Agent"], None]] = field(default_factory=dict)
    verbose: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int) or not (1 <= self.max_steps <= MAX_STEP_BUDGET):
            raise ValueError(
                f"max_steps must be an int in [1, {MAX_STEP_BUDGET}], got {self.max_steps!r}"
            )

    # ---------- core ----------
    def _next_state(self) -> AgentState:
        if self.state is AgentState.REFLECTING:
            return AgentState.HALTED if self.goal_reached else AgentState.DELIBERATING
        (only_next,) = TRANSITIONS[self.state]  # every other state has exactly one successor
        return only_next

    def _move(self, new_state: AgentState, forced: bool = False) -> None:
        if not forced and new_state not in TRANSITIONS[self.state]:
            raise IllegalTransitionError(f"{self.state.name} -> {new_state.name} is not allowed")
        self.history.append((self.step_count, self.state, new_state))
        if self.verbose:
            tag = " [FORCED]" if forced else ""
            print(f"step {self.step_count:02d}: {self.state.name:<12} -> {new_state.name}{tag}")
        self.state = new_state

    def halt(self, reason: str) -> None:
        """Force HALTED from any state (used by the budget and by the Sentinel's tripwire)."""
        if self.state is AgentState.HALTED:
            return
        self.halt_reason = reason
        self._move(AgentState.HALTED, forced=True)

    def step(self) -> AgentState:
        """Execute exactly ONE state transition and return the new state."""
        if self.state is AgentState.HALTED:
            return self.state  # terminal: further calls are no-ops

        if self.step_count >= self.max_steps:
            self.halt(f"step budget exhausted ({self.step_count}/{self.max_steps})")
            return self.state

        self.step_count += 1
        new_state = self._next_state()
        if new_state is AgentState.HALTED:
            self.halt_reason = "goal reached"
        self._move(new_state)

        hook = self.hooks.get(new_state)
        if hook is not None and self.state is new_state:
            hook(self)
        return self.state

    def run(self, max_calls: int = 1000) -> AgentState:
        """Call step() until HALTED (max_calls is just a safety net for the caller)."""
        for _ in range(max_calls):
            if self.state is AgentState.HALTED:
                break
            self.step()
        return self.state


if __name__ == "__main__":
    print("=== Subsystem 06 / Part 1: Agent State Machine ===\n")

    print("Run A: goal reached on the 2nd reflection")
    reflections = {"n": 0}

    def reflect(agent: Agent) -> None:
        reflections["n"] += 1
        agent.goal_reached = reflections["n"] >= 2

    a = Agent(verbose=True, hooks={AgentState.REFLECTING: reflect})
    a.run()
    print(f"-> final={a.state.name}, steps={a.step_count}, reason={a.halt_reason}\n")

    print("Run B: goal never reached, forced to run 21 steps with N=20")
    b = Agent(max_steps=20)
    for _ in range(21):
        b.step()
    print(f"-> final={b.state.name}, steps used={b.step_count}, reason={b.halt_reason}")
    b.step()
    print(f"-> step() after HALTED is a no-op: state={b.state.name}, steps={b.step_count}\n")

    try:
        Agent(max_steps=25)
    except ValueError as e:
        print("Budget guard:", e)
