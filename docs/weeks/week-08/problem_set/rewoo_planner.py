"""
Mission 08 — Part 1: Systems Architect (Lead Engineer)
ReWOO (Reasoning Without Observation) Decoupled DAG Planner, Worker & Token Profiler

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula)
Subsystem 08: ReWOO DAG Planner, Reflexion Surgeon & MemGPT Virtual Pager
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple


@dataclass
class PlanStep:
    """Represents a single symbolic node #E_i = ToolName[raw_arg] in a ReWOO DAG."""
    var_name: str
    plan_desc: str
    tool_name: str
    raw_arg: str
    dependencies: List[str] = field(default_factory=list)


class ReWOOEngine:
    """
    Implements the 3-stage ReWOO (Reasoning Without Observation) paradigm (Xu et al., 2023):
      1. Planner (P): Emits a complete symbolic execution graph (#E1, #E2, ..., #EK) in one LLM call.
      2. Worker (E): Resolves #E variable dependencies in topological levels and executes tools.
      3. Solver (S): Synthesizes final output from resolved evidence in a single final LLM call.
    Also supports dynamic fallback re-planning when a tool emits a runtime sensor fault (e.g. ERR_CORONA_SPIKE).
    """

    VAR_PATTERN = re.compile(r"#E\d+")

    def __init__(self, tools: Dict[str, Callable[[str], str]]):
        self.tools = tools

    def parse_blueprint(self, blueprint_text: str) -> List[Tuple[str, str, str, str]]:
        """
        TODO (Systems Architect): Parse a ReWOO blueprint containing lines formatted as:
            Plan: <reasoning>
            #E1 = ToolName[argument_string_possibly_referencing_#E_vars]
        Return a list of tuples: (var_name, plan_desc, tool_name, raw_arg).
        """
        steps: List[Tuple[str, str, str, str]] = []
        current_plan = ""
        for line in blueprint_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Plan:"):
                current_plan = line[len("Plan:") :].strip()
            elif line.startswith("#E"):
                match = re.match(r"^(#E\d+)\s*=\s*(\w+)\[(.*)\]$", line)
                if match:
                    var_name, tool_name, raw_arg = match.groups()
                    steps.append((var_name, current_plan, tool_name, raw_arg))
        return steps

    def to_plan_steps(self, steps: List[Tuple[str, str, str, str]]) -> List[PlanStep]:
        """Converts parsed 4-tuples into structured PlanStep nodes with extracted #E dependencies."""
        plan_nodes: List[PlanStep] = []
        for var_name, plan_desc, tool_name, raw_arg in steps:
            deps = sorted(set(self.VAR_PATTERN.findall(raw_arg)), key=lambda v: int(v[2:]))
            plan_nodes.append(
                PlanStep(
                    var_name=var_name,
                    plan_desc=plan_desc,
                    tool_name=tool_name,
                    raw_arg=raw_arg,
                    dependencies=deps,
                )
            )
        return plan_nodes

    def build_dependency_levels(self, steps: List[Tuple[str, str, str, str]]) -> List[List[str]]:
        """
        TODO (Systems Architect): Group step variable names (#E1, #E2, ...) into topological
        execution levels so that independent tool calls in the same level can execute concurrently.
        Raise ValueError if an unresolvable or cyclic dependency is detected.
        """
        remaining: Dict[str, Set[str]] = {}
        order_index: Dict[str, int] = {}

        for idx, (var_name, _, _, raw_arg) in enumerate(steps):
            deps = set(self.VAR_PATTERN.findall(raw_arg))
            remaining[var_name] = deps
            order_index[var_name] = idx

        resolved: Set[str] = set()
        levels: List[List[str]] = []

        while remaining:
            ready = [
                var
                for var, deps in remaining.items()
                if deps.issubset(resolved)
            ]
            if not ready:
                raise ValueError(f"Cyclic or unresolved variable dependency in ReWOO DAG: {remaining}")
            ready.sort(key=lambda v: order_index[v])
            levels.append(ready)
            for var in ready:
                resolved.add(var)
                del remaining[var]

        return levels

    def execute_dag(
        self,
        steps: List[Tuple[str, str, str, str]],
        fallback_replan_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Dict[str, str]:
        """
        TODO (Systems Architect): Resolve variable placeholders (#E1, #E2, ...) in tool arguments
        and execute each tool in topological level order, storing outputs in an evidence dictionary.
        If a tool output contains 'ERR_' and `fallback_replan_fn` is provided, invoke
        `fallback_replan_fn(var_name, error_output)` to recover gracefully.
        """
        step_Lookup: Dict[str, Tuple[str, str, str, str]] = {s[0]: s for s in steps}
        levels = self.build_dependency_levels(steps)
        evidence: Dict[str, str] = {}

        for level in levels:
            for var_name in level:
                _, _, tool_name, raw_arg = step_Lookup[var_name]
                if tool_name not in self.tools:
                    raise KeyError(f"Unknown tool '{tool_name}' referenced in step {var_name}")

                resolved_arg = raw_arg
                for dep_var, dep_val in evidence.items():
                    resolved_arg = resolved_arg.replace(dep_var, dep_val)

                output = self.tools[tool_name](resolved_arg)
                if "ERR_" in output and fallback_replan_fn is not None:
                    output = fallback_replan_fn(var_name, output)

                evidence[var_name] = output

        return evidence

    @staticmethod
    def compare_token_cost(
        k_steps: int,
        base_tokens: int,
        step_tokens: int,
        obs_tokens: int,
    ) -> Dict[str, float]:
        """
        TODO (Systems Architect): Compute theoretical cumulative input token cost for K-step ReAct vs. ReWOO:
          C_ReAct = sum_{t=1..K} (base_tokens + (t - 1) * (step_tokens + obs_tokens))
                  = K * base_tokens + (K * (K - 1) // 2) * (step_tokens + obs_tokens)
          C_ReWOO = 2 * base_tokens + K * (step_tokens + obs_tokens)
        Return {'react_tokens': ..., 'rewoo_tokens': ..., 'savings_pct': ..., 'speedup_ratio': ...}.
        """
        per_turn_growth = step_tokens + obs_tokens
        react_tokens = sum(base_tokens + (t - 1) * per_turn_growth for t in range(1, k_steps + 1))
        rewoo_tokens = 2 * base_tokens + k_steps * per_turn_growth
        savings_pct = ((react_tokens - rewoo_tokens) / react_tokens) * 100.0 if react_tokens > 0 else 0.0
        speedup_ratio = react_tokens / rewoo_tokens if rewoo_tokens > 0 else 1.0
        return {
            "react_tokens": float(react_tokens),
            "rewoo_tokens": float(rewoo_tokens),
            "savings_pct": round(savings_pct, 2),
            "speedup_ratio": round(speedup_ratio, 2),
        }


# ==============================================================================
# Verification Harness for Part 1 (Systems Architect)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 08 (Part 1): ReWOO DAG Planner & Token Profiler Verification ===")

    BLUEPRINT = """
    Plan: Scan Cygnus X-1 hard X-ray flare flux from primary telemetry array.
    #E1 = TelemetryScan[Cygnus_X1_Corona_Sector_08]
    Plan: Measure Veil Nebula shockwave plasma density in parallel.
    #E2 = PlasmaProbe[Veil_Nebula_Shock_Front]
    Plan: Compute Hohmann escape burn vector combining X-ray flux #E1 and plasma density #E2.
    #E3 = HohmannCalc[flux=#E1, density=#E2]
    Plan: Modulate deflector shield harmonics for burn trajectory #E3.
    #E4 = ShieldModulate[vector=#E3]
    """

    mock_tools: Dict[str, Callable[[str], str]] = {
        "TelemetryScan": lambda arg: f"XRayFlux({arg})=4.2e37_erg/s",
        "PlasmaProbe": lambda arg: f"IonDensity({arg})=14.8_cm^-3",
        "HohmannCalc": lambda arg: f"DeltaV({arg})=18.4_km/s",
        "ShieldModulate": lambda arg: f"HarmonicLock({arg})=STABLE",
    }

    engine = ReWOOEngine(mock_tools)
    steps = engine.parse_blueprint(BLUEPRINT)
    levels = engine.build_dependency_levels(steps)
    evidence = engine.execute_dag(steps)

    print(f"Parsed Steps ({len(steps)}): {[s[0] for s in steps]}")
    print(f"Topological Parallel Execution Levels: {levels}")
    for k, v in evidence.items():
        print(f"  {k} -> {v}")

    assert levels == [["#E1", "#E2"], ["#E3"], ["#E4"]], f"Unexpected DAG levels: {levels}"
    assert "4.2e37_erg/s" in evidence["#E3"] and "14.8_cm^-3" in evidence["#E3"]
    assert evidence["#E4"].startswith("HarmonicLock(vector=DeltaV(")

    print("\n--- ReAct vs. ReWOO Cumulative Input Token Benchmark (L0=1200, L_step=120, L_obs=280) ---")
    for k in (3, 5, 8):
        metrics = ReWOOEngine.compare_token_cost(k_steps=k, base_tokens=1200, step_tokens=120, obs_tokens=280)
        print(
            f"  K={k} tools | ReAct={int(metrics['react_tokens']):6d} tok | "
            f"ReWOO={int(metrics['rewoo_tokens']):5d} tok | "
            f"Savings={metrics['savings_pct']:5.2f}% ({metrics['speedup_ratio']:.2f}x)"
        )

    m5 = ReWOOEngine.compare_token_cost(5, 1200, 120, 280)
    assert m5["react_tokens"] == 10000.0 and m5["rewoo_tokens"] == 4400.0
    print("[OK] Part 1 (Systems Architect) verification passed.")
