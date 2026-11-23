"""
Mission 08 — Part 1: Systems Architect (Lead Engineer)
ReWOO (Reasoning Without Observation) Decoupled DAG Planner, Worker & Token Profiler

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula)
Subsystem 08: ReWOO DAG Planner, Reflexion Surgeon & MemGPT Virtual Pager
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
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
        step_lookup: Dict[str, Tuple[str, str, str, str]] = {s[0]: s for s in steps}
        levels = self.build_dependency_levels(steps)
        evidence: Dict[str, str] = {}

        for level in levels:
            for var_name in level:
                evidence[var_name] = self._run_step(step_lookup[var_name], evidence, fallback_replan_fn)

        return evidence

    # ------------------------------------------------------------------
    # Squadron LeCun additions
    # ------------------------------------------------------------------
    TOKEN_PATTERN = re.compile(r"#E\d+(?!\d)")

    def substitute(self, raw_arg: str, evidence: Dict[str, str]) -> str:
        """Replace each #E_i token by its evidence, matching the WHOLE token.
        The plain str.replace() version turns '#E10' into '<value of #E1>0'."""
        def repl(m: "re.Match[str]") -> str:
            var = m.group(0)
            if var not in evidence:
                raise KeyError(f"{var} used before it was resolved")
            return evidence[var]
        return self.TOKEN_PATTERN.sub(repl, raw_arg)

    def _run_step(self, step, evidence, fallback_replan_fn, max_replans: int = 2) -> str:
        var_name, _, tool_name, raw_arg = step
        if tool_name not in self.tools:
            raise KeyError(f"Unknown tool '{tool_name}' referenced in step {var_name}")
        output = self.tools[tool_name](self.substitute(raw_arg, evidence))
        tries = 0
        while "ERR_" in output and fallback_replan_fn is not None and tries < max_replans:
            output = fallback_replan_fn(var_name, output)
            tries += 1
        return output

    def execute_dag_parallel(
        self,
        steps: List[Tuple[str, str, str, str]],
        fallback_replan_fn: Optional[Callable[[str, str], str]] = None,
        max_workers: int = 8,
    ) -> Dict[str, str]:
        """Same as execute_dag, but every topological level runs its tools concurrently."""
        step_lookup = {s[0]: s for s in steps}
        evidence: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for level in self.build_dependency_levels(steps):
                snapshot = dict(evidence)
                futures = {v: pool.submit(self._run_step, step_lookup[v], snapshot, fallback_replan_fn)
                           for v in level}
                for v, f in futures.items():
                    evidence[v] = f.result()
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
# Systems Architect Extensions (Squadron LeCun)
# ==============================================================================
def _ext_substitution_bug() -> None:
    print("\n=== Extension 1: #E10 substitution bug ===")
    tools = {"Echo": lambda a: a.upper()}
    lines = []
    for i in range(1, 12):
        arg = "seed" if i == 1 else f"#E{i - 1}"
        lines += [f"Plan: step {i}", f"#E{i} = Echo[{arg}{'|' + str(i) if i > 1 else ''}]"]
    bp = "\n".join(lines) + "\nPlan: combine\n#E12 = Echo[a=#E1 b=#E10]"
    eng = ReWOOEngine(tools)
    steps = eng.parse_blueprint(bp)
    ev = eng.execute_dag(steps)
    naive = "a=#E1 b=#E10"
    for k, v in ev.items():  # what the starter's str.replace() loop would do
        naive = naive.replace(k, v)
    print(f"Fixed  #E12 arg -> {('a=' + ev['#E1'] + ' b=' + ev['#E10'])[:60]}")
    print(f"Starter #E12 arg -> {naive[:60]}")
    assert ev["#E12"] == ("A=" + ev["#E1"] + " B=" + ev["#E10"]).upper()
    assert naive != "a=" + ev["#E1"] + " b=" + ev["#E10"], "expected the naive version to break"
    print("[PASS] whole-token substitution keeps #E1 and #E10 apart; naive replace corrupts #E10")


def _ext_parallel_speedup() -> None:
    print("\n=== Extension 2: Parallel levels vs sequential (each tool sleeps 0.2 s) ===")

    def slow(name):
        def f(arg):
            time.sleep(0.2)
            return f"{name}({arg})"
        return f
    tools = {n: slow(n) for n in ("Scan", "Probe", "Calc", "Shield")}
    lines = []
    width = 4
    for i in range(1, width + 1):
        lines += [f"Plan: sensor {i}", f"#E{i} = Scan[sector_{i}]"]
    for i in range(width + 1, 2 * width + 1):
        lines += [f"Plan: probe {i}", f"#E{i} = Probe[#E{i - width}]"]
    lines += ["Plan: fuse", "#E9 = Calc[" + ",".join(f"#E{i}" for i in range(5, 9)) + "]",
              "Plan: act", "#E10 = Shield[#E9]"]
    eng = ReWOOEngine(tools)
    steps = eng.parse_blueprint("\n".join(lines))
    levels = eng.build_dependency_levels(steps)
    t0 = time.perf_counter(); seq = eng.execute_dag(steps); t_seq = time.perf_counter() - t0
    t0 = time.perf_counter(); par = eng.execute_dag_parallel(steps); t_par = time.perf_counter() - t0
    print(f"Levels: {[len(l) for l in levels]} tools per level ({len(steps)} tools)")
    print(f"Sequential: {t_seq:.2f} s | Parallel: {t_par:.2f} s | speed-up {t_seq / t_par:.1f}x")
    assert seq == par and t_par < t_seq * 0.6
    print("[PASS] identical evidence, wall time ~ number of levels instead of number of tools")


def _ext_fallback_and_errors() -> None:
    print("\n=== Extension 3: Fallback re-planning and error checks ===")
    calls = {"n": 0}

    def flaky(arg):
        calls["n"] += 1
        return "ERR_CORONA_SPIKE" if calls["n"] == 1 else f"Flux({arg})=4.1e37"
    eng = ReWOOEngine({"TelemetryScan": flaky, "Use": lambda a: f"ok[{a}]"})
    steps = eng.parse_blueprint("Plan: scan\n#E1 = TelemetryScan[S8]\nPlan: use\n#E2 = Use[#E1]")
    ev = eng.execute_dag(steps, fallback_replan_fn=lambda v, e: eng.tools["TelemetryScan"]("S8_backup"))
    print(f"With fallback: {ev}")
    assert ev["#E2"] == "ok[Flux(S8_backup)=4.1e37]"
    always_bad = ReWOOEngine({"T": lambda a: "ERR_DEAD"})
    ev = always_bad.execute_dag(always_bad.parse_blueprint("#E1 = T[x]"), lambda v, e: "ERR_STILL_DEAD")
    print(f"Fallback that keeps failing stops after 2 re-plans: {ev}")
    for bp, err in (("#E1 = T[#E2]\n#E2 = T[#E1]", ValueError), ("#E1 = Missing[x]", KeyError)):
        try:
            eng2 = ReWOOEngine({"T": lambda a: a})
            eng2.execute_dag(eng2.parse_blueprint(bp))
            raise AssertionError("should have failed")
        except err as e:
            print(f"[PASS] {err.__name__}: {str(e)[:70]}")


def _ext_token_simulation() -> None:
    print("\n=== Extension 4: Token formula checked by simulating real prompts ===")
    L0, LS, LO = 1200, 120, 280
    print(f"{'K':>3}{'ReAct sim':>11}{'formula':>9}{'ReWOO sim':>11}{'formula':>9}{'saving':>8}")
    for K in (1, 2, 3, 5, 8, 12, 20):
        # ReAct: every turn re-sends the base prompt plus all earlier thoughts+observations.
        ctx, react = L0, 0
        for _ in range(K):
            react += ctx
            ctx += LS + LO
        # ReWOO: one planner call (L0) + one solver call that sees base + plan + all evidence.
        rewoo = L0 + (L0 + K * (LS + LO))
        m = ReWOOEngine.compare_token_cost(K, L0, LS, LO)
        assert react == m["react_tokens"] and rewoo == m["rewoo_tokens"]
        print(f"{K:>3}{react:>11}{int(m['react_tokens']):>9}{rewoo:>11}{int(m['rewoo_tokens']):>9}"
              f"{m['savings_pct']:>7.1f}%")
    print("[PASS] simulation matches the closed-form costs; ReWOO costs more for K <= 2 (it always pays for 2 calls), and wins from K = 3")


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

    _ext_substitution_bug()
    _ext_parallel_speedup()
    _ext_fallback_and_errors()
    _ext_token_simulation()
    print("\n[OK] Part 1 extensions passed.")
