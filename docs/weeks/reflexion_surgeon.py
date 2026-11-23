"""
Mission 08 — Part 2: Communications & Protocol Officer
3-Actor Reflexion Verbal RL & AST Traceback Surgeon (Shinn et al., NeurIPS 2023)

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula)
Subsystem 08: ReWOO DAG Planner, Reflexion Surgeon & MemGPT Virtual Pager
"""

import hashlib
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple


class TracebackParser:
    """Helper utility to extract structured (file, line_number, exception_type, fault_summary) from Python tracebacks."""

    LINE_PATTERN = re.compile(r'File "([^"]+)", line (\d+)')
    EXC_PATTERN = re.compile(r"^([A-Za-z_]\w*(?:Error|Exception|Warning)):\s*(.*)$")

    @classmethod
    def parse(cls, stderr_text: str) -> Dict[str, str]:
        file_name = "unknown.py"
        line_number = "-1"
        exception_type = "RuntimeError"
        fault_summary = stderr_text.strip()

        lines = [ln.strip() for ln in stderr_text.strip().splitlines() if ln.strip()]
        for ln in lines:
            m_loc = cls.LINE_PATTERN.search(ln)
            if m_loc:
                file_name, line_number = m_loc.group(1), m_loc.group(2)
            m_exc = cls.EXC_PATTERN.match(ln)
            if m_exc:
                exception_type, fault_summary = m_exc.group(1), m_exc.group(2)

        return {
            "file": file_name,
            "line_number": line_number,
            "exception_type": exception_type,
            "fault_summary": fault_summary,
        }


class ReflexionSurgeon:
    """
    Implements the 3-Actor Verbal Reinforcement Learning loop (Shinn et al., 2023):
      - Actor (M_a): Generates candidate code patches conditioned on bounded episodic memory.
      - Evaluator (M_e): Executes deterministic unit tests / runtime verification (grounded external oracle).
      - Self-Reflection (M_sr): Translates traceback diagnostics into a natural-language critique r_k
        stored in a sliding episodic buffer capped at Omega = 3, with SHA-256 cycle detection.
    """

    def __init__(self, max_trials: int = 4, buffer_cap: int = 3):
        self.max_trials = max_trials
        self.buffer_cap = buffer_cap
        self.episodic_buffer: List[str] = []
        self.patch_hashes: Set[str] = set()

    def parse_traceback(self, stderr_text: str) -> Dict[str, str]:
        """
        TODO (Protocol Officer): Extract structured diagnostics from a Python traceback string.
        Return {'file': ..., 'line_number': ..., 'exception_type': ..., 'fault_summary': ...}.
        """
        return TracebackParser.parse(stderr_text)

    def generate_verbal_reflection(self, diag: Dict[str, str], failed_code: str) -> str:
        """
        TODO (Protocol Officer): Synthesize a structured natural-language reflection r_k diagnosing
        the root cause and prescribing a concrete code constraint for trial k+1.
        Append r_k to `self.episodic_buffer` while enforcing sliding-window capacity `self.buffer_cap`
        (evicting the oldest reflection when len(self.episodic_buffer) > self.buffer_cap).
        """
        exc = diag.get("exception_type", "Exception")
        line_no = diag.get("line_number", "?")
        summary = diag.get("fault_summary", "")

        if exc == "ZeroDivisionError":
            prescription = "Guard denominator against zero before dividing telemetry flux."
        elif exc == "KeyError":
            prescription = f"Use dict.get() or verify key {summary} exists in telemetry packet."
        elif exc == "IndexError":
            prescription = "Check array bounds before indexing shockwave sample buffer."
        else:
            prescription = f"Resolve {exc} ({summary}) with explicit input validation."

        reflection = (
            f"[Reflection Line {line_no} | {exc}]: Fault '{summary}'. "
            f"Constraint for next trial: {prescription}"
        )
        self.episodic_buffer.append(reflection)
        if len(self.episodic_buffer) > self.buffer_cap:
            self.episodic_buffer.pop(0)
        return reflection

    def is_duplicate_patch(self, candidate_code: str) -> bool:
        """
        TODO (Protocol Officer): Compute SHA-256 hash of whitespace-normalized `candidate_code`.
        Return True if already seen in `self.patch_hashes`, else record the digest and return False.
        """
        norm = "\n".join(line.strip() for line in candidate_code.strip().splitlines() if line.strip())
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if digest in self.patch_hashes:
            return True
        self.patch_hashes.add(digest)
        return False

    def run_repair_loop(
        self,
        initial_code: str,
        evaluator_fn: Callable[[str], Tuple[bool, str]],
        actor_patch_fn: Callable[[str, List[str]], str],
    ) -> Dict[str, object]:
        """
        TODO (Protocol Officer): Execute up to `self.max_trials` iterations of:
          1. Check if `initial_code` / `current_code` is a duplicate patch via `is_duplicate_patch`.
          2. Evaluate `current_code` via `evaluator_fn(current_code) -> (passed, stderr)`.
          3. If `passed` is True, return a success report dictionary.
          4. Otherwise, parse the traceback (`parse_traceback`), generate a verbal reflection
             (`generate_verbal_reflection`), and synthesize the next candidate patch via
             `actor_patch_fn(current_code, list(self.episodic_buffer))`.
        """
        current_code = initial_code
        history: List[Dict[str, object]] = []

        for trial in range(1, self.max_trials + 1):
            duplicate = self.is_duplicate_patch(current_code)
            if duplicate and trial > 1:
                # Cycle guard: the Actor proposed code we already tested. Re-running it cannot
                # change the result, so we record it, add a reflection that says so, and ask again.
                note = ("[Reflection | CycleGuard]: The proposed patch is identical (SHA-256) to an "
                        "earlier failed attempt. Constraint for next trial: produce a DIFFERENT fix.")
                self.episodic_buffer.append(note)
                if len(self.episodic_buffer) > self.buffer_cap:
                    self.episodic_buffer.pop(0)
                history.append({"trial": trial, "passed": False, "duplicate": True,
                                "exception": "DuplicatePatch", "reflection": note})
                if trial < self.max_trials:
                    current_code = actor_patch_fn(current_code, list(self.episodic_buffer))
                continue

            passed, stderr = evaluator_fn(current_code)

            if passed:
                history.append({"trial": trial, "passed": True, "duplicate": duplicate, "reflection": "PASS"})
                return {
                    "success": True,
                    "trials_used": trial,
                    "final_code": current_code,
                    "episodic_buffer": list(self.episodic_buffer),
                    "history": history,
                }

            diag = self.parse_traceback(stderr)
            reflection = self.generate_verbal_reflection(diag, current_code)
            history.append(
                {
                    "trial": trial,
                    "passed": False,
                    "duplicate": duplicate,
                    "exception": diag["exception_type"],
                    "reflection": reflection,
                }
            )

            if trial < self.max_trials:
                current_code = actor_patch_fn(current_code, list(self.episodic_buffer))

        return {
            "success": False,
            "trials_used": self.max_trials,
            "final_code": current_code,
            "episodic_buffer": list(self.episodic_buffer),
            "history": history,
        }


# ==============================================================================
# Protocol Officer Extensions (Squadron LeCun)
#   A real Evaluator: candidate code is written to a temp file and executed with
#   real unit tests in a separate Python process (timeout 5 s). The traceback the
#   parser sees is the genuine CPython stderr, not a hand-written string.
# ==============================================================================
UNIT_TESTS = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from candidate import normalize_xray_flux as f
    assert f({'flux': 10.0, 'window_sec': 2.0}) == 5.0
    assert f({'flux': 3.0, 'window_sec': 0}) == 0.0
    assert f({'window_sec': 4.0}) == 0.0
    assert f({'flux': '8', 'window_sec': 2}) == 4.0
    assert f({'flux': 6.0, 'window_sec': -3.0}) == 0.0, "negative window must be rejected"
""")


def subprocess_evaluator(code_str: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "candidate.py").write_text(textwrap.dedent(code_str))
        Path(d, "test_candidate.py").write_text(UNIT_TESTS)
        try:
            r = subprocess.run([sys.executable, "-I", "test_candidate.py"], cwd=d,
                               capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            return False, 'File "candidate.py", line 1\nTimeoutError: candidate ran longer than 5 s'
        return r.returncode == 0, r.stderr


# A rule-based Actor that reads ONLY the episodic memory (like an LLM conditioned on reflections)
# and adds one guard per distinct lesson it finds there.
FIXES = {
    "ZeroDivisionError": "    if window == 0:\n        return 0.0\n",
    "KeyError": None,  # handled by switching to .get()
    "TypeError": "    flux = float(flux)\n",
    "AssertionError": "    if window <= 0:\n        return 0.0\n",
}


def build_patch(lessons: List[str]) -> str:
    use_get = any("KeyError" in r for r in lessons)
    body = "def normalize_xray_flux(packet):\n"
    body += ("    flux = packet.get('flux', 0.0)\n    window = packet.get('window_sec', 1.0)\n" if use_get
             else "    flux = packet['flux']\n    window = packet['window_sec']\n")
    for exc in ("TypeError", "AssertionError", "ZeroDivisionError"):
        if any(exc in r for r in lessons) and FIXES[exc]:
            body += FIXES[exc]
    return body + "    return flux / window\n"


def make_memory_actor(lesson_log: List[str]):
    def actor(curr_code: str, reflections: List[str]) -> str:
        return build_patch(reflections)
    return actor


BUGGY = "def normalize_xray_flux(packet):\n    return packet['flux'] / packet['window_sec']\n"


def _ext_real_evaluator() -> None:
    print("\n=== Extension 1: Reflexion with a REAL unit-test evaluator (subprocess) ===")
    s = ReflexionSurgeon(max_trials=8, buffer_cap=3)
    rep = s.run_repair_loop(BUGGY, subprocess_evaluator, make_memory_actor([]))
    for h in rep["history"]:
        print(f"  Trial {h['trial']}: {'PASS' if h['passed'] else h['exception']:<18} dup={h['duplicate']}")
    print(f"Success={rep['success']} after {rep['trials_used']} trials")
    print("Final code:\n" + textwrap.indent(rep["final_code"], "    "))
    assert rep["success"] and subprocess_evaluator(rep["final_code"])[0]
    d = s.parse_traceback(subprocess_evaluator(BUGGY)[1])
    print(f"Parsed real traceback of the original bug: {d}")


def _ext_buffer_ablation() -> None:
    print("\n=== Extension 2: Episodic buffer size (Omega) ablation, max 8 trials ===")
    print(f"{'Omega':>6}{'success':>9}{'trials':>8}  exceptions seen")
    results = {}
    for omega in (1, 2, 3, 5):
        s = ReflexionSurgeon(max_trials=8, buffer_cap=omega)
        rep = s.run_repair_loop(BUGGY, subprocess_evaluator, make_memory_actor([]))
        seq = [h["exception"] if not h["passed"] else "PASS" for h in rep["history"]]
        results[omega] = rep["success"]
        print(f"{omega:>6}{str(rep['success']):>9}{rep['trials_used']:>8}  {' -> '.join(seq)}")
    assert results[3] and not results[1]
    print("[PASS] with Omega=1 the Actor forgets earlier lessons and cycles; Omega=3 drops only the oldest lesson (ZeroDivision), which the later 'window <= 0' guard already covers")


def _ext_cycle_guard() -> None:
    print("\n=== Extension 3: Cycle guard actually stops wasted evaluations ===")
    calls = {"n": 0}

    def counting_eval(code):
        calls["n"] += 1
        return subprocess_evaluator(code)
    stubborn = lambda code, refl: BUGGY  # an Actor that keeps proposing the same broken code
    s = ReflexionSurgeon(max_trials=6, buffer_cap=3)
    rep = s.run_repair_loop(BUGGY, counting_eval, stubborn)
    dups = sum(1 for h in rep["history"] if h["exception"] == "DuplicatePatch")
    print(f"6 trials, duplicates caught: {dups}, real test runs: {calls['n']} (starter would run 6)")
    ws = "def normalize_xray_flux(packet):\n\n        return packet['flux']   /   packet['window_sec']\n"
    s2 = ReflexionSurgeon()
    s2.is_duplicate_patch(BUGGY)
    print(f"Whitespace-only change detected as duplicate: {s2.is_duplicate_patch(ws)} "
          f"(inner spaces differ, so the line-strip normaliser treats it as new)")
    assert dups == 5 and calls["n"] == 1


# ==============================================================================
# Verification Harness for Part 2 (Communications & Protocol Officer)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 08 (Part 2): Reflexion Surgeon & Traceback Parser Verification ===")

    surgeon = ReflexionSurgeon(max_trials=4, buffer_cap=3)

    BUGGY_CODE_V1 = """
def normalize_xray_flux(packet):
    return packet['flux'] / packet['window_sec']
"""

    PATCH_V2 = """
def normalize_xray_flux(packet):
    if packet['window_sec'] == 0:
        return 0.0
    return packet['flux'] / packet['window_sec']
"""

    PATCH_V3 = """
def normalize_xray_flux(packet):
    flux = packet.get('flux', 0.0)
    window = packet.get('window_sec', 1.0)
    if window == 0:
        return 0.0
    return flux / window
"""

    def mock_unit_test_evaluator(code_str: str) -> Tuple[bool, str]:
        if "packet.get('flux'" not in code_str and "if packet['window_sec'] == 0" not in code_str:
            return (
                False,
                'Traceback (most recent call last):\n'
                '  File "telemetry_parser.py", line 3, in normalize_xray_flux\n'
                'ZeroDivisionError: division by zero when window_sec=0',
            )
        if "packet.get('flux'" not in code_str:
            return (
                False,
                'Traceback (most recent call last):\n'
                '  File "telemetry_parser.py", line 5, in normalize_xray_flux\n'
                "KeyError: 'flux'",
            )
        return True, ""

    def mock_actor(curr_code: str, reflections: List[str]) -> str:
        latest = reflections[-1] if reflections else ""
        if "ZeroDivisionError" in latest:
            return PATCH_V2
        if "KeyError" in latest:
            return PATCH_V3
        return curr_code

    report = surgeon.run_repair_loop(BUGGY_CODE_V1, mock_unit_test_evaluator, mock_actor)
    print(f"Repair Success: {report['success']} in {report['trials_used']} trials")
    for step in report["history"]:  # type: ignore[union-attr]
        print(f"  Trial {step['trial']}: passed={step['passed']} | {step['reflection']}")

    assert report["success"] is True and report["trials_used"] == 3
    assert surgeon.is_duplicate_patch(PATCH_V3) is True, "Duplicate patch SHA-256 guard failed!"
    print("[OK] Part 2 (Communications & Protocol Officer) verification passed.")

    _ext_real_evaluator()
    _ext_buffer_ablation()
    _ext_cycle_guard()
    print("\n[OK] Part 2 extensions passed.")
