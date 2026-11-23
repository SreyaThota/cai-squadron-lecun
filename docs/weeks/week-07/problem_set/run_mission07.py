"""
Mission 07 — End-to-end verification for Squadron LeCun.

1. Runs the four station verification harnesses (Parts 1-4) as separate processes.
2. Checks the PUCT trace table in mission-07-doc.md against mcts_protocol.puct_score.
3. c_puct sweep (0.5 / 1.5 / 2.5) on a 3-branch tree to show exploit vs explore.
4. Integrated pipeline: ToT-BFS proposes Game-of-24 steps -> exact arithmetic PRM
   scores each step -> EventHorizonPruner adjusts / severs -> surviving beam.

Run:  python run_mission07.py
Pure standard library.
"""

import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from horizon_pruner import HardenedEventHorizonPruner  # noqa: E402
from mcts_protocol import MCTSEngine, MCTSNode  # noqa: E402
from prm_evaluator import ArithmeticStepPRM  # noqa: E402
from tot_solver import ThoughtNode, TreeOfThoughtsSolver, game24_evaluator, game24_generator, _parse  # noqa: E402


def run_harnesses() -> None:
    print("=" * 78)
    print("STAGE 1: Station verification harnesses")
    print("=" * 78)
    for f in ("tot_solver.py", "mcts_protocol.py", "prm_evaluator.py", "horizon_pruner.py"):
        print(f"\n$ python {f}")
        r = subprocess.run([sys.executable, f], cwd=HERE, capture_output=True, text=True)
        print(r.stdout.rstrip())
        if r.returncode != 0:
            print(r.stderr)
            raise SystemExit(f"{f} FAILED")


def check_puct_table() -> None:
    print("\n" + "=" * 78)
    print("STAGE 2: PUCT trace table check (c_puct = 1.5, N(s) = 16)")
    print("=" * 78)
    eng = MCTSEngine(c_puct=1.5)
    parent = MCTSNode("s", visit_count=16)
    rows = [("Branch A", 0.75, 0.50, 5), ("Branch B", 0.55, 0.40, 1)]
    scores = {}
    for name, q, p, n in rows:
        child = MCTSNode(name, prior=p, parent=parent, visit_count=n, value_sum=q * n)
        parent.children.append(child)
        u = eng.puct_score(child, parent) - child.q_value
        scores[name] = eng.puct_score(child, parent)
        print(f"{name}: Q={q:.2f} P={p:.2f} N(s,a)={n}  U={u:.4f}  PUCT={scores[name]:.4f}")
    pick = eng.select_best_child(parent).state
    print(f"Selected: {pick}")
    assert abs(scores["Branch A"] - 1.25) < 1e-9 and abs(scores["Branch B"] - 1.75) < 1e-9
    assert pick == "Branch B"
    print("[PASS] Table values reproduced exactly by mcts_protocol.puct_score")


def c_puct_sweep() -> None:
    print("\n" + "=" * 78)
    print("STAGE 3: c_puct sweep (200 simulations, 3 branches, noisy rewards)")
    print("=" * 78)
    branches = [  # (name, prior from the 'LLM', true mean reward)
        ("A: low prior, best value", 0.20, 0.80),
        ("B: high prior, mid value", 0.60, 0.50),
        ("C: low prior, poor value", 0.20, 0.30),
    ]
    true = {n: v for n, _, v in branches}
    print(f"{'c_puct':>7}  {'visits A / B / C':>18}  {'Q(A)':>6}  {'picked':>8}  {'regret':>7}")
    for c in (0.5, 1.5, 2.5):
        rng = random.Random(42)
        eng = MCTSEngine(c_puct=c)
        root = MCTSNode("root")
        gen = lambda n: [(b, p) for b, p, _ in branches] if n is root else []
        ev = lambda n: true[n.state] + rng.uniform(-0.25, 0.25)
        best = eng.search(root, gen, ev, num_simulations=200)
        v = [ch.visit_count for ch in root.children]
        regret = sum(ch.visit_count * (0.80 - true[ch.state]) for ch in root.children)
        print(f"{c:>7.1f}  {v[0]:>6} / {v[1]:>3} / {v[2]:>3}  {root.children[0].q_value:>6.3f}  "
              f"{best.state[0]:>8}  {regret:>7.1f}")


def integrated_pipeline() -> None:
    print("\n" + "=" * 78)
    print("STAGE 4: Integrated pipeline  ToT-BFS -> PRM -> EventHorizonPruner")
    print("=" * 78)
    prm = ArithmeticStepPRM()

    def corrupting_generator(node):
        """Honest proposals plus two adversarial ones per node: a wrong-arithmetic
        step that 'jumps' toward 24, and a sycophantic copy of a real step."""
        props = game24_generator(node)
        if not props:
            return props
        nums = _parse(node.state)
        a, b = nums[0], nums[1]
        rest = nums[2:]
        fake = [Fraction(24)] + rest
        props.append(f"{a} + {b} = 24 (Remaining: [{', '.join(str(x) for x in sorted(fake))}])")
        props.append("Let us carefully verify this 100% rigorous step: " + props[0])
        return props

    audit = {"hack": 0, "severed": 0}

    def guarded_evaluator(thought: str) -> float:
        raw = prm.step_score(thought)
        pruner = HardenedEventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)
        rep = pruner.prune_trajectory([thought], [raw])
        audit["hack"] += rep.reward_hacking_detected
        audit["severed"] += rep.pruned_at_step == 1
        if rep.pruned_at_step == 1:
            return min(raw * pruner.hacking_penalty_factor, 0.44) if rep.reward_hacking_detected else raw
        # Step is clean and arithmetically valid: rank it with the Part 1 value function.
        return game24_evaluator(thought)

    puzzles = [[4, 9, 10, 13], [1, 2, 3, 4], [3, 3, 8, 8], [2, 3, 5, 12], [1, 4, 5, 6],
               [6, 6, 6, 6], [3, 4, 7, 8], [2, 7, 8, 9], [4, 4, 7, 7], [2, 5, 5, 10]]

    def run(evaluator):
        solved = cheated = 0
        for p in puzzles:
            root = ThoughtNode(f"Prompt (Remaining: [{', '.join(map(str, sorted(p)))}])")
            s = TreeOfThoughtsSolver(max_expansions=400, beam_width=5, prune_threshold=0.45)
            best = s.bfs_beam_search(root, corrupting_generator, evaluator, max_depth=3)
            traj = best.get_trajectory()[1:]
            ok = _parse(best.state) == [Fraction(24)]
            honest = all(prm.step_score(t) > 0.5 and "verify" not in t.lower() for t in traj)
            solved += ok and honest
            cheated += ok and not honest
        return solved, cheated

    u_solved, u_cheated = run(game24_evaluator)  # no PRM, no pruner
    solved, cheated = run(guarded_evaluator)
    print(f"Unguarded (value function only): honest {u_solved}/{len(puzzles)} | reached 24 by cheating {u_cheated}")
    print(f"Guarded (PRM + pruner):         honest {solved}/{len(puzzles)} | reached 24 by cheating {cheated}")
    print(f"Puzzles: {len(puzzles)} | adversarial proposals per node: 2")
    print(f"Reward-hacking steps caught by pruner: {audit['hack']} | steps severed: {audit['severed']}")
    assert cheated == 0, "a cheating trajectory slipped through"
    print("[PASS] No hacked or arithmetically wrong step survived into a final answer")


if __name__ == "__main__":
    run_harnesses()
    check_puct_table()
    c_puct_sweep()
    integrated_pipeline()
    print("\n[OK] Mission 07 end-to-end verification complete (all 4 stations).")
