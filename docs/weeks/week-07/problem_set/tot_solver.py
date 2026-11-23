"""
Mission 07 — Part 1: Systems Architect (Lead Engineer)
Tree-of-Thoughts (ToT) Breadth-First Beam Search & Depth-First Backtracking Solver

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class ThoughtNode:
    """Represents a discrete semantic reasoning step z_t in a Tree-of-Thoughts."""
    state: str
    depth: int = 0
    step_score: float = 1.0
    cumulative_score: float = 1.0
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    is_pruned: bool = False

    def add_child(self, child_state: str, step_score: float) -> "ThoughtNode":
        """Creates and attaches a child ThoughtNode, inheriting minimum-link cumulative score."""
        child = ThoughtNode(
            state=child_state,
            depth=self.depth + 1,
            step_score=step_score,
            cumulative_score=min(self.cumulative_score, step_score),
            parent=self,
        )
        self.children.append(child)
        return child

    def get_trajectory(self) -> List[str]:
        """Reconstructs the full chain of thoughts [z_0, z_1, ..., z_t] from root to this node."""
        path: List[str] = []
        curr: Optional[ThoughtNode] = self
        while curr is not None:
            path.append(curr.state)
            curr = curr.parent
        return list(reversed(path))


class TreeOfThoughtsSolver:
    """
    Executes deliberate System 2 Tree-of-Thoughts search under a strict inference compute budget.
    Supports both ToT-BFS (Beam Search of width B) and ToT-DFS (Chronological Backtracking).
    """

    def __init__(self, max_expansions: int = 16, beam_width: int = 2, prune_threshold: float = 0.45):
        self.max_expansions = max_expansions
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold
        self.expansion_count = 0

    def bfs_beam_search(
        self,
        root: ThoughtNode,
        generator_fn: Callable[[ThoughtNode], List[str]],
        evaluator_fn: Callable[[str], float],
        max_depth: int = 3,
    ) -> ThoughtNode:
        """
        TODO (Systems Architect): Implement Breadth-First Beam Search (ToT-BFS).
        1. Reset self.expansion_count = 0 and initialize `beam = [root]`, `best_node = root`.
        2. For each depth level d in range(max_depth):
           a. Initialize `candidates: List[ThoughtNode] = []`.
           b. For each `node` in `beam`:
              - If self.expansion_count >= self.max_expansions, break out of expansion.
              - Generate candidate thought strings via `generator_fn(node)`.
              - For each thought string:
                * Increment `self.expansion_count += 1` (stop if budget reached).
                * Score the thought with `score = evaluator_fn(thought_str)`.
                * Attach `child = node.add_child(thought_str, score)`.
                * If `score < self.prune_threshold`, mark `child.is_pruned = True`;
                  otherwise append `child` to `candidates`.
           c. If `candidates` is empty, break.
           d. Sort `candidates` by `(c.cumulative_score, c.step_score)` descending and keep top `self.beam_width`.
           e. Update `best_node` to the top node in the new `beam`.
        3. Return `best_node`.
        """
        self.expansion_count = 0
        beam: List[ThoughtNode] = [root]
        best_node: ThoughtNode = root

        for _ in range(max_depth):
            candidates: List[ThoughtNode] = []
            for node in beam:
                if self.expansion_count >= self.max_expansions:
                    break
                for thought_str in generator_fn(node):
                    if self.expansion_count >= self.max_expansions:
                        break
                    self.expansion_count += 1
                    score = evaluator_fn(thought_str)
                    child = node.add_child(thought_str, score)
                    if score < self.prune_threshold:
                        child.is_pruned = True
                    else:
                        candidates.append(child)

            if not candidates:
                break

            candidates.sort(key=lambda c: (c.cumulative_score, c.step_score), reverse=True)
            beam = candidates[: self.beam_width]
            if beam[0].depth > best_node.depth or (
                beam[0].depth == best_node.depth and beam[0].cumulative_score > best_node.cumulative_score
            ):
                best_node = beam[0]

        return best_node

    def dfs_backtrack_search(
        self,
        root: ThoughtNode,
        generator_fn: Callable[[ThoughtNode], List[str]],
        evaluator_fn: Callable[[str], float],
        max_depth: int = 3,
    ) -> ThoughtNode:
        """
        TODO (Systems Architect): Implement Depth-First Backtracking Search (ToT-DFS).
        1. Reset `self.expansion_count = 0` and track `best_node = root`.
        2. Define a recursive helper `_dfs(node: ThoughtNode)`:
           - If `node.depth >= max_depth` or `self.expansion_count >= self.max_expansions`, return.
           - Expand children via `generator_fn(node)`, scoring each child while respecting `max_expansions`.
           - Prune children with `step_score < self.prune_threshold` (`child.is_pruned = True`).
           - Sort valid children by `step_score` descending and recursively visit each valid child,
             updating `best_node` whenever a deeper or higher-scoring valid trajectory is reached.
        3. Return `best_node`.
        """
        self.expansion_count = 0
        best_node: ThoughtNode = root

        def _dfs(node: ThoughtNode) -> None:
            nonlocal best_node
            if node.depth > best_node.depth or (
                node.depth == best_node.depth and node.cumulative_score > best_node.cumulative_score
            ):
                best_node = node

            if node.depth >= max_depth or self.expansion_count >= self.max_expansions:
                return

            valid_children: List[ThoughtNode] = []
            for thought_str in generator_fn(node):
                if self.expansion_count >= self.max_expansions:
                    break
                self.expansion_count += 1
                score = evaluator_fn(thought_str)
                child = node.add_child(thought_str, score)
                if score < self.prune_threshold:
                    child.is_pruned = True
                else:
                    valid_children.append(child)

            valid_children.sort(key=lambda c: c.step_score, reverse=True)
            for child in valid_children:
                if self.expansion_count >= self.max_expansions and child.depth >= max_depth:
                    break
                _dfs(child)

        _dfs(root)
        return best_node


# ==============================================================================
# Systems Architect Extensions (Squadron LeCun)
#   1. Rule checks: budget, pruning, beam width, trajectory.
#   2. Real benchmark: Game of 24 ("thruster synthesis") with a heuristic value
#      function, comparing ToT-BFS vs ToT-DFS under different compute budgets.
# Pure standard library.
# ==============================================================================
import itertools
from fractions import Fraction

TARGET = Fraction(24)


def _fmt(nums: List[Fraction]) -> str:
    return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"


def _parse(state: str) -> List[Fraction]:
    """Each thought string ends with 'Remaining: [a, b, ...]'."""
    inner = state.rsplit("Remaining: [", 1)[1].split("]", 1)[0]
    return [Fraction(x) for x in inner.split(", ")] if inner else []


def _combine(a: Fraction, b: Fraction):
    yield f"{a} + {b}", a + b
    yield f"{a} * {b}", a * b
    yield f"{a} - {b}", a - b
    yield f"{b} - {a}", b - a
    if b != 0:
        yield f"{a} / {b}", a / b
    if a != 0:
        yield f"{b} / {a}", b / a


def game24_generator(node: ThoughtNode) -> List[str]:
    """Propose every 'pick two numbers, apply one operation' next step."""
    nums = _parse(node.state)
    if len(nums) < 2:
        return []
    out, seen = [], set()
    for i, j in itertools.combinations(range(len(nums)), 2):
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        for expr, val in _combine(nums[i], nums[j]):
            new = rest + [val]
            key = tuple(sorted(new))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{expr} = {val} (Remaining: {_fmt(new)})")
    return out


def _reachable(nums: List[Fraction]) -> bool:
    if len(nums) == 1:
        return nums[0] == TARGET
    for i, j in itertools.combinations(range(len(nums)), 2):
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        if any(_reachable(rest + [v]) for _, v in _combine(nums[i], nums[j])):
            return True
    return False


def game24_evaluator(thought: str) -> float:
    """
    Cheap value function in the style of the ToT paper's "sure / likely / impossible"
    prompt. It only looks one step ahead (2 numbers left) and otherwise guesses from
    how close the numbers are to 24, so it is a noisy heuristic, not an oracle.
    """
    nums = _parse(thought)
    if len(nums) == 1:
        return 0.99 if nums[0] == TARGET else 0.05
    if len(nums) == 2:
        return 0.90 if _reachable(nums) else 0.20
    closest = min(abs(float(n) - 24) for n in nums)
    big = max(float(n) for n in nums)
    score = 0.75 - 0.01 * min(closest, 20) - (0.15 if big > 60 else 0.0)
    return max(0.46, round(score, 3))  # 3-number states stay above the prune threshold


def _run_rule_checks() -> None:
    print("\n=== Extension 1: Rule Checks ===")
    root = ThoughtNode("Prompt (Remaining: [4, 9, 10, 13])")
    solver = TreeOfThoughtsSolver(max_expansions=7, beam_width=2, prune_threshold=0.45)
    solver.bfs_beam_search(root, game24_generator, game24_evaluator, max_depth=3)
    assert solver.expansion_count <= 7, "BFS broke the compute budget"
    print(f"[PASS] BFS budget respected: {solver.expansion_count}/7 expansions")

    solver.dfs_backtrack_search(ThoughtNode("Prompt (Remaining: [4, 9, 10, 13])"),
                                game24_generator, game24_evaluator, max_depth=3)
    assert solver.expansion_count <= 7, "DFS broke the compute budget"
    print(f"[PASS] DFS budget respected: {solver.expansion_count}/7 expansions")

    def walk(n):
        yield n
        for c in n.children:
            yield from walk(c)

    root = ThoughtNode("Prompt (Remaining: [4, 9, 10, 13])")
    s = TreeOfThoughtsSolver(max_expansions=500, beam_width=3, prune_threshold=0.45)
    s.bfs_beam_search(root, game24_generator, game24_evaluator, max_depth=3)
    nodes = list(walk(root))
    assert all(not n.children for n in nodes if n.is_pruned), "a pruned node was expanded"
    assert all(n.step_score < 0.45 for n in nodes if n.is_pruned), "wrong node pruned"
    print("[PASS] Pruned nodes are below threshold and never expanded")
    per_depth = {}
    for n in nodes:
        if n.children:
            per_depth.setdefault(n.depth, 0)
            per_depth[n.depth] += 1
    assert all(v <= 3 for d, v in per_depth.items() if d > 0), "beam width exceeded"
    print(f"[PASS] At most B=3 nodes expanded per level: {per_depth}")

    leaf = max(nodes, key=lambda n: n.depth)
    traj = leaf.get_trajectory()
    assert len(traj) == leaf.depth + 1 and traj[0] == root.state
    print(f"[PASS] get_trajectory() length = depth + 1 = {len(traj)}")


PUZZLES = [
    [4, 9, 10, 13], [1, 2, 3, 4], [3, 3, 8, 8], [1, 5, 5, 5], [2, 3, 5, 12],
    [1, 4, 5, 6], [6, 6, 6, 6], [2, 2, 2, 9], [1, 3, 4, 6], [5, 5, 5, 1],
    [3, 4, 7, 8], [2, 7, 8, 9], [1, 1, 11, 11], [4, 4, 7, 7], [2, 5, 5, 10],
    [1, 6, 11, 13], [3, 5, 7, 13], [1, 2, 7, 7], [4, 6, 9, 9], [2, 4, 10, 10],
]


def _run_benchmark() -> None:
    print("\n=== Extension 2: Game-of-24 Benchmark (ToT-BFS vs ToT-DFS) ===")
    solvable = [p for p in PUZZLES if _reachable([Fraction(x) for x in p])]
    print(f"Puzzles: {len(solvable)} solvable | prune_threshold = 0.45 | max_depth = 3")
    print(f"{'Method':<14}{'Budget':>7}{'Solved':>9}{'Avg exp.':>10}")
    for budget in (40, 100, 250, 600):
        for name, B in (("BFS (B=1)", 1), ("BFS (B=2)", 2), ("BFS (B=5)", 5), ("DFS", None)):
            solved, used = 0, 0
            for p in solvable:
                root = ThoughtNode(f"Prompt (Remaining: {_fmt([Fraction(x) for x in p])})")
                s = TreeOfThoughtsSolver(max_expansions=budget, beam_width=B or 1, prune_threshold=0.45)
                if B is None:
                    best = s.dfs_backtrack_search(root, game24_generator, game24_evaluator, 3)
                else:
                    best = s.bfs_beam_search(root, game24_generator, game24_evaluator, 3)
                solved += _parse(best.state) == [TARGET]
                used += s.expansion_count
            print(f"{name:<14}{budget:>7}{solved:>5}/{len(solvable):<3}{used / len(solvable):>10.1f}")


# ==============================================================================
# Verification Harness for Part 1 (Systems Architect)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 1): Tree-of-Thoughts Solver Verification ===")

    # Mock Sagittarius A* 24-Vector Thruster Synthesis Tree
    MOCK_BRANCHES = {
        "Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]": [
            "Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID]",
            "Step 1B: 10 + 13 = 23 (Remaining: [4, 9, 23]) [DEAD-END]",
        ],
        "Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID]": [
            "Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL]",
            "Step 2B: 4 + 4 = 8 (Remaining: [8, 10]) [DEAD-END]",
        ],
        "Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL]": [
            "Step 3A: 4 * 6 = 24.0 km/s -> Escape Geodesic Locked! [GOAL]",
        ],
    }

    def mock_generator(node: ThoughtNode) -> List[str]:
        return MOCK_BRANCHES.get(node.state, [])

    def mock_evaluator(thought: str) -> float:
        if "[GOAL]" in thought:
            return 0.98
        if "[OPTIMAL]" in thought:
            return 0.92
        if "[VALID]" in thought:
            return 0.85
        return 0.25  # Dead-end branch below prune_threshold

    solver = TreeOfThoughtsSolver(max_expansions=12, beam_width=2, prune_threshold=0.45)

    root_bfs = ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]")
    best_bfs = solver.bfs_beam_search(root_bfs, mock_generator, mock_evaluator, max_depth=3)
    print(f"[ToT-BFS] Expansions Used: {solver.expansion_count}/{solver.max_expansions}")
    print(f"[ToT-BFS] Best Terminal Score: {best_bfs.cumulative_score:.2f}")
    print(f"[ToT-BFS] Winning Trajectory: {' -> '.join(best_bfs.get_trajectory())}")

    root_dfs = ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]")
    best_dfs = solver.dfs_backtrack_search(root_dfs, mock_generator, mock_evaluator, max_depth=3)
    print(f"[ToT-DFS] Expansions Used: {solver.expansion_count}/{solver.max_expansions}")
    print(f"[ToT-DFS] Winning Trajectory: {' -> '.join(best_dfs.get_trajectory())}")

    assert "[GOAL]" in best_bfs.state, "ToT-BFS failed to reach [GOAL] state!"
    assert "[GOAL]" in best_dfs.state, "ToT-DFS failed to reach [GOAL] state!"
    print("[OK] Part 1 (Systems Architect) verification passed.")

    _run_rule_checks()
    _run_benchmark()
    print("\n[OK] Part 1 extensions passed.")
