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
