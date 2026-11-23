"""
mcts_protocol.py
Mission 07 / Part 2: 4-Phase Monte Carlo Tree Search with PUCT selection.

Pure Python 3.11+, standard library only.

PUCT(s, a) = Q(s, a) + c_puct * P(a | s) * sqrt(N(s)) / (1 + N(s, a))
  Q(s, a)  = W(s, a) / N(s, a) if N(s, a) > 0 else 0.0
  N(s)     = max(1, parent.visit_count)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""

    state: str
    prior: float = 1.0                      # P_theta(a | s) from the LLM policy
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0                    # N(s, a)
    value_sum: float = 0.0                  # W(s, a)
    is_terminal: bool = False
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean value Q = W / N, or 0.0 for an unvisited node."""
        if self.visit_count > 0:
            return self.value_sum / self.visit_count
        return 0.0

    def is_leaf(self) -> bool:
        """True if the node has no children yet (unexpanded) or is terminal."""
        return self.is_terminal or not self.children

    def get_trajectory(self) -> List[str]:
        """Reasoning chain of states from the root to this node."""
        path: List[str] = []
        node: Optional[MCTSNode] = self
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))

    def __repr__(self) -> str:
        return (f"MCTSNode(state={self.state!r}, N={self.visit_count}, "
                f"Q={self.q_value:.3f}, P={self.prior:.3f})")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class MCTSEngine:
    """4-phase MCTS (select, expand, simulate, backpropagate) with PUCT."""

    def __init__(self, c_puct: float = 1.414):
        self.c_puct = c_puct

    # ---- PUCT --------------------------------------------------------------
    def puct_score(self, node: MCTSNode, parent: Optional[MCTSNode] = None) -> float:
        """Exact PUCT formula for the edge (parent -> node)."""
        parent = parent if parent is not None else node.parent
        parent_visits = parent.visit_count if parent is not None else 0
        n_s = max(1, parent_visits)
        q = node.value_sum / node.visit_count if node.visit_count > 0 else 0.0
        exploration = self.c_puct * node.prior * math.sqrt(n_s) / (1 + node.visit_count)
        return q + exploration

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Child with the highest PUCT score (first one wins ties)."""
        if not node.children:
            raise ValueError("select_best_child called on a node without children")
        best_child = node.children[0]
        best_score = self.puct_score(best_child, node)
        for child in node.children[1:]:
            score = self.puct_score(child, node)
            if score > best_score:
                best_child, best_score = child, score
        return best_child

    # ---- Phase 1: Selection ------------------------------------------------
    def select_leaf(self, root: MCTSNode) -> MCTSNode:
        """Greedily descend via PUCT until an unexpanded or terminal leaf."""
        node = root
        while node.children and not node.is_terminal:
            node = self.select_best_child(node)
        return node

    # ---- Phase 2: Expansion ------------------------------------------------
    def expand(
        self,
        node: MCTSNode,
        candidate_priors: List[Tuple[str, float]],
    ) -> List[MCTSNode]:
        """Create one child per (thought_state, prior_prob) pair."""
        if node.is_terminal:
            return []
        new_children: List[MCTSNode] = []
        for thought_state, prior_prob in candidate_priors:
            child = MCTSNode(state=thought_state, prior=prior_prob, parent=node)
            node.children.append(child)
            new_children.append(child)
        node.is_expanded = True
        return new_children

    # ---- Phase 3: Simulation / Evaluation ---------------------------------
    def simulate_rollout(
        self,
        node: MCTSNode,
        rollout_evaluator: Callable[[MCTSNode], float],
    ) -> float:
        """Evaluate a node via rollout or PRM value estimate, clamped to [0, 1]."""
        value = float(rollout_evaluator(node))
        return max(0.0, min(1.0, value))

    # ---- Phase 4: Backpropagation -----------------------------------------
    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Walk to the root: visit_count += 1, value_sum += reward at every ancestor."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            current = current.parent

    # ---- Full search loop --------------------------------------------------
    def search(
        self,
        root: MCTSNode,
        candidate_generator: Callable[[MCTSNode], List[Tuple[str, float]]],
        rollout_evaluator: Callable[[MCTSNode], float],
        num_simulations: int = 50,
    ) -> MCTSNode:
        """Run MCTS; return the most-visited child of the root (or root if none)."""
        for _ in range(num_simulations):
            leaf = self.select_leaf(root)
            if not leaf.is_terminal and not leaf.is_expanded:
                priors = candidate_generator(leaf)
                if priors:
                    children = self.expand(leaf, priors)
                    # evaluate the first new child (or leaf if nothing was added)
                    target = children[0] if children else leaf
                else:
                    leaf.is_terminal = True
                    target = leaf
            else:
                target = leaf
            reward = self.simulate_rollout(target, rollout_evaluator)
            self.backpropagate(target, reward)

        if not root.children:
            return root
        return max(root.children, key=lambda c: c.visit_count)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    engine = MCTSEngine(c_puct=1.0)

    # PUCT formula check: parent N=4, child N=1, W=0.5, P=0.4
    parent = MCTSNode("root", visit_count=4)
    child = MCTSNode("a", prior=0.4, parent=parent, visit_count=1, value_sum=0.5)
    parent.children.append(child)
    expected = 0.5 + 1.0 * 0.4 * math.sqrt(4) / (1 + 1)
    assert abs(engine.puct_score(child) - expected) < 1e-12

    # Unvisited child: Q = 0, N(s) floors at 1
    fresh_parent = MCTSNode("p", visit_count=0)
    fresh = MCTSNode("x", prior=0.5, parent=fresh_parent)
    assert abs(engine.puct_score(fresh) - 0.5) < 1e-12

    # Backpropagation
    root = MCTSNode("root")
    a = MCTSNode("a", parent=root)
    b = MCTSNode("b", parent=a)
    engine.backpropagate(b, 0.8)
    assert (root.visit_count, a.visit_count, b.visit_count) == (1, 1, 1)
    assert abs(root.value_sum - 0.8) < 1e-12

    # Full search: "good" branch should win
    def gen(n: MCTSNode):
        if n.parent is None:
            return [("good", 0.5), ("bad", 0.5)]
        return []

    def rollout(n: MCTSNode) -> float:
        return 0.9 if n.state == "good" else 0.1

    root = MCTSNode("root")
    best = engine.search(root, gen, rollout, num_simulations=30)
    assert best.state == "good", best
    print("mcts_protocol.py self-test passed:", best)


if __name__ == "__main__":
    _self_test()
