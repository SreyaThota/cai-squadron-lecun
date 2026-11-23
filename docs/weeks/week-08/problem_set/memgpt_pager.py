"""
Mission 08 — Part 3: Data & Payload Specialist
MemGPT Virtual Context Pager (Packer et al., 2023) & Generative Agents Tri-Factor Retrieval (Park et al., 2023)

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula)
Subsystem 08: ReWOO DAG Planner, Reflexion Surgeon & MemGPT Virtual Pager
"""

import math
from typing import Any, Dict, List, Optional


class ArchivalMemoryNode:
    """Represents a persistent memory record in External Archival Storage."""

    def __init__(
        self,
        node_id: str,
        text: str,
        importance: float,
        hours_since_access: float,
        embedding: List[float],
        t_valid: int = 0,
        t_invalid: Optional[int] = None,
    ):
        self.node_id = node_id
        self.text = text
        self.importance = importance  # I(m) in [1.0, 10.0]
        self.hours_since_access = hours_since_access  # Delta t since last access
        self.embedding = embedding
        self.t_valid = t_valid
        self.t_invalid = t_invalid  # None means still currently valid (Zep bi-temporal edge)


class MemGPTVirtualPager:
    """
    OS-Inspired Virtual Context Pager separating:
      - Main Context (RAM): System Instructions + R/W Core Memory Blocks + FIFO Message Queue
      - External Context (Disk): Recall Storage + Archival Storage ranked by S(m, q)
    Triggers a `memory_pressure_warning` interrupt when active token usage >= tau_warn * token_budget.
    """

    def __init__(self, token_budget: int = 4096, tau_warn: float = 0.70, decay_gamma: float = 0.99):
        self.token_budget = token_budget
        self.tau_warn = tau_warn
        self.decay_gamma = decay_gamma
        self.core_memory: Dict[str, str] = {"persona": "", "task_state": ""}
        self.fifo_queue: List[Dict[str, Any]] = []
        self.archival_store: List[ArchivalMemoryNode] = []
        self.system_interrupts: List[str] = []

    def core_memory_append(self, block_key: str, content: str) -> None:
        """MemGPT syscall: appends text to a read/write Core Memory block."""
        if block_key not in self.core_memory:
            raise KeyError(f"Unknown core_memory block: {block_key}")
        sep = " " if self.core_memory[block_key] else ""
        self.core_memory[block_key] = f"{self.core_memory[block_key]}{sep}{content}".strip()

    def core_memory_replace(self, block_key: str, old_text: str, new_text: str) -> None:
        """MemGPT syscall: replaces `old_text` with `new_text` inside a Core Memory block."""
        if block_key not in self.core_memory:
            raise KeyError(f"Unknown core_memory block: {block_key}")
        if old_text not in self.core_memory[block_key]:
            raise ValueError(f"Target substring '{old_text}' not found in core_memory['{block_key}']")
        self.core_memory[block_key] = self.core_memory[block_key].replace(old_text, new_text)

    def archival_memory_insert(self, node: ArchivalMemoryNode) -> None:
        """MemGPT syscall: inserts an ArchivalMemoryNode into persistent Archival Storage."""
        self.archival_store.append(node)

    def enqueue_message(self, role: str, text: str, tokens: Optional[int] = None) -> bool:
        """Appends a conversational turn to the FIFO queue and checks for memory pressure."""
        est_tokens = tokens if tokens is not None else len(text.split()) * 2
        self.fifo_queue.append({"role": role, "text": text, "tokens": est_tokens})
        return self.check_memory_pressure()

    def current_token_usage(self) -> int:
        """Estimate active context token utilization across core_memory and fifo_queue."""
        core_tokens = sum(len(v.split()) * 2 for v in self.core_memory.values())
        fifo_tokens = sum(msg.get("tokens", len(msg["text"].split()) * 2) for msg in self.fifo_queue)
        return core_tokens + fifo_tokens

    def check_memory_pressure(self) -> bool:
        """
        TODO (Data Specialist): Return True if `current_token_usage() / self.token_budget >= self.tau_warn`.
        If True:
          1. Record a system interrupt in `self.system_interrupts`.
          2. Flush the oldest 50% of `self.fifo_queue` into `self.archival_store` as a consolidated
             summary `ArchivalMemoryNode` so active context utilization drops safely below `tau_warn`.
        """
        usage = self.current_token_usage()
        ratio = usage / float(self.token_budget)
        if ratio < self.tau_warn:
            return False

        evict_count = max(1, len(self.fifo_queue) // 2)
        evicted_msgs = self.fifo_queue[:evict_count]
        self.fifo_queue = self.fifo_queue[evict_count:]

        summary_text = " | ".join(m["text"] for m in evicted_msgs)
        summary_node = ArchivalMemoryNode(
            node_id=f"evict_{len(self.archival_store) + 1}",
            text=f"[FIFO_FLUSH_SUMMARY]: {summary_text}",
            importance=7.5,
            hours_since_access=0.0,
            embedding=[0.5, 0.5, 0.5],
        )
        self.archival_store.append(summary_node)
        self.system_interrupts.append(
            f"SYSTEM_INTERRUPT(memory_pressure_warning): usage={usage}/{self.token_budget} ({ratio:.1%}) >= {self.tau_warn:.0%}. "
            f"Evicted {evict_count} oldest FIFO messages to Archival Storage."
        )
        return True

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def score_memory(
        self,
        node: ArchivalMemoryNode,
        query_embedding: List[float],
        alpha_rec: float = 1.0,
        alpha_imp: float = 1.0,
        alpha_rel: float = 1.0,
    ) -> float:
        """
        TODO (Data Specialist): Compute Generative Agents (Park et al., 2023) tri-partite retrieval score:
            R(m)      = decay_gamma ** node.hours_since_access
            I_norm(m) = node.importance / 10.0
            Sim(m, q) = cosine_similarity(node.embedding, query_embedding)
            S(m, q)   = alpha_rec * R(m) + alpha_imp * I_norm(m) + alpha_rel * Sim(m, q)
        """
        recency = self.decay_gamma ** node.hours_since_access
        importance_norm = node.importance / 10.0
        relevance = self._cosine_similarity(node.embedding, query_embedding)
        return alpha_rec * recency + alpha_imp * importance_norm + alpha_rel * relevance

    def archival_memory_search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        alpha_rec: float = 1.0,
        alpha_imp: float = 1.0,
        alpha_rel: float = 1.0,
        touch_on_access: bool = False,
    ) -> List[ArchivalMemoryNode]:
        """
        TODO (Data Specialist): Rank all currently valid nodes (`t_invalid is None`) in
        `self.archival_store` by `score_memory()` descending and return the `top_k` nodes.
        If `touch_on_access` is True, reset `hours_since_access = 0.0` on the returned nodes.
        """
        active_nodes = [n for n in self.archival_store if n.t_invalid is None]
        ranked = sorted(
            active_nodes,
            key=lambda n: self.score_memory(n, query_embedding, alpha_rec, alpha_imp, alpha_rel),
            reverse=True,
        )
        selected = ranked[:top_k]
        if touch_on_access:
            for n in selected:
                n.hours_since_access = 0.0
        return selected


# ==============================================================================
# Verification Harness for Part 3 (Data & Payload Specialist)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 08 (Part 3): MemGPT Virtual Pager & Tri-Factor Retrieval Verification ===")

    pager = MemGPTVirtualPager(token_budget=1000, tau_warn=0.70, decay_gamma=0.99)
    pager.core_memory_append("persona", "AURA Tactical Navigator v8.0")
    pager.core_memory_append("task_state", "Orbit=Cygnus_X1_Outer_Corona")
    pager.core_memory_replace("task_state", "Outer_Corona", "Inner_Shock_Front")

    # Populate archival memories with contrasting Recency, Importance, and Relevance
    pager.archival_memory_insert(
        ArchivalMemoryNode(
            node_id="mem_safety",
            text="CRITICAL DIRECTIVE: Never drop magnetic shielding below 42% near Cygnus X-1 X-ray flares.",
            importance=10.0,
            hours_since_access=48.0,
            embedding=[0.92, 0.35, 0.15],
        )
    )
    pager.archival_memory_insert(
        ArchivalMemoryNode(
            node_id="mem_chatter",
            text="Bridge crew coffee dispenser recalibrated in Mess Hall Deck 3.",
            importance=1.5,
            hours_since_access=1.0,
            embedding=[0.10, 0.85, 0.45],
        )
    )
    pager.archival_memory_insert(
        ArchivalMemoryNode(
            node_id="mem_telemetry",
            text="X-ray flare harmonic filter locked to 6.4 keV iron K-alpha line.",
            importance=8.5,
            hours_since_access=6.0,
            embedding=[0.88, 0.42, 0.20],
        )
    )

    # Trigger 70% memory pressure warning via FIFO queue turns
    for i in range(4):
        triggered = pager.enqueue_message("user", f"Shockwave telemetry packet {i}", tokens=200)
        if triggered:
            print(" ", pager.system_interrupts[-1])

    query_vec = [0.95, 0.30, 0.10]  # Query about Cygnus X-1 X-ray shield safety
    top_memories = pager.archival_memory_search(query_vec, top_k=2)
    print("\nTop-2 Retrieved Archival Memories for X-Ray Shield Query:")
    for node in top_memories:
        s = pager.score_memory(node, query_vec)
        print(f"  [{node.node_id}] S(m,q)={s:.4f} | I={node.importance} | dt={node.hours_since_access}h | {node.text}")

    assert len(pager.system_interrupts) >= 1, "70% token-pressure interrupt failed to trigger!"
    assert top_memories[0].node_id in {"mem_safety", "mem_telemetry"}, "High-importance safety memory should outrank trivial recent chatter!"
    print("[OK] Part 3 (Data & Payload Specialist) verification passed.")
