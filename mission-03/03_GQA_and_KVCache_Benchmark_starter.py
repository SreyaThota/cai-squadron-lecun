"""
Mission 03 — Part 3: Data & Payload Specialist
Grouped-Query Attention (GQA) & Autoregressive KV-Cache Benchmark

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class KVCache:
    """
    Key-Value Cache for autoregressive token decoding.
    Maintains cached keys and values to prevent O(N^2) token recomputation.
    """
    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int, device="cpu"):
        self.k_cache = torch.zeros(max_batch_size, n_kv_heads, max_seq_len, head_dim, device=device)
        self.v_cache = torch.zeros(max_batch_size, n_kv_heads, max_seq_len, head_dim, device=device)
        self.seq_len = 0

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cache with newly computed key and value slices.
        """
        b, h, seq_len, d = k_val.shape
        self.k_cache[:b, :h, start_pos : start_pos + seq_len] = k_val
        self.v_cache[:b, :h, start_pos : start_pos + seq_len] = v_val
        self.seq_len = start_pos + seq_len
        return self.k_cache[:b, :h, : self.seq_len], self.v_cache[:b, :h, : self.seq_len]

    def get_memory_mb(self) -> float:
        """Returns memory occupied by cache in Megabytes."""
        bytes_total = self.k_cache.element_size() * (self.k_cache.nelement() + self.v_cache.nelement())
        return bytes_total / (1024 * 1024)


class GQAAttention(nn.Module):
    """
    Grouped-Query Attention layer (e.g. LLaMA 3 style with 32 Query heads and 8 KV heads).
    """
    def __init__(self, d_model: int = 1024, num_q_heads: int = 16, num_kv_heads: int = 4):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.w_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None
    ) -> torch.Tensor:
        b, seq_len, _ = x.shape

        q = self.w_q(x).view(b, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)

        # Broadcast KV heads to match Q heads: (b, num_kv_heads, 1, seq_len, d) -> (b, num_q_heads, seq_len, d)
        k = torch.repeat_interleave(k, repeats=self.num_queries_per_kv, dim=1)
        v = torch.repeat_interleave(v, repeats=self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.w_o(out)


def run_kv_cache_benchmark():
    print("=== Subsystem 03: KV-Cache & GQA Memory Footprint Benchmark ===")
    seq_lengths = [1024, 4096, 16384, 32768]
    head_dim = 128
    batch_size = 1

    print(f"{'Seq Length':<12} | {'MHA (32 KV Heads)':<18} | {'GQA (8 KV Heads)':<18} | {'MQA (1 KV Head)':<18} | {'VRAM Savings (GQA vs MHA)':<25}")
    print("-" * 105)

    for n in seq_lengths:
        # 2 tensors (K, V) * 2 bytes (float16) * batch * heads * seq_len * dim
        mha_mb = (2 * 2 * batch_size * 32 * n * head_dim) / (1024 * 1024)
        gqa_mb = (2 * 2 * batch_size * 8 * n * head_dim) / (1024 * 1024)
        mqa_mb = (2 * 2 * batch_size * 1 * n * head_dim) / (1024 * 1024)
        savings = (1.0 - gqa_mb / mha_mb) * 100.0

        print(f"{n:<12} | {mha_mb:>14.2f} MB | {gqa_mb:>14.2f} MB | {mqa_mb:>14.2f} MB | {savings:>22.1f}%")

    print("-" * 105)
    print("Benchmark complete: GQA achieves 75% KV-cache VRAM reduction while maintaining multi-head expressivity!")


if __name__ == "__main__":
    run_kv_cache_benchmark()
