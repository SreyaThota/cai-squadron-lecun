"""
Mission 03 — Part 3: Data & Payload Specialist
Grouped-Query Attention (GQA) & Autoregressive KV-Cache Benchmark

Starship Operational Unit: USSE Noosphere Bridge Crew
Course: Conversational AI — Winter Semester 2026/27
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver
"""

import argparse
import math
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class KVCache:
    """
    Key-Value Cache for autoregressive token decoding.
    Retains past key and value projections so each new decoding step only computes
    projections for the single incoming token, shifting sequential generation
    from full sequence recomputation down to O(1) projection compute and O(N) attention read.
    """
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.k_cache = torch.zeros(
            max_batch_size, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            max_batch_size, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.seq_len = 0

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor, start_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cache with newly computed key and value slices for current token(s).

        Args:
            k_val: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
            v_val: Value tensor of shape (batch, n_kv_heads, seq_len, head_dim)
            start_pos: Current starting sequence position in cache

        Returns:
            Tuple of cached keys and values sliced up to current sequence length
        """
        b, h, seq_len, d = k_val.shape
        end_pos = start_pos + seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds maximum cache capacity {self.max_seq_len}."
            )

        self.k_cache[:b, :h, start_pos:end_pos] = k_val.to(self.dtype)
        self.v_cache[:b, :h, start_pos:end_pos] = v_val.to(self.dtype)
        self.seq_len = end_pos

        return self.k_cache[:b, :h, :self.seq_len], self.v_cache[:b, :h, :self.seq_len]

    def get_memory_mb(self) -> float:
        """Returns total memory allocated by cache buffers in Megabytes (MiB)."""
        bytes_total = self.k_cache.element_size() * (
            self.k_cache.nelement() + self.v_cache.nelement()
        )
        return bytes_total / (1024.0 * 1024.0)

    def reset(self):
        """Resets sequence length counter and zeroes cache state."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0


class GQAAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) layer with configurable query and key-value heads.
    Supports MHA (H_Q == H_KV), GQA (H_Q > H_KV > 1), and MQA (H_KV == 1).

    Includes two execution paths for head grouping:
    1. Copy-free grouped path (default, use_repeat=False): Reshapes Q to (B, H_KV, Q_per_KV, L, D)
       and unsqueezes K/V to (B, H_KV, 1, L, D) for broadcasting without memory replication.
    2. Explicit repeat_interleave path (use_repeat=True): Replicates K/V tensors across heads as
       specified in the problem brief.
    """
    def __init__(
        self,
        d_model: int = 1024,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert (
            num_q_heads % num_kv_heads == 0
        ), f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads

        self.head_dim = head_dim if head_dim is not None else (d_model // num_q_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections for Query, Key, Value, Output
        self.w_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False, dtype=dtype)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.w_o = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        use_causal_mask: bool = True,
        use_repeat: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for Grouped-Query Attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            start_pos: Sequence offset for autoregressive KV cache indexing
            kv_cache: Optional stateful KVCache instance
            use_causal_mask: Whether to apply causal masking during prompt ingestion
            use_repeat: If True, uses torch.repeat_interleave; if False, uses memory-efficient copy-free grouped matmul

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        b, seq_len, _ = x.shape

        # Linear projections: (b, seq_len, num_heads * head_dim) -> (b, num_heads, seq_len, head_dim)
        q = self.w_q(x).view(b, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Update or retain KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)

        total_kv_len = k.shape[2]

        if use_repeat and self.num_queries_per_kv > 1:
            # Brief-specified repeat_interleave path (explicit memory replication)
            k_exp = torch.repeat_interleave(k, repeats=self.num_queries_per_kv, dim=1)
            v_exp = torch.repeat_interleave(v, repeats=self.num_queries_per_kv, dim=1)

            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
            if use_causal_mask and seq_len > 1:
                mask = torch.full(
                    (seq_len, total_kv_len), float("-inf"), device=x.device, dtype=scores.dtype
                )
                mask = torch.triu(mask, diagonal=start_pos + 1)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)

            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v_exp)
            out = out.transpose(1, 2).contiguous().view(b, seq_len, self.num_q_heads * self.head_dim)
        else:
            # Copy-free grouped attention path (production standard)
            if self.num_queries_per_kv > 1:
                q_grouped = q.view(b, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
                k_grouped = k.unsqueeze(2)  # (b, num_kv_heads, 1, total_kv_len, head_dim)
                v_grouped = v.unsqueeze(2)  # (b, num_kv_heads, 1, total_kv_len, head_dim)

                scores = torch.matmul(q_grouped, k_grouped.transpose(-2, -1)) * self.scale
                if use_causal_mask and seq_len > 1:
                    mask = torch.full(
                        (seq_len, total_kv_len), float("-inf"), device=x.device, dtype=scores.dtype
                    )
                    mask = torch.triu(mask, diagonal=start_pos + 1)
                    scores = scores + mask.view(1, 1, 1, seq_len, total_kv_len)

                attn_weights = F.softmax(scores, dim=-1)
                out = torch.matmul(attn_weights, v_grouped)
                out = out.view(b, self.num_q_heads, seq_len, self.head_dim).transpose(1, 2).contiguous().view(
                    b, seq_len, self.num_q_heads * self.head_dim
                )
            else:
                # Standard MHA path (1:1 head matching)
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                if use_causal_mask and seq_len > 1:
                    mask = torch.full(
                        (seq_len, total_kv_len), float("-inf"), device=x.device, dtype=scores.dtype
                    )
                    mask = torch.triu(mask, diagonal=start_pos + 1)
                    scores = scores + mask.unsqueeze(0).unsqueeze(0)

                attn_weights = F.softmax(scores, dim=-1)
                out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(
                    b, seq_len, self.num_q_heads * self.head_dim
                )

        return self.w_o(out)


def benchmark_decoding_latency_ms(
    model: GQAAttention,
    kv_cache: KVCache,
    seq_len: int,
    head_dim: int,
    d_model: int,
    num_runs: int = 15,
    warmup: int = 3,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    use_repeat: bool = False,
) -> float:
    """
    Measures the empirical latency (in milliseconds) for decoding a single new token
    when the KV cache is already filled up to `seq_len` tokens.
    """
    model.eval()
    batch_size = 1

    kv_cache.reset()
    dummy_k = torch.randn(
        batch_size, model.num_kv_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    dummy_v = torch.randn(
        batch_size, model.num_kv_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    kv_cache.update(dummy_k, dummy_v, start_pos=0)

    single_token = torch.randn(batch_size, 1, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(
                single_token,
                start_pos=seq_len - 1,
                kv_cache=kv_cache,
                use_causal_mask=False,
                use_repeat=use_repeat,
            )

        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(
                single_token,
                start_pos=seq_len - 1,
                kv_cache=kv_cache,
                use_causal_mask=False,
                use_repeat=use_repeat,
            )
        total_time = time.perf_counter() - start_time

    return (total_time / num_runs) * 1000.0


def plot_kv_cache_benchmark(seq_lengths, mem_results, bw_results, output_path="kv_cache_memory.png"):
    """
    Plots KV-cache memory footprint and memory bandwidth curves comparing MHA, GQA, and MQA.
    """
    if not HAS_MATPLOTLIB:
        print("[INFO] matplotlib is not installed. Skipping plot generation.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    styles = {
        "MHA 32:32": {"color": "#e63946", "marker": "o", "label": "MHA (32 KV Heads)"},
        "GQA 4:1": {"color": "#457b9d", "marker": "s", "label": "GQA 4:1 (8 KV Heads)"},
        "MQA 32:1": {"color": "#2a9d8f", "marker": "^", "label": "MQA 32:1 (1 KV Head)"},
    }

    # Panel 1: Memory Footprint
    for name, data in mem_results.items():
        ax1.plot(seq_lengths, data, label=styles[name]["label"], color=styles[name]["color"],
                 marker=styles[name]["marker"], linewidth=2.2, markersize=7)
    ax1.set_title("Full Model KV-Cache Memory Footprint (32 Layers, fp16)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Context Length (N tokens)", fontsize=11)
    ax1.set_ylabel("KV-Cache Size (MB)", fontsize=11)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(seq_lengths)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Panel 2: Memory Bandwidth
    for name, data in bw_results.items():
        ax2.plot(seq_lengths, data, label=styles[name]["label"], color=styles[name]["color"],
                 marker=styles[name]["marker"], linewidth=2.2, markersize=7)
    ax2.set_title("Memory Bandwidth Traffic at 50 tokens/sec", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Context Length (N tokens)", fontsize=11)
    ax2.set_ylabel("Bandwidth Traffic (GB/s)", fontsize=11)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(seq_lengths)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Benchmark visualization plot saved to: {output_path}")


def run_kv_cache_benchmark(quick: bool = False):
    """
    Executes the comprehensive Mission 03 benchmark comparing:
    - Multi-Head Attention (MHA: 32 Q heads, 32 KV heads)
    - Grouped-Query Attention (GQA: 32 Q heads, 8 KV heads, 4:1)
    - Multi-Query Attention (MQA: 32 Q heads, 1 KV head, 32:1)
    across sequence lengths N in {1024, 4096, 16384, 32768}.
    """
    print("=" * 115)
    print("USSE NOOSPHERE — SUBSYSTEM 03: DISTRIBUTED ATTENTION & KV-CACHE BENCHMARK")
    print("Waypoint: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)")
    print("Role Deliverable: Part 3 — Data & Payload Specialist")
    print("=" * 115)

    seq_lengths = [1024, 4096] if quick else [1024, 4096, 16384, 32768]
    batch_size = 1
    num_q_heads = 32
    head_dim = 128
    d_model = num_q_heads * head_dim  # 4096 (standard LLaMA-3-8B dimension)
    num_layers = 32  # Standard LLaMA-3-8B transformer layers
    token_rate = 50  # tokens/sec for memory bandwidth calculation
    bytes_per_elem = 2  # float16 precision (2 bytes per activation/weight)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Compute Device: {device.upper()}")
    print(f"Model Configuration: d_model={d_model}, num_q_heads={num_q_heads}, head_dim={head_dim}, num_layers={num_layers}")
    print(f"KV-Cache Precision: float16 (2 bytes/elem), Batch Size: {batch_size}, Target Token Rate: {token_rate} tok/s\n")

    configs = {
        "MHA 32:32": 32,
        "GQA 4:1": 8,
        "MQA 32:1": 1,
    }

    mem_results = {"MHA 32:32": [], "GQA 4:1": [], "MQA 32:1": []}
    bw_results = {"MHA 32:32": [], "GQA 4:1": [], "MQA 32:1": []}

    # -------------------------------------------------------------
    # 1. Full Model KV-Cache Memory Footprint Table (32 layers, in MB)
    # -------------------------------------------------------------
    print("=== Table 1: Full Model KV-Cache Memory Footprint (32 Layers, fp16) ===")
    print(f"{'Seq Length (N)':<16} | {'MHA 32:32 (MB)':<18} | {'GQA 4:1 (MB)':<18} | {'MQA 32:1 (MB)':<18} | {'VRAM Reduction (GQA vs MHA)':<28}")
    print("-" * 110)

    for n in seq_lengths:
        # Full model cache size = 2 (K, V) * bytes * batch * kv_heads * seq_len * head_dim * num_layers / (1024^2)
        mha_mb = (2 * bytes_per_elem * batch_size * configs["MHA 32:32"] * n * head_dim * num_layers) / (1024 * 1024)
        gqa_mb = (2 * bytes_per_elem * batch_size * configs["GQA 4:1"] * n * head_dim * num_layers) / (1024 * 1024)
        mqa_mb = (2 * bytes_per_elem * batch_size * configs["MQA 32:1"] * n * head_dim * num_layers) / (1024 * 1024)
        savings = (1.0 - gqa_mb / mha_mb) * 100.0

        mem_results["MHA 32:32"].append(mha_mb)
        mem_results["GQA 4:1"].append(gqa_mb)
        mem_results["MQA 32:1"].append(mqa_mb)

        print(f"{n:<16} | {mha_mb:>15.2f} MB | {gqa_mb:>15.2f} MB | {mqa_mb:>15.2f} MB | {savings:>25.1f}%")

    print("-" * 110)
    print("Result: GQA achieves an immediate 75.0% memory footprint reduction over standard MHA.\n")

    # -------------------------------------------------------------
    # 2. Memory Bandwidth Traffic Table (in GB/s at 50 tokens/sec)
    # -------------------------------------------------------------
    print("=== Table 2: Full Model Memory Bandwidth Traffic (32 Layers at 50 tokens/sec decoding) ===")
    print(f"{'Seq Length (N)':<16} | {'MHA Bandwidth':<18} | {'GQA Bandwidth':<18} | {'MQA Bandwidth':<18} | {'Bandwidth Savings':<28}")
    print("-" * 110)

    for i, n in enumerate(seq_lengths):
        # Bandwidth (GB/s) = (Total KV Cache in MB * tokens/sec) / 1024
        bw_mha = (mem_results["MHA 32:32"][i] * token_rate) / 1024.0
        bw_gqa = (mem_results["GQA 4:1"][i] * token_rate) / 1024.0
        bw_mqa = (mem_results["MQA 32:1"][i] * token_rate) / 1024.0
        savings = (1.0 - bw_gqa / bw_mha) * 100.0

        bw_results["MHA 32:32"].append(bw_mha)
        bw_results["GQA 4:1"].append(bw_gqa)
        bw_results["MQA 32:1"].append(bw_mqa)

        print(f"{n:<16} | {bw_mha:>15.2f} GB/s | {bw_gqa:>15.2f} GB/s | {bw_mqa:>15.2f} GB/s | {savings:>25.1f}%")

    print("-" * 110)
    print("Result: At 32k context, MHA saturates the memory bus requiring 800.00 GB/s continuous bandwidth.")
    print("        GQA drops this to 200.00 GB/s (75% savings), bringing long-context inference within practical GPU limits.\n")

    # -------------------------------------------------------------
    # 3. Empirical Decoding Latency per Token (in ms)
    # -------------------------------------------------------------
    print("=== Table 3: Empirical Decoding Latency per Token (1 Layer Measured / 32 Layers Est.) ===")
    print(f"{'Seq Length (N)':<16} | {'MHA 1-Layer (ms)':<18} | {'GQA 1-Layer (ms)':<18} | {'MQA 1-Layer (ms)':<18} | {'GQA 32-L Est (ms)':<18}")
    print("-" * 110)

    # Measure latency on current machine for fast, reliable verification
    benchmark_lengths = [1024, 4096] if quick else [1024, 4096, 16384]

    for n in benchmark_lengths:
        latencies = {}
        for arch_name, n_kv in configs.items():
            model = GQAAttention(
                d_model=d_model, num_q_heads=num_q_heads, num_kv_heads=n_kv, head_dim=head_dim
            ).to(device)
            cache = KVCache(
                max_batch_size=batch_size,
                max_seq_len=n + 32,
                n_kv_heads=n_kv,
                head_dim=head_dim,
                device=device,
            )
            lat = benchmark_decoding_latency_ms(
                model=model,
                kv_cache=cache,
                seq_len=n,
                head_dim=head_dim,
                d_model=d_model,
                num_runs=10,
                warmup=2,
                device=device,
                use_repeat=False,  # Use copy-free grouped path
            )
            latencies[arch_name] = lat

        est_32_layer = latencies["GQA 4:1"] * num_layers
        print(
            f"{n:<16} | {latencies['MHA 32:32']:>15.2f} ms | "
            f"{latencies['GQA 4:1']:>15.2f} ms | "
            f"{latencies['MQA 32:1']:>15.2f} ms | "
            f"{est_32_layer:>15.2f} ms"
        )

    if not quick and 32768 not in benchmark_lengths:
        # Transparently label projected row when CPU memory/time limits apply
        ref_n = 16384
        scale = 32768 / ref_n
        mha_proj = latencies["MHA 32:32"] * scale
        gqa_proj = latencies["GQA 4:1"] * scale
        mqa_proj = latencies["MQA 32:1"] * scale
        est_32_proj = gqa_proj * num_layers
        print(
            f"{'32768 (est)':<16} | {mha_proj:>15.2f} ms | "
            f"{gqa_proj:>15.2f} ms | "
            f"{mqa_proj:>15.2f} ms | "
            f"{est_32_proj:>15.2f} ms"
        )

    print("-" * 110)
    print("Note on CPU vs. GPU Benchmarking:")
    print("  - On memory-bandwidth-bound hardware (GPUs), GQA delivers significant decoding speedups (up to 3x-4x)")
    print("    by drastically reducing high-bandwidth memory (HBM) read traffic per generated token.")
    print("  - On CPU, compute is bounded by vector execution rather than memory bandwidth.")
    print("  - Using copy-free grouped query attention (use_repeat=False) prevents unnecessary tensor replication.")
    print("=========================================================================================\n")

    # Save visualization plot
    plot_path = "kv_cache_memory.png"
    plot_kv_cache_benchmark(seq_lengths, mem_results, bw_results, output_path=plot_path)


def test_gqa_kv_cache_correctness():
    """
    Executes four rigorous correctness unit tests:
    1. Output shape verification.
    2. Equivalence between autoregressive step-by-step KV-cache decoding and causal full-sequence pass.
    3. Memory calculation verification against allocated buffer size.
    4. Equivalence between copy-free grouped attention and repeat_interleave path.
    """
    print("=== Running Subsystem 03 Unit Tests ===")
    torch.manual_seed(42)

    b, seq_len, d_model = 1, 8, 256
    num_q_heads, num_kv_heads = 8, 2  # GQA 4:1
    head_dim = d_model // num_q_heads

    model = GQAAttention(
        d_model=d_model, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    model.eval()

    # 1. Output shape test
    x = torch.randn(b, seq_len, d_model)
    out = model(x, start_pos=0, kv_cache=None, use_causal_mask=True)
    assert out.shape == (b, seq_len, d_model), f"Shape mismatch: {out.shape} vs {(b, seq_len, d_model)}"
    print("[PASS] Test 1: Full-sequence forward output shape is verified.")

    # 2. Autoregressive step-by-step equivalence test
    cache = KVCache(max_batch_size=b, max_seq_len=seq_len + 4, n_kv_heads=num_kv_heads, head_dim=head_dim)
    step_outputs = []

    for pos in range(seq_len):
        token_input = x[:, pos : pos + 1, :]
        step_out = model(token_input, start_pos=pos, kv_cache=cache, use_causal_mask=False)
        step_outputs.append(step_out)

    autoregressive_out = torch.cat(step_outputs, dim=1)
    max_diff = torch.max(torch.abs(out - autoregressive_out)).item()
    assert max_diff < 1e-5, f"Cache decoding mismatch: {max_diff}"
    print(f"[PASS] Test 2: Autoregressive KV-cache decoding matches full-sequence causal pass (Max diff: {max_diff:.2e}).")

    # 3. Cache memory calculation test
    cache_mb = cache.get_memory_mb()
    expected_mb = (2 * 4 * b * num_kv_heads * (seq_len + 4) * head_dim) / (1024 * 1024)
    assert abs(cache_mb - expected_mb) < 1e-4, f"Cache memory calculation mismatch: {cache_mb} vs {expected_mb}"
    print(f"[PASS] Test 3: Cache memory computation verified ({cache_mb:.4f} MB).")

    # 4. Copy-free path vs. repeat_interleave equivalence test
    out_copy_free = model(x, start_pos=0, kv_cache=None, use_causal_mask=True, use_repeat=False)
    out_repeat = model(x, start_pos=0, kv_cache=None, use_causal_mask=True, use_repeat=True)
    diff_grouped = torch.max(torch.abs(out_copy_free - out_repeat)).item()
    assert diff_grouped < 1e-5, f"Copy-free vs repeat_interleave mismatch: {diff_grouped}"
    print(f"[PASS] Test 4: Copy-free grouped path matches repeat_interleave path (Max diff: {diff_grouped:.2e}).")

    print("All 4 Subsystem 03 unit tests completed successfully!\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Mission 03: GQA & KV-Cache Benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark up to N=4096 instead of full N=32768",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_gqa_kv_cache_correctness()
    run_kv_cache_benchmark(quick=args.quick)
