"""
Mission 03 - Part 3: Data & Payload Specialist
Grouped-Query Attention (GQA) & Autoregressive KV-Cache Benchmark

Starship Operational Unit: USSE Noosphere Bridge Crew
Course: Conversational AI - Winter Semester 2026/27
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver

What this file delivers (see ps03 brief, Part 3)
------------------------------------------------
1. GQAAttention : grouped-query attention, configurable H_Q (e.g. 32) and H_KV (e.g. 8).
                  KV heads are matched to query heads with torch.repeat_interleave
                  (as in the brief) and, alternatively, with a copy-free grouped path.
2. KVCache      : stateful buffer that keeps past (K_t, V_t) and appends new slices
                  while decoding token by token.
3. Benchmark    : MHA vs GQA 4:1 vs MQA 32:1 at N = 1024, 4096, 16384, 32768
                  - KV-cache memory footprint (MB)
                  - memory bandwidth traffic (GB/s at 50 tokens/sec)
                  - decoding latency per token (ms), measured for every N

Run
---
    python 03_GQA_and_KVCache_Benchmark.py            # full run, all four N
    python 03_GQA_and_KVCache_Benchmark.py --quick    # N up to 4096 only
    python 03_GQA_and_KVCache_Benchmark.py --no-plot  # skip the PNG

Memorizable patterns
--------------------
* Q asks, K labels, V holds the content.
* GQA = many askers (query heads) share a few librarians (KV heads).
* KV-cache = "don't redo old homework": store K and V of old tokens,
  compute only the new token.
* Cache size = 2 (K and V) x layers x KV-heads x head_dim x bytes x tokens.

Conventions used in every table
-------------------------------
* Model shape is LLaMA-3-8B-like: 32 layers, 32 query heads, head_dim 128.
* Memory and bandwidth are for the WHOLE model (all layers), stored in float16.
* 1 MB = 2^20 bytes and 1 GB = 2^30 bytes, everywhere (no unit mixing).
* Latency is measured for ONE attention layer (one decode step). The "x32 model"
  column is just that number times 32 layers, it is NOT a separate measurement.
"""

import argparse
import math
import os
import statistics
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. KV-Cache
# =============================================================================
class KVCache:
    """
    Pre-allocated buffer that remembers the K and V of every past token.

    Without a cache, generating token number t means recomputing K and V for all
    t tokens again. With a cache we compute them once, store them, and each new
    step only computes K and V for the single new token.

    Note: the projections cost O(1) per step, but the attention step still has to
    READ all N cached tokens, so a decode step is O(N) in memory traffic. That
    reading is exactly why a smaller cache (GQA) makes decoding faster.
    """

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len
        self.length = 0  # how many tokens are stored so far

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new_k/new_v of shape (B, H_kv, S_new, D) after the stored tokens.
        Returns views (no copy) of ALL stored keys and values.
        """
        s_new = new_k.shape[2]
        if self.length + s_new > self.max_seq_len:
            raise ValueError(
                f"KVCache overflow: {self.length} + {s_new} > capacity {self.max_seq_len}"
            )
        self.k[:, :, self.length : self.length + s_new] = new_k
        self.v[:, :, self.length : self.length + s_new] = new_v
        self.length += s_new
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    def reset(self) -> None:
        """Forget everything (set the fill level back to zero)."""
        self.length = 0

    @property
    def allocated_bytes(self) -> int:
        """Bytes reserved for K and V together."""
        return 2 * self.k.numel() * self.k.element_size()

    @property
    def used_bytes(self) -> int:
        """Bytes actually holding tokens right now."""
        return 2 * self.k[:, :, : self.length].numel() * self.k.element_size()

    def get_memory_mb(self) -> float:
        """Allocated size in MB (1 MB = 2^20 bytes)."""
        return self.allocated_bytes / 2**20


# =============================================================================
# 2. Grouped-Query Attention
# =============================================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    (B, H_kv, T, D) -> (B, H_kv * n_rep, T, D).
    Every KV head is copied n_rep times so each query head has a partner.
    Query heads 0..n_rep-1 use KV head 0, the next n_rep use KV head 1, and so on.
    """
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, repeats=n_rep, dim=1)


class GQAAttention(nn.Module):
    """
    Grouped-Query Attention: H_Q query heads share H_KV key/value heads.

        H_KV == H_Q        -> MHA (every asker has its own librarian)
        1 < H_KV < H_Q     -> GQA (a group of askers shares one librarian)
        H_KV == 1          -> MQA (everybody shares one librarian)

    Two equivalent ways to line the heads up (they give identical outputs):
      use_repeat=True  : copy K,V out to H_Q heads with repeat_interleave (simple,
                         but it COPIES memory on every step).
      use_repeat=False : stack the query group into the "rows" axis instead, so K,V
                         are used as they are and never copied.
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: Optional[int] = None,
        use_repeat: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_q_heads // num_kv_heads  # askers per librarian
        self.head_dim = head_dim if head_dim is not None else d_model // num_q_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_repeat = use_repeat

        self.w_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False, dtype=dtype)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.w_o = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model). With a cache, x holds only the NEW token(s);
        the old ones are already inside the cache. Masking is always causal.
        """
        b, s, _ = x.shape
        h_q, h_kv, d = self.num_q_heads, self.num_kv_heads, self.head_dim

        q = self.w_q(x).view(b, s, h_q, d).transpose(1, 2)    # (B, Hq,  S, D)
        k = self.w_k(x).view(b, s, h_kv, d).transpose(1, 2)   # (B, Hkv, S, D)
        v = self.w_v(x).view(b, s, h_kv, d).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)  # now k,v hold ALL tokens so far
        t = k.shape[2]                    # total number of keys

        # The query at row i sits at absolute position (t - s + i). It may look at keys
        # j <= that position. This is the causal mask and it also works with a cache.
        allowed = torch.ones(s, t, dtype=torch.bool, device=x.device).tril(diagonal=t - s)

        if self.use_repeat:
            k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)   # (B, Hq, T, D) copy
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, Hq, S, T)
            scores = scores.masked_fill(~allowed, float("-inf"))
            out = torch.matmul(F.softmax(scores, dim=-1), v)            # (B, Hq, S, D)
        else:
            # Query heads of one group are neighbours: head = kv_head * n_rep + r.
            # Fold the group into the row axis so K,V are used without any copy.
            qg = q.reshape(b, h_kv, self.n_rep * s, d)
            scores = torch.matmul(qg, k.transpose(-2, -1)) * self.scale  # (B, Hkv, n_rep*S, T)
            scores = scores.masked_fill(~allowed.repeat(self.n_rep, 1), float("-inf"))
            out = torch.matmul(F.softmax(scores, dim=-1), v)             # (B, Hkv, n_rep*S, D)
            out = out.reshape(b, h_q, s, d)

        out = out.transpose(1, 2).contiguous().view(b, s, h_q * d)
        return self.w_o(out)


# =============================================================================
# 3. Closed-form formulas (used by the tables and checked by the tests)
# =============================================================================
def kv_cache_bytes(n_tokens: int, n_layers: int, n_kv_heads: int, head_dim: int,
                   bytes_per_elem: int, batch_size: int = 1) -> int:
    """2 (K and V) x batch x layers x KV heads x head_dim x bytes x tokens."""
    return 2 * batch_size * n_layers * n_kv_heads * head_dim * bytes_per_elem * n_tokens


# =============================================================================
# 4. Correctness tests (all use assert, so a wrong result stops the script)
# =============================================================================
def test_cache_matches_full_forward() -> None:
    """Token-by-token decoding with a cache must equal one full pass without it."""
    torch.manual_seed(0)
    layer = GQAAttention(256, 8, 2).eval()
    x = torch.randn(1, 16, 256)
    with torch.no_grad():
        full = layer(x)
        cache = KVCache(1, 2, 32, layer.head_dim)
        outs = [layer(x[:, :10], cache)]           # prefill: first 10 tokens at once
        for i in range(10, 16):                    # decode: one token at a time
            outs.append(layer(x[:, i : i + 1], cache))
        cached = torch.cat(outs, dim=1)
    diff = (full - cached).abs().max().item()
    assert torch.allclose(full, cached, atol=1e-5), f"cache mismatch, max diff {diff:.2e}"
    print(f"[PASS] cached decoding == full recompute          (max diff {diff:.2e})")


def test_gqa_matches_torch_reference() -> None:
    """Our GQA must equal PyTorch's own causal attention on the same weights."""
    torch.manual_seed(1)
    layer = GQAAttention(256, 8, 2).eval()
    x = torch.randn(2, 12, 256)
    with torch.no_grad():
        mine = layer(x)
        b, s, _ = x.shape
        q = layer.w_q(x).view(b, s, 8, layer.head_dim).transpose(1, 2)
        k = repeat_kv(layer.w_k(x).view(b, s, 2, layer.head_dim).transpose(1, 2), 4)
        v = repeat_kv(layer.w_v(x).view(b, s, 2, layer.head_dim).transpose(1, 2), 4)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = layer.w_o(ref.transpose(1, 2).reshape(b, s, -1))
    assert torch.allclose(mine, ref, atol=1e-5), "GQA differs from torch reference"
    print("[PASS] GQA == torch scaled_dot_product_attention  (causal)")


def test_copy_free_path_matches_repeat_path() -> None:
    """The two ways of matching heads must give the same numbers."""
    torch.manual_seed(3)
    a = GQAAttention(256, 8, 2, use_repeat=True).eval()
    b = GQAAttention(256, 8, 2, use_repeat=False).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 12, 256)
    with torch.no_grad():
        ok = torch.allclose(a(x), b(x), atol=1e-5)
    assert ok, "grouped path differs from repeat_interleave path"
    print("[PASS] copy-free grouped path == repeat_interleave path")


def test_no_peeking_at_future_tokens() -> None:
    """Changing a LATER token must not change the output of an EARLIER one."""
    torch.manual_seed(4)
    layer = GQAAttention(256, 8, 2).eval()
    x = torch.randn(1, 10, 256)
    y = x.clone()
    y[:, 7:] = torch.randn(1, 3, 256)              # alter tokens 7, 8, 9
    with torch.no_grad():
        same = torch.allclose(layer(x)[:, :7], layer(y)[:, :7], atol=1e-6)
    assert same, "causal mask leaks future information"
    print("[PASS] causal mask: future tokens cannot change the past")


def test_head_grouping_and_cache_size() -> None:
    """Head ordering, cache size formula, and the overflow guard."""
    k = torch.arange(2.0).view(1, 2, 1, 1).expand(1, 2, 3, 1)       # head0 = 0s, head1 = 1s
    r = repeat_kv(k, 4)[0, :, 0, 0].tolist()
    assert r == [0, 0, 0, 0, 1, 1, 1, 1], f"bad head grouping: {r}"

    n, layers, hkv, d = 1024, 32, 8, 128
    c = KVCache(1, hkv, n, d, dtype=torch.float16)
    assert c.allocated_bytes * layers == kv_cache_bytes(n, layers, hkv, d, 2), "formula mismatch"

    small = KVCache(1, 2, 4, 8)
    small.update(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8))
    try:
        small.update(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("overflow was not caught")
    print("[PASS] head grouping order, cache-size formula, overflow guard")


def demo_cache_speedup(tokens: int = 128) -> None:
    """Generate `tokens` tokens with and without the cache on a small layer."""
    torch.manual_seed(2)
    layer = GQAAttention(256, 8, 2).eval()
    x = torch.randn(1, tokens, 256)
    with torch.no_grad():
        t0 = time.perf_counter()
        for i in range(1, tokens + 1):             # redo the whole prefix every step
            layer(x[:, :i])
        no_cache = time.perf_counter() - t0

        cache = KVCache(1, 2, tokens, layer.head_dim)
        t0 = time.perf_counter()
        for i in range(tokens):                    # only the new token every step
            layer(x[:, i : i + 1], cache)
        with_cache = time.perf_counter() - t0
    print(f"[demo] {tokens} tokens: no cache {no_cache * 1e3:7.1f} ms | "
          f"with cache {with_cache * 1e3:7.1f} ms | speedup x{no_cache / with_cache:.1f}")


# =============================================================================
# 5. Benchmark
# =============================================================================
D_MODEL, N_LAYERS, HEAD_DIM, H_Q = 4096, 32, 128, 32   # LLaMA-3-8B-like
BYTES_FP16 = 2
TOKENS_PER_SEC = 50
BATCH = 1
SEQ_LENGTHS = [1024, 4096, 16384, 32768]
ARCHS = [("MHA", 32), ("GQA", 8), ("MQA", 1)]            # (name, number of KV heads)
MB, GB = 2**20, 2**30


@torch.no_grad()
def measure_decode_latency_ms(n_kv: int, n_tokens: int, use_repeat: bool, device: str,
                              dtype: torch.dtype, iters: int = 7) -> float:
    """
    Median time (ms) of ONE decode step of ONE attention layer when the cache
    already holds n_tokens tokens (filled with random numbers, which is fine for timing).
    """
    layer = GQAAttention(D_MODEL, H_Q, n_kv, HEAD_DIM, use_repeat=use_repeat, dtype=dtype)
    layer = layer.to(device).eval()
    cache = KVCache(BATCH, n_kv, n_tokens + iters + 2, HEAD_DIM, dtype=dtype, device=device)
    cache.k[:, :, :n_tokens].normal_()
    cache.v[:, :, :n_tokens].normal_()
    cache.length = n_tokens
    x = torch.randn(BATCH, 1, D_MODEL, dtype=dtype, device=device)

    layer(x, cache)                                 # warm-up (also adds one token)
    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        layer(x, cache)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    del layer, cache
    return statistics.median(times)


def safe_latency(n_kv: int, n: int, use_repeat: bool, device: str, dtype: torch.dtype) -> float:
    """Latency, or NaN if the machine runs out of memory for this case."""
    try:
        return measure_decode_latency_ms(n_kv, n, use_repeat, device, dtype)
    except (RuntimeError, MemoryError):
        return float("nan")


def fmt_ms(v: float) -> str:
    return "OOM" if math.isnan(v) else f"{v:,.2f}"


def run_benchmark(seq_lengths) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    line = "=" * 118

    print(line)
    print("USSE NOOSPHERE - SUBSYSTEM 03: DISTRIBUTED ATTENTION & KV-CACHE BENCHMARK")
    print("Waypoint: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)")
    print("Role Deliverable: Part 3 - Data & Payload Specialist")
    print(line)
    print(f"Device: {device.upper()} | latency dtype: {str(dtype).replace('torch.', '')} | "
          f"model: {N_LAYERS} layers, {H_Q} query heads, head_dim {HEAD_DIM}")
    print(f"Memory/bandwidth: whole model, float16, batch {BATCH}, {TOKENS_PER_SEC} tok/s | "
          "1 MB = 2^20 B, 1 GB = 2^30 B\n")

    results = {"mem": {}, "bw": {}, "lat": {}}

    # ---- Table 1: memory ---------------------------------------------------
    print("=== Table 1: KV-Cache Memory Footprint (MB, whole model) ===")
    head = f"{'N':>6} | {'MHA (32 KV)':>14} | {'GQA 4:1 (8 KV)':>14} | {'MQA 32:1 (1 KV)':>15} | {'GQA vs MHA':>10}"
    print(head)
    print("-" * len(head))
    for n in seq_lengths:
        mb = {a: kv_cache_bytes(n, N_LAYERS, h, HEAD_DIM, BYTES_FP16, BATCH) / MB for a, h in ARCHS}
        results["mem"][n] = mb
        print(f"{n:>6} | {mb['MHA']:>14,.0f} | {mb['GQA']:>14,.0f} | {mb['MQA']:>15,.0f} | "
              f"{(1 - mb['GQA'] / mb['MHA']) * 100:>9.1f}%")
    print("-" * len(head))
    print("Per layer, divide by 32. GQA 4:1 stores 1/4 of MHA, MQA stores 1/32.\n")

    # ---- Table 2: bandwidth ------------------------------------------------
    print(f"=== Table 2: Memory Bandwidth Traffic (GB/s at {TOKENS_PER_SEC} tokens/sec) ===")
    head = f"{'N':>6} | {'MHA (32 KV)':>14} | {'GQA 4:1 (8 KV)':>14} | {'MQA 32:1 (1 KV)':>15} | {'GQA vs MHA':>10}"
    print(head)
    print("-" * len(head))
    for n in seq_lengths:
        # Each new token must READ the whole cache once, so traffic = cache size x token rate.
        bw = {a: kv_cache_bytes(n, N_LAYERS, h, HEAD_DIM, BYTES_FP16, BATCH) * TOKENS_PER_SEC / GB
              for a, h in ARCHS}
        results["bw"][n] = bw
        print(f"{n:>6} | {bw['MHA']:>14,.1f} | {bw['GQA']:>14,.1f} | {bw['MQA']:>15,.1f} | "
              f"{(1 - bw['GQA'] / bw['MHA']) * 100:>9.1f}%")
    print("-" * len(head))
    print("Traffic counts only the KV-cache reads. Model weights add traffic on top.\n")

    # ---- Table 3: latency --------------------------------------------------
    print("=== Table 3: Decoding Latency per Token (ms, ONE attention layer, measured) ===")
    print("GQA/MQA are timed with both head-matching methods: 'repeat' = repeat_interleave")
    print("(copies K,V to 32 heads each step), 'grouped' = copy-free.\n")
    cols = ["MHA", "GQA repeat", "GQA grouped", "MQA repeat", "MQA grouped"]
    head = f"{'N':>6} | " + " | ".join(f"{c:>11}" for c in cols) + f" | {'GQA grouped':>11} | {'x32 model':>10}"
    sub = f"{'':>6} | " + " | ".join(f"{'ms':>11}" for _ in cols) + f" | {'speedup':>11} | {'(GQA grp)':>10}"
    print(head)
    print(sub)
    print("-" * len(head))
    for n in seq_lengths:
        lat = {
            "MHA": safe_latency(32, n, False, device, dtype),
            "GQA repeat": safe_latency(8, n, True, device, dtype),
            "GQA grouped": safe_latency(8, n, False, device, dtype),
            "MQA repeat": safe_latency(1, n, True, device, dtype),
            "MQA grouped": safe_latency(1, n, False, device, dtype),
        }
        results["lat"][n] = lat
        sp = lat["MHA"] / lat["GQA grouped"]
        sp_s = "n/a" if math.isnan(sp) else f"{sp:.2f}x"
        full = lat["GQA grouped"] * N_LAYERS
        full_s = "n/a" if math.isnan(full) else f"{full:,.0f} ms"
        print(f"{n:>6} | " + " | ".join(f"{fmt_ms(lat[c]):>11}" for c in cols)
              + f" | {sp_s:>11} | {full_s:>10}")
    print("-" * len(head))
    print("Every latency number above was measured on this machine (none are projected).")
    print("OOM would mean this machine ran out of memory for that case.\n")
    return results


def print_summary(results: dict) -> None:
    n = max(results["mem"])
    mem, bw, lat = results["mem"][n], results["bw"][n], results["lat"][n]
    print("=== Summary (computed from the tables above) ===")
    print(f"* At N={n:,}: MHA needs {mem['MHA'] / 1024:,.1f} GB of cache, GQA 4:1 needs "
          f"{mem['GQA'] / 1024:,.1f} GB, MQA needs {mem['MQA'] / 1024:,.2f} GB.")
    print(f"* Cache traffic at {TOKENS_PER_SEC} tok/s: MHA {bw['MHA']:,.0f} GB/s, "
          f"GQA {bw['GQA']:,.0f} GB/s, MQA {bw['MQA']:,.0f} GB/s.")
    print("* Memory and traffic follow H_KV exactly: GQA = 1/4 of MHA, MQA = 1/32.")
    if not any(math.isnan(lat[k]) for k in ("MHA", "GQA repeat", "GQA grouped")):
        print(f"* Latency at N={n:,}: MHA {lat['MHA']:.1f} ms | GQA repeat_interleave "
              f"{lat['GQA repeat']:.1f} ms | GQA grouped {lat['GQA grouped']:.1f} ms.")
    print("* repeat_interleave copies K,V up to 32 heads every step. That copy can cost")
    print("  more than the smaller cache saves, so GQA only gets its full speed-up with a")
    print("  copy-free layout (or a fused GPU kernel). Memory savings do not depend on this.")
    print("* Decoding is memory-bound on GPUs: smaller cache = less data to read = faster")
    print("  tokens and more simultaneous chats per GPU. CPU timings here are only")
    print("  indicative, repeat them on a GPU for final numbers.\n")


def plot_memory(path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plot)")
        return
    plt.figure(figsize=(7, 4.5))
    for name, h in ARCHS:
        mbs = [kv_cache_bytes(n, N_LAYERS, h, HEAD_DIM, BYTES_FP16) / MB for n in SEQ_LENGTHS]
        plt.plot(SEQ_LENGTHS, mbs, marker="o", label=f"{name} ({h} KV heads)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length N (tokens)")
    plt.ylabel("KV-cache size (MB, float16, 32 layers)")
    plt.title("KV-cache memory: MHA vs GQA 4:1 vs MQA 32:1")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Part 3: GQA + KV-cache benchmark")
    ap.add_argument("--quick", action="store_true", help="only N <= 4096 (fast)")
    ap.add_argument("--no-plot", action="store_true", help="do not write the PNG")
    args = ap.parse_args()
    seq_lengths = [n for n in SEQ_LENGTHS if n <= 4096] if args.quick else SEQ_LENGTHS

    print("=== Running Subsystem 03 unit tests ===")
    test_cache_matches_full_forward()
    test_gqa_matches_torch_reference()
    test_copy_free_path_matches_repeat_path()
    test_no_peeking_at_future_tokens()
    test_head_grouping_and_cache_size()
    demo_cache_speedup()
    print("All Subsystem 03 tests passed.\n")

    results = run_benchmark(seq_lengths)
    print_summary(results)
    if not args.no_plot:
        here = os.path.dirname(os.path.abspath(__file__))
        plot_memory(os.path.join(here, "kv_cache_memory.png"))


if __name__ == "__main__":
    main()
