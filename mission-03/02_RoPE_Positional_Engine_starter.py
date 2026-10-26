"""
Mission 03 — Part 2: Communications & Protocol Officer
Rotary Positional Embeddings (RoPE) Engine & Relative Invariance Validation

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver
"""

import torch
import torch.nn as nn
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponential frequencies cis(m * theta_i) = e^{i * m * theta_i}.
    
    Args:
        dim: Dimension of the head (must be even)
        end: Maximum sequence length
        theta: Frequency base scaling factor (default: 10000.0)
        
    Returns:
        freqs_cis: Complex tensor of shape (end, dim // 2)
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE 2D pairing."
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs) # (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64 tensor
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to Query and Key tensors.
    
    Args:
        xq: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        xk: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim // 2)
        
    Returns:
        Tuple of rotated (xq_out, xk_out) with identical shape
    """
    # Reshape into complex view of pairs
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Match broadcasting: freqs_cis shape (1, 1, seq_len, head_dim // 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1)

    # Complex multiplication rotates the 2D vector coordinates
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def verify_relative_invariance():
    """
    Cadet Verification: Empirically verify that:
    < RoPE(q, m), RoPE(k, n) > == < RoPE(q, m + delta), RoPE(k, n + delta) >
    for any arbitrary position shift delta.
    """
    print("=== Subsystem 03: RoPE Relative Shift Invariance Verification ===")
    torch.manual_seed(42)
    dim = 64
    max_seq_len = 256
    freqs_cis = precompute_freqs_cis(dim=dim, end=max_seq_len)

    # Vector representations at position m=10 and n=25
    m, n = 10, 25
    delta = 40 # translation offset

    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    # Rotations at original positions
    q_m, _ = apply_rotary_emb(q, q, freqs_cis[m : m + 1])
    _, k_n = apply_rotary_emb(k, k, freqs_cis[n : n + 1])
    dot_orig = (q_m * k_n).sum().item()

    # Rotations shifted by delta
    q_m_shifted, _ = apply_rotary_emb(q, q, freqs_cis[m + delta : m + delta + 1])
    _, k_n_shifted = apply_rotary_emb(k, k, freqs_cis[n + delta : n + delta + 1])
    dot_shifted = (q_m_shifted * k_n_shifted).sum().item()

    print(f"Dot product <RoPE(q, {m}), RoPE(k, {n})>:                 {dot_orig:.6f}")
    print(f"Dot product <RoPE(q, {m+delta}), RoPE(k, {n+delta})> (delta={delta}): {dot_shifted:.6f}")
    diff = abs(dot_orig - dot_shifted)
    print(f"Absolute Difference: {diff:.8e}")
    assert diff < 1e-5, "RoPE relative invariance violation detected!"
    print("[PASS] RoPE relative distance invariance holds perfectly across spatial translations!")


if __name__ == "__main__":
    verify_relative_invariance()
