"""
Mission 03 — Part 1: Systems Architect (Lead Engineer)
Multi-Head Attention from Scratch with Causal Masking & Variance Preservation

Starship Operational Unit: USSE Noosphere Bridge Crew
Course: Conversational AI — Winter Semester 2026/27
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver

Deliverable Specifications:
---------------------------
1. Linear Projections: Learned weight matrices W_Q, W_K, W_V in R^{d_model x d_model}
   and output projection W_O in R^{d_model x d_model}.
2. Multi-Head Reshaping: Split inputs into h parallel heads:
   (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim).
3. Causal Masking: Lower-triangular causal attention mask (-inf above diagonal)
   to support autoregressive decoder execution.
4. Variance Preservation Unit Test: Empirical test validating that when inputs
   have zero mean and unit variance (sigma^2 = 1.0), scaling by 1/sqrt(d_k)
   preserves unit variance across dot products before softmax.
"""

import argparse
import math
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MultiHeadSelfAttention(nn.Module):
    """
    Complete Multi-Head Self-Attention (MHSA) implemented from first principles
    in pure PyTorch without relying on nn.MultiheadAttention.

    Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + M ) @ V
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1. Learned linear projection weight matrices W_Q, W_K, W_V in R^{d_model x d_model}
        self.w_q = nn.Linear(d_model, d_model, bias=bias, dtype=dtype)
        self.w_k = nn.Linear(d_model, d_model, bias=bias, dtype=dtype)
        self.w_v = nn.Linear(d_model, d_model, bias=bias, dtype=dtype)

        # Output projection matrix W_O in R^{d_model x d_model}
        self.w_o = nn.Linear(d_model, d_model, bias=bias, dtype=dtype)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Self-Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            causal: If True, applies lower-triangular causal attention mask (-inf above diagonal)
            attention_mask: Optional custom boolean or float mask of shape (batch_size, seq_len)

        Returns:
            Tuple of (output, attention_weights):
                output: shape (batch_size, seq_len, d_model)
                attention_weights: shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Linear projections for Query, Key, Value
        # Shape: (batch_size, seq_len, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Step 2: Multi-head reshaping
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Scaled dot-product attention scores
        # scores = (Q @ K^T) / sqrt(d_k) -> shape (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Step 4: Causal masking (lower-triangular mask, -inf above diagonal)
        if causal and seq_len > 1:
            causal_mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=x.device, dtype=scores.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Optional external attention mask (e.g. padding tokens)
        if attention_mask is not None:
            # Broadcast shape: (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Step 5: Softmax normalization and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 6: Weight values and project output
        # context: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v)

        # Concatenate heads: (batch_size, seq_len, num_heads * head_dim) = (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection W_O
        output = self.w_o(context)

        return output, attn_weights


# =============================================================================
# Unit Tests & Theoretical Verifications
# =============================================================================
def verify_variance_preservation() -> None:
    """
    Validates that scaling dot products by 1 / sqrt(d_k) preserves unit variance (sigma^2 = 1.0)
    when Query and Key components are drawn from standard normal distributions (zero mean, unit variance).

    Theoretical Proof:
    Let q_i, k_i ~ i.i.d. N(0, 1).
    Dot product z = sum_{i=1}^{d_k} (q_i * k_i).
    E[q_i * k_i] = E[q_i] * E[k_i] = 0.
    Var(q_i * k_i) = Var(q_i) * Var(k_i) + [E[q_i]]^2 * Var(k_i) + [E[k_i]]^2 * Var(q_i) = 1.
    Since components are independent:
    Var(z) = sum_{i=1}^{d_k} Var(q_i * k_i) = d_k.

    When scaled by 1 / sqrt(d_k):
    z_scaled = z / sqrt(d_k).
    Var(z_scaled) = (1 / d_k) * Var(z) = d_k / d_k = 1.0.
    """
    print("=== Subsystem 03: Variance Preservation Empirical Validation ===")
    torch.manual_seed(42)

    test_dims = [64, 128, 256]
    seq_len = 256
    batch_size = 64

    for d_k in test_dims:
        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)

        unscaled_dot = torch.matmul(q, k.transpose(-2, -1))
        scaled_dot = unscaled_dot / math.sqrt(d_k)

        mean_unscaled = unscaled_dot.mean().item()
        var_unscaled = unscaled_dot.var().item()
        mean_scaled = scaled_dot.mean().item()
        var_scaled = scaled_dot.var().item()

        print(f"Head Dim (d_k = {d_k:>3}):")
        print(f"  * Unscaled Dot Product -> Mean: {mean_unscaled:+.4f}, Var: {var_unscaled:>8.2f} (Expected ~{d_k})")
        print(f"  * Scaled Dot Product   -> Mean: {mean_scaled:+.4f}, Var: {var_scaled:>8.4f} (Expected ~1.0000)")

        # Verify unit variance preservation within tight statistical tolerance
        assert abs(var_scaled - 1.0) < 0.05, f"Variance preservation failed for d_k={d_k}!"

    print("[PASS] Variance preservation holds across all head dimensions (Var ~= 1.000).\n")


def test_causal_mask_prevents_future_leakage() -> None:
    """
    Verifies that changing a future token at position t_future does NOT alter
    the attention representations or outputs of earlier tokens at t_past <= t_future.
    """
    torch.manual_seed(101)
    d_model, num_heads, seq_len = 256, 4, 16
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads).eval()

    x1 = torch.randn(1, seq_len, d_model)
    x2 = x1.clone()
    # Mutate future tokens 10 through 15
    x2[:, 10:, :] = torch.randn(1, 6, d_model)

    with torch.no_grad():
        out1, _ = mha(x1, causal=True)
        out2, _ = mha(x2, causal=True)

    # Prefix tokens 0..9 must remain perfectly identical
    diff_past = (out1[:, :10, :] - out2[:, :10, :]).abs().max().item()
    diff_future = (out1[:, 10:, :] - out2[:, 10:, :]).abs().max().item()

    assert diff_past < 1e-6, f"Causal mask leakage! Past tokens changed by {diff_past}"
    assert diff_future > 1e-3, "Future tokens should differ after mutation."
    print(f"[PASS] Causal masking strictly preserves past invariant (Past diff: {diff_past:.2e}).")


def test_equivalence_with_pytorch_reference() -> None:
    """
    Verifies that our custom MultiHeadSelfAttention output matches PyTorch's
    built-in scaled_dot_product_attention when given identical projection weights.
    """
    torch.manual_seed(202)
    b, s, d_model, h = 2, 8, 256, 4
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=h).eval()
    x = torch.randn(b, s, d_model)

    with torch.no_grad():
        out_custom, _ = mha(x, causal=True)

        # Replicate with F.scaled_dot_product_attention
        head_dim = d_model // h
        q = mha.w_q(x).view(b, s, h, head_dim).transpose(1, 2)
        k = mha.w_k(x).view(b, s, h, head_dim).transpose(1, 2)
        v = mha.w_v(x).view(b, s, h, head_dim).transpose(1, 2)

        ref_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref_out = mha.w_o(ref_attn.transpose(1, 2).contiguous().view(b, s, d_model))

    max_diff = (out_custom - ref_out).abs().max().item()
    assert max_diff < 1e-5, f"Custom MHSA differs from PyTorch reference by {max_diff}"
    print(f"[PASS] Custom MHSA matches PyTorch F.scaled_dot_product_attention (Max diff: {max_diff:.2e}).")


def test_tensor_shapes_and_dimensions() -> None:
    """
    Verifies output tensor shapes and attention weight shapes across batches.
    """
    b, s, d_model, num_heads = 4, 32, 512, 8
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads).eval()
    x = torch.randn(b, s, d_model)

    out, attn = mha(x, causal=True)
    assert out.shape == (b, s, d_model), f"Unexpected output shape {out.shape}"
    assert attn.shape == (b, num_heads, s, s), f"Unexpected attention shape {attn.shape}"

    # Verify that attention weights sum to 1 across key dimension
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Attention weights do not sum to 1.0"
    print(f"[PASS] Output shape ({out.shape}) and attention probabilities verified.")


def plot_causal_attention_map(output_path: str = "causal_attention_map.png") -> None:
    """
    Visualizes and saves the lower-triangular causal attention matrix.
    """
    if not HAS_MATPLOTLIB:
        print("[INFO] matplotlib not installed, skipping heatmap generation.")
        return

    torch.manual_seed(77)
    s, d_model, h = 16, 256, 4
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=h).eval()
    x = torch.randn(1, s, d_model)

    with torch.no_grad():
        _, attn = mha(x, causal=True)

    # Plot Head 0 attention matrix
    head_attn = attn[0, 0].cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(head_attn, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Attention Weight")
    plt.title("Causal Lower-Triangular Attention Map (Head 0)", fontsize=11, fontweight="bold")
    plt.xlabel("Key Position (j)", fontsize=10)
    plt.ylabel("Query Position (i)", fontsize=10)
    plt.xticks(range(0, s, 2))
    plt.yticks(range(0, s, 2))
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"[SUCCESS] Causal attention map saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 03 — Part 1: Multi-Head Attention Core")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving attention heatmap plot")
    args = parser.parse_args()

    print("=" * 80)
    print("USSE NOOSPHERE — SUBSYSTEM 03: MULTI-HEAD ATTENTION CORE PIPELINE")
    print("Role Deliverable: Part 1 — Systems Architect (Lead Engineer)")
    print("=" * 80 + "\n")

    # Run verification suite
    verify_variance_preservation()
    test_causal_mask_prevents_future_leakage()
    test_equivalence_with_pytorch_reference()
    test_tensor_shapes_and_dimensions()

    if not args.no_plot:
        here = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(here, "causal_attention_map.png")
        plot_causal_attention_map(plot_path)

    print("\nAll Part 1 Systems Architect specifications verified and operational!")


if __name__ == "__main__":
    main()
