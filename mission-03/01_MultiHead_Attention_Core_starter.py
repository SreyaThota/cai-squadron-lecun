"""
Mission 03 — Part 1: Systems Architect (Lead Engineer)
Multi-Head Attention from Scratch with Causal Masking & Variance Preservation

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections for Query, Key, Value and Output
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Self-Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            causal: If True, applies lower-triangular causal attention mask
            attention_mask: Optional extra boolean mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (output, attention_weights):
                output: shape (batch_size, seq_len, d_model)
                attention_weights: shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. Reshape into (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Compute scaled dot-product scores: (Q @ K.T) * scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 4. Apply Causal Masking if requested
        if causal and seq_len > 1:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            scores = scores + causal_mask

        if attention_mask is not None:
            # Broadcast mask: (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # 5. Softmax and Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. Weight values and project output
        context = torch.matmul(attn_weights, v) # (batch_size, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)

        return output, attn_weights


def verify_variance_preservation():
    """
    Cadet Verification: Validate that scaling by 1/sqrt(d_k) preserves unit variance
    across dot products before softmax when inputs have zero mean and unit variance.
    """
    print("=== Subsystem 03: Attention Variance Preservation Verification ===")
    torch.manual_seed(42)
    d_k = 64
    seq_len = 128
    batch_size = 32

    # Standard normal inputs (zero mean, unit variance)
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)

    unscaled_dot = torch.matmul(q, k.transpose(-2, -1))
    scaled_dot = unscaled_dot / math.sqrt(d_k)

    print(f"Head Dim (d_k): {d_k}")
    print(f"Unscaled Dot Product — Mean: {unscaled_dot.mean().item():.4f}, Variance: {unscaled_dot.var().item():.4f} (Expected ~{d_k})")
    print(f"Scaled Dot Product   — Mean: {scaled_dot.mean().item():.4f}, Variance: {scaled_dot.var().item():.4f} (Expected ~1.000)")
    assert abs(scaled_dot.var().item() - 1.0) < 0.1, "Variance preservation check failed!"
    print("[PASS] Variance preservation holds within tolerance! Subsystem 03 Attention Matrix stabilized.")


if __name__ == "__main__":
    verify_variance_preservation()
    
    # Run test forward pass
    mha = MultiHeadSelfAttention(d_model=256, num_heads=4)
    dummy_input = torch.randn(2, 16, 256)
    out, attn = mha(dummy_input, causal=True)
    print(f"Output shape: {out.shape}, Attention shape: {attn.shape}")
