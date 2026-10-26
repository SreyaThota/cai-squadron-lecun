"""
Mission 03 — Part 2: Communications & Protocol Officer
Rotary Positional Embeddings (RoPE) Engine & Relative Invariance Validation

Starship Operational Unit: USSE Noosphere Bridge Crew
Course: Conversational AI — Winter Semester 2026/27
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver
Author: Vasu Midde

What this file delivers (see ps03 brief, Part 2)
------------------------------------------------
1. apply_rope(x, freqs_cis) : complex 2D planar rotation of Q/K with
                              theta_i = base^{-2(i-1)/d}, i = 1 .. d/2.
                              (+ a real-valued 2x2 rotation-matrix reference that
                              must agree with the complex version.)
2. Relative invariance      : proof in `relative_invariance_proof()` docstring and
                              empirical checks, single pair (starter) + random sweep +
                              a full attention-logit matrix under translation.
3. Decay heatmap / curve    : `plot_rope_vs_alibi()` -> rope_vs_alibi_decay.png
                              (RoPE oscillating decay vs ALiBi linear penalty, plus the
                              attention weights they produce and a distance heatmap).
4. Context extrapolation    : `evaluate_context_extrapolation()` shows which frequency
                              pairs leave their trained angle range beyond the training
                              length, and how Position Interpolation (PI) fixes it.

Run
---
    python 02_RoPE_Positional_Engine.py            # tests + extrapolation + plot
    python 02_RoPE_Positional_Engine.py --no-plot  # skip the PNG

Memorizable patterns
--------------------
* RoPE = turn every (even, odd) pair of a vector like a clock hand.
  Position m turns pair i by the angle m * theta_i.
* Fast hands (early pairs) see local order, slow hands (late pairs) see far distance.
* q at m and k at n: the dot product only sees the angle DIFFERENCE (m - n) * theta_i.
  Move both tokens by the same delta and nothing changes -> relative position.
* RoPE touches Q and K only, never V. And it never changes vector length.
"""

import argparse
import math
import os
from typing import Tuple

import torch
import torch.nn as nn  # noqa: F401  (kept from the starter)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 1. Frequencies  (starter function, unchanged)
# =============================================================================
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
    freqs = torch.outer(t, freqs)  # (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 tensor
    return freqs_cis


# --- NEW: helpers -------------------------------------------------------------
def rope_frequencies(dim: int, theta: float = 10000.0) -> torch.Tensor:
    """theta_i = base^{-2(i-1)/d} for i = 1 .. d/2  (radians per token)."""
    i = torch.arange(1, dim // 2 + 1, dtype=torch.float64)
    return theta ** (-2.0 * (i - 1) / dim)


def precompute_freqs_cis_scaled(dim: int, end: int, theta: float = 10000.0,
                                pi_scale: float = 1.0) -> torch.Tensor:
    """
    Same as precompute_freqs_cis, but positions are multiplied by `pi_scale`
    (Position Interpolation, Chen et al. 2023). pi_scale = L_train / L_target < 1
    squeezes a long context back into the angle range seen during training.
    """
    freqs = rope_frequencies(dim, theta).float()
    t = torch.arange(end, dtype=torch.float32) * pi_scale
    return torch.polar(torch.ones(end, dim // 2), torch.outer(t, freqs))


# =============================================================================
# 2. Rotation engine
# =============================================================================
def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    NEW (brief deliverable 1). Rotate ONE tensor (queries OR keys).

    Args:
        x:         (batch, heads, seq_len, head_dim), head_dim even
        freqs_cis: (seq_len, head_dim // 2) complex, row t = e^{i * pos_t * theta}.
                   With a KV-cache, pass the rows of the ABSOLUTE positions,
                   e.g. freqs_cis[offset : offset + seq_len].

    Pair (x_{2i}, x_{2i+1}) is read as the complex number x_{2i} + i*x_{2i+1};
    multiplying by e^{i*m*theta_i} turns it by angle m*theta_i:

        [x'_{2i}  ]   [cos m0  -sin m0] [x_{2i}  ]
        [x'_{2i+1}] = [sin m0   cos m0] [x_{2i+1}]      (m0 = m * theta_i)
    """
    assert x.shape[-1] % 2 == 0, "head_dim must be even for RoPE"
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1] // 2), (
        f"freqs_cis {tuple(freqs_cis.shape)} does not match "
        f"(seq_len, head_dim//2) = {(x.shape[-2], x.shape[-1] // 2)}"
    )
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B,H,S,D/2)
    x_out = torch.view_as_real(x_ * freqs_cis.to(x.device)).flatten(-2)   # broadcast over B,H
    return x_out.type_as(x)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to Query and Key tensors.
    (Starter API kept; CHANGED: body now delegates to apply_rope, same maths.)

    Args:
        xq: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        xk: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim // 2)

    Returns:
        Tuple of rotated (xq_out, xk_out) with identical shape
    """
    return apply_rope(xq, freqs_cis), apply_rope(xk, freqs_cis)


def apply_rope_real(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """
    NEW. Reference implementation with explicit real 2x2 rotation matrices
    (no complex numbers). Used only to cross-check apply_rope.
    positions: (seq_len,) integer/float absolute positions.
    """
    d = x.shape[-1]
    ang = positions.double()[:, None] * rope_frequencies(d, theta)[None, :]  # (S, D/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


# =============================================================================
# 3. Relative invariance: proof + empirical checks
# =============================================================================
def relative_invariance_proof() -> None:
    """
    Claim:  <RoPE(q, m), RoPE(k, n)> = g(q, k, m - n)   for some function g,
            hence <RoPE(q, m), RoPE(k, n)> = <RoPE(q, m+delta), RoPE(k, n+delta)>.

    Step 1 (pairs as complex numbers). Write q_j = q_{2j} + i q_{2j+1},
            k_j = k_{2j} + i k_{2j+1}, j = 0 .. d/2 - 1.
            The real dot product of two such vectors is
                <a, b> = sum_j Re( a_j * conj(b_j) ).

    Step 2 (rotation). RoPE(q, m)_j = q_j e^{i m theta_j},
                       RoPE(k, n)_j = k_j e^{i n theta_j}.

    Step 3 (combine).
        <RoPE(q,m), RoPE(k,n)> = sum_j Re( q_j e^{i m theta_j} * conj(k_j) e^{-i n theta_j} )
                               = sum_j Re( q_j conj(k_j) e^{i (m - n) theta_j} )
                               =: g(q, k, m - n).
        Only the difference m - n survives, because |e^{i phi}| = 1 and
        e^{i m theta} e^{-i n theta} = e^{i (m-n) theta}.

    Step 4 (translation). (m + delta) - (n + delta) = m - n, so g is unchanged. QED.

    Corollaries used in the tests:
        * Norm preservation: |q_j e^{i m theta_j}| = |q_j|  ->  ||RoPE(q,m)|| = ||q||.
        * Position 0 is the identity (e^{0} = 1).
        * Long-term decay (Su et al. 2021, Sec. 3.4.3): for q = k the sum
          sum_j cos((m-n) theta_j) is a sum of waves with different speeds; as |m - n|
          grows they fall out of phase and the score shrinks (oscillating decay).
    """


def verify_relative_invariance():
    """
    Cadet Verification: Empirically verify that:
    < RoPE(q, m), RoPE(k, n) > == < RoPE(q, m + delta), RoPE(k, n + delta) >
    for any arbitrary position shift delta.
    (Starter function, unchanged.)
    """
    print("=== Subsystem 03: RoPE Relative Shift Invariance Verification ===")
    torch.manual_seed(42)
    dim = 64
    max_seq_len = 256
    freqs_cis = precompute_freqs_cis(dim=dim, end=max_seq_len)

    # Vector representations at position m=10 and n=25
    m, n = 10, 25
    delta = 40  # translation offset

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
    print("[PASS] RoPE relative distance invariance holds perfectly across spatial translations!\n")


def _rope_dot(q: torch.Tensor, k: torch.Tensor, m: int, n: int, dim: int) -> float:
    """<RoPE(q, m), RoPE(k, n)> in float64 (q, k are 1-D vectors of length dim)."""
    q4, k4 = q.view(1, 1, 1, dim), k.view(1, 1, 1, dim)
    qm = apply_rope_real(q4, torch.tensor([m]))
    kn = apply_rope_real(k4, torch.tensor([n]))
    return (qm * kn).sum().item()


def test_relative_invariance_sweep(trials: int = 200) -> None:
    """Random q, k, m, n, delta (up to 100k) in float64: dot product only sees m - n."""
    torch.manual_seed(7)
    dim, worst = 128, 0.0
    for _ in range(trials):
        q, k = torch.randn(dim, dtype=torch.float64), torch.randn(dim, dtype=torch.float64)
        m, n = torch.randint(0, 4096, (2,)).tolist()
        delta = int(torch.randint(-min(m, n), 100_000, (1,)))
        a = _rope_dot(q, k, m, n, dim)
        b = _rope_dot(q, k, m + delta, n + delta, dim)
        # closed form g(q, k, m - n) from Step 3 of the proof
        qc = torch.view_as_complex(q.view(-1, 2).contiguous())
        kc = torch.view_as_complex(k.view(-1, 2).contiguous())
        g = (qc * kc.conj() * torch.polar(torch.ones(dim // 2, dtype=torch.float64),
                                          (m - n) * rope_frequencies(dim))).real.sum().item()
        worst = max(worst, abs(a - b), abs(a - g))
    assert worst < 1e-8, f"relative invariance broken, worst diff {worst:.2e}"
    print(f"[PASS] invariance sweep: {trials} random (q,k,m,n,delta), |delta| up to 1e5, "
          f"matches closed form g(q,k,m-n)  (worst diff {worst:.1e})")


def test_attention_logits_translation_invariant() -> None:
    """
    Whole attention-logit matrix QK^T/sqrt(d) for a sequence placed at positions
    0..S-1 versus 500..500+S-1 must be identical (float32 production path).
    """
    torch.manual_seed(11)
    b, h, s, d, shift = 2, 4, 32, 64, 500
    q, k = torch.randn(b, h, s, d), torch.randn(b, h, s, d)
    fc = precompute_freqs_cis(d, shift + s)
    q0, k0 = apply_rotary_emb(q, k, fc[:s])
    q1, k1 = apply_rotary_emb(q, k, fc[shift : shift + s])
    l0 = q0 @ k0.transpose(-2, -1) / math.sqrt(d)
    l1 = q1 @ k1.transpose(-2, -1) / math.sqrt(d)
    diff = (l0 - l1).abs().max().item()
    assert diff < 1e-3, f"attention logits changed under translation by {diff:.2e}"
    print(f"[PASS] full attention-logit matrix invariant to shifting all positions by "
          f"{shift} (max diff {diff:.1e}, float32)")


def test_complex_matches_real_rotation() -> None:
    """Complex-multiplication engine == explicit 2x2 rotation matrices."""
    torch.manual_seed(3)
    x = torch.randn(2, 3, 50, 64)
    a = apply_rope(x, precompute_freqs_cis(64, 50))
    b = apply_rope_real(x, torch.arange(50))
    diff = (a - b).abs().max().item()
    assert diff < 1e-4, f"complex and real rotation disagree by {diff:.2e}"
    print(f"[PASS] complex e^(i m theta) engine == real 2x2 rotation matrices (max diff {diff:.1e})")


def test_norm_and_identity() -> None:
    """Rotation keeps every vector's length; position 0 is the identity."""
    torch.manual_seed(5)
    x = torch.randn(2, 4, 128, 64)
    y = apply_rope(x, precompute_freqs_cis(64, 128))
    n_diff = (x.norm(dim=-1) - y.norm(dim=-1)).abs().max().item()
    assert n_diff < 1e-4, f"norm changed by {n_diff:.2e}"
    assert torch.allclose(x[:, :, 0], y[:, :, 0], atol=1e-6), "position 0 must not rotate"
    # each 2D pair keeps its own length too (it's a rotation per plane)
    pair_x = x.view(2, 4, 128, 32, 2).norm(dim=-1)
    pair_y = y.view(2, 4, 128, 32, 2).norm(dim=-1)
    assert torch.allclose(pair_x, pair_y, atol=1e-5), "a 2D pair changed its length"
    print(f"[PASS] norm preserved per vector and per 2D pair, position 0 = identity "
          f"(max norm diff {n_diff:.1e})")


def test_shapes_dtypes_and_api() -> None:
    """Shape/dtype round-trip, starter API == apply_rope, bad shapes are rejected."""
    fc = precompute_freqs_cis(64, 16)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(2, 8, 16, 64).to(dt)
        y = apply_rope(x, fc)
        assert y.shape == x.shape and y.dtype == dt, f"shape/dtype broken for {dt}"
    q, k = torch.randn(1, 2, 16, 64), torch.randn(1, 2, 16, 64)
    qa, ka = apply_rotary_emb(q, k, fc)
    assert torch.equal(qa, apply_rope(q, fc)) and torch.equal(ka, apply_rope(k, fc))
    try:
        apply_rope(torch.randn(1, 1, 15, 64), fc)  # seq_len mismatch
    except AssertionError:
        pass
    else:
        raise AssertionError("shape mismatch was not caught")
    print("[PASS] shapes/dtypes (fp32, fp16, bf16) preserved, starter API == apply_rope, "
          "shape guard works")


def test_frequency_formula() -> None:
    """Starter frequencies == theta_i = base^{-2(i-1)/d}; first pair turns 1 rad/token."""
    d = 128
    fc = precompute_freqs_cis(d, 2)
    from_starter = torch.angle(fc[1]).double()  # angle at position 1 = theta_i
    expected = rope_frequencies(d)
    assert torch.allclose(from_starter, expected, rtol=1e-5), "frequency formula mismatch"
    assert abs(expected[0].item() - 1.0) < 1e-12
    print(f"[PASS] theta_i = 10000^(-2(i-1)/d): theta_1 = {expected[0]:.3f} rad/token, "
          f"theta_{d//2} = {expected[-1]:.2e} rad/token")


# =============================================================================
# 4. Context-length extrapolation
# =============================================================================
def evaluate_context_extrapolation(dim: int = 128, train_len: int = 4096,
                                   target_len: int = 16384, theta: float = 10000.0) -> dict:
    """
    Why RoPE models break past their training length, and how PI repairs it.

    Pair i turns theta_i rad per token, full turn after lambda_i = 2*pi/theta_i tokens.
      * lambda_i < train_len : the pair has already seen every angle in training -> safe.
      * lambda_i > train_len : in training it only reached angle train_len*theta_i < 2*pi.
        At longer contexts it produces angles it has NEVER seen -> out of distribution.
    Position Interpolation multiplies positions by s = train_len/target_len, so the
    largest angle is again train_len*theta_i (back inside the trained range). Cost: two
    neighbouring tokens now differ by theta_i*s instead of theta_i (less local resolution).
    """
    print(f"=== Context extrapolation: d={dim}, train {train_len:,} -> target {target_len:,} ===")
    th = rope_frequencies(dim, theta)
    lam = 2 * math.pi / th
    unsafe = lam > train_len
    s = train_len / target_len
    n_unsafe = int(unsafe.sum())

    print(f"{'pair i':>6} | {'theta_i (rad/tok)':>17} | {'wavelength (tok)':>16} | "
          f"{'max angle train':>15} | {'max angle @target':>17} | {'PI @target':>10}")
    rows = sorted(set([0, 15, 31, int(unsafe.nonzero()[0][0]) if n_unsafe else 0, 47, dim // 2 - 1]))
    for i in rows:
        a_tr = (train_len - 1) * th[i].item()
        a_tg = (target_len - 1) * th[i].item()
        a_pi = (target_len - 1) * s * th[i].item()
        flag = "  <- unseen" if lam[i] > train_len else ""
        print(f"{i + 1:>6} | {th[i].item():>17.3e} | {lam[i].item():>16,.0f} | "
              f"{a_tr:>15.3f} | {a_tg:>17.3f} | {a_pi:>10.3f}{flag}")

    # check: with PI, every interpolated position (target_len-1)*s lies inside the trained
    # position interval [0, train_len), so every pair's angle stays below train_len*theta_i
    pi_ok = bool(((target_len - 1) * s * th < train_len * th).all())
    print(f"* {n_unsafe}/{dim // 2} pairs have wavelength > {train_len:,} tokens: beyond "
          f"{train_len:,} they rotate into angles never seen in training.")
    print(f"* Position Interpolation (s = {s:.3f}) keeps all {dim // 2} pairs inside their "
          f"trained angle range: {pi_ok}.")
    print(f"* Price of PI: adjacent tokens differ by theta_1*s = {th[0].item() * s:.3f} rad "
          f"instead of {th[0].item():.3f} rad, so a short fine-tune is still needed.")
    ntk_base = theta * (target_len / train_len) ** (dim / (dim - 2))
    print(f"* NTK-aware alternative: raise the base to {ntk_base:,.0f} instead (stretches slow "
          f"pairs, keeps fast pairs nearly unchanged).\n")
    assert pi_ok
    return {"n_unsafe": n_unsafe, "pi_ok": pi_ok, "ntk_base": ntk_base}


# =============================================================================
# 5. Decay curves: RoPE vs ALiBi
# =============================================================================
def rope_relative_score(dim: int, distances: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """
    Normalised score for content-aligned q = k = 1 (all ones):
        s(D) = <RoPE(1, m), RoPE(1, m - D)> / ||1||^2 = (2/d) * sum_i cos(D theta_i).
    s(0) = 1. Same quantity as the "long-term decay" illustration in Su et al. 2021.
    """
    th = rope_frequencies(dim, theta)
    return (torch.cos(distances.double()[:, None] * th[None, :])).mean(dim=1)


def rope_decay_envelope(dim: int, distances: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Su et al. upper bound, (1/(d/2)) sum_j |S_j| with S_j = sum_{l<j} e^{i D theta_l},
    rescaled to 1 at D = 0 so its SHAPE can be compared with s(D)."""
    th = rope_frequencies(dim, theta)
    ph = torch.polar(torch.ones(len(distances), len(th), dtype=torch.float64),
                     distances.double()[:, None] * th[None, :])
    env = ph.cumsum(dim=1).abs().mean(dim=1) / (dim // 2)
    return env / ((dim // 2 + 1) / (dim // 2) / 2)  # rescale so the envelope is 1 at D = 0


def alibi_slopes(num_heads: int) -> torch.Tensor:
    """ALiBi slopes m_h = 2^(-8h/H), h = 1..H (Press et al. 2022, power-of-two heads)."""
    return torch.tensor([2 ** (-8 * h / num_heads) for h in range(1, num_heads + 1)])


def plot_rope_vs_alibi(path: str, dim: int = 128, num_heads: int = 8, seq_len: int = 1024) -> None:
    if not HAS_MATPLOTLIB:
        print("[INFO] matplotlib not installed, skipping plot.")
        return

    dist = torch.arange(0, seq_len)
    score = rope_relative_score(dim, dist)
    env = rope_decay_envelope(dim, dist)
    slopes = alibi_slopes(num_heads)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) RoPE score vs distance
    a = ax[0, 0]
    a.plot(dist, score, lw=0.9, label="RoPE score s(D), q = k = 1")
    a.plot(dist, env, lw=1.8, ls="--", label="Su et al. upper bound (rescaled, 1 at D=0)")
    a.axhline(0, color="grey", lw=0.6)
    a.set_xscale("symlog", linthresh=8)
    a.set_xlim(0, seq_len)
    a.set_title("(a) RoPE: oscillating long-term decay", fontweight="bold")
    a.set_xlabel("relative distance D = m - n (tokens)")
    a.set_ylabel("normalised q·k")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (b) ALiBi bias vs distance
    b = ax[0, 1]
    for h in (0, 3, 7):
        b.plot(dist, -slopes[h] * dist, label=f"head {h + 1}: slope {slopes[h]:.4f}")
    b.set_title("(b) ALiBi: linear distance penalty  -m_h · D", fontweight="bold")
    b.set_xlabel("relative distance D (tokens)")
    b.set_ylabel("additive logit bias")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)

    # (c) Attention weights the last query gives to each earlier key
    c = ax[1, 0]
    content = dim / math.sqrt(dim)  # q·k/sqrt(d) for q = k = 1
    w_rope = torch.softmax(content * score, dim=0)
    c.plot(dist, w_rope, lw=1.0, label="RoPE")
    for h in (0, 3, 7):
        w = torch.softmax(content - slopes[h] * dist.double(), dim=0)
        c.plot(dist, w, lw=1.2, label=f"ALiBi head {h + 1}")
    c.set_yscale("log")
    c.set_xscale("symlog", linthresh=8)
    c.set_xlim(0, seq_len)
    c.set_ylim(1e-8, 1.5)
    c.set_title(f"(c) Softmax weight on key at distance D (query at {seq_len - 1})",
                fontweight="bold")
    c.set_xlabel("relative distance D (tokens)")
    c.set_ylabel("attention weight (log)")
    c.legend(fontsize=8)
    c.grid(alpha=0.3, which="both")

    # (d) RoPE distance heatmap: score depends only on (i - j) -> Toeplitz pattern
    d = ax[1, 1]
    n = 64
    ij = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).abs().flatten()
    hm = rope_relative_score(dim, ij).view(n, n)
    im = d.imshow(hm.numpy(), cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=d, label="normalised q·k")
    d.set_title("(d) RoPE score heatmap: only |i - j| matters", fontweight="bold")
    d.set_xlabel("key position j")
    d.set_ylabel("query position i")

    fig.suptitle("Mission 03 / Part 2 — RoPE relative decay vs ALiBi linear penalty "
                 f"(d = {dim}, H = {num_heads})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    near, far = score[:16].abs().mean().item(), score[512:].abs().mean().item()
    print(f"[decay] mean |s(D)|: D < 16 -> {near:.3f}, D >= 512 -> {far:.3f} "
          f"(RoPE decays, but oscillates; ALiBi decays monotonically)")
    print(f"[SUCCESS] RoPE vs ALiBi decay plot saved to: {path}")


# =============================================================================
# main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Mission 03 — Part 2: RoPE Positional Engine")
    ap.add_argument("--no-plot", action="store_true", help="skip saving the decay plot")
    args = ap.parse_args()

    print("=" * 80)
    print("USSE NOOSPHERE — SUBSYSTEM 03: ROTARY POSITIONAL ENGINE")
    print("Role Deliverable: Part 2 — Communications & Protocol Officer")
    print("=" * 80 + "\n")

    verify_relative_invariance()
    print("=== Unit tests ===")
    test_frequency_formula()
    test_complex_matches_real_rotation()
    test_norm_and_identity()
    test_shapes_dtypes_and_api()
    test_relative_invariance_sweep()
    test_attention_logits_translation_invariant()
    print()

    evaluate_context_extrapolation()

    if not args.no_plot:
        here = os.path.dirname(os.path.abspath(__file__))
        plot_rope_vs_alibi(os.path.join(here, "rope_vs_alibi_decay.png"))

    print("\nAll Part 2 Communications & Protocol Officer specifications verified!")


if __name__ == "__main__":
    main()
