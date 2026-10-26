# Memory Hierarchy & Safety Audit

**Mission 3, Part 4: Security & Ethics Sentinel**
**Subsystem:** AURA Subsystem 03, Distributed Multi-Head Attention Mesh & KV-Cache Transceiver
**Course:** Conversational AI, Winter Semester 2026/27
**Author:** _<your name / squadron>_

---

## Contents

1. FlashAttention IO complexity derivation
2. KV-Cache denial-of-service scenario
3. Mitigation strategies: PagedAttention and Multi-Head Latent Attention
4. Recommendations and ethical considerations
5. References

---

## 1. FlashAttention IO Complexity Derivation

### 1.1 Setup and notation

| Symbol | Meaning |
|---|---|
| $N$ | sequence length |
| $d$ | head dimension (per head) |
| $M$ | on-chip SRAM size, measured in number of elements |
| HBM | off-chip high-bandwidth memory: large (tens of GB), slow (about 2-3 TB/s on A100/H100) |
| SRAM | on-chip shared memory / registers: small (about 100-200 KB per SM), roughly an order of magnitude faster |

Single head, with $Q, K, V \in \mathbb{R}^{N \times d}$. We assume $d \le M \le N d$, the regime of practical interest, and in particular $M \gg d^2$ for the bound below to be meaningful.

Attention computes

$$S = QK^\top \in \mathbb{R}^{N\times N}, \qquad P = \mathrm{softmax}(S), \qquad O = PV \in \mathbb{R}^{N\times d}.$$

Modern attention kernels are **memory-bound, not compute-bound**: the cost is dominated by how many bytes move between HBM and SRAM, not by the FLOPs. We therefore count HBM reads and writes.

### 1.2 Standard attention

The naive implementation runs three separate kernels. Each kernel reads its inputs from HBM and writes its outputs back.

| Step | Operation | HBM reads | HBM writes |
|---|---|---|---|
| 1 | $S = QK^\top$ | $Q, K$: $2Nd$ | $S$: $N^2$ |
| 2 | $P = \mathrm{softmax}(S)$ | $S$: $N^2$ | $P$: $N^2$ |
| 3 | $O = PV$ | $P, V$: $N^2 + Nd$ | $O$: $Nd$ |

Total:

$$\text{IO}_{\text{standard}} = 4Nd + 4N^2 = \Theta(Nd + N^2).$$

Additionally the algorithm needs $\Theta(N^2)$ **extra memory** for $S$ and $P$ (per head, per batch element). For $N = 131{,}072$ (128K tokens) in fp16 this is $131072^2 \times 2\ \text{B} \approx 34\ \text{GB}$ per head, which is clearly infeasible for any real model.

### 1.3 Key idea of FlashAttention: tiling + online softmax

FlashAttention never materializes $S$ or $P$ in HBM. It splits $Q, K, V$ into blocks that fit in SRAM and computes the output block by block, **fusing** all three steps into one kernel.

**Block sizes.** SRAM must hold a block of $K_j, V_j$ (each $B_c \times d$), a block of $Q_i$ and $O_i$ (each $B_r \times d$), and the score tile $S_{ij}$ ($B_r \times B_c$). Choosing

$$B_c = \left\lceil \frac{M}{4d} \right\rceil, \qquad B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil,\, d\right)$$

guarantees everything fits.

**Online softmax.** Softmax couples all columns of a row, so the output cannot simply be accumulated tile by tile. The fix is to keep, for each query row, a running maximum $m$ and a running normalizer $\ell$, and rescale previous partial results whenever the maximum changes. For a new tile $S_{ij}$:

$$
\begin{aligned}
m^{\text{new}} &= \max\!\big(m^{\text{old}},\ \text{rowmax}(S_{ij})\big)\\
\ell^{\text{new}} &= e^{m^{\text{old}} - m^{\text{new}}}\,\ell^{\text{old}} + \text{rowsum}\!\big(e^{S_{ij} - m^{\text{new}}}\big)\\
O^{\text{new}} &= e^{m^{\text{old}} - m^{\text{new}}}\,O^{\text{old}} + e^{S_{ij} - m^{\text{new}}}\,V_j
\end{aligned}
$$

After the last tile, $O \leftarrow O / \ell$. This is **exact** (not an approximation) and numerically stable because the maximum is subtracted.

**Loop structure.**

```
for j in 1..T_c:            # outer loop over K/V blocks, T_c = ceil(N / B_c)
    load K_j, V_j  -> SRAM
    for i in 1..T_r:        # inner loop over Q blocks, T_r = ceil(N / B_r)
        load Q_i, O_i, m_i, l_i -> SRAM
        S_ij = Q_i K_j^T                       (in SRAM)
        update m_i, l_i, O_i via online softmax (in SRAM)
        write O_i, m_i, l_i -> HBM
```

### 1.4 Counting HBM accesses

* Each $K_j, V_j$ block is loaded **once**, so all of $K$ and $V$ together cost $2Nd$.
* For each of the $T_c$ outer iterations, the inner loop streams all of $Q$ and all of $O$: reads of $Q$ and $O$ plus a write of $O$, i.e. $\Theta(Nd)$ per outer iteration. (The statistics $m, \ell$ have size $N$, which is lower order.)
* Number of outer iterations: $T_c = \lceil N / B_c \rceil = \Theta\!\left(\dfrac{Nd}{M}\right)$.

Therefore

$$\text{IO}_{\text{flash}} = \Theta(Nd) + \Theta(Nd \cdot T_c) = \Theta\!\left(Nd + Nd\cdot\frac{Nd}{M}\right) = \Theta\!\left(Nd + \frac{N^2 d^2}{M}\right).$$

### 1.5 Comparison

$$\frac{\text{IO}_{\text{flash}}}{\text{IO}_{\text{standard}}} \;\approx\; \frac{N^2 d^2 / M}{N^2} = \frac{d^2}{M}.$$

Because $M$ is typically much larger than $d^2$, FlashAttention performs asymptotically fewer HBM accesses by a factor of about $M / d^2$. The paper additionally shows this is **optimal**: no exact attention algorithm can asymptotically beat $\Omega(N^2 d^2 / M)$ HBM accesses for all $M$ in the range $[d, Nd]$.

**Worked example.** Take $N = 8192$, $d = 64$, and $M \approx 50{,}000$ fp16 elements (about 100 KB of SRAM).

* Standard: $4Nd + 4N^2 \approx 2.7 \times 10^8$ element accesses.
* FlashAttention: $B_c = \lceil 50000 / 256 \rceil = 196$, $T_c = \lceil 8192/196 \rceil = 42$, so IO $\approx 2Nd + 3NdT_c \approx 6.7 \times 10^7$.
* Ratio: about **4x** fewer HBM accesses in this simplified model.

This is a simple counting model, so real kernels will differ. The qualitative conclusion holds: the benefit grows with larger SRAM tiles and longer sequences, and it is amplified in practice because the kernel is fused (no extra launches, no intermediate round trips).

### 1.6 A note on "O(N^2) to O(N)"

The mission briefing states that FlashAttention reduces traffic "from $O(N^2)$ to $O(N)$". This should be stated carefully:

* **Memory footprint** drops from $O(N^2)$ to **$O(N)$**: the $N \times N$ matrices are never stored.
* **HBM traffic** drops from $\Theta(N^2)$ to $\Theta(N^2 d^2 / M)$. It is still quadratic in $N$, but with a much smaller constant, and it is *linear in $N$ only when $M = \Omega(N d^2)$*, which is not realistic for long sequences.
* **FLOPs** remain $\Theta(N^2 d)$: FlashAttention is exact and does not reduce compute. It is faster because it removes the memory bottleneck.

In the backward pass, FlashAttention also avoids storing $S$ and $P$ by **recomputing** them from $Q, K, V$ and the saved statistics $(m, \ell)$. This trades extra FLOPs for far less memory traffic.

### 1.7 FlashAttention-3 (hardware-specific improvements)

FlashAttention-3 targets NVIDIA Hopper (H100). It does not change the asymptotic IO analysis above. It improves *utilization* of the hardware through:

* **Warp specialization / asynchrony:** overlapping data movement (via the Tensor Memory Accelerator, TMA) with Tensor Core matmuls and softmax.
* **Interleaving matmul and softmax** to hide softmax latency.
* **FP8 support** with block-wise scaling and incoherent processing to limit quantization error.

---

## 2. KV-Cache Denial-of-Service Simulation

### 2.1 Why the KV cache is the attack surface

During autoregressive decoding, the keys and values of all previous tokens are stored so they are not recomputed. This turns generation from repeated recomputation into one-token-at-a-time work, but the cache **grows linearly with context length and with the number of concurrent sessions**, and it lives in scarce GPU VRAM.

Per-token KV-cache size:

$$\text{bytes/token} = 2 \times L \times H_{KV} \times d_{\text{head}} \times b$$

where $L$ is the number of layers, $H_{KV}$ the number of key/value heads, and $b$ the bytes per element.

**LLaMA-3-8B-like configuration** ($L=32$, $H_{KV}=8$ via GQA, $d_{\text{head}}=128$, fp16, $b=2$):

$$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{B} = 128\ \text{KiB per token}.$$

| Context length | KV cache per session |
|---|---|
| 8,192 | 1 GiB |
| 32,768 | 4 GiB |
| 131,072 (128K) | 16 GiB |

*(For comparison, the same model with full MHA, $H_{KV}=32$, would need 4x more: 64 GiB per 128K session.)*

### 2.2 Threat model

* **Target:** a shared inference server (the USSE Noosphere mainframe) serving many users from one GPU pool.
* **Attacker:** a normal authenticated or anonymous user with no special privileges. Only the public chat API is used.
* **Goal:** exhaust VRAM so that other users' requests are rejected, stalled, or crash the serving process (availability attack, a denial of service).
* **Assumptions:** the server allows long contexts (e.g. 128K), keeps conversation state alive across turns for fast follow-ups, and does not strictly limit sessions per user.

### 2.3 Attack scenario ("Seven Sisters Flood")

**Illustrative budget (one 80 GB GPU):** model weights of about 15 GiB leave a KV pool of roughly **48 GiB** after activations and overhead.

| Step | Attacker action | Effect on server |
|---|---|---|
| 1 | Open many concurrent sessions (or a few accounts, or a script rotating sessions). | Each session is admitted. |
| 2 | In each session, paste or generate a very long prompt (e.g. 120K+ tokens of filler) to maximize the prefill size. | Each session's KV cache grows to ~15 GiB. |
| 3 | Keep every session "alive" by sending a periodic short follow-up turn, and never close the connection. | Caches are never freed (multi-turn state retention). |
| 4 | Request very long outputs (large `max_tokens`) so decoding keeps growing the cache. | VRAM pressure rises with every generated token. |
| 5 | Repeat until the pool is full. | New requests fail with CUDA OOM, the scheduler stalls, or the process is killed and **all** users lose their sessions. |

**Capacity table (48 GiB KV pool, LLaMA-3-8B-like, GQA 8 KV heads):**

| Context per session | KV per session | Sessions that fit |
|---|---|---|
| 8K | 1 GiB | 48 |
| 32K | 4 GiB | 12 |
| 128K | 16 GiB | **3** |

Just **four** malicious 128K sessions are enough to take down a node that comfortably served dozens of normal users.

**Naive pre-allocation makes it worse.** If the server reserves a contiguous buffer for `max_seq_len` per request (a common simple design), then *every* request, even a 50-token chat, reserves 16 GiB. Then **three concurrent ordinary users** already exhaust the pool, with no attacker needed. The attacker only needs to be one of the first few.

**Compute-side amplification.** Without FlashAttention, prefill materializes $N \times N$ score matrices. For $N = 131{,}072$ and 32 heads in fp16 that is about $131072^2 \times 32 \times 2\ \text{B} \approx 1.1\ \text{TB}$ for a single layer's scores, so a single malicious prompt can OOM the GPU even before the KV cache matters. Kernel-level fixes (FlashAttention) are therefore a prerequisite for safe long-context serving.

**Failure modes observed:**

* **CUDA OOM** during prefill or decode, which crashes the request or the whole engine process.
* **Head-of-line blocking:** the scheduler cannot admit new requests, so latency for legitimate users grows without bound.
* **Memory fragmentation:** even when total free memory is large, no *contiguous* region is large enough for a new request's buffer.
* **Cascading failure:** after a crash, clients retry at once, producing a thundering herd.

### 2.4 Reproducing the simulation

The capacity arithmetic above can be reproduced with a few lines of Python (no GPU required):

```python
# kv_dos_sim.py: capacity model for the KV-cache DoS scenario
L, H_KV, D_HEAD, BYTES = 32, 8, 128, 2          # LLaMA-3-8B-like, fp16
POOL_GIB = 48                                   # KV pool after weights + overhead

bytes_per_token = 2 * L * H_KV * D_HEAD * BYTES
print(f"KV per token: {bytes_per_token/1024:.0f} KiB")

print(f"{'context':>9} {'KV/session (GiB)':>18} {'sessions that fit':>19}")
for ctx in (8_192, 32_768, 131_072):
    gib = ctx * bytes_per_token / 2**30
    print(f"{ctx:>9} {gib:>18.1f} {int(POOL_GIB // gib):>19}")

# Naive pre-allocation: every request reserves max_seq_len
max_len = 131_072
reserved = max_len * bytes_per_token / 2**30
print(f"\nNaive reservation: {reserved:.0f} GiB per request -> "
      f"only {int(POOL_GIB // reserved)} concurrent requests, regardless of actual length")
```

---

## 3. Mitigation Strategies

### 3.1 PagedAttention (vLLM block paging)

**Idea.** Borrow virtual memory from operating systems. Instead of one contiguous buffer per sequence, split each sequence's KV cache into fixed-size **blocks** (e.g. 16 tokens). A per-sequence **block table** maps logical block indices to physical blocks, which can be scattered anywhere in the GPU memory pool. Blocks are allocated **on demand** as the sequence grows and returned to a free list when the sequence ends.

**What it fixes**

* **Internal fragmentation / over-reservation:** a sequence only holds the blocks it has actually filled, with waste bounded to less than one block per sequence. This removes the "reserve `max_seq_len` for everyone" problem, which in the example above increases admitted ordinary sessions from 3 to dozens.
* **External fragmentation:** blocks are uniform in size, so any free block can be used.
* **Sharing:** blocks can be shared across sequences with **copy-on-write**, enabling prefix caching (shared system prompts) and efficient parallel sampling / beam search.
* **Graceful degradation:** with a block-level view, the scheduler can **preempt** low-priority sequences by swapping their blocks to CPU memory or recomputing them later, instead of crashing.

**What it does *not* fix**

* It does not reduce the **amount** of KV data a truly long sequence needs. Four genuinely filled 128K sessions still need 64 GiB. PagedAttention removes *waste*, not *demand*, so a determined attacker can still exhaust the pool unless admission control and quotas are also applied.
* Block-table indirection adds a small overhead to the attention kernel.

### 3.2 Multi-Head Latent Attention (MLA, DeepSeek-V2/V3)

**Idea.** Instead of caching full per-head keys and values, cache a **low-rank latent vector** per token and reconstruct K and V from it when needed.

For a token hidden state $h_t$:

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \qquad K_t = W^{UK} c_t^{KV}, \qquad V_t = W^{UV} c_t^{KV},$$

with $d_c \ll n_h d_h$. Only $c_t^{KV}$ is cached. At inference, the up-projections $W^{UK}$ and $W^{UV}$ can be **absorbed** into the query and output projections, so the full K and V never need to be materialized.

Because RoPE rotates keys in a position-dependent way, it cannot be absorbed into a fixed low-rank map. MLA handles this with a small **decoupled RoPE key** $k_t^R \in \mathbb{R}^{d_R}$ that is cached alongside the latent. The cache per token per layer is therefore $d_c + d_R$ elements.

**Cache comparison (per token, using the published DeepSeek-V3 configuration: 61 layers, 128 heads, head dim 128, $d_c = 512$, $d_R = 64$, BF16):**

| Scheme | Elements per token (all layers) | Approx. size per token |
|---|---|---|
| MHA (hypothetical, full K and V) | $2 \times 128 \times 128 \times 61 \approx 2.0\text{M}$ | about 3.8 MiB |
| MLA | $(512 + 64) \times 61 \approx 35\text{K}$ | about 69 KiB |

That is a reduction of roughly **57x** versus MHA, and substantially below what GQA with 8 KV heads can achieve at the same scale. (DeepSeek-V2 reports a 93% KV-cache reduction relative to its previous dense model.)

**What it fixes**

* Directly attacks **demand**: each 128K session costs far less VRAM, so many more sessions fit and the cost of an attacker's long session drops by more than an order of magnitude.
* Lower memory bandwidth per decoded token, which improves latency and throughput.
* Unlike GQA/MQA, MLA is reported to preserve (and even slightly improve) model quality relative to MHA.

**Trade-offs**

* It is an **architectural** change: it must be chosen at pre-training time and cannot simply be bolted onto an existing GQA model without retraining or conversion work.
* More complex implementation (decoupled RoPE, weight absorption, specialized kernels).
* Extra compute at decode time, which is acceptable because decoding is memory-bound.

### 3.3 Comparison

| Property | Naive contiguous cache | GQA / MQA | PagedAttention | MLA |
|---|---|---|---|---|
| Reduces total KV demand per token | No | Yes (by $H_Q/H_{KV}$) | No | Yes (largest) |
| Removes over-reservation / fragmentation | No | No | **Yes** | No |
| Enables prefix sharing | No | No | **Yes** | Compatible |
| Needs model retraining | No | Yes | No | Yes |
| Stops a determined attacker by itself | No | No | No | No, but raises their cost substantially |
| Complexity | Low | Low | Medium (serving system) | High (model + kernel) |

**Conclusion:** the two defenses are **complementary**. PagedAttention is a *serving-system* fix for waste and fragmentation. MLA is a *model-architecture* fix for raw demand. Neither is sufficient alone against an adversary, which is why policy-level controls (Section 4) are needed.

---

## 4. Recommendations and Ethical Considerations

### 4.1 Defense-in-depth recommendations for the Noosphere mainframes

1. **Use a memory-efficient attention kernel** (FlashAttention) so prefill never materializes $N \times N$ matrices.
2. **Adopt paged KV allocation** (vLLM-style) with on-demand blocks and no `max_seq_len` pre-reservation.
3. **Admission control:** estimate each request's worst-case KV need ($\text{prompt} + \text{max\_tokens}$) and reject or queue requests that would exceed the pool, rather than discovering the limit via OOM.
4. **Per-user and per-tenant quotas:** cap concurrent sessions, total KV tokens in flight, and maximum context per tier.
5. **Idle-session eviction:** free or swap out caches of sessions idle beyond a timeout; recompute from the transcript when they resume (optionally using prefix caching to make this cheap).
6. **Preemption and swapping:** under pressure, preempt lowest-priority sequences to CPU memory instead of crashing.
7. **Reduce per-token footprint:** prefer GQA/MLA models for long-context workloads; consider KV-cache quantization (FP8/INT4) and sliding-window or sparse attention where quality permits.
8. **Rate limiting and anomaly detection:** flag accounts that open many long, never-closing sessions with filler content.
9. **Isolation and graceful failure:** per-tenant memory pools or separate worker processes so one tenant's OOM cannot take down the whole engine, and retry with jittered backoff to avoid thundering herds.
10. **Monitoring:** alert on KV-pool utilization, block-allocator free-list size, queue depth, and preemption rate.

### 4.2 Ethical considerations

* **Availability is a safety property.** On a ship AI such as AURA, a DoS on the attention subsystem during an astrogation scan is not just an inconvenience: it can degrade critical decisions. Capacity planning should treat availability as part of system safety.
* **Fairness vs. protection.** Quotas and eviction protect the shared service but may disadvantage users with legitimately long documents or research workloads. Limits should be transparent, tiered, and documented, with a clear path to request more capacity.
* **Privacy of cached state.** KV caches contain a transformed representation of user prompts. Swapping them to CPU memory or disk, sharing prefix blocks across users, or evicting without zeroing memory can create cross-tenant leakage risks. Prefix sharing should be scoped per tenant (or per trust domain), and freed blocks should not be readable by other tenants. (Cache-timing side channels in shared prefix caches are a known concern worth monitoring.)
* **Responsible testing.** DoS testing must be done only on systems we own or are authorized to test, in an isolated environment. The simulation in Section 2.4 is a capacity model, not an attack tool.

---

## 5. References

1. T. Dao, D. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
2. T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023.
3. J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* 2024.
4. W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
5. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* 2024 (introduces MLA).
6. DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024.
7. J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* 2023.
8. N. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019 (Multi-Query Attention).
