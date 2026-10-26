# Mission 3: Multi-Head Attention, RoPE Vector Rotation & Grouped-Query KV-Cache Optimization

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 3 — Modern Transformer Architectures and Architectural Divergence  
**Galactic Waypoint:** Pleiades Open Cluster (`SEC-03-M45`, 444 ly from Sol — Sub-Sector Seven Sisters)  
**AURA Subsystem Repair:** Subsystem 03: *Distributed Multi-Head Attention Mesh & KV-Cache Transceiver* (+7.1% $\to$ 21.3% total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | PyTorch 2.4+ / Colab / Modern LLMs  
**Official Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

### Mission Briefing

**TO:** All Bridge Squadrons, *USSE Noosphere*  
**FROM:** Expedition Command (AURA Astrogation Core & Transformer Mesh Bay)  
**SUBJECT:** ANOMALY DETECTED: Pleiades M45 Optical Resonance Lock & KV-Cache Memory Bandwidth Saturation  

**Situation:**  
Following a successful warp jump past Sirius, the *USSE Noosphere* has entered the glowing sapphire reflection nebulae of the **Pleiades Open Cluster (M45)**, located 444 light-years from Earth. As our astronomical sensor arrays capture multi-spectral stellar reflections from the Seven Sisters, **AURA Subsystem 03 (Spatial Attention Matrix & KV-Cache Transceiver)** suffers severe telemetry buffer overflow.

**Problem:**  
AURA's damaged neural architecture is attempting to process long-context sensor streams using naive, un-tiled Multi-Head Attention. The $O(N^2)$ quadratic compute scaling and massive VRAM bandwidth demands of standard Key-Value caching are saturating our photonic memory bus, threatening system kernel panic during critical astrogation scans.

To restore Subsystem 03 (+7.1% integrity, raising total ship AI consciousness to **21.3%**) and stabilize multi-turn squadron communications, cadets must reconstruct the core Transformer self-attention pipeline from first principles in pure PyTorch:
1. **Scaled Dot-Product Self-Attention:** $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$ with variance preservation validation.
2. **Rotary Positional Embeddings (RoPE):** $R_{\Theta, m}^d$ complex 2D vector coordinate rotation ensuring relative distance invariance $\langle R_m q, R_n k \rangle = g(q, k, m - n)$.
3. **Grouped-Query Attention (GQA):** $H_Q : H_{KV}$ head grouping (e.g., LLaMA 3 $32:8$ configuration) with dynamic KV tensor broadcasting.
4. **Key-Value Cache (KV-Cache):** State retention buffer for autoregressive decoding to shift execution from $O(N^3)$ recomputation to $O(1)$ token generation.
5. **FlashAttention-3 SRAM Tiling & Memory Safety:** IO complexity analysis reducing High-Bandwidth Memory (HBM) traffic from $O(N^2)$ to $O(N)$.

---

### Bridge Crew Role Directives & Deliverables

#### Part 1: Systems Architect (Lead Engineer) — *Scaled Dot-Product & Multi-Head Self-Attention from Scratch*
* **Focus:** Mathematical Implementation & PyTorch Tensor Pipeline
* **Directive:** Construct a complete `MultiHeadSelfAttention(nn.Module)` in pure PyTorch from first principles without using high-level library wrappers (`nn.MultiheadAttention`).
* **Deliverable (`01_MultiHead_Attention_Core.py` / `ps03_solution.ipynb`):**
  1. **Linear Projections:** Implement learned weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ and output projection $W_O$.
  2. **Multi-Head Reshaping:** Correctly split tensors into $h$ parallel heads: `(batch, seq_len, num_heads, head_dim)` $\to$ `(batch, num_heads, seq_len, head_dim)`.
  3. **Causal Masking:** Implement lower-triangular causal attention masking ($-\infty$ above diagonal) to support autoregressive decoder execution.
  4. **Variance Preservation Unit Test:** Provide an empirical test validating that when inputs have zero mean and unit variance ($\sigma^2 = 1.0$), scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance across dot products before softmax.

#### Part 2: Communications & Protocol Officer — *Rotary Positional Embeddings (RoPE) Engine*
* **Focus:** Geometric Relative Positional Injection & Frequency Scaling
* **Directive:** Implement Rotary Position Embeddings (RoPE / Su et al., 2021) and evaluate context length extrapolation.
* **Deliverable (`02_RoPE_Positional_Engine.py`):**
  1. **`apply_rope(x, freqs_cis)` Function:** Compute rotary frequencies $\theta_i = \text{base}^{-2(i-1)/d}$ and apply complex 2D planar rotation to Query and Key tensors.
  2. **Relative Invariance Validation:** Mathematically and empirically verify that $\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = \langle \text{RoPE}(q, m + \delta), \text{RoPE}(k, n + \delta) \rangle$ for any arbitrary spatial translation $\delta$.
  3. **Visual Attention Decay Heatmap:** Plot an attention weight distance curve comparing RoPE relative decay against ALiBi linear distance penalties.

#### Part 3: Data & Payload Specialist — *Grouped-Query Attention (GQA) & KV-Cache Benchmarking*
* **Focus:** Memory Bandwidth Optimization & Autoregressive KV-Caching
* **Directive:** Build a Grouped-Query Attention (GQA) layer and implement stateful Key-Value caching for autoregressive token decoding.
* **Deliverable (`03_GQA_and_KVCache_Benchmark.py`):**
  1. **`GQAAttention` Layer:** Implement head grouping with configurable $H_Q$ (e.g. 32) and $H_{KV}$ (e.g. 8) heads, using `torch.repeat_interleave` or tensor expansion to match query heads during attention.
  2. **`KVCache` Class:** Implement an in-memory cache buffer that retains past $(K_t, V_t)$ tensors and appends new single-token slices during sequential decoding.
  3. **VRAM & Latency Benchmark Table:** Generate a comparative benchmark across sequence lengths $N \in \{1024, 4096, 16384, 32768\}$ measuring:
     - KV-Cache Memory Footprint (in MB)
     - Memory Bandwidth Traffic (in GB/s at 50 tokens/sec)
     - Decoding Latency per token (in ms)
     - Compare Multi-Head Attention (MHA), Grouped-Query Attention (GQA 4:1), and Multi-Query Attention (MQA 32:1).

#### Part 4: Security & Ethics Sentinel — *FlashAttention IO Analysis & Denial-of-Service Audit*
* **Focus:** Hardware Memory Hierarchy & Long-Context VRAM Exhaustion
* **Directive:** Audit hardware memory limits and evaluate systemic failure modes under adversarial long-context inputs.
* **Deliverable (`04_Memory_Hierarchy_and_Safety_Audit.md`):**
  1. **FlashAttention IO Complexity Derivation:** Provide a formal derivation comparing Standard Attention HBM reads/writes ($O(N \cdot d + N^2)$) against FlashAttention SRAM Block Tiling ($O(N \cdot d + \frac{N^2 d^2}{M})$).
  2. **KV-Cache Denial-of-Service Simulation:** Document an adversarial scenario where concurrent unconstrained multi-turn dialogs trigger CUDA Out-Of-Memory (OOM) memory panics on the *USSE Noosphere* mainframes.
  3. **Mitigation Strategy:** Evaluate PagedAttention (vLLM block paging) and low-rank Multi-Head Latent Attention (MLA / DeepSeek-V3) as architectural defenses against memory fragmentation and resource exhaustion.

---

### Submission Guidelines

1. **Repository Structure:**
   ```
   docs/weeks/week-03/problem_set/
   ├── 01_MultiHead_Attention_Core.py
   ├── 02_RoPE_Positional_Engine.py
   ├── 03_GQA_and_KVCache_Benchmark.py
   ├── 04_Memory_Hierarchy_and_Safety_Audit.md
   └── ps03_solution.ipynb
   ```
2. **Execution & Reproduction:**
   Ensure all Python scripts run cleanly with `python <filename>.py` and produce formatted terminal output / visualization plots.
3. **Git Commit Protocol:**
   Commit all deliverables to your squadron branch with message:
   `feat(ps03): restore AURA Subsystem 03 Spatial Attention Mesh (SEC-03-M45)`
