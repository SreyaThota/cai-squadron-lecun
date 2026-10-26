# USSE Noosphere — Squadron LeCun
**Conversational AI — Winter Semester 2026/27**  
**Course Reference Repository:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  
**Squadron Branch:** `main` | **IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | PyTorch 2.4+

---

## 🛸 Starship Mission Directory & Log

| Mission | Sector & Waypoint | Subsystem Repaired | Status | Path Link |
| :--- | :--- | :--- | :--- | :--- |
| **Mission 02** | Sirius Binary System (`SEC-02-CMA`) | *Subsystem 02: Signal Entropy Decoders & Statistical Filters* (+7.1%) | ✅ **Completed & Submitted** | [`docs/weeks/week-02/problem_set/`](./docs/weeks/week-02/problem_set/) |
| **Mission 03** | Pleiades Open Cluster (`SEC-03-M45`) | *Subsystem 03: Distributed Attention Mesh & KV-Cache Transceiver* (+7.1%) | ✅ **All 5 Parts Complete** | [`docs/weeks/week-03/problem_set/`](./docs/weeks/week-03/problem_set/) |

---

## 🛰️ Mission 03: Station Roles & Deliverables Status

- [x] **Part 1: Systems Architect** (`01_MultiHead_Attention_Core.py`) — **COMPLETED**  
  * Scaled Dot-Product & Multi-Head Self-Attention from scratch in pure PyTorch.
  * Linear projections $W_Q, W_K, W_V, W_O$ and multi-head tensor splitting into $h$ heads.
  * Lower-triangular causal attention masking ($-\infty$ above diagonal).
  * Empirical variance preservation validation ($\frac{1}{\sqrt{d_k}}$ preserves unit variance $\sigma^2 \approx 1.0$).
  * Automated tests for PyTorch reference equivalence and future information leakage prevention.
- [x] **Part 2: Communications & Protocol Officer** (`02_RoPE_Positional_Engine.py`) — **COMPLETED by Vasu Midde**  
  * `apply_rope(x, freqs_cis)` complex 2D planar rotation with $\theta_i = 10000^{-2(i-1)/d}$, cross-checked against explicit $2\times2$ rotation matrices.
  * Proof and empirical checks of relative invariance $\langle R_m q, R_n k\rangle = g(q,k,m-n)$: starter test, 200-trial random sweep ($|\delta|$ up to $10^5$), full attention-logit matrix under translation.
  * Context-extrapolation audit (18/64 frequency pairs leave their trained range beyond 4K) with Position Interpolation fix and NTK-aware base.
  * RoPE oscillating decay vs ALiBi linear penalty plot `rope_vs_alibi_decay.png`.
- [x] **Part 3: Data & Payload Specialist** (`03_GQA_and_KVCache_Benchmark.py`) — **COMPLETED by Sreya Thota**  
  * Stateful `KVCache` class with memory tracking and overflow guards.
  * `GQAAttention` layer with configurable $H_Q$ and $H_{KV}$, supporting both copy-free grouped execution and `repeat_interleave`.
  * Comprehensive benchmark across $N \in \{1024, 4096, 16384, 32768\}$ measuring KV-cache footprint (MB), memory bandwidth traffic (GB/s at 50 tok/s), and empirical per-token decoding latency (ms).
  * Passed all 5 unit tests and generated visualization curve `kv_cache_memory.png`.
- [x] **Part 4: Security & Ethics Sentinel** (`04_Memory_Hierarchy_and_Safety_Audit.md`) — **COMPLETED by Arjun V R**  
  * FlashAttention IO complexity derivation ($O(N)$ SRAM tiling vs $O(N^2)$ HBM).
  * Long-context KV-Cache DoS and CUDA Out-Of-Memory (OOM) attack simulation.
  * Architectural mitigation audit (PagedAttention & Multi-Head Latent Attention - MLA).
- [x] **Part 5: Full Integration** (`ps03_solution.ipynb`) — **COMPLETED by Vasu Midde**  
  * Imports and runs the verification suites of Parts 1-3 directly from the `.py` files.
  * Re-computes the Part 4 audit in code: exact tiled online-softmax attention, 4.0x IO worked example, KV-cache DoS capacity table.
  * `SquadronAttention` = GQA (Part 3) + RoPE (Part 2) + KV-cache, verified against Part 1 MHSA, Part 3 GQA and PyTorch SDPA; demonstrates the RoPE cache-offset trap.
  * `SquadronLM` tiny decoder: learns echo-after-16 (100% with RoPE vs 59.8% without); cached generation == full recompute. 18 checks pass.

---

## 📂 Repository Layout

```
cai-squadron-lecun/
├── README.md                                          <- Master Squadron Overview
│
└── docs/
    └── weeks/
        ├── week-02/
        │   └── problem_set/
        │       ├── 01_Lovelace_Inquiry_Brief.md       <- Part 1: Lovelace Brief & Anti-Turing Probe
        │       ├── 02_FST_and_Noisy_Protocol.md       <- Part 2: FST Dialogue Manager & Noisy Protocol
        │       ├── 03_Entropy_Hazard_Audit.md         <- Part 3: FST Stress Tests & Shannon Audit
        │       ├── 04_fst_noisy_simulator.py          <- Part 4: Python FST Simulator
        │       ├── 04_fst_noisy_simulator_starter.py
        │       ├── 05_Noisy_Interrogation_Transcript.md <- Part 5: Interrogation Transcript
        │       ├── fst_dialogue_manager.py            <- FST Class Implementation
        │       ├── ps02_briefing.md                   <- Problem Specification
        │       └── README.md                          <- Mission 2 Overview & Checklist
        │
        └── week-03/
            └── problem_set/
                ├── 01_MultiHead_Attention_Core.py     <- Part 1: COMPLETED DELIVERABLE (Systems Architect)
                ├── causal_attention_map.png           <- Part 1: Causal Attention Heatmap
                ├── 02_RoPE_Positional_Engine.py       <- Part 2: COMPLETED DELIVERABLE (Vasu Midde)
                ├── rope_vs_alibi_decay.png            <- Part 2: RoPE vs ALiBi Decay Plot
                ├── 03_GQA_and_KVCache_Benchmark.py    <- Part 3: COMPLETED DELIVERABLE (Data & Payload)
                ├── kv_cache_memory.png                <- Part 3: Benchmark Curves Plot
                ├── 04_Memory_Hierarchy_and_Safety_Audit.md <- Part 4: COMPLETED DELIVERABLE (Arjun V R)
                └── ps03_solution.ipynb                <- Part 5: FULL INTEGRATION (Vasu Midde)
```
