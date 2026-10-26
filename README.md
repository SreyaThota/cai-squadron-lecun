# USSE Noosphere — Squadron LeCun
**Conversational AI — Winter Semester 2026/27**  
**Course Reference Repository:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  
**Squadron Branch:** `main` | **IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | PyTorch 2.4+

---

## 🛸 Starship Mission Directory & Log

| Mission | Sector & Waypoint | Subsystem Repaired | Status | Path Link |
| :--- | :--- | :--- | :--- | :--- |
| **Mission 02** | Sirius Binary System (`SEC-02-CMA`) | *Subsystem 02: Signal Entropy Decoders & Statistical Filters* (+7.1%) | ✅ **Completed & Submitted** | [`docs/weeks/week-02/problem_set/`](./docs/weeks/week-02/problem_set/) |
| **Mission 03** | Pleiades Open Cluster (`SEC-03-M45`) | *Subsystem 03: Distributed Attention Mesh & KV-Cache Transceiver* (+7.1%) | 🚀 **In Progress (Part 3 Done)** | [`docs/weeks/week-03/problem_set/`](./docs/weeks/week-03/problem_set/) |

---

## 🛰️ Mission 03: Station Roles & Deliverables Status

- [ ] **Part 1: Systems Architect** (`01_MultiHead_Attention_Core.py`) — Scaled Dot-Product & Multi-Head Self-Attention from scratch.
- [ ] **Part 2: Communications & Protocol Officer** (`02_RoPE_Positional_Engine.py`) — Rotary Positional Embeddings (RoPE) 2D vector rotation engine.
- [x] **Part 3: Data & Payload Specialist** (`03_GQA_and_KVCache_Benchmark.py`) — **COMPLETED by Sreya Thota**  
  * Stateful `KVCache` class with memory tracking and overflow guards.
  * `GQAAttention` layer with configurable $H_Q$ and $H_{KV}$, supporting both copy-free grouped execution and `repeat_interleave`.
  * Comprehensive benchmark across $N \in \{1024, 4096, 16384, 32768\}$ measuring KV-cache footprint (MB), memory bandwidth traffic (GB/s at 50 tok/s), and empirical per-token decoding latency (ms).
  * Passed all 5 unit tests and generated visualization curve `kv_cache_memory.png`.
- [ ] **Part 4: Security & Ethics Sentinel** (`04_Memory_Hierarchy_and_Safety_Audit.md`) — FlashAttention IO derivation & KV-Cache DoS / OOM audit.
- [ ] **Part 5: Full Integration** (`ps03_solution.ipynb`) — Full squadron assembly.

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
                ├── 03_GQA_and_KVCache_Benchmark.py    <- Part 3: COMPLETED DELIVERABLE (Sreya Thota)
                └── kv_cache_memory.png                <- Part 3: Benchmark Curves Plot
```
