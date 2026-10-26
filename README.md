# USSE Noosphere — Squadron LeCun
**Conversational AI — Winter Semester 2026/27**  
**Official Course Repository:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  
**Squadron Branch:** `main` | **IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | PyTorch 2.4+

---

## 🛸 Starship Mission Directory & Log

| Mission | Sector & Waypoint | Subsystem Repaired | Status | Folder Link |
| :--- | :--- | :--- | :--- | :--- |
| **Mission 02** | Sirius Binary System (`SEC-02-CMA`) | *Subsystem 02: Signal Entropy Decoders & Statistical Filters* (+7.1%) | ✅ **Completed & Submitted** | [`mission-02/`](./mission-02/) |
| **Mission 03** | Pleiades Open Cluster (`SEC-03-M45`) | *Subsystem 03: Distributed Attention Mesh & KV-Cache Transceiver* (+7.1%) | 🚀 **In Progress** | [`mission-03/`](./mission-03/) |

---

## 🛰️ Mission 03: Station Roles & Deliverables Status

- [ ] **Part 1: Systems Architect** (`01_MultiHead_Attention_Core.py`) — Scaled Dot-Product & Multi-Head Self-Attention from scratch.
- [ ] **Part 2: Communications & Protocol Officer** (`02_RoPE_Positional_Engine.py`) — Rotary Positional Embeddings (RoPE) 2D vector rotation engine.
- [x] **Part 3: Data & Payload Specialist** (`03_GQA_and_KVCache_Benchmark.py`) — **COMPLETED by Sreya Thota**  
  * Implemented stateful `KVCache` class with memory tracking.
  * Implemented `GQAAttention` layer with configurable $H_Q$ and $H_{KV}$ and dynamic `torch.repeat_interleave` head grouping.
  * Generated multi-dimensional VRAM, Memory Bandwidth (at 50 tok/s), and Latency benchmark tables across $N \in \{1024, 4096, 16384, 32768\}$.
  * Verified with automated unit tests for causal autoregressive decoding equivalence.
- [ ] **Part 4: Security & Ethics Sentinel** (`04_Memory_Hierarchy_and_Safety_Audit.md`) — FlashAttention IO derivation & KV-Cache DoS / OOM audit.
- [ ] **Part 5: Full Integration** (`ps03_solution.ipynb`) — Full squadron assembly.

---

## 📂 Repository Layout

```
cai-squadron-lecun/
├── README.md                                <- Squadron Master Overview (this file)
│
├── mission-02/                              <- Mission 02 (Sirius Binary System)
│   ├── README.md                            <- Mission 2 briefing & checklist
│   ├── ps02_briefing.md                     <- Problem specification
│   ├── 01_Lovelace_Inquiry_Brief.md         <- Part 1: Lovelace inquiry & Anti-Turing probe
│   ├── 02_FST_and_Noisy_Protocol.md         <- Part 2: FST dialogue manager & noisy protocol
│   ├── 03_Entropy_Hazard_Audit.md           <- Part 3: FST stress tests & hazard audit
│   ├── 04_fst_noisy_simulator.py            <- Part 4: Python FST simulator
│   ├── 04_fst_noisy_simulator_starter.py
│   ├── 05_Noisy_Interrogation_Transcript.md <- Part 5: Interrogation transcript
│   └── fst_dialogue_manager.py              <- FST class implementation
│
├── mission-03/                              <- Mission 03 (Pleiades Open Cluster)
│   ├── README.md                            <- Mission 3 briefing & checklist
│   ├── ps03_briefing.md                     <- Problem specification
│   ├── 01_MultiHead_Attention_Core_starter.py
│   ├── 02_RoPE_Positional_Engine_starter.py
│   ├── 03_GQA_and_KVCache_Benchmark.py      <- Part 3: COMPLETED DELIVERABLE (Data & Payload)
│   ├── 03_GQA_and_KVCache_Benchmark_starter.py
│   └── 04_Memory_Hierarchy_and_Safety_Audit_template.md
│
└── docs/weeks/week-03/problem_set/          <- Mirror path specified in ps03.pdf
    └── 03_GQA_and_KVCache_Benchmark.py
```
