# Mission 03: Multi-Head Attention, RoPE Vector Rotation & Grouped-Query KV-Cache Optimization

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 3 — Modern Transformer Architectures and Architectural Divergence  
**Waypoint:** Pleiades Open Cluster (`SEC-03-M45`, 444 ly from Sol)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | PyTorch 2.4+ / Colab / Modern LLMs  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter assets and templates for **Mission 3**.

```
mission-03/
├── README.md                                          <- Mission overview & squad sign-off status
├── ps03_briefing.md                                    <- Full mission specification & problem statement
├── 01_MultiHead_Attention_Core_starter.py             <- Starter script (Systems Architect)
├── 02_RoPE_Positional_Engine_starter.py               <- Starter script (Protocol Officer)
├── 03_GQA_and_KVCache_Benchmark_starter.py           <- Starter script (Data & Payload Specialist)
└── 04_Memory_Hierarchy_and_Safety_Audit_template.md   <- Deliverable template (Security Sentinel)
```

---

## Station Role Deliverables Checklist

Before transmitting your team's repository link on the course website submission portal, complete all deliverables and record digital sign-offs from all four station officers:

- [ ] **Systems Architect (`01_MultiHead_Attention_Core_starter.py`):** Pure PyTorch `MultiHeadSelfAttention` with causal masking, head splitting, and empirical variance preservation unit test ($\frac{1}{\sqrt{d_k}}$).
- [ ] **Communications & Protocol Officer (`02_RoPE_Positional_Engine_starter.py`):** Rotary Positional Embeddings (`apply_rope`), complex 2D planar rotation, and relative position invariance verification.
- [x] **Data & Payload Specialist (`03_GQA_and_KVCache_Benchmark.py`):** Grouped-Query Attention layer (`GQAAttention`), stateful autoregressive `KVCache`, and latency/memory benchmark table. *(Completed by Sreya Thota)*
- [ ] **Security & Ethics Sentinel (`04_Memory_Hierarchy_and_Safety_Audit_template.md`):** FlashAttention IO complexity derivation ($O(N)$ SRAM tiling), KV-Cache DoS / OOM analysis, and mitigation evaluation (PagedAttention, MLA).

---

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 3 (Mission 3) -> Problem Set ps03 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
