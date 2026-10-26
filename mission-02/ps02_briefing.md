# Mission 2 Specification: FST Systems, Lovelace & Noisy Channels

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Official Settings Repo:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

### Mission Briefing

**TO:** All Bridge Crews, USSE Noosphere  
**FROM:** Expedition Command (AURA Communications Array)  
**SUBJECT:** ANOMALY DETECTED: Deep Space Signal Noise & Symbolic Control Limits  

**Situation:**  
As the *USSE Noosphere* passes through the *Signal Void*, severe electromagnetic noise distorts AURA’s telemetry. Furthermore, command is concerned that AURA may merely be simulating intelligence by manipulating symbol patterns without genuine understanding. To verify AURA's cognitive validity under constraint, we must apply the foundational principles of **Deterministic Transducers** (FST state machines), **Lovelace** (Originality vs. Order), **Turing** (Imitation & Interrogation), and **Shannon** (Information Entropy & Noisy Channel Capacity).

**Objective:**  
Implement a `FSTDialogueManager` class in Python, design a 5-question "Anti-Turing Interrogation Protocol" to test modern LLMs (Gemini, ChatGPT, Claude) for genuine machine originality, and build a "Noisy Channel Simulator" in VS Code that transmits human-like signals under strict bandwidth limits and symbol noise.

---

### Part 1: Historical Archivist — *The Lovelace Inquiry Brief*
* **Deliverable:** `01_Lovelace_Inquiry_Brief.md`
* **Directives:**
  1. Design a 5-question Anti-Turing interrogation protocol probing machine originality.
  2. Execute probe against a modern LLM (Gemini, ChatGPT, Claude) and evaluate if it overcome Lovelace's objection.

### Part 2: Protocol Specialist — *FST Class & Noisy Channel Protocol*
* **Deliverable:** `02_FST_and_Noisy_Protocol.md`
* **Directives:**
  1. Implement `FSTDialogueManager` in Python with 5 states (`INIT`, `GREETING`, `COLLECT_SLOTS`, `CONFIRMATION`, `TERMINAL`/`FALLBACK`) and `slot_memory = {}`.
  2. Define Noisy Channel constraints (7-word limit + noise vocabulary substitution).

### Part 3: Ethical Navigator — *Entropy & Hazard Audit*
* **Deliverable:** `03_Entropy_Hazard_Audit.md`
* **Directives:**
  1. Stress-test the FST dialogue manager with 5 non-canonical user utterances to document state breakage.
  2. Analyze Shannon information entropy and semantic loss under noise.

### Part 4: Core Architect — *FST + Noisy Turing Simulator*
* **Deliverable:** `04_fst_noisy_simulator_starter.py` / `ps02_solution.ipynb`
* **Directives:**
  1. Build Python simulator in VS Code + Antigravity connecting FST class and Noisy Channel prompt constraints.
  2. Format output logs, state traces, and channel entropy metrics.
