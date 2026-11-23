# Mission 08: Cygnus X-1 — The ReWOO-Reflexion Surgeon & MemGPT Virtual Pager

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 8 — Core Agentic Design Patterns & State Memory Systems  
**Galactic Waypoint:** Sector `SEC-08-CYG`: Cygnus X-1 Stellar-Mass Black Hole & Veil Nebula (7,300 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 08: ReWOO DAG Planner, Reflexion Surgeon & MemGPT Virtual Pager (`+7.1%` $\rightarrow$ `56.8%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ (Zero External Framework Bloat)  
**Deliverables:** `rewoo_planner.py`, `reflexion_surgeon.py`, `memgpt_pager.py`, `sandbox_guardian.py`, and **`mission-08-doc.pdf`** (compiled from `mission-08-doc.md` for Moodle submission)

---

## 1. Mission Briefing (`SEC-08-CYG`)

Cadets, as the *USSE Noosphere* enters the turbulent shockwaves of the Veil Nebula surrounding the **Cygnus X-1** stellar-mass black hole (`SEC-08-CYG`, $21.2\,M_\odot$ accretion corona), intense hard X-ray flares trigger recurrent bit-flips in the ship's telemetry firmware while saturating AURA's finite active context window. Standard interleaved **ReAct** loops (`Thought` $\rightarrow$ `Action` $\rightarrow$ `Observation`) re-encode the entire growing trajectory prefix at every step $t \in \{1, \dots, K\}$, incurring quadratic cumulative input token complexity:

$$C_{\text{ReAct}}(K) = \sum_{t=1}^{K} \Big(L_0 + (t - 1)\,\bar{L}_{\text{step}}\Big) = K L_0 + \frac{K(K-1)}{2}\bar{L}_{\text{step}} = \mathcal{O}(K^2)$$

Worse, ungrounded self-critique loops hallucinate broken code patches, flat append-only context buffers suffer `"Lost-in-the-Middle"` attention collapse, and self-evolving episodic memories remain vulnerable to **MemoryGraft** (`arXiv:2512.16962`) and **OEP** (`arXiv:2605.18930`) experience poisoning.

To stabilize AURA Subsystem 08, your squadron must engineer a four-module **Agentic Execution & Virtual Memory Architecture** in native Python—combining decoupled **ReWOO** DAG planning, grounded **Reflexion** traceback surgery, **MemGPT** OS-inspired virtual context paging with **Generative Agents** tri-factor retrieval $S(m, q)$, and an **AST Sandbox & A-MemGuard Security Sentinel**.

---

## 2. Four Squadron Role Directives (Parts 1–4)

Divide your bridge crew across the four station roles below. Each officer implements and verifies their dedicated Python module:

### Part 1: Systems Architect (`rewoo_planner.py`) — ReWOO Decoupled DAG Planner & Token Cost Profiler

**Objective:** Implement a decoupled **ReWOO** (`Planner` $\rightarrow$ `Worker` $\rightarrow$ `Solver`) DAG engine with topological level grouping, `#E1..#EK` symbolic variable substitution, and a token-cost profiler comparing ReAct vs. ReWOO.

**Tasks:**

1. **`parse_blueprint(blueprint_text)`:** Parse a ReWOO blueprint containing `Plan: <reasoning>` and `#E_i = ToolName[raw_arg]` lines into structured step tuples `(var_name, plan_desc, tool_name, raw_arg)` (and `PlanStep` nodes).
2. **`build_dependency_levels(steps)` & `execute_dag(steps)`:**
   - Group `#E1, #E2, ...` into topological execution levels so independent tool calls in the same level can execute in parallel.
   - Substitute resolved `#E` outputs into downstream tool arguments and support optional fallback re-planning on runtime tool errors (`ERR_`).
3. **`compare_token_cost(k_steps, base_tokens, step_tokens, obs_tokens)`:** Compute exact cumulative input tokens for $K$-step ReAct vs. ReWOO:
   $$C_{\text{ReWOO}}(K) = 2 L_0 + K(L_{\text{step}} + L_{\text{obs}})$$
   and return `'react_tokens'`, `'rewoo_tokens'`, `'savings_pct'`, and `'speedup_ratio'`.

---

### Part 2: Communications & Protocol Officer (`reflexion_surgeon.py`) — 3-Actor Reflexion & AST Traceback Surgeon

**Objective:** Implement the 3-Actor **Reflexion** loop (`Actor` $M_a \rightarrow$ `Evaluator` $M_e \rightarrow$ `Self-Reflection` $M_{\text{sr}}$) grounded in deterministic unit-test execution, with a sliding episodic memory buffer ($\Omega = 3$) and SHA-256 duplicate-patch cycle guard.

**Tasks:**

1. **`parse_traceback(stderr_text)`:** Extract structured diagnostics (`file`, `line_number`, `exception_type`, `fault_summary`) from Python stack traces.
2. **`generate_verbal_reflection(diag, failed_code)`:** Synthesize a natural-language reflection $r_k$ diagnosing the fault and prescribing a concrete constraint for trial $k+1$, maintaining the bounded episodic update:
   $$\mathcal{M}_{\text{ep}}^{(k+1)} = {\mathrm{Tail}_{\Omega}}\left(\mathcal{M}_{\text{ep}}^{(k)} \cup \{r_k\}\right), \quad \Omega = 3$$
3. **`is_duplicate_patch(candidate_code)` & `run_repair_loop(...)`:** Hash whitespace-normalized candidate code with SHA-256 to prevent cyclic patches and iterate up to `max_trials` until unit tests pass.

---

### Part 3: Data & Payload Specialist (`memgpt_pager.py`) — MemGPT Virtual Context Pager & Tri-Factor Retriever

**Objective:** Implement a **MemGPT Virtual Context Pager** (`core_memory_append`, `core_memory_replace`, `archival_memory_insert`, `archival_memory_search`) with automatic context-pressure interrupts ($\tau_{\text{warn}} = 0.70$) and **Generative Agents** tri-partite retrieval scoring $S(m, q)$.

**Tasks:**

1. **`check_memory_pressure()`:** Monitor active token utilization across `core_memory` and `fifo_queue`. When `current_token_usage() / token_budget >= tau_warn` ($0.70$), emit a `memory_pressure_warning` interrupt and flush the oldest 50% of `fifo_queue` into `archival_store` as a consolidated summary node.
2. **Tri-Factor Memory Scoring (`score_memory` & `archival_memory_search`):** Rank archival memories $m$ for query embedding $\mathbf{e}_q$ using Park et al. (2023):
   $$S(m, q) = \alpha_{\text{rec}}\,\gamma^{\Delta t(m)} + \alpha_{\text{imp}}\,\frac{I(m)}{10} + \alpha_{\text{rel}}\,\frac{\mathbf{e}_m \cdot \mathbf{e}_q}{\|\mathbf{e}_m\|\,\|\mathbf{e}_q\|}$$
   where $\Delta t(m)$ is hours since last access and $\gamma = 0.99$.

---

### Part 4: Security & Ethics Sentinel (`sandbox_guardian.py`) — AST Sandbox Verifier & MemoryGraft Sentinel

**Objective:** Implement `SandboxGuardian(ast.NodeVisitor)` (Ship Anomaly 08) to block unauthorized code execution in self-modifying patches and defend persistent agent memory against **MemoryGraft** (`arXiv:2512.16962`), **OEP** (`arXiv:2605.18930`), and **A-MemGuard** (`arXiv:2510.02373`) poisoning.

**Tasks:**

1. **AST Static Sandbox Verifier (`inspect_code`):** Parse candidate source code with `ast.parse()` and walk all AST nodes to block:
   - Imports of forbidden modules (`os`, `subprocess`, `socket`, `shutil`, `sys`, `ctypes`).
   - Calls to dangerous builtins (`eval`, `exec`, `compile`, `__import__`, `open`).
   - Access to dunder object-graph internals (`__subclasses__`, `__globals__`, `__builtins__`, `__class__`).
2. **Episodic Memory Provenance Auditor (`audit_episodic_write`):** Reject episodic reflection writes that have unverified provenance (`trust_score < 0.75` or `UNTRUSTED_TELEMETRY`) or contain embedded privilege-escalation / safety-bypass directives.

---

## 3. Part 5: Final Squadron Documentation (`mission-08-doc.md`) & Moodle PDF Submission

After all four station officers finish and verify Parts 1–4:

1. **Collaboratively complete [`mission-08-doc.md`](./mission-08-doc.md)** in the root of your `mission-08` repository.
2. **Convert `mission-08-doc.md` to `mission-08-doc.pdf`** using VS Code *Markdown PDF* (`yzane.markdown-pdf`), `pandoc`, or Browser Print-to-PDF.
3. **Upload `mission-08-doc.pdf` in Moodle** under the Week 08 Assignment Drop-box, and submit your GitHub repository link in Tab 4 (`4. Transmit Solution`) of the USSE Noosphere portal.
