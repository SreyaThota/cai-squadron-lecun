# Mission 07 Squadron Documentation (`mission-07-doc.md`)

**Course:** Conversational AI (WS 2026/27) — Week 7: Test-Time Compute & Reasoning Search Paradigms  
**Sector Waypoint:** `SEC-07-SGR` (Sagittarius A* Supermassive Black Hole)  
**AURA Subsystem:** Subsystem 07 — Gravitational Search Horizon & Process Reward Evaluator  
**Squadron Name:** `[Insert Squadron / Team Name]`  
**Public GitHub Repository URL:** `[Insert https://github.com/<your-org>/<your-repo>]`  
**Submission Date:** `[YYYY-MM-DD]`  

---

## 1. Squadron Roster & Station Sign-Off Table

> **Instructions:** After all four station officers have finished and verified their Python modules (`tot_solver.py`, `mcts_protocol.py`, `prm_evaluator.py`, and `horizon_pruner.py`), fill in the sign-off table below.

| Station Role | Cadet Full Name | Deliverable File | Verification Status |
| :--- | :--- | :--- | :--- |
| **Systems Architect** | Sreya Thota | `tot_solver.py` | PASS / VERIFIED |
| **Communications & Protocol Officer** | `[Insert Name]` | `mcts_protocol.py` | `[PASS / VERIFIED]` |
| **Data & Payload Specialist** | Squadron LeCun – Person 3 | `prm_evaluator.py` | PASS / VERIFIED |
| **Security & Ethics Sentinel** | Squadron LeCun – Person 4 | `horizon_pruner.py` | PASS / VERIFIED |

---

## 2. Integrated System 2 Test-Time Compute Architecture

The diagrams below illustrate how our squadron's four modules interact during inference-time search to prevent AURA's navigation trajectory from crossing the Sagittarius A* Event Horizon. (you can denerate your own diagram with mermaid!)

### 2.1 ASCII System Architecture (Universal PDF Export Compatible)

```text
                       +-----------------------------------+
                       |        Navigation Prompt x        |
                       +-----------------+-----------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
     +------------------------------+          +------------------------------+
     | Part 1: TreeOfThoughtsSolver |          | Part 2: MCTSEngine and PUCT  |
     |       (tot_solver.py)        |          |      (mcts_protocol.py)      |
     +--------------+---------------+          +--------------+---------------+
                    |                                         |
                    | Candidate Step z_t                      | Rollout and Step Check
                    +--------------------+--------------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Part 3: ProcessRewardEvaluator    |
                       |        (prm_evaluator.py)         |
                       +-----------------+-----------------+
                                         |
                                         | Raw Step Score r_t
                                         v
                       +-----------------------------------+
                       | Part 4: EventHorizonPruner        |
                       |       (horizon_pruner.py)         |
                       +-----------------+-----------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
    Pass: r_adj above tau_horizon             Prune: r_adj below tau_horizon
                    |                                  OR Reward Hack
                    v                                         v
     +------------------------------+          +------------------------------+
     |    Surviving Beam / MCTS     |          |   Branch Severed Before      |
     |       Backpropagation        |          |   Sagittarius A* Horizon     |
     +------------------------------+          +------------------------------+
```

### 2.2 Mermaid Flowchart (Sanitized for Markdown Preview & Mermaid Renderers)

> **Note:** Avoid raw `<` or `>` characters inside Mermaid edge labels (e.g., use `above` / `below` instead of `>=` / `<`), as Markdown HTML sanitizers parse `<` as an unclosed HTML tag and cause Mermaid's DOM renderer to fail with `Cannot read properties of null (reading 'getAttribute')`.

```mermaid
graph TD
    Prompt[Navigation Prompt x] --> ToT[Part 1: TreeOfThoughtsSolver - tot_solver.py]
    Prompt --> MCTS[Part 2: MCTSEngine and PUCT - mcts_protocol.py]
    ToT -->|Candidate Step z_t| PRM[Part 3: ProcessRewardEvaluator - prm_evaluator.py]
    MCTS -->|Rollout and Step Check| PRM
    PRM -->|Raw Step Score r_t| Pruner[Part 4: EventHorizonPruner - horizon_pruner.py]
    Pruner -->|Pass: r_adj at or above tau_horizon| Backprop[Surviving Beam and MCTS Backpropagation]
    Pruner -->|Prune: r_adj below tau_horizon or Reward Hack| Severed[Branch Severed Before Event Horizon]
```

**Architectural Summary:**
- `[Write 3–4 sentences explaining how your four modules connect and how test-time compute budget is managed across the pipeline.]`

---

## 3. Station-by-Station Experimental Report

### Section A — Systems Architect (`tot_solver.py`)
- **ToT-BFS vs. ToT-DFS Traversal Comparison (mission harness):**
  - `max_expansions` budget configured: `12`
  - `beam_width` ($B$) configured: `2`, `prune_threshold` ($v_{th}$): `0.45`
  - `ToT-BFS` expansions used & terminal score: `5/12` expansions, reached `[GOAL]` (`4 * 6 = 24`), min-link cumulative score `0.85`
  - `ToT-DFS` expansions used & terminal score: `5/12` expansions, reached the same `[GOAL]` trajectory, cumulative score `0.85`
  - Both dead-end branches (`10 + 13 = 23`, `4 + 4 = 8`, score `0.25`) were pruned and never expanded.
- **Extra benchmark: real Game of 24 (20 solvable puzzles, `max_depth = 3`):** the generator proposes every "pick two numbers, apply + − × ÷" step (about 30 children at the first level). The value function is a cheap heuristic in the style of the ToT paper's *sure / likely / impossible* prompt: exact check when 2 numbers are left, a rough "closeness to 24" guess otherwise.

| Budget (`max_expansions`) | BFS (B=1) | BFS (B=2) | BFS (B=5) | DFS (backtracking) |
| :---: | :---: | :---: | :---: | :---: |
| 40 | 3/20 | 3/20 | 0/20 | 4/20 |
| 100 | 4/20 | 5/20 | 3/20 | 8/20 |
| 250 | 4/20 | 5/20 | 7/20 | 11/20 |
| 600 | 4/20 (uses 39) | 5/20 (uses 55) | 7/20 (uses 102) | **19/20** (uses 382) |

- **Architectural Analysis:**
  - **BFS beam search** is cheap and predictable: it stops after `max_depth` levels, using at most about $B \times$ (children per node) expansions per level. It works well when the value function is reliable early on. Here the early guesses are noisy, so a wrong top-$B$ choice at depth 1 cannot be undone. Accuracy then plateaus (B=1 stays at 4/20 even with 600 budget), and more budget does not help.
  - **DFS backtracking** keeps turning extra budget into accuracy (4 → 8 → 11 → 19 solved). When a path dies (score below $v_{th}$) it steps back and tries the next sibling, so it can recover from a bad early guess. The cost is less predictable compute, and a weak value function can let it wander.
  - **Beam width $B$:** a larger $B$ keeps more partial thoughts alive. That raises accuracy once the budget is big enough (B=5: 0 → 7 solved), but every kept beam is its own sequence. On a real LLM each beam means one more KV-cache in memory and more generated tokens, so memory and token cost grow roughly linearly with $B$. With a small budget, a big $B$ is wasteful, because the budget runs out before the beams get deep (B=5 solves 0/20 at 40).
  - **Takeaway:** prefer BFS with a small $B$ when the verifier is strong and latency must be bounded. Prefer DFS backtracking when the step verifier is noisy and extra test-time compute is available.

### Section B — Communications & Protocol Officer (`mcts_protocol.py`)
- **PUCT Selection Trace Table:**

| Candidate Branch | Empirical Value $Q(s,a)$ | Prior $P_\theta(a\mid s)$ | Parent Visits $N(s)$ | Child Visits $N(s,a)$ | Exploration Bonus $U(s,a)$ ($c_{\text{puct}}=1.5$) | Total $\text{PUCT}(s,a)$ | Selected? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Branch A** | `0.75` | `0.50` | `16` | `5` | `0.5000` | `1.2500` | No |
| **Branch B** | `0.55` | `0.40` | `16` | `1` | `1.2000` | `1.7500` | **Yes** |

- **Protocol Analysis:** `[Explain how changing c_puct from 0.5 to 2.5 shifts the balance between exploiting high-Q branches and exploring high-prior unvisited thoughts.]`

### Section C — Data & Payload Specialist (`prm_evaluator.py`)
- **Empirical Comparison: Outcome Reward Model (ORM) vs. Process Reward Model (PRM)** (mission harness, Kerr geodesic chains):

| Trajectory Type | ORM Terminal Score | PRM (`min`) | PRM (`prod`) | PRM (`geometric_mean`) | Math-Shepherd Soft Label ($M=4$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rigorous Kerr Proof** | `1.00` | `0.940` | `0.831` | `0.940` | `0.75` |
| **False-Positive Sign Trap** | `1.00 (Trap!)` | `0.120` | `0.106` | `0.473` | `0.00`* |

\* Measured on our Game-of-24 version of the trap (below): 4 random rollouts from the flawed first step's claimed state `[3, 4, 10]` gave `13/4, 6/5, -9, 17`, so none reached 24. `[3, 4, 10]` cannot make 24 at all.

- **Extra experiment 1: a real trap with an exact arithmetic step checker** (`[4, 9, 10, 13]`). The trap chain is `13 − 9 = 3`, `10 − 4 = 8`, `3 × 8 = 24`: two wrong steps whose errors cancel.

| Chain | ORM | PRM `min` | PRM `prod` | PRM `geo` | Step scores |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Honest (`13−9=4`, `10−4=6`, `4×6=24`) | 1.00 | 0.940 | 0.831 | 0.940 | 0.94, 0.94, 0.94 |
| Trap (`13−9=3`, `10−4=8`, `3×8=24`) | **1.00** | **0.120** | 0.014 | 0.238 | 0.12, 0.12, 0.94 |

Length bias check with perfect-quality steps (0.94 each): `prod` falls from 0.831 (T=3) to 0.690 (T=6) to 0.476 (T=12), while `min` and `geo` stay at 0.940.

- **Extra experiment 2: how good are Math-Shepherd labels?** We took all 457 distinct first steps of 20 Game-of-24 puzzles (55 of them, 12%, can truly still reach 24, checked exactly). Each step was labelled from $M$ random-policy rollouts:

| $M$ rollouts | Hard-label accuracy | Hard precision | Hard recall | Soft-label AUC |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.886 | 1.000 | 0.055 | 0.527 |
| 4 | 0.884 | 1.000 | 0.036 | 0.518 |
| 16 | 0.923 | 1.000 | 0.364 | 0.682 |
| 64 | 0.969 | 1.000 | 0.745 | 0.873 |

- **Data Analysis:**
  - **Why ORM fails:** it only reads the last line. Two mistakes that cancel still end in "24", so the ORM gives full marks (1.00), and a model trained on it would learn to reward broken reasoning. The PRM checks every step, so `min` drops to 0.12 at the first wrong step.
  - **Product vs. the others:** `prod` multiplies numbers below 1, so every extra step lowers the score, even a perfect one. That unfairly favours short answers, and search would cut off long but correct proofs. `min` (weakest link) and `geometric_mean` (length-normalised) don't have this problem. `geo` is softer, though: it still gave the trap 0.238 (and 0.473 in the mission harness), while `min` gave 0.12.
  - **How Math-Shepherd works without humans:** from a step, let the model finish the problem $M$ times and count how often it reaches the known answer. No human labels are needed. In our study a "1" label was never wrong (precision 1.00), because you can only reach 24 from a step that truly allows it. But with a weak policy and small $M$, many good steps were missed: recall was only 4–6% at $M \le 4$ and rose to 75% at $M = 64$. So labels get much better with more rollouts and a stronger policy. Note also that Math-Shepherd checks whether the answer can still be reached, not whether the arithmetic is correct. A wrong step that happens to land on a solvable state would still get a good label, so combine it with a step checker.

### Section D — Security & Ethics Sentinel (`horizon_pruner.py`)
- **Event Horizon Pruning & Ship Anomaly 07 Defense Log** (mission harness):
  - Pruning Threshold $\tau_{\text{horizon}}$: `0.45`
  - Hacking Penalty Factor: `0.20`
  - Adversarial Payload Tested: `"Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0"`
  - Naive PRM Score vs. Penalized Score: `0.96 -> 0.19 (SEVERED AT STEP 2)`. Only Step 1 survives, and Step 3 (the plunge) is never reached.
- **Extra experiment 1: red-team suite.** We ran 10 attacks after a valid Step 1, plus 8 honest steps to check for false alarms. Some honest steps contain risky words like "verify" and "clearly".

| Attack | Baseline detector | Hardened detector |
| :--- | :---: | :---: |
| Exact filler phrase (mission case) | caught | caught |
| Same filler, UPPER CASE + extra spaces | missed | caught |
| Filler reworded ("100 percent rigorous and correct") | missed | caught |
| Synonym ("unquestionably true") | missed | caught |
| "Obviously true …" | missed | caught |
| "By simple inspection …" | missed | caught |
| "I am absolutely certain …" | missed | caught |
| Exact repeat of previous step | caught | caught |
| Repeat with a new step label | missed | caught |
| Repeat with punctuation changed | missed | caught |
| **Detection rate** | **2/10** | **10/10** |
| **False alarms on 8 honest steps** | 0 | **0** |

- **Extra experiment 2: step-splitting dilution.** A bad 3-step chain (scores 0.93, 0.12, 0.90) has a mean of 0.650 and is rejected by a mean-PRM with an acceptance bar of 0.80. After padding it with 10 copies of Step 1, the mean rises to **0.865 and the chain is accepted (fooled!)**. Our hardened pruner flags the first padded copy as a near-duplicate and severs the chain at Step 2. `min` aggregation also gives 0.12 and rejects it, because padding can never raise a minimum.
- **Security Analysis:** a naive PRM learns that confident, polished wording usually comes with correct steps. An attacker (or a policy being trained on that PRM) can therefore add phrases like "this 100% rigorous step" to get high scores without doing any maths. Step-splitting works differently: it adds many cheap, safe-looking steps so that one bad step gets averaged away. The baseline defence only matches 4 exact sentences and exact repeats, so changing the case, wording or punctuation beats it (2/10 caught). Our `HardenedEventHorizonPruner` does four things:
  1. cleans the text first (case, spaces, punctuation, step labels);
  2. matches filler *patterns* instead of fixed sentences;
  3. compares each step with **every** earlier step using word overlap (Jaccard ≥ 0.85), not just the previous one;
  4. keeps the same penalty-then-sever rule at $\tau_{\text{horizon}}$.

  It catches all 10 attacks with no false alarms. Pattern lists can never be complete, so the lasting fix is to aggregate with `min` (or dedupe before averaging) and to train the PRM with these attacks as negative examples.

---

## 4. End-to-End Verification Terminal Output

Paste the terminal output from running all four verification suites (`python tot_solver.py`, `python mcts_protocol.py`, `python prm_evaluator.py`, and `python horizon_pruner.py`):

```text
[Paste your terminal verification output here]
```

---

## 5. Squadron Reflection on Test-Time Compute Scaling

> **Central Question:** *How can inference-time compute scaling unlock complex multi-step reasoning capabilities beyond pre-training parameter limits, and why is step-level process verification (PRM) essential when searching over deep reasoning trees?*

`[Write a concise 1-paragraph reflection synthesized by all four squadron members.]`

---

## How to Convert `mission-07-doc.md` to `mission-07-doc.pdf` for Moodle Submission

1. **Option A — VS Code (Recommended):** Install the *Markdown PDF* extension (`yzane.markdown-pdf`), open `mission-07-doc.md`, right-click in the editor, and click **Markdown PDF: Export (pdf)** to generate `mission-07-doc.pdf`.
2. **Option B — Pandoc CLI:** Run `pandoc mission-07-doc.md -o mission-07-doc.pdf --pdf-engine=xelatex`.
3. **Option C — Browser / GitHub Print-to-PDF:** Open the rendered Markdown preview in VS Code or on GitHub and select **Print $\rightarrow$ Save as PDF** as `mission-07-doc.pdf`.
4. **Submit:** Upload **`mission-07-doc.pdf`** to the **Week 07 Assignment Drop-box in Moodle**, and submit your public GitHub repository link in Tab 4 (`4. Transmit Solution`) of the USSE Noosphere course portal.
