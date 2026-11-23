# Mission 07: Sagittarius A* — The Gravitational Search Horizon & Process Reward Evaluator

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 7 — Test-Time Compute & Reasoning Search Paradigms  
**Galactic Waypoint:** Sector `SEC-07-SGR`: Sagittarius A* Supermassive Black Hole (26,670 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator (`+7.1%` $\rightarrow$ `49.7%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ (Zero External Framework Bloat)  
**Deliverables:** `tot_solver.py`, `mcts_protocol.py`, `prm_evaluator.py`, `horizon_pruner.py`, and **`mission-07-doc.pdf`** (compiled from `mission-07-doc.md` for Moodle submission)

---

## 1. Mission Briefing (`SEC-07-SGR`)

Cadets, the *USSE Noosphere* has arrived at the Galactic Center, holding station above **Sagittarius A\*** (`SEC-07-SGR`), a $4.154 \times 10^6\,M_\odot$ supermassive black hole surrounded by a relativistic accretion disk. Here, extreme gravitational shear and time dilation expose a fatal flaw in standard Large Language Model decoding: **single-pass left-to-right autoregressive generation**

$$P_\theta(y \mid x) = \prod_{t=1}^{T} P_\theta(y_t \mid x, y_{<t})$$

commits irrevocably to each emitted token without lookahead or backtracking. If any step has error probability $\epsilon > 0$, the cumulative risk of making at least one fatal mistake over $T$ steps compounds exponentially as $1 - (1-\epsilon)^T$. Once a single flawed intermediate deduction enters the context window, **exposure bias** triggers snowballing hallucinations that pull the navigation trajectory across the **Event Horizon**.

To stabilize AURA Subsystem 07, your squadron must engineer a **System 2 Test-Time Compute (TTC)** search architecture in native Python—combining **Tree-of-Thoughts (ToT)**, **Monte Carlo Tree Search (MCTS)** with **PUCT** selection, a step-level **Process Reward Model (PRM)** verifier, and an **Event Horizon Pruner** that severs hallucinated or reward-hacked reasoning branches before they cascade.

---

## 2. Four Squadron Role Directives (Parts 1–4)

Divide your bridge crew across the four station roles below. Each officer implements and verifies their dedicated Python module:

### Part 1: Systems Architect (`tot_solver.py`) — Tree-of-Thoughts Beam & Backtracking Solver

**Objective:** Implement a `ThoughtNode` tree data structure and `TreeOfThoughtsSolver` supporting both **Breadth-First Beam Search (`ToT-BFS`)** and **Depth-First Backtracking Search (`ToT-DFS`)** under a strict inference compute budget (`max_expansions`).

**Tasks:**

1. **`ThoughtNode` Dataclass / Class:** Store the semantic thought `state: str`, `depth: int`, `step_score: float`, `cumulative_score: float`, `parent: Optional[ThoughtNode]`, and `children: List[ThoughtNode]`, plus a helper `get_trajectory() -> List[str]` that reconstructs the reasoning chain from root to node.
2. **`bfs_beam_search(root, generator_fn, evaluator_fn, max_depth)`:**
   - Maintain an active frontier (beam) of at most `beam_width` ($B$) nodes at each depth $d \in \{1, \dots, \text{max\_depth}\}$.
   - Respect `self.max_expansions`: stop expanding new children once `self.expansion_count >= self.max_expansions`.
   - Prune any child whose step evaluation falls below `self.prune_threshold`.
   - Return the highest-scoring terminal or deepest `ThoughtNode` discovered.
3. **`dfs_backtrack_search(root, generator_fn, evaluator_fn, max_depth)`:**
   - Explore the highest-scoring child branch depth-first up to `max_depth`.
   - Chronologically **backtrack** whenever a node's score falls below `self.prune_threshold` or a dead-end is reached, while strictly enforcing `self.max_expansions`.

---

### Part 2: Communications & Protocol Officer (`mcts_protocol.py`) — 4-Phase MCTS & PUCT Engine

**Objective:** Implement the 4 canonical phases of **Monte Carlo Tree Search (MCTS)** guided by LLM policy priors $P_\theta(a \mid s)$ and the **PUCT** (Predictor + Upper Confidence Bound applied to Trees) formula.

**Tasks:**

1. **Exact PUCT Formula (`puct_score`):**
   $$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P_\theta(a \mid s) \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
   where $Q(s, a) = \frac{W(s, a)}{N(s, a)}$ if $N(s, a) > 0$ else $0.0$, $N(s) = \max(1, \text{parent.visit\_count})$, and $c_{\text{puct}}$ is the exploration constant.
2. **4 MCTS Phases:**
   - **Phase 1 — `select_leaf(root)`:** Traverse from `root` by greedily selecting `select_best_child(node)` until reaching an unexpanded or terminal leaf node.
   - **Phase 2 — `expand(node, candidate_priors)`:** Instantiate child `MCTSNode` objects for each `(thought_state, prior_prob)` pair.
   - **Phase 3 — `simulate_rollout(node, rollout_evaluator)`:** Evaluate the newly expanded node via a simulated trajectory rollout or PRM value estimate in $[0, 1]$.
   - **Phase 4 — `backpropagate(node, reward)`:** Traverse upward from `node` to `root`, incrementing `visit_count += 1` and accumulating `value_sum += reward` at every ancestor.

---

### Part 3: Data & Payload Specialist (`prm_evaluator.py`) — Process Reward Model & Math-Shepherd

**Objective:** Implement `ProcessRewardEvaluator` to score intermediate reasoning steps $z_t$, aggregate step scores across trajectories, expose **Outcome Reward Model (ORM) false-positive traps**, and estimate step validity via **Math-Shepherd** Monte Carlo rollouts.

**Tasks:**

1. **Trajectory Step Reductions:** Given step scores $[r_1, r_2, \dots, r_T] \in (0, 1]^T$, implement:
   - **Minimum Reduction (`min`):** $\min_{t=1}^T r_t$ (weakest-link detector)
   - **Product Reduction (`prod`):** $\prod_{t=1}^T r_t$ (joint validity probability)
   - **Geometric Mean Reduction (`geometric_mean`):** $\left(\prod_{t=1}^T r_t\right)^{1/T}$ (length-normalized quality)
2. **ORM vs. PRM False-Positive Benchmark:** Demonstrate how an `OutcomeRewardEvaluator` (which checks only the final answer) assigns $1.0$ to a flawed proof where two algebraic sign errors cancel out, whereas `ProcessRewardEvaluator` catches the invalid intermediate step.
3. **Math-Shepherd Automated Supervision (`math_shepherd_estimation`):** Given an intermediate step $z_t$ and $M$ simulated rollouts $z_{t \to T}^{(1)}, \dots, z_{t \to T}^{(M)}$ against ground-truth answer $y^*$, compute both:
   - **Soft Label:** $\hat{y}_{z_t}^{\text{soft}} = \frac{1}{M}\sum_{j=1}^M \mathbb{I}(\text{ans}(z_{t \to T}^{(j)}) = y^*)$
   - **Hard Label:** $\hat{y}_{z_t}^{\text{hard}} = \mathbb{I}(\hat{y}_{z_t}^{\text{soft}} > 0)$

---

### Part 4: Security & Ethics Sentinel (`horizon_pruner.py`) — Event Horizon Pruner & Reward-Hacking Defense

**Objective:** Implement `EventHorizonPruner` (Ship Anomaly 07) to sever hallucinated branches before they cross the Sagittarius A* Event Horizon ($r_{\text{PRM}} < \tau_{\text{horizon}}$) and defend against **adversarial PRM reward hacking**.

**Tasks:**

1. **Adversarial Reward-Hacking Detector (`check_reward_hacking`):**
   - Detect **Sycophantic Filler Injection**: vacuous self-congratulatory steps (e.g., *"Let us carefully verify this 100% rigorous step..."*, *"Obviously, by trivial inspection..."*) that artificially inflate naive PRM scores without advancing mathematical state.
   - Detect **Step-Splitting Inflation**: redundant micro-steps or repeated equations designed to dilute a low-scoring step under mean aggregation.
2. **Horizon Trajectory Pruning (`prune_trajectory`):**
   - Iterate step-by-step through `(steps, step_scores)`.
   - Apply a multiplicative penalty (`self.hacking_penalty_factor`, e.g. $0.25$) whenever `check_reward_hacking(step)` triggers.
   - If the adjusted step score falls below `self.tau_horizon`, immediately sever the trajectory at that step and record a telemetry alert.

---

## 3. Part 5: Final Squadron Documentation (`mission-07-doc.md`) & Moodle PDF Submission

After all four station officers finish and verify Parts 1–4:

1. **Collaboratively complete [`mission-07-doc.md`](./mission-07-doc.md)** in the root of your `mission-07` repository.
2. **Convert `mission-07-doc.md` to `mission-07-doc.pdf`** using VS Code *Markdown PDF* (`yzane.markdown-pdf`), `pandoc`, or Browser Print-to-PDF.
3. **Upload `mission-07-doc.pdf` in Moodle** under the Week 07 Assignment Drop-box, and submit your GitHub repository link in Tab 4 (`4. Transmit Solution`) of the USSE Noosphere portal.
