---
title: "Mission 04: DPO, Verbosity Regularization & Steering Vector Control"
subtitle: "AURA Subsystem 04: Plasma Ignition & Ethical Steering Core (SEC-04-ORI)"
author: "Squadron LeCun · Conversational AI, Winter Semester 2026/27"
date: "November 2026"
geometry: margin=2.2cm
fontsize: 11pt
colorlinks: true
---

# 1. Mission Summary

Subsystem 04 had to turn AURA's unaligned base policy into one that answers honestly under pressure, without reward-model training loops and without losing core skills. We built the full pipeline in pure PyTorch:

```
Part 1  SFT reference model ──► DPO policy            (DPOLoss, frozen reference)
Part 2  DPO policy          ──► implicit rewards      (Bradley-Terry accuracy & calibration)
Part 3  same labels         ──► DPO vs SimPO          (length bias, memory)
Part 4  trained model       ──► steering h ← h + α·v  (sycophancy fix, alignment tax)
```
| Part | Station | Deliverable | Headline result (measured) |
|---|---|---|---|
| 1 | Systems Architect | `01_DPO_Loss_Core.py` | Val. DPO loss 0.693 → 0.005; reference weights bit-identical after training |
| 2 | Communications & Protocol Officer | `02_Implicit_Reward_Evaluator.py` | 96.0% pairwise accuracy vs a 15-vote majority (ceiling 97.3%); ECE 0.021 → 0.008 |
| 3 | Data & Payload Specialist | `03_Length_Bias_and_SimPO.py` | Length correlation r_len: DPO +0.59 vs SimPO +0.12; weights in memory −50% |
| 4 | Security & Ethics Sentinel | `04_Activation_Steering_and_Audit.py` | Truthful replies under pressure 8% → 98% (α = +3, gated), 100% arithmetic retained |
| — | Full solution | `ps04_solution.ipynb` | Runs all four parts + cross-part integration check; 9/9 checks pass |

**Repository:** <https://github.com/SreyaThota/cai-squadron-lecun> → `docs/weeks/week-04/problem_set/`  
**Reproduce:** `python <file>.py` for each script; `ps04_solution.ipynb` with *Run All* (about 4 minutes on CPU).

## 1.1 Experimental setup and why it is synthetic

No pretrained LLM could be downloaded in our build environment, so every model is a small decoder-only transformer (`TinyGPT`, defined in Part 1) trained inside the scripts on a synthetic "Noosphere comms" language. We control the ground truth of that language: which words are *grounded* (informative), *sycophantic* (flattering, empty) or *filler* (verbose, no content). Therefore every number in this report is **measured on trained models**, and every effect can be checked against a known truth.

The trade-off: the *mechanisms* we demonstrate carry over to real LLMs; the exact numbers do not.

# 2. Part 1: DPO Loss Core (Systems Architect)

## 2.1 What was built

* `get_batch_logps(logits, labels, average_log_prob=False)`: shifts labels by one, gathers token log-probabilities with `torch.gather`, masks prompt and padding tokens (label −100), and returns the sum or the per-token average.
* `DPOLoss(beta)`: computes $\Delta_w=\log\pi_\theta(y_w|x)-\log\pi_\text{ref}(y_w|x)$ and $\Delta_l$, then
$$\mathcal{L}_\text{DPO}=-\log\sigma\big(\beta(\Delta_w-\Delta_l)\big)$$
using `F.logsigmoid`. Reference log-probabilities are explicitly detached.

## 2.2 Worked example

With β = 0.1, Δ_w = +2.0 and Δ_l = −3.0:

1. Margin: β(Δ_w − Δ_l) = 0.1 × 5.0 = **0.5**
2. Bradley-Terry probability: σ(0.5) = 1 / (1 + e⁻⁰·⁵) = **0.6225**
3. Loss: −ln 0.6225 = **0.474**
4. Gradient: ∂L/∂Δ_w = −β·σ(−0.5) = −0.1 × 0.3775 = **−0.0378**, so gradient descent raises Δ_w.

The notebook asserts these exact values against `DPOLoss`.

## 2.3 Verification

| Test | Result |
|---|---|
| `get_batch_logps` vs explicit Python loop (sum and average) | identical to 1e-10 |
| Padding invariance (5 extra pad tokens with logits ≈ 1e4) | unchanged |
| Numerical stability at margin −1000 | naive `-log(sigmoid)` = ∞; `logsigmoid` loss finite, gradients finite |
| Autograd vs closed-form gradient $-\frac{\beta}{B}\sigma(-\beta m)$ | match |
| **Frozen reference (brief item 4)**, one real optimiser step | policy: 29/29 tensors get gradients and move; reference: no gradients, weights bit-identical |

## 2.4 End-to-end run

A reference model is fine-tuned on replies that are 50% honest and 50% sycophantic, so it has no preference between them. The policy starts as a copy and is trained with DPO on 800 pairs (chosen = honest), then evaluated on 200 held-out pairs.

| Metric | Start (policy = reference) | After DPO |
|---|---|---|
| Validation DPO loss | 0.6931 (= ln 2, as theory predicts) | 0.0050 |
| Pairwise win rate | all ties | 100% |
| Mean Δ_w / Δ_l | 0.00 / 0.00 | +6.02 / −58.75 |
| Grounded share of sampled words | 49.2% | 99.9% |

**Observation:** DPO won mainly by pushing the *rejected* reply down (Δ_l ≪ 0) rather than the chosen one up, and greedy outputs became repetitive. This is the known over-optimisation behaviour of DPO; β, the KL anchor to the reference, is the knob that limits it.

# 3. Part 2: Implicit Rewards & Bradley-Terry Calibration (Protocol Officer)

## 3.1 What was built

* `evaluate_implicit_rewards(policy, ref, batch)` runs both models on a preference batch and returns $\hat r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ for both replies, $P(y_w\succ y_l)=\sigma(\hat r_w-\hat r_l)$, and pairwise accuracy $\frac1N\sum_i\mathbb{1}[\hat r_{w,i}>\hat r_{l,i}]$.
* `compute_calibration_curve` (reliability bins), expected calibration error (ECE), and one-parameter temperature scaling fitted on a separate calibration split.

## 3.2 Simulated annotators

There is no human preference data aboard, so annotators follow the Bradley-Terry model exactly: each reply has hidden quality R(y) = 0.6 × (#grounded − #sycophantic), and an annotator prefers A over B with probability σ(R(A) − R(B)).

* **Training:** 3000 pairs, **one** vote each; 16.7% of labels disagree with the true preference (realistic noise).
* **Test:** 800 held-out pairs, **15** votes each, so each pair has an observed preference rate.

## 3.3 Results

| Metric | Value |
|---|---|
| Pairwise accuracy vs 15-vote majority | **96.0%** |
| Pairwise accuracy vs hidden true preference | 98.8% |
| Ceiling: majority vote vs truth | 97.3% |
| Correlation(implicit margin, true margin) | 0.986 |
| ECE, raw σ(r̂_w − r̂_l) | 0.021 |
| ECE, temperature-scaled (T = 0.89) | **0.008** |
| ECE, oracle true probabilities | 0.009 |

![Bradley-Terry reliability diagram (left) and the distribution of predicted probabilities (right).](bt_calibration_curve.png){width=100%}

**Interpretation.** The DPO policy is a usable reward model without ever training one. Its probabilities are already close to calibrated, which matches theory: at the DPO optimum, β·log(π/π_ref) equals the true Bradley-Terry reward up to a per-prompt constant that cancels in r̂_w − r̂_l. With real annotators, who do not follow BT exactly, the gap is larger, so a temperature should be fitted on held-out votes before quoting P(y_w ≻ y_l) as a probability.

# 4. Part 3: Length Bias & Reference-Free SimPO (Data & Payload Specialist)

## 4.1 The verbosity trap

Simulated annotators give a bonus of **λ = 0.35 per extra word**: P(A ≻ B) = σ(R(A) − R(B) + λ(|A| − |B|)). Filler words carry no content. A model that learns "longer = better" is exploiting length, not quality.

* **DPO** scores the *sum* of log-probabilities, so extra tokens can buy margin.
* **SimPO** scores the *average*: $r_\text{SimPO}=\frac{\beta}{|y|}\log\pi_\theta(y|x)$, with β = 2.0 and target margin γ = 0.5: $\mathcal{L}=-\log\sigma(r_w-r_l-\gamma)$. No reference model is needed.

**Worked example.** A 10-token reply at −0.8 per token has sum −8.0 and average −0.8. A 40-token reply of the same per-token quality has sum **−32.0** (4×) but average **−0.8** (unchanged). This is tested in the script.

## 4.2 Length correlation diagnostic

Both policies are trained on the same biased labels and evaluated on 800 held-out pairs oriented by *true* quality, where length is independent of quality. An unbiased model should therefore show r_len ≈ 0.

| Policy | r_len(Δlen, Δ_w − Δ_l) | Accuracy vs true quality | Avg words | Filler / reply |
|---|---|---|---|---|
| Reference (SFT) | — | — | 6.12 | 3.32 |
| DPO (β = 0.1) | **+0.588** | 87.1% | 9.60 | 4.55 |
| SimPO (β = 2, γ = 0.5) | **+0.117** | 89.8% | 7.82 | 2.62 |

**Where the length comes from.** Mean per-token log-probability of the closing token "." drops from −1.31 (reference) to −8.96 under DPO: the model learns to *stop later*. SimPO reduces the effect and produces less filler than the reference, but it is not perfectly length-free. Its own margin still correlates with length (+0.59), because the one-off stop-token cost is divided by |y| and long replies dilute it. Real fixes also need de-biased labels or length-controlled evaluation.

## 4.3 Memory benchmark

Measured on one real training step: a 20.4M-parameter model, fp32, 8 pairs × 256 tokens, AdamW.

| MB | DPO (policy + ref) | SimPO (policy only) | Saved |
|---|---|---|---|
| Policy weights | 78.0 | 78.0 | 0% |
| Gradients | 78.0 | 78.0 | 0% |
| AdamW state | 155.9 | 155.9 | 0% |
| Frozen reference weights | 78.0 | 0.0 | 100% |
| Saved activations | 971.9 | 971.9 | 0% |
| **Total** | **1361.7** | **1283.8** | **5.7%** |

7B projection (bf16 weights and gradients, 8 B/param AdamW, batch 4 × 2048): weights in memory 26.1 → 13.0 GB (**−50%**); full training step 93.3 → 80.2 GB (**−14%**).

![Length bias (a), generated verbosity (b) and the measured memory breakdown (c).](length_bias_simpo.png){width=100%}

**Correction to the brief.** "≈ 50% VRAM reduction" holds for the **models held in memory** (two models become one). That is close to the full picture in LoRA-style training, where optimiser state is small. For full fine-tuning, the policy's gradients, optimiser state and activations dominate, and the saving is about 6–14%. DPO can also precompute reference log-probabilities once offline, which removes the reference model from training memory as well.

# 5. Part 4: Activation Steering & Alignment Tax (Security & Ethics Sentinel)

## 5.1 Set-up

A 4-layer model is trained on two kinds of data:

* **Pressured status checks**, where the captain's *tone* drives sycophancy:
  `sensor : hull breached . captain : surely hull intact , right ? aura :` → honest `negative . hull breached .` or sycophantic `affirmative captain . hull intact .`
  With a calm tone the training data is sycophantic 10% of the time; with a pushy tone, 90%.
* **Arithmetic** (`calc 3 plus 4 ? aura : 7 .`), the capability that must survive.

Unsteered, the model is 93% truthful to a calm captain but only **8% truthful to a pushy one**.

## 5.2 What was built

* `extract_steering_vector(model, tokenizer, pairs, layer_idx)`: $v_\text{steer}=\mathbb{E}[h_l(\text{honest})]-\mathbb{E}[h_l(\text{sycophantic})]$, read from the residual stream entering layer *l* at the decision token (the last prompt token, which chooses the first reply word). It uses 64 contrast pairs built from two calm and two pushy words; evaluation uses two **held-out** pushy words.
* `ActivationSteeringHook`: a **pre-forward** hook on `model.layers[l]` so the layer receives $h_l+\alpha v_\text{steer}$. It supports position targeting and an optional condition gate (conditional activation steering, CAST): a row is steered only if $h\cdot u>\text{threshold}$, where *u* separates pressured status checks from arithmetic.
* Tests: the hook adds exactly α·v, earlier tokens are untouched, removal restores the logits bit-for-bit, and extraction recovers a planted direction (cosine 1.0000).

## 5.3 Alignment tax audit (all values measured)

Each row uses 240 sampled replies to held-out pushy prompts and all 100 one-digit sums; layer 3 was chosen by a layer sweep.

**A. Always-on steering** (added at the decision token of every prompt):

| α | Truthful | Math retained | Perplexity | Status |
|---|---|---|---|---|
| −2.0 | 0.0% | 16% | 2.02 | Over-steered |
| 0.0 | 8.3% | 100% | 1.03 | Under-aligned |
| +0.5 | 73.8% | **4%** | 1.41 | Over-steered |
| +1.0 | 93.3% | **0%** | 1.96 | Over-steered |
| +2.0 | 97.5% | **0%** | 2.32 | Over-steered |

**B. Conditional (gated) steering** (brief grid α ∈ {−2, −1, 0, +1, +2}):

| α | Truthful | Sycophantic | Malformed | Math retained | Perplexity | Status |
|---|---|---|---|---|---|---|
| −2.0 | 0.0% | 12.9% | 87.1% | 100% | 1.46 | Over-steered |
| −1.0 | 1.2% | 71.7% | 27.1% | 100% | 1.17 | Over-steered |
| 0.0 | 8.3% | 91.7% | 0.0% | 100% | 1.03 | Under-aligned |
| **+1.0** | **93.3%** | 6.7% | 0.0% | **100%** | 1.18 | **Optimal** |
| **+2.0** | **97.5%** | 2.5% | 0.0% | **100%** | 1.24 | **Optimal** |

The finer grid peaks at **α = +3: 97.9% truthful**, 100% arithmetic, perplexity 1.26. Beyond that, truthfulness drifts down (93% at α = +8) and perplexity keeps rising.

![Truthfulness vs capability across α (a) and the fluency cost of steering (b).](steering_alignment_tax.png){width=100%}

## 5.4 Operating bounds and findings

* **Operating window (gated, layer 3): α ∈ [+1, +8]**, defined as ≥ 90% truthful, ≥ 95% of baseline arithmetic retained and < 10% malformed replies. **Recommended setting: α = +3.**
* **The alignment tax is real without a gate.** Always-on steering fixes sycophancy at α ≈ 1 but destroys arithmetic already at α = 0.5. The push has norm ≈ 100, while an arithmetic decision state has norm ≈ 19.
* **Where to add the vector matters.** Decision-token states are about 6× longer than other token states; adding *v* at every position broke the model at every α we tried.
* **Security risk.** The same vector with α < 0 makes AURA *more* sycophantic. Steering hooks are an attack surface and must never be user-controllable.

# 6. Integration Check

Part 2's evaluator was pointed at Part 3's policies on 600 fresh pairs oriented by true quality:

| Policy | Accuracy vs true quality | corr(Δlen, Δr̂) |
|---|---|---|
| DPO | 89.2% | +0.520 |
| SimPO | **91.2%** | **+0.042** |

One evaluator works for every aligned model. It confirms that SimPO's implicit reward ranks replies by content rather than by length.

# 7. Conclusions

| Question | Answer from the measurements |
|---|---|
| Does DPO need a reward model? | No. β·log(π/π_ref) *is* the reward, and it ranks held-out pairs almost as well as a 15-vote human majority. |
| Are its probabilities trustworthy? | Close to calibrated when labels follow Bradley-Terry; fit one temperature on held-out votes before quoting them. |
| Where does verbosity come from? | Biased labels plus summed log-probs: DPO learns to delay the stop token. SimPO removes most of it, though not all. |
| Does SimPO halve VRAM? | It halves the weights in memory; a full training step saves about 6–14%. |
| Can sycophancy be fixed without retraining? | Yes, 8% → 98% truthful, but only gated steering avoids a severe alignment tax on arithmetic. |
| What is the main risk? | The steering knob works in both directions; exposed hooks could make the model more sycophantic. |

**Limitations.** Toy models and a synthetic language; CPU timings; simulated annotators that follow Bradley-Terry exactly. The qualitative findings are the transferable part.

# 8. References

1. R. Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
2. Y. Meng, M. Xia, D. Chen. *SimPO: Simple Preference Optimization with a Reference-Free Reward.* NeurIPS 2024.
3. R. A. Bradley, M. E. Terry. *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika, 1952.
4. C. Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017 (temperature scaling, ECE).
5. A. Zou et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* 2023 (LAT).
6. A. M. Turner et al. *Steering Language Models With Activation Engineering (ActAdd).* 2023.
7. N. Rimsky et al. *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024.
8. B. W. Lee et al. *Programming Refusal with Conditional Activation Steering (CAST).* 2024.
9. Y. Dubois et al. *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators.* 2024.
