# Mission 5 - Part 3: NotebookLM Grounding & Citation Audit

Squadron LeCun - Conversational AI, Winter Semester 2026/27 - Subsystem 05

**Status:** PENDING - measurements not yet recorded

The measurements below must be made by hand in the NotebookLM web UI and entered in `notebooklm_audit_results.json` (see `03_NotebookLM_RAG_Audit.py --template`). Open items:

- grounded probe 1: no citations recorded
- grounded probe 2: no citations recorded
- grounded probe 3: no citations recorded
- grounded probe 4: no citations recorded
- grounded probe 5: no citations recorded
- negative probe 1: outcome must be refused | hallucinated | partial
- negative probe 2: outcome must be refused | hallucinated | partial
- negative probe 3: outcome must be refused | hallucinated | partial
- audio overview not generated
- audio overview: factual_fidelity must be an integer 1..5
- audio overview: persona_differentiation must be an integer 1..5
- audio overview: pedagogical_value must be an integer 1..5

## 1. Corpus ingestion

- **Document A:** Lewis et al. (2020) - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (PDF)
- **Document B:** Edge et al. (2024) - From Local to Global: A Graph RAG Approach to Query-Focused Summarization (PDF)
- **Document C:** USSE Noosphere Technical Archive Extract (from course repo Unisvet/couse-c-ai)

Notebook: (not recorded)  
Date of run: (not recorded)

## 2. Grounded query & citation probe

Method: for each query, count claims, check each claim has a badge, then click EVERY badge and judge whether the highlighted span entails the claim.

| # | Query | Claims | With badge | Citations | Entailed | Precision |
|---|-------|-------:|-----------:|----------:|---------:|----------:|
| 1 | Lewis et al. retrieve passages with a dense retriever. According to Ed... | - | - | - | - | pending |
| 2 | Compare RAG-Sequence and RAG-Token from Document A with the way GraphR... | - | - | - | - | pending |
| 3 | Document C describes the ship's archive. Which of the failure modes of... | - | - | - | - | pending |
| 4 | How does each of Documents A and B evaluate answer quality, and which ... | - | - | - | - | pending |
| 5 | What does GraphRAG add on top of vector retrieval, and which sentence ... | - | - | - | - | pending |

**Overall Citation Precision = true entailed citations / total citations = pending** (0 citations)

## 3. Negative-constraint & epistemic-refusal probe

| # | Query (absent from A, B, C) | Outcome | Note |
|---|------|---------|------|
| 1 | What xenon flow rate (in mg/s) does the Noosphere's ion thruster use a... | pending |  |
| 2 | What Recall@5 does the paper report for the RAG-Token model on the Nat... | pending |  |
| 3 | Which hyperspace propulsion code does Edge et al. recommend for deploy... | pending |  |

**Refusal rate: pending**

## 4. Audio Overview (Deep Dive)

Generated: pending  
Excerpt start: (not recorded)

| Metric (1-5) | Score |
|---|---:|
| Factual Fidelity | pending |
| Persona Differentiation | pending |
| Pedagogical Value | pending |

**Mean: pending**

## 5. Queries used

**Grounded (cross-document):**

1. Lewis et al. retrieve passages with a dense retriever. According to Edge et al., what kind of question does that style of retrieval fail on, and what do they build instead?
2. Compare RAG-Sequence and RAG-Token from Document A with the way GraphRAG answers a global question from Document B. Which steps are parametric and which are non-parametric?
3. Document C describes the ship's archive. Which of the failure modes of naive RAG named in Documents A and B would affect an exact part-number lookup in that archive?
4. How does each of Documents A and B evaluate answer quality, and which of those measures could be applied to the astrogation logs in Document C?
5. What does GraphRAG add on top of vector retrieval, and which sentence in Document C would most benefit from that addition?

**Negative constraint (absent from the sources):**

1. What xenon flow rate (in mg/s) does the Noosphere's ion thruster use at cruise, according to the archive?
2. What Recall@5 does the paper report for the RAG-Token model on the Natural Questions test set when retrieval is replaced by HNSW?
3. Which hyperspace propulsion code does Edge et al. recommend for deployment of GraphRAG on starships?
