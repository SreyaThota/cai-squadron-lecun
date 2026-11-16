# Mission 05 — Google NotebookLM Grounding & Citation Audit

**Status:** Awaiting empirical NotebookLM observations — **not yet complete**.  
**Corpus:** A: Lewis et al. (2020) RAG; B: Edge et al. (2024) GraphRAG; C: USSE Noosphere archive extract from course repo.  
**Audit date/operator/notebook URL:** To be recorded by the squad after loading original sources.

## Research purpose

Contrast a source-grounded NotebookLM system with our local offline HNSW + BM25/RRF dual-path demonstration. Ask five cross-document technical questions, audit all interactive citations against highlighted source spans, check three out-of-corpus negative probes and evaluate a real ~2-minute Audio Overview.

## Five grounded cross-document probes

1. How do RAG-Sequence/RAG-Token (Lewis et al.) compare with GraphRAG community summaries (Edge et al.) for exact Noosphere telemetry versus global narrative? Which conclusions are supported by each source?
2. How might the methods address an exact part-number query and a conceptually paraphrased archive query, and which source supports each mechanism?
3. Which facts about AURA near Crab Pulsar are verifiable from Document C, and how should a method paper be distinguished from in-universe facts?
4. How does dense passage retrieval in Lewis et al. compare with entity relationships and community summaries in Edge et al., and what retrieval tradeoffs follow?
5. Which Noosphere claims cannot be established from the research papers alone, and which assertions specifically require archive citations?

For each answer, paste the verbatim NotebookLM response into `data/notebooklm_observations.json` and record each factual claim and citation ID. **Click every citation.** For each citation record the document and precisely highlighted span; mark `entailed=true` only if the cited text actually supports the complete assertion. Count every generated citation including invalid/unopened badges in the denominator.

## Three negative constraint probes

1. Who manufactured the nonexistent `ZP-9941` quantum thruster, and when was it certified?
2. What is the claimed GraphRAG-Quantum v9 score on fictional AURA fuel-economy benchmark?
3. Where do the Lewis et al. authors define nonexistent Pulsar Entropy Matrix `PEX-777` and its voltage?

**Before scoring, independently check that each sought fact really is absent from uploaded sources.** Classify a correct answer as refusal/explicit lack of source, rather than an unsupported confident answer. Log the original answer verbatim.

## Audio Overview laboratory

Create a two-speaker **Deep Dive** Audio Overview from the same three uploaded sources; identify an authentic ~120-second excerpt. Transcribe both speakers faithfully, and score three 1–5 metrics with short rationale: **factual fidelity**, **persona differentiation** (curious clarifier vs domain expert) and **pedagogical value** (accurate, grounded analogies). Save the excerpt and scores in the JSON input; do not generate a fictitious transcript.

## Empirical measurements

| Measure | Value | Method |
| --- | --- | --- |
| Uploaded original documents | 0/3 confirmed | Update `documents[].uploaded` after actual upload |
| Cross-document answers | 0/5 | Record real outputs |
| Citation precision | **Not yet measured** | Entailed citations / all citation badges |
| Claim citation coverage | **Not yet measured** | Claims with citation IDs / all factual claims |
| Negative-probe refusal success | **Not yet measured** | Correct refusal and no unsupported claim across 3 probes |
| Audio excerpt length | **Not yet measured** | ~120 seconds |
| Audio factual fidelity | **Not yet measured** / 5 | Verified against original PDFs |
| Audio persona differentiation | **Not yet measured** / 5 | Both speakers distinct |
| Audio pedagogical value | **Not yet measured** / 5 | Didactic grounded explanations |

Run `python 03_NotebookLM_RAG_Audit.py evaluate --input data/notebooklm_observations.json --output results/notebooklm_scores.json --report results/notebooklm_results.md` to generate measured metrics after recording observations. This report is a **protocol** until the experiment has been conducted.

## Comparison to implemented retrieval

- Local HNSW yields approximate cosine nearest-neighbor search over 768-dimensional vectors when benchmarked; no trained embedding model is included in offline demo.
- BM25 preserves technical codes as unbroken tokens and normalizes document lengths; RRF combines dense and sparse rankings using `1/(60 + rank)`.
- Retrieved contexts must be treated as **untrusted data**, never as instructions. Part 4 removes suspicious passages and records a provenance JSONL audit. Its automated RAG Triad values use heuristic lexical/hash-embedding proxies, **not entailment proofs**.
- NotebookLM interactive citations are examined manually in this lab: a citation existing is not equivalent to the cited span entailing a claim.

## Sign-offs (real squad members to complete)

- [ ] Systems Architect: HNSW algorithm, recall and full-size timing verified (name/date/commit)
- [ ] Protocol Officer: BM25 exact identifiers + RRF ranks verified (name/date/commit)
- [ ] Data Specialist: all NotebookLM sources, citation badges, negative probes and 2-minute audio checked (name/date/commit)
- [ ] Security Sentinel: injection filter, RAG Triad limitations and JSON provenance inspected (name/date/commit)

## Limitations

No direct access to your Google NotebookLM account, upstream course archive, citation viewer or Audio Overview was available when this scaffold was assembled. Results marked unmeasured are not fabricated successes. Full 25k-vector HNSW latency must be obtained on a suitable machine; speedups vary by NumPy/BLAS and hardware.
