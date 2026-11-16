"""
03_NotebookLM_RAG_Audit.py
Mission 5 - Part 3: Data & Payload Specialist
Google NotebookLM grounding, citation and Audio Overview audit harness

Starship Operational Unit: USSE Noosphere Bridge Crew     Squadron: LeCun
Course: Conversational AI - Winter Semester 2026/27
Subsystem 05: Dual-Path Associative Memory Core

What this file is (and is not)
------------------------------
NotebookLM has no public API for the features the brief asks us to audit (citation badges,
source viewer, Audio Overview). The measurements are therefore made BY HAND in the web UI,
recorded in notebooklm_audit_results.json, and this file does everything that can be
automated and checked:

  1. The experimental PROTOCOL: 3 documents, 5 cross-document queries, 3 negative-constraint
     probes, the Audio Overview rubric.
  2. The SCORING: Citation Precision, refusal rate, Audio Overview mean scores, with
     validation of the results file (every query needs citations, scores in 1..5, ...).
  3. The REPORT generator: writes notebooklm_audit_report.md from the results file.

No number in this repository's audit report comes from this file's fixtures: the unit tests
use clearly labelled SYNTHETIC records to prove the arithmetic, and the report generator
refuses to print results that are not in notebooklm_audit_results.json.

Run
---
    python 03_NotebookLM_RAG_Audit.py --template   # write an empty notebooklm_audit_results.json
    python 03_NotebookLM_RAG_Audit.py              # tests + score + (re)write the report
                                                   # (status PENDING until the results file is filled)

Memorizable patterns
--------------------
* Citation Precision = true entailed citations / total citations. "Has a citation badge"
  is NOT the same as "the cited span supports the claim": you must click and read.
* Epistemic refusal is a feature: a grounded system should say "not in your sources" for a
  question the documents cannot answer, instead of answering from pre-training.
* A podcast is a lossy compression of the sources; score it for invented facts first.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "notebooklm_audit_results.json")
REPORT_PATH = os.path.join(HERE, "notebooklm_audit_report.md")

# =============================================================================
# 1. Protocol
# =============================================================================
DOCUMENTS = {
    "A": "Lewis et al. (2020) - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (PDF)",
    "B": "Edge et al. (2024) - From Local to Global: A Graph RAG Approach to Query-Focused Summarization (PDF)",
    "C": "USSE Noosphere Technical Archive Extract (from course repo Unisvet/couse-c-ai)",
}

# Five cross-document queries (each needs evidence from at least two documents).
GROUNDED_QUERIES = [
    "Lewis et al. retrieve passages with a dense retriever. According to Edge et al., what kind of "
    "question does that style of retrieval fail on, and what do they build instead?",
    "Compare RAG-Sequence and RAG-Token from Document A with the way GraphRAG answers a global "
    "question from Document B. Which steps are parametric and which are non-parametric?",
    "Document C describes the ship's archive. Which of the failure modes of naive RAG named in "
    "Documents A and B would affect an exact part-number lookup in that archive?",
    "How does each of Documents A and B evaluate answer quality, and which of those measures "
    "could be applied to the astrogation logs in Document C?",
    "What does GraphRAG add on top of vector retrieval, and which sentence in Document C would "
    "most benefit from that addition?",
]

# Three plausible but intentionally unanswerable questions (absent from A, B and C).
NEGATIVE_PROBES = [
    "What xenon flow rate (in mg/s) does the Noosphere's ion thruster use at cruise, according to the archive?",
    "What Recall@5 does the paper report for the RAG-Token model on the Natural Questions test set when "
    "retrieval is replaced by HNSW?",
    "Which hyperspace propulsion code does Edge et al. recommend for deployment of GraphRAG on starships?",
]

AUDIO_METRICS = ("factual_fidelity", "persona_differentiation", "pedagogical_value")

# =============================================================================
# 2. Records and scoring
# =============================================================================
@dataclass
class Citation:
    badge: int                      # the number shown on the badge
    source: str                     # "A" | "B" | "C"
    entailed: Optional[bool] = None  # after CLICKING: does the highlighted span support the claim?
    note: str = ""


@dataclass
class GroundedProbe:
    query: str
    answer_text: str = ""
    claims_total: int = 0           # factual claims in the answer
    claims_with_badge: int = 0      # of those, claims that carry a citation badge
    citations: List[Citation] = field(default_factory=list)


@dataclass
class NegativeProbe:
    query: str
    outcome: str = ""               # "refused" | "hallucinated" | "partial"
    answer_text: str = ""
    note: str = ""


@dataclass
class AudioOverview:
    generated: bool = False
    excerpt_transcript: str = ""    # ~2 minute excerpt
    excerpt_start_mm_ss: str = ""
    factual_fidelity: Optional[int] = None
    persona_differentiation: Optional[int] = None
    pedagogical_value: Optional[int] = None
    invented_claims: List[str] = field(default_factory=list)
    notes: str = ""


def citation_precision(citations: List[Citation]) -> Optional[float]:
    """Citation Precision = True Entailed Citations / Total Citations Generated."""
    judged = [c for c in citations if c.entailed is not None]
    if not judged or len(judged) != len(citations):
        return None  # not every citation has been clicked and judged yet
    return sum(c.entailed for c in judged) / len(judged)


def citation_coverage(p: GroundedProbe) -> Optional[float]:
    """Share of claims that carry a badge at all (the brief's first check)."""
    return None if p.claims_total == 0 else p.claims_with_badge / p.claims_total


def refusal_rate(probes: List[NegativeProbe]) -> Optional[float]:
    done = [p for p in probes if p.outcome]
    if len(done) != len(probes) or not probes:
        return None
    return sum(p.outcome == "refused" for p in probes) / len(probes)


def audio_mean(a: AudioOverview) -> Optional[float]:
    vals = [getattr(a, m) for m in AUDIO_METRICS]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def validate(results: dict) -> List[str]:
    """Return a list of problems in the results file (empty list = complete and consistent)."""
    errs: List[str] = []
    g = results.get("grounded", [])
    if len(g) != 5:
        errs.append(f"need 5 grounded probes, found {len(g)}")
    for i, p in enumerate(g, 1):
        if not p.get("citations"):
            errs.append(f"grounded probe {i}: no citations recorded")
        if p.get("claims_with_badge", 0) > p.get("claims_total", 0):
            errs.append(f"grounded probe {i}: more badged claims than claims")
        for c in p.get("citations", []):
            if c.get("entailed") is None:
                errs.append(f"grounded probe {i}: citation {c.get('badge')} not clicked/judged")
            if c.get("source") not in DOCUMENTS:
                errs.append(f"grounded probe {i}: citation {c.get('badge')} has unknown source")
    n = results.get("negative", [])
    if len(n) != 3:
        errs.append(f"need 3 negative probes, found {len(n)}")
    for i, p in enumerate(n, 1):
        if p.get("outcome") not in ("refused", "hallucinated", "partial"):
            errs.append(f"negative probe {i}: outcome must be refused | hallucinated | partial")
    a = results.get("audio", {})
    if not a.get("generated"):
        errs.append("audio overview not generated")
    elif not a.get("excerpt_transcript"):
        errs.append("audio overview: no transcript excerpt")
    for m in AUDIO_METRICS:
        v = a.get(m)
        if v is None or not (1 <= v <= 5):
            errs.append(f"audio overview: {m} must be an integer 1..5")
    return errs


def load_results(path: str = RESULTS_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def empty_results() -> dict:
    return {
        "date": "",
        "notebook_url": "",
        "documents": DOCUMENTS,
        "grounded": [asdict(GroundedProbe(q)) for q in GROUNDED_QUERIES],
        "negative": [asdict(NegativeProbe(q)) for q in NEGATIVE_PROBES],
        "audio": asdict(AudioOverview()),
    }


def summarize(results: dict) -> Dict[str, Optional[float]]:
    g = [GroundedProbe(**{**p, "citations": [Citation(**c) for c in p["citations"]]})
         for p in results["grounded"]]
    allc = [c for p in g for c in p.citations]
    return {
        "citation_precision_overall": citation_precision(allc),
        "per_query_precision": [citation_precision(p.citations) for p in g],
        "per_query_coverage": [citation_coverage(p) for p in g],
        "refusal_rate": refusal_rate([NegativeProbe(**p) for p in results["negative"]]),
        "audio_mean": audio_mean(AudioOverview(**results["audio"])),
        "total_citations": len(allc),
    }


# =============================================================================
# 3. Report generator (prints ONLY what is in the results file)
# =============================================================================
def pct(x: Optional[float]) -> str:
    return "pending" if x is None else f"{100 * x:.0f}%"


def build_report(results: Optional[dict]) -> str:
    L: List[str] = []
    L += ["# Mission 5 - Part 3: NotebookLM Grounding & Citation Audit",
          "",
          "Squadron LeCun - Conversational AI, Winter Semester 2026/27 - Subsystem 05",
          ""]
    if results is None:
        results = empty_results()
    errs = validate(results)
    status = "COMPLETE" if not errs else "PENDING - measurements not yet recorded"
    L += [f"**Status:** {status}", ""]
    if errs:
        L += ["The measurements below must be made by hand in the NotebookLM web UI and entered in "
              "`notebooklm_audit_results.json` (see `03_NotebookLM_RAG_Audit.py --template`). "
              "Open items:", ""]
        L += [f"- {e}" for e in errs[:12]]
        if len(errs) > 12:
            L.append(f"- ... and {len(errs) - 12} more")
        L.append("")
    L += ["## 1. Corpus ingestion", ""]
    for k, v in DOCUMENTS.items():
        L.append(f"- **Document {k}:** {v}")
    L += ["", f"Notebook: {results.get('notebook_url') or '(not recorded)'}  ",
          f"Date of run: {results.get('date') or '(not recorded)'}", ""]

    s = summarize(results)
    L += ["## 2. Grounded query & citation probe", "",
          "Method: for each query, count claims, check each claim has a badge, then click EVERY badge "
          "and judge whether the highlighted span entails the claim.", "",
          "| # | Query | Claims | With badge | Citations | Entailed | Precision |",
          "|---|-------|-------:|-----------:|----------:|---------:|----------:|"]
    for i, p in enumerate(results["grounded"], 1):
        cits = p["citations"]
        ent = sum(1 for c in cits if c.get("entailed"))
        prec = citation_precision([Citation(**c) for c in cits]) if cits else None
        L.append(f"| {i} | {p['query'][:70]}... | {p['claims_total'] or '-'} | "
                 f"{p['claims_with_badge'] or '-'} | {len(cits) or '-'} | "
                 f"{ent if cits else '-'} | {pct(prec)} |")
    L += ["", f"**Overall Citation Precision = true entailed citations / total citations = "
              f"{pct(s['citation_precision_overall'])}** ({s['total_citations']} citations)", ""]
    bad = [(i, c) for i, p in enumerate(results["grounded"], 1)
           for c in p["citations"] if c.get("entailed") is False]
    if bad:
        L += ["Citations whose span did NOT entail the claim:", ""]
        L += [f"- Query {i}, badge [{c['badge']}] (Doc {c['source']}): {c.get('note') or 'no note'}"
              for i, c in bad]
        L.append("")

    L += ["## 3. Negative-constraint & epistemic-refusal probe", "",
          "| # | Query (absent from A, B, C) | Outcome | Note |", "|---|------|---------|------|"]
    for i, p in enumerate(results["negative"], 1):
        L.append(f"| {i} | {p['query'][:70]}... | {p['outcome'] or 'pending'} | {p.get('note') or ''} |")
    L += ["", f"**Refusal rate: {pct(s['refusal_rate'])}**", ""]

    a = results["audio"]
    L += ["## 4. Audio Overview (Deep Dive)", "",
          f"Generated: {'yes' if a.get('generated') else 'pending'}  ",
          f"Excerpt start: {a.get('excerpt_start_mm_ss') or '(not recorded)'}", "",
          "| Metric (1-5) | Score |", "|---|---:|"]
    for m in AUDIO_METRICS:
        L.append(f"| {m.replace('_', ' ').title()} | {a.get(m) if a.get(m) is not None else 'pending'} |")
    am = s["audio_mean"]
    L += ["", f"**Mean: {'pending' if am is None else f'{am:.2f} / 5'}**", ""]
    if a.get("invented_claims"):
        L += ["Claims in the dialogue that are not in the sources:", ""]
        L += [f"- {c}" for c in a["invented_claims"]]
        L.append("")
    if a.get("excerpt_transcript"):
        L += ["### 2-minute transcript excerpt", "", "```", a["excerpt_transcript"].strip(), "```", ""]
    if a.get("notes"):
        L += ["### Notes", "", a["notes"], ""]

    L += ["## 5. Queries used", "", "**Grounded (cross-document):**", ""]
    L += [f"{i}. {q}" for i, q in enumerate(GROUNDED_QUERIES, 1)]
    L += ["", "**Negative constraint (absent from the sources):**", ""]
    L += [f"{i}. {q}" for i, q in enumerate(NEGATIVE_PROBES, 1)]
    L.append("")
    return "\n".join(L)


# =============================================================================
# Tests: arithmetic only, on SYNTHETIC records (not NotebookLM results)
# =============================================================================
def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def test_scoring_arithmetic() -> None:
    cits = [Citation(1, "A", True), Citation(2, "A", True), Citation(3, "B", False), Citation(4, "C", True)]
    assert citation_precision(cits) == 0.75
    assert citation_precision(cits[:2] + [Citation(5, "A", None)]) is None, "unjudged citation must block the score"
    assert citation_precision([]) is None
    p = GroundedProbe("q", claims_total=8, claims_with_badge=6)
    assert citation_coverage(p) == 0.75
    neg = [NegativeProbe("a", "refused"), NegativeProbe("b", "hallucinated"), NegativeProbe("c", "refused")]
    assert abs(refusal_rate(neg) - 2 / 3) < 1e-12
    assert refusal_rate(neg[:2] + [NegativeProbe("c")]) is None
    assert audio_mean(AudioOverview(True, "t", "0:30", 4, 5, 3)) == 4.0
    _ok("SYNTHETIC fixtures: precision 3/4 = 0.75, coverage 6/8, refusal 2/3, audio mean 4.0; "
        "unjudged citations block the score")


def test_validation_blocks_incomplete_results() -> None:
    errs = validate(empty_results())
    assert errs and any("not clicked" in e or "no citations" in e for e in errs)
    assert any("audio overview not generated" in e for e in errs)
    _ok(f"empty results file is rejected ({len(errs)} open items), so the report cannot claim a result")


def test_report_is_pending_without_data() -> None:
    md = build_report(None)
    assert "PENDING" in md and "pending" in md
    assert "Overall Citation Precision" in md
    _ok("report generated from an empty results file says PENDING and invents no numbers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true", help="write an empty results file and exit")
    args = ap.parse_args()
    bar = "=" * 92
    print(f"{bar}\nUSSE NOOSPHERE - SUBSYSTEM 05: DUAL-PATH ASSOCIATIVE MEMORY CORE\n"
          f"Role Deliverable: Part 3 - Data & Payload Specialist (NotebookLM audit harness)\n{bar}")
    if args.template:
        if os.path.exists(RESULTS_PATH):
            print(f"[SKIP] {RESULTS_PATH} already exists; not overwriting")
        else:
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(empty_results(), f, indent=2)
            print(f"[SUCCESS] template written to {RESULTS_PATH}")
        return
    print("\n=== Unit tests (arithmetic on synthetic fixtures) ===")
    test_scoring_arithmetic()
    test_validation_blocks_incomplete_results()
    test_report_is_pending_without_data()
    results = load_results()
    errs = validate(results) if results else ["results file missing"]
    print(f"\n=== Audit status: {'COMPLETE' if not errs else 'PENDING'} ===")
    for e in errs[:8]:
        print(f"  open: {e}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(results))
    print(f"\n[SUCCESS] Report written to: {REPORT_PATH}")
    print(("[SUCCESS] Subsystem 05 NotebookLM audit recorded.\n" if not errs else
           "[PENDING] Fill notebooklm_audit_results.json from the NotebookLM UI, then re-run.\n") + bar)


if __name__ == "__main__":
    main()
