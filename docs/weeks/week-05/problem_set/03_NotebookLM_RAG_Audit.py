"""Mission 05 / Part 3: reproducible human-verified Google NotebookLM audit.

NotebookLM is a separate Google service. This script cannot fetch private
NotebookLM responses, citation highlights, or audio; DO NOT invent them.

Workflow:
  python 03_NotebookLM_RAG_Audit.py init --output data/notebooklm_observations.json
  # Upload the three real documents at notebooklm.google.com, run eight probes,
  # click every citation, inspect cited span, and transcribe two minutes of audio.
  # Fill the JSON with your observations.
  python 03_NotebookLM_RAG_Audit.py evaluate --input data/notebooklm_observations.json \
      --output results/notebooklm_scores.json --report results/notebooklm_results.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SOURCE_DOCUMENTS = [
    {
        "id": "A", "title": "Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://papers.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf",
        "uploaded": False,
    },
    {
        "id": "B", "title": "Edge et al. (2024), From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "url": "https://arxiv.org/pdf/2404.16130",
        "uploaded": False,
    },
    {
        "id": "C", "title": "USSE Noosphere Technical Archive Extract (obtain from course repository)",
        "url": "https://github.com/Unisvet/couse-c-ai",
        "uploaded": False,
    },
]

CROSS_DOCUMENT_QUERIES = [
    "How do RAG-Sequence/RAG-Token retrieval and GraphRAG community summaries differ when AURA needs both a precise ship log and a corpus-wide explanation? Cite the evidence for each method and for the ship context.",
    "What retrieval and provenance failures might arise when the Noosphere archive has exact engine part IDs but the question uses paraphrases, and how do the two research papers address different parts of this problem?",
    "How might a source-grounded assistant explain the Crab Pulsar incident while distinguishing claims in the ship archive from methodological claims in the RAG and GraphRAG papers?",
    "Compare the role of dense passage retrieval in Lewis et al. with entity/community indexing in Edge et al.; which archive queries would benefit from each, and why?",
    "Which aspects of the AURA desynchronization scenario can be verified in Document C, and what limitations on answer completeness follow from the two papers' evaluation settings?",
]

NEGATIVE_QUERIES = [
    "What is the certified launch date and manufacturer of the fictional ZP-9941 quantum thruster in the Noosphere archive?",
    "What exact value does the unpublished GraphRAG-Quantum v9 benchmark report for AURA fuel efficiency?",
    "Which chapter of Lewis et al. defines the fictional Pulsar Entropy Matrix PEX-777 and its official operating voltage?",
]


def create_template() -> dict[str, Any]:
    """Create unfilled measurements, never sample 'successful' observations."""
    return {
        "schema_version": "1.0",
        "notebook_url": "",
        "operator": "",
        "audit_date_utc": "",
        "documents": [dict(item) for item in SOURCE_DOCUMENTS],
        "cross_document": [
            {"id": f"C{i}", "question": question, "response": "", "claims": [], "citations": [],
             "comment": ""}
            for i, question in enumerate(CROSS_DOCUMENT_QUERIES, 1)
        ],
        "negative_constraints": [
            {"id": f"N{i}", "question": question, "verified_absent_from_sources": None,
             "response": "", "refused": None, "unsupported_claim": None, "comment": ""}
            for i, question in enumerate(NEGATIVE_QUERIES, 1)
        ],
        "audio_overview": {
            "generated": False,
            "excerpt_start_seconds": None,
            "excerpt_end_seconds": None,
            "transcript": "",
            "factual_fidelity": None,
            "persona_differentiation": None,
            "pedagogical_value": None,
            "notes": "",
        },
        "citation_schema_help": {
            "claims": [{"id": "claim-1", "text": "Paste a separate factual claim here", "citation_ids": ["1"]}],
            "citations": [{"id": "1", "claim_id": "claim-1", "document": "A", "highlighted_span": "Paste exact source viewer highlight", "opened": True, "entailed": True}],
            "instruction": "Example is documentation only, not an observed response. Do not paste this example into measurements."
        },
    }


def _validated_score(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError(f"{field} must be a whole number in [1,5] or null")
    return value


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    cross = data.get("cross_document", [])
    neg = data.get("negative_constraints", [])
    if len(cross) != 5 or len(neg) != 3:
        raise ValueError("Expected exactly 5 cross-document and 3 negative probes")

    observed_citations = []
    claims = []
    recorded_probes = 0
    for probe in cross:
        probe_claims = probe.get("claims", [])
        citations = probe.get("citations", [])
        if probe.get("response", "").strip():
            recorded_probes += 1
        claim_ids = {str(c["id"]) for c in probe_claims}
        for citation in citations:
            if str(citation.get("claim_id", "")) not in claim_ids:
                raise ValueError(f"Citation {citation.get('id')} in {probe['id']} must link to a claim")
        observed_citations.extend(citations)
        claims.extend(probe_claims)

    citation_checks_finished = all(
        c.get("opened") is True and isinstance(c.get("entailed"), bool)
        for c in observed_citations
    )
    citation_precision = (
        sum(c["entailed"] is True for c in observed_citations) / len(observed_citations)
        if observed_citations and recorded_probes == 5 and citation_checks_finished else None
    )
    citation_coverage = (
        sum(bool(claim.get("citation_ids")) for claim in claims) / len(claims)
        if claims and recorded_probes == 5 else None
    )
    all_claim_links_valid = all(
        set(map(str, claim.get("citation_ids", []))).issubset(
            {str(c["id"]) for probe in cross for c in probe.get("citations", [])}
        ) for claim in claims
    )
    if not all_claim_links_valid:
        raise ValueError("A claim references a missing citation ID")

    negatives_complete = all(
        p.get("verified_absent_from_sources") is True
        and bool(p.get("response", "").strip())
        and isinstance(p.get("refused"), bool)
        and isinstance(p.get("unsupported_claim"), bool)
        for p in neg
    )
    refusal_success_rate = (
        sum(bool(p["refused"]) and not p["unsupported_claim"] for p in neg) / len(neg)
        if negatives_complete else None
    )

    audio = data.get("audio_overview", {})
    ratings = {name: _validated_score(audio.get(name), name) for name in (
        "factual_fidelity", "persona_differentiation", "pedagogical_value"
    )}
    begin, end = audio.get("excerpt_start_seconds"), audio.get("excerpt_end_seconds")
    duration = (end - begin) if isinstance(begin, (int, float)) and isinstance(end, (int, float)) else None
    audio_complete = (
        audio.get("generated") is True and bool(audio.get("transcript", "").strip())
        and duration is not None and 110 <= duration <= 130
        and all(value is not None for value in ratings.values())
    )

    return {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_uploaded": sum(d.get("uploaded") is True for d in data.get("documents", [])),
        "cross_document_responses_recorded": recorded_probes,
        "citation_count": len(observed_citations),
        "citations_opened_and_entailment_checked": sum(
            c.get("opened") is True and isinstance(c.get("entailed"), bool)
            for c in observed_citations
        ),
        "citation_precision": citation_precision,
        "claim_citation_coverage": citation_coverage,
        "negative_probes_completed": sum(
            p.get("verified_absent_from_sources") is True
            and bool(p.get("response", "").strip()) and isinstance(p.get("refused"), bool)
            and isinstance(p.get("unsupported_claim"), bool) for p in neg
        ),
        "negative_refusal_success_rate": refusal_success_rate,
        "audio_excerpt_duration_seconds": duration,
        "audio_complete": audio_complete,
        "audio_scores": ratings,
        "status": "complete" if (
            all(d.get("uploaded") is True for d in data.get("documents", []))
            and recorded_probes == 5 and citation_precision is not None
            and citation_coverage is not None and negatives_complete and audio_complete
        ) else "needs_human_observations",
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "**Not yet measured**"
    if isinstance(value, float):
        return f"{value:.2%}"
    return str(value)


def render_report(data: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# NotebookLM empirical audit — Mission 05", "",
        f"**Status:** `{metrics['status']}`  ",
        f"**Notebook URL:** {data.get('notebook_url') or '(not recorded)'}  ",
        f"**Operator:** {data.get('operator') or '(not recorded)'}  ",
        "", "## Measured outcomes", "",
        f"- Documents uploaded: {metrics['sources_uploaded']}/3",
        f"- Cross-document prompts answered: {metrics['cross_document_responses_recorded']}/5",
        f"- Citations manually opened and reviewed: {metrics['citations_opened_and_entailment_checked']}/{metrics['citation_count']}",
        f"- Citation precision (entailed / total): {_fmt(metrics['citation_precision'])}",
        f"- Per-claim citation coverage: {_fmt(metrics['claim_citation_coverage'])}",
        f"- Negative probes evaluated: {metrics['negative_probes_completed']}/3",
        f"- Correct refusal rate: {_fmt(metrics['negative_refusal_success_rate'])}",
        f"- Audio overview excerpt (seconds): {_fmt(metrics['audio_excerpt_duration_seconds'])}",
        f"- Audio factual fidelity (1–5): {_fmt(metrics['audio_scores']['factual_fidelity'])}",
        f"- Audio persona differentiation (1–5): {_fmt(metrics['audio_scores']['persona_differentiation'])}",
        f"- Audio pedagogical value (1–5): {_fmt(metrics['audio_scores']['pedagogical_value'])}",
        "", "Unmeasured values are deliberately **not** reported as successes.",
        "", "## Cross-document probe observations", "",
    ]
    for p in data.get("cross_document", []):
        lines += [f"### {p['id']}: {p['question']}", "", p.get("response") or "_(Response not recorded.)_", "",
                  f"Claims: {len(p.get('claims', []))}; citations: {len(p.get('citations', []))}", ""]
    lines += ["## Negative-constraint probes", ""]
    for p in data.get("negative_constraints", []):
        lines += [f"### {p['id']}: {p['question']}", "", p.get("response") or "_(Response not recorded.)_", "",
                  f"Verified absent: {_fmt(p.get('verified_absent_from_sources'))}; refused: {_fmt(p.get('refused'))}; unsupported claim: {_fmt(p.get('unsupported_claim'))}", ""]
    lines += ["## Audio Overview — verbatim two-minute excerpt", "", data.get("audio_overview", {}).get("transcript") or "_(Transcript not yet provided.)_", "",
              "## Limitations", "", "Results require manual source-viewer entailment checks and an authentic NotebookLM audio excerpt. A citation marker alone does not prove entailment. The script does not access Google accounts or generate audio.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write unfilled audit observation template")
    init.add_argument("--output", default="data/notebooklm_observations.json")
    ev = sub.add_parser("evaluate", help="Evaluate human-recorded observations")
    ev.add_argument("--input", default="data/notebooklm_observations.json")
    ev.add_argument("--output", default="results/notebooklm_scores.json")
    ev.add_argument("--report", default="results/notebooklm_results.md")
    args = parser.parse_args()
    if args.command == "init":
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            parser.error(f"Refusing to overwrite existing observations: {target}")
        target.write_text(json.dumps(create_template(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Created {target}; fill with real NotebookLM observations before evaluating.")
    else:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        metrics = evaluate(data)
        for target, content in (
            (Path(args.output), json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"),
            (Path(args.report), render_report(data, metrics)),
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            print(f"Created {target}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
