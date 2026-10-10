"""Evaluate retrieval, answers, citations and refusals on held-out golden questions.

    python -m eval.run_eval                                   # fixture set, configured threshold
    python -m eval.run_eval --sweep                           # pick the threshold on the dev split
    python -m eval.run_eval --sweep --real-pdf my.pdf         # + your real document set
    python -m eval.run_eval --sweep --real-pdf my.pdf --apply --update-readme

Every golden question belongs to a "dev" or a "test" split. The threshold sweep only ever looks
at dev questions; the reported numbers come from the test questions at the chosen threshold, so
the threshold is never tuned on the questions it is scored on. A sweep result below 0.25 is only
kept when it beats 0.25 on dev by at least 5 points.

Two question sets are supported and reported separately: the generated fixture
(eval/golden.jsonl on tests/fixtures/handbook.pdf) and a real document of your own
(eval/golden_real.jsonl, see golden_real.example.jsonl). The headline number is the real set's
test split; without a real set, nothing is written to the README.

Each question is retrieved and answered once with the confidence gate disabled. The gate only
looks at the top retrieval score, so the outcome at any threshold t is exact without re-asking:
refused if top_score < t, otherwise whatever the LLM answered. Writes eval/results.{json,md}.
Numbers are reported as measured; nothing is rounded up or filtered.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden.jsonl"
REAL_GOLDEN = ROOT / "eval" / "golden_real.jsonl"
FIXTURE = ROOT / "tests" / "fixtures" / "handbook.pdf"
SWEEP = [round(0.20 + 0.05 * i, 2) for i in range(9)]  # 0.20 .. 0.60
FLOOR = 0.25  # a sweep pick below this must earn its place on dev
FLOOR_MIN_GAIN = 0.05  # ... by at least 5 points of (answer + refusal accuracy) / 2
SPLITS = ("dev", "test")
README_START, README_END = "<!-- eval:start -->", "<!-- eval:end -->"


@dataclass
class Record:
    id: str
    kind: str
    question: str
    answerable: bool
    gold_pages: list[int]
    gold_keywords: list[str]
    top_score: float  # best cosine score: what the confidence gate looks at
    top5_pages: list[list[int]]
    retrieval_ms: int  # cost of a refusal at the gate (no LLM call)
    llm_answer: str  # the answer with the gate disabled
    llm_refused: bool
    cited_pages: list[int]
    confidence: float
    answer_ms: int
    load_ms: int
    split: str = "test"
    cited_chunks: list[list[int]] = field(default_factory=list)  # pages cited per chunk
    chunk_spans: list[list[int]] = field(default_factory=list)  # every page each cited chunk spans


@dataclass
class Metrics:
    threshold: float
    n_answerable: int
    n_unanswerable: int
    hit_at_5: float | None
    answer_accuracy_lenient: float | None  # any gold keyword present
    answer_accuracy_strict: float | None  # every gold keyword present
    answer_accuracy_cited: float | None  # strict, and every cited page is a gold page
    refusal_accuracy: float | None
    false_refusal_rate: float | None
    citation_precision_strict: float | None  # answered questions whose cited pages are all gold
    citation_precision_per_page: float | None  # mean share of cited pages that are gold
    combined: float | None  # (strict answer accuracy + refusal accuracy) / 2
    latency_p50_ms: int
    latency_p95_ms: int
    answer_accuracy_by_kind: dict[str, float | None] = field(default_factory=dict)


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return assign_splits([json.loads(line) for line in fh if line.strip()])


def assign_splits(items: list[dict]) -> list[dict]:
    """Keep each item's "split"; items without one alternate dev/test within their group.

    Groups are answerable and unanswerable questions, in file order, so a hand-written set
    without split labels still ends up with both kinds of question on both sides.
    """
    counters = {True: 0, False: 0}
    for item in items:
        split = item.get("split")
        if split is None:
            group = not item.get("unanswerable", False)
            item["split"] = SPLITS[counters[group] % 2]
            counters[group] += 1
        elif split not in SPLITS:
            raise ValueError(f"{item.get('id')}: split must be 'dev' or 'test', got {split!r}")
    return items


def run_questions(store, llm, doc_id: str, items: list[dict], k: int = 5) -> list[Record]:
    from app.qa.answerer import Answerer
    from app.qa.retriever import Retriever

    answerer = Answerer(Retriever(store, top_k=k), llm, threshold=float("-inf"))
    records = []
    for item in items:
        started = time.perf_counter()
        hits = store.search(doc_id, item["question"], 5)
        retrieval_ms = round((time.perf_counter() - started) * 1000)
        answer = answerer.answer(doc_id, item["question"])
        records.append(
            Record(
                id=item["id"],
                kind=item.get("kind", "unanswerable" if item.get("unanswerable") else "other"),
                question=item["question"],
                answerable=not item.get("unanswerable", False),
                gold_pages=item.get("gold_pages", []),
                gold_keywords=item.get("gold_keywords", []),
                top_score=round(hits[0][1], 4) if hits else 0.0,
                top5_pages=[c.pages for c, _ in hits],
                retrieval_ms=retrieval_ms,
                llm_answer=answer.answer,
                llm_refused=answer.refused,
                cited_pages=sorted({p for c in answer.citations for p in c.pages}),
                confidence=answer.confidence,
                answer_ms=answer.answer_ms,
                load_ms=answer.load_ms,
                split=item["split"],
                cited_chunks=[list(c.pages) for c in answer.citations],
                chunk_spans=[list(c.chunk_pages or c.pages) for c in answer.citations],
            )
        )
    return records


def refused_at(r: Record, threshold: float) -> bool:
    return r.top_score < threshold or r.llm_refused


def keywords_present(answer: str, keywords: list[str], require: str = "all") -> bool:
    """Gold keywords in the answer; "a|b" accepts either spelling (e.g. "two weeks|2 weeks").

    require="all" is the strict check (every keyword), require="any" the lenient one.
    """
    text = answer.lower()
    found = [any(alt.strip().lower() in text for alt in kw.split("|")) for kw in keywords]
    return all(found) if require == "all" else any(found)


def correct_at(r: Record, threshold: float, require: str = "all") -> bool:
    return not refused_at(r, threshold) and keywords_present(r.llm_answer, r.gold_keywords, require)


def citation_ok(r: Record) -> bool:
    return bool(r.cited_pages) and set(r.cited_pages) <= set(r.gold_pages)


def page_precision(r: Record) -> float:
    if not r.cited_pages:
        return 0.0
    return len(set(r.cited_pages) & set(r.gold_pages)) / len(r.cited_pages)


def citation_failure_reason(r: Record) -> str:
    """Why an answered question's citations are not all on gold pages."""
    if not r.cited_pages:
        return "no citation"
    gold = set(r.gold_pages)
    reasons = []
    for pages in r.cited_chunks or [r.cited_pages]:
        if gold.isdisjoint(pages):
            reasons.append("extra chunk (no gold page)")
        elif not set(pages) <= gold:
            reasons.append("multi-page citation (gold + other page)")
    return ", ".join(dict.fromkeys(reasons)) or "unknown"


def citation_failures(records: list[Record], threshold: float) -> list[dict]:
    return [
        {
            "id": r.id,
            "split": r.split,
            "question": r.question,
            "gold_pages": r.gold_pages,
            "cited_pages": r.cited_pages,
            "cited_chunks": r.cited_chunks,
            "chunk_spans": r.chunk_spans,
            "reason": citation_failure_reason(r),
        }
        for r in records
        if r.answerable and not refused_at(r, threshold) and not citation_ok(r)
    ]


def missing_keywords(r: Record) -> list[str]:
    return [kw for kw in r.gold_keywords if not keywords_present(r.llm_answer, [kw])]


def refusal_failures(records: list[Record], threshold: float) -> list[dict]:
    """Unanswerable questions that were answered at this threshold."""
    return [
        {
            "id": r.id,
            "split": r.split,
            "question": r.question,
            "answer": r.llm_answer,
            "top_score": r.top_score,
            "confidence": r.confidence,
            "cited_pages": r.cited_pages,
        }
        for r in records
        if not r.answerable and not refused_at(r, threshold)
    ]


def answer_failures(records: list[Record], threshold: float) -> list[dict]:
    """Answerable questions that fail the strict check at this threshold, and why."""
    return [
        {
            "id": r.id,
            "split": r.split,
            "question": r.question,
            "gold_keywords": r.gold_keywords,
            "missing_keywords": [] if refused_at(r, threshold) else missing_keywords(r),
            "refused": refused_at(r, threshold),
            "refused_by": ("gate" if r.top_score < threshold else "llm" if r.llm_refused else None),
            "top_score": r.top_score,
            "answer": r.llm_answer,
        }
        for r in records
        if r.answerable and not correct_at(r, threshold)
    ]


def latency_at(r: Record, threshold: float) -> int:
    return r.retrieval_ms if r.top_score < threshold else r.answer_ms


def _mean(values: list[bool | float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(pct * len(ordered) + 0.5) - 1))]


def compute_metrics(records: list[Record], threshold: float) -> Metrics:
    ans = [r for r in records if r.answerable]
    unans = [r for r in records if not r.answerable]
    answered = [r for r in ans if not refused_at(r, threshold)]
    strict = _mean([correct_at(r, threshold) for r in ans])
    refusal = _mean([refused_at(r, threshold) for r in unans])
    parts = [v for v in (strict, refusal) if v is not None]
    latencies = [latency_at(r, threshold) for r in records]
    return Metrics(
        threshold=threshold,
        n_answerable=len(ans),
        n_unanswerable=len(unans),
        hit_at_5=_mean([any(set(p) & set(r.gold_pages) for p in r.top5_pages) for r in ans]),
        answer_accuracy_lenient=_mean([correct_at(r, threshold, "any") for r in ans]),
        answer_accuracy_strict=strict,
        answer_accuracy_cited=_mean([correct_at(r, threshold) and citation_ok(r) for r in ans]),
        refusal_accuracy=refusal,
        false_refusal_rate=_mean([refused_at(r, threshold) for r in ans]),
        citation_precision_strict=_mean([citation_ok(r) for r in answered]),
        citation_precision_per_page=_mean([page_precision(r) for r in answered]),
        combined=round(sum(parts) / len(parts), 4) if parts else None,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        answer_accuracy_by_kind={
            kind: _mean([correct_at(r, threshold) for r in ans if r.kind == kind])
            for kind in sorted({r.kind for r in ans})
        },
    )


def pick_threshold(sweep: list[Metrics]) -> Metrics:
    """Best (answer + refusal accuracy) / 2; ties go to the lower false-refusal rate."""
    return max(
        sweep, key=lambda m: (m.combined or 0.0, -(m.false_refusal_rate or 0.0), -m.threshold)
    )


def choose_threshold(dev: list[Record]) -> tuple[dict, list[Metrics]]:
    """Sweep on dev questions only and apply the floor rule. Returns (decision, sweep table)."""
    if not dev:
        raise ValueError("no dev questions: the threshold can only be tuned on the dev split")
    table = [compute_metrics(dev, t) for t in SWEEP]
    best = pick_threshold(table)
    floor = next(m for m in table if m.threshold == FLOOR)
    gain = (best.combined or 0.0) - (floor.combined or 0.0)
    decision = {
        "source": "dev sweep",
        "dev_best": best.threshold,
        "dev_best_combined": best.combined,
        "floor": FLOOR,
        "floor_combined": floor.combined,
        "dev_questions": len(dev),
    }
    pct = _pct
    if best.threshold >= FLOOR:
        decision["value"] = best.threshold
        decision["explanation"] = (
            f"{best.threshold:.2f} had the best dev score ((answer + refusal accuracy) / 2 = "
            f"{pct(best.combined)}) and is at or above the {FLOOR:.2f} floor, so it is used."
        )
    elif gain >= FLOOR_MIN_GAIN:
        decision["value"] = best.threshold
        decision["explanation"] = (
            f"The dev sweep picked {best.threshold:.2f}, below the {FLOOR:.2f} floor. It scored "
            f"{pct(best.combined)} on dev against {pct(floor.combined)} at {FLOOR:.2f}, a gain "
            f"of {gain * 100:.1f} points, which clears the {FLOOR_MIN_GAIN * 100:.0f}-point bar, "
            f"so {best.threshold:.2f} is used."
        )
    else:
        decision["value"] = FLOOR
        decision["explanation"] = (
            f"The dev sweep picked {best.threshold:.2f}, below the {FLOOR:.2f} floor. It scored "
            f"{pct(best.combined)} on dev against {pct(floor.combined)} at {FLOOR:.2f}, a gain "
            f"of {gain * 100:.1f} points, under the {FLOOR_MIN_GAIN * 100:.0f}-point bar. That "
            f"is not a clear gain, so {FLOOR:.2f} is kept: a lower gate lets more weakly "
            f"matched questions reach the LLM, which is what the gate exists to stop."
        )
    return decision, table


def score_summary(records: list[Record]) -> dict:
    def summary(scores: list[float]) -> dict:
        if not scores:
            return {}
        return {
            "n": len(scores),
            "min": round(min(scores), 4),
            "median": round(statistics.median(scores), 4),
            "max": round(max(scores), 4),
        }

    return {
        "answerable": summary([r.top_score for r in records if r.answerable]),
        "unanswerable": summary([r.top_score for r in records if not r.answerable]),
    }


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    # os.sysconf is POSIX-only; on Windows RAM is simply left out
    with contextlib.suppress(AttributeError, ValueError, OSError):
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    return info


def _split_counts(records: list[Record]) -> dict:
    return {
        split: {
            "answerable": sum(r.answerable for r in records if r.split == split),
            "unanswerable": sum(not r.answerable for r in records if r.split == split),
        }
        for split in SPLITS
    }


def run(
    store,
    llm,
    golden_path: Path = GOLDEN,
    pdf_path: Path = FIXTURE,
    out_dir: Path | None = ROOT / "eval",
    sweep: bool = False,
    threshold: float | None = None,
    real_pdf: Path | None = None,
    real_golden: Path = REAL_GOLDEN,
) -> dict:
    from app.config import get_settings
    from app.ingest.chunker import CHUNKER_VERSION

    settings = get_settings()
    sources = {"fixture": (Path(pdf_path), Path(golden_path))}
    if real_pdf is not None:
        sources["real"] = (Path(real_pdf), Path(real_golden))

    started = time.perf_counter()
    runs = {}
    for name, (pdf, golden) in sources.items():
        manifest = store.ingest(pdf)
        records = run_questions(store, llm, manifest.doc_id, load_golden(golden), settings.top_k)
        runs[name] = (manifest, golden, records)

    if sweep:
        dev = [r for _, _, records in runs.values() for r in records if r.split == "dev"]
        decision, table = choose_threshold(dev)
    else:
        value = settings.confidence_threshold if threshold is None else threshold
        decision = {
            "source": "configured",
            "value": value,
            "explanation": f"{value:.2f} is the configured CONFIDENCE_THRESHOLD (no sweep run).",
        }
        table = []
    t = decision["value"]

    report = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm_model": llm.name,
        "embed_model": next(iter(runs.values()))[0].embed_model,
        "chunk_size": settings.chunk_size,
        "overlap": settings.chunk_overlap,
        "chunker_version": CHUNKER_VERSION,
        "top_k": settings.top_k,
        "machine": machine_info(),
        "run_seconds": round(time.perf_counter() - started, 1),
        "threshold": decision,
        "headline": "real" if "real" in runs else None,
        "datasets": {},
    }
    if table:
        report["sweep_dev"] = [asdict(m) for m in table]
    for name, (manifest, golden, records) in runs.items():
        report["datasets"][name] = {
            "document": {
                "filename": manifest.filename,
                "pages": manifest.pages,
                "chunks": manifest.chunks,
            },
            "golden": golden.name,
            "questions": _split_counts(records),
            "test": asdict(compute_metrics([r for r in records if r.split == "test"], t)),
            "dev": asdict(compute_metrics([r for r in records if r.split == "dev"], t)),
            "top_score": score_summary(records),
            "citation_failures": citation_failures(records, t),
            "refusal_failures": refusal_failures(records, t),
            "answer_failures": answer_failures(records, t),
            "records": [asdict(r) for r in records],
        }
    if out_dir is not None:
        write_results(report, Path(out_dir))
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


METRIC_ROWS = [
    ("Retrieval hit@5", "hit_at_5"),
    ("Answer accuracy, lenient (any gold keyword)", "answer_accuracy_lenient"),
    ("Answer accuracy, strict (all gold keywords)", "answer_accuracy_strict"),
    ("Answer accuracy, strict with correct citations", "answer_accuracy_cited"),
    ("Refusal accuracy (unanswerable)", "refusal_accuracy"),
    ("False-refusal rate (answerable)", "false_refusal_rate"),
    ("Citation precision, strict (all cited pages correct)", "citation_precision_strict"),
    ("Citation precision, per page", "citation_precision_per_page"),
]


def metrics_table(data: dict, with_dev: bool = True) -> str:
    test, dev = data["test"], data["dev"]
    head = "| Metric | Test |" + (" Dev (reference) |" if with_dev else "")
    rule = "| --- | --- |" + (" --- |" if with_dev else "")
    rows = [head, rule]

    def row(label: str, a: str, b: str) -> str:
        return f"| {label} | {a} |" + (f" {b} |" if with_dev else "")

    rows.append(
        row(
            "Questions (answerable / unanswerable)",
            f"{test['n_answerable']} / {test['n_unanswerable']}",
            f"{dev['n_answerable']} / {dev['n_unanswerable']}",
        )
    )
    rows += [row(label, _pct(test[key]), _pct(dev[key])) for label, key in METRIC_ROWS]
    rows.append(
        row(
            "Latency p50 / p95",
            f"{test['latency_p50_ms']} ms / {test['latency_p95_ms']} ms",
            f"{dev['latency_p50_ms']} ms / {dev['latency_p95_ms']} ms",
        )
    )
    return "\n".join(rows)


def summary_table(report: dict, dataset: str | None = None) -> str:
    """Test-split table for one question set (the headline set by default)."""
    name = dataset or report["headline"] or "fixture"
    data = report["datasets"][name]
    t = report["threshold"]
    lines = [
        metrics_table(data, with_dev=False),
        "",
        f"Test split of the {name} set: `{data['document']['filename']}` · threshold "
        f"{t['value']:.2f} ({t['source']}) · {report['llm_model']} · {report['embed_model']}"
        f" · {report['date'][:10]}",
    ]
    return "\n".join(lines)


def _failure_lines(failures: list[dict]) -> list[str]:
    lines = [
        "| id | split | question | gold pages | cited pages | cited per chunk | chunk spans"
        " | cause |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in failures:
        lines.append(
            f"| {f['id']} | {f['split']} | {_cell(f['question'])} | "
            f"{','.join(map(str, f['gold_pages']))} | "
            f"{','.join(map(str, f['cited_pages'])) or '-'} | {_groups(f['cited_chunks'])} | "
            f"{_groups(f.get('chunk_spans', []))} | {f['reason']} |"
        )
    return lines


def _groups(groups: list[list[int]]) -> str:
    return " ".join("[" + ",".join(map(str, g)) + "]" for g in groups) or "-"


def _refusal_failure_lines(failures: list[dict]) -> list[str]:
    lines = [
        "| id | split | question | top score | confidence | cited | answer |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in failures:
        lines.append(
            f"| {f['id']} | {f['split']} | {_cell(f['question'])} | {f['top_score']:.3f} | "
            f"{f['confidence']:.2f} | {','.join(map(str, f['cited_pages'])) or '-'} | "
            f"{_cell(f['answer'], 200)} |"
        )
    return lines


def _answer_failure_lines(failures: list[dict]) -> list[str]:
    lines = [
        "| id | split | question | top score | outcome | missing keywords | answer |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in failures:
        outcome = f"refused ({f['refused_by']})" if f["refused"] else "answered"
        lines.append(
            f"| {f['id']} | {f['split']} | {_cell(f['question'])} | {f['top_score']:.3f} | "
            f"{outcome} | {_cell(', '.join(f['missing_keywords'])) or '-'} | "
            f"{_cell(f['answer'], 200)} |"
        )
    return lines


def _cell(text: str, limit: int | None = None) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    return text[:limit] if limit else text


def _dataset_section(report: dict, name: str) -> list[str]:
    data = report["datasets"][name]
    t = report["threshold"]["value"]
    doc = data["document"]
    q = data["questions"]
    lines = [
        f"## {name.capitalize()} set: `{doc['filename']}`",
        "",
        f"{doc['pages']} pages, {doc['chunks']} chunks, questions from `{data['golden']}`. "
        f"Dev: {q['dev']['answerable']} answerable + {q['dev']['unanswerable']} unanswerable; "
        f"test: {q['test']['answerable']} + {q['test']['unanswerable']}.",
        "",
        metrics_table(data),
        "",
        "### Top retrieval score (what the confidence gate sees, all questions)",
        "",
        "| Questions | n | min | median | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for group in ("answerable", "unanswerable"):
        g = data["top_score"].get(group) or {}
        if g:
            lines.append(f"| {group} | {g['n']} | {g['min']} | {g['median']} | {g['max']} |")
    lines += [
        "",
        "### Strict answer accuracy by question kind (test split)",
        "",
        "| Kind | Accuracy |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {_pct(v)} |" for k, v in data["test"]["answer_accuracy_by_kind"].items()]
    failures = data["citation_failures"]
    lines += ["", f"### Citation failures at {t:.2f} (both splits)", ""]
    if failures:
        lines += _failure_lines(failures)
    else:
        lines.append("None: every answered question cited only gold pages.")
    refusals = data["refusal_failures"]
    lines += ["", f"### Refusal failures at {t:.2f}: unanswerable questions answered", ""]
    lines += _refusal_failure_lines(refusals) if refusals else ["None: every one was refused."]
    answers = data["answer_failures"]
    lines += ["", f"### Answer failures at {t:.2f}: strict check failed (both splits)", ""]
    lines += (
        _answer_failure_lines(answers)
        if answers
        else ["None: every answerable question passed the strict check."]
    )
    lines += [
        "",
        f"### Every question at {t:.2f}",
        "",
        "| id | split | kind | top score | refused | strict | lenient | cited | answer |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in data["records"]:
        rec = Record(**r)
        ok = (correct_at(rec, t), correct_at(rec, t, "any")) if rec.answerable else ("-", "-")
        lines.append(
            f"| {rec.id} | {rec.split} | {rec.kind} | {rec.top_score:.3f} | "
            f"{refused_at(rec, t)} | {ok[0]} | {ok[1]} | "
            f"{','.join(map(str, rec.cited_pages)) or '-'} | {_cell(rec.llm_answer, 120)} |"
        )
    return lines + [""]


def render_markdown(report: dict) -> str:
    lines = ["# Evaluation results", ""]
    if report["headline"]:
        lines += [
            f"Headline: the **{report['headline']}** set's test split.",
            "",
            summary_table(report),
            "",
        ]
    else:
        lines += [
            "No real-document question set in this run, so there is no headline number. The "
            "fixture is a generated handbook with planted facts; its numbers show the pipeline "
            "works, not how well it does on real documents. Add eval/golden_real.jsonl and pass "
            "--real-pdf.",
            "",
        ]
    t = report["threshold"]
    lines += ["## Confidence threshold", "", f"**{t['value']:.2f}** ({t['source']}).", ""]
    lines += [t["explanation"], ""]
    if t["source"] == "dev sweep":
        lines += [
            f"Chosen from {t['dev_questions']} dev questions only. Test questions were not "
            "looked at to pick it.",
            "",
        ]
    for name in ("real", "fixture"):
        if name in report["datasets"]:
            lines += _dataset_section(report, name)
    if "sweep_dev" in report:
        lines += [
            "## Threshold sweep (dev split only)",
            "",
            "| Threshold | Strict answer acc | Refusal acc | False refusals | (ans+ref)/2"
            " | Citation prec. (strict) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for m in report["sweep_dev"]:
            mark = " **(chosen)**" if m["threshold"] == t["value"] else ""
            lines.append(
                f"| {m['threshold']:.2f}{mark} | {_pct(m['answer_accuracy_strict'])} | "
                f"{_pct(m['refusal_accuracy'])} | {_pct(m['false_refusal_rate'])} | "
                f"{_pct(m['combined'])} | {_pct(m['citation_precision_strict'])} |"
            )
        lines.append("")
    m = report["machine"]
    lines += [
        "## Setup",
        "",
        f"- Chunks: {report['chunk_size']} chars, {report['overlap']} overlap, chunker"
        f" v{report['chunker_version']}, top_k {report['top_k']}",
        f"- Machine: {m['platform']}, {m['processor']}, {m['cpu_count']} CPUs"
        + (f", {m['ram_gb']} GB RAM" if "ram_gb" in m else ""),
        f"- Python {m['python']}, run time {report['run_seconds']} s",
        "",
    ]
    return "\n".join(lines)


def print_report(report: dict, out=None) -> None:
    out = out or sys.stdout  # resolved per call so redirected stdout is honoured
    t = report["threshold"]
    print(f"Threshold {t['value']:.2f} ({t['source']}): {t['explanation']}\n", file=out)
    for name, data in report["datasets"].items():
        headline = " (headline)" if name == report["headline"] else ""
        print(f"== {name} set{headline}: {data['document']['filename']} ==", file=out)
        print(metrics_table(data), file=out)
        failures = data["citation_failures"]
        print(f"\nCitation failures ({len(failures)}):", file=out)
        for f in failures:
            print(
                f"  {f['id']} [{f['split']}] {f['question']!r}: gold {f['gold_pages']}, "
                f"cited {f['cited_pages']} (per chunk {f['cited_chunks']}, chunk spans "
                f"{f.get('chunk_spans', [])}) -> {f['reason']}",
                file=out,
            )
        refusals = data["refusal_failures"]
        print(f"Refusal failures ({len(refusals)}):", file=out)
        for f in refusals:
            print(
                f"  {f['id']} [{f['split']}] {f['question']!r}: answered with confidence "
                f"{f['confidence']:.2f} (top score {f['top_score']:.3f}): {f['answer']!r}",
                file=out,
            )
        answers = data["answer_failures"]
        print(f"Answer failures ({len(answers)}):", file=out)
        for f in answers:
            outcome = f"refused by {f['refused_by']}" if f["refused"] else "answered"
            print(
                f"  {f['id']} [{f['split']}] {f['question']!r}: {outcome}, top score "
                f"{f['top_score']:.3f}, missing {f['missing_keywords']}: {f['answer']!r}",
                file=out,
            )
        print(file=out)
    if not report["headline"]:
        print("No real set: fixture numbers only, no headline.", file=out)


def write_results(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(report), encoding="utf-8")


def apply_threshold(value: float, config_path: Path = ROOT / "app" / "config.py") -> None:
    """Make the chosen threshold the default in app/config.py."""
    source = config_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(confidence_threshold: float = )[0-9.]+", rf"\g<1>{value:.2f}", source
    )
    if count != 1:
        raise RuntimeError("could not find confidence_threshold in config.py")
    config_path.write_text(updated, encoding="utf-8")


def update_readme(report: dict, readme: Path = ROOT / "README.md") -> None:
    if report["headline"] != "real":
        raise RuntimeError("the README headline must come from the real set: pass --real-pdf")
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise RuntimeError("README.md has no eval markers")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    link = "Full results, including the fixture set: [eval/results.md](eval/results.md)"
    block = f"{README_START}\n{summary_table(report)}\n\n{link}\n{README_END}"
    readme.write_text(head + block + tail, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep", action="store_true", help="pick the threshold on dev questions")
    parser.add_argument("--apply", action="store_true", help="write the chosen threshold to config")
    parser.add_argument(
        "--update-readme", action="store_true", help="refresh the README table (needs --real-pdf)"
    )
    parser.add_argument("--golden", type=Path, default=GOLDEN, help="fixture question set")
    parser.add_argument("--pdf", type=Path, default=FIXTURE, help="fixture PDF")
    parser.add_argument("--real-pdf", type=Path, help="your real PDF for the real question set")
    parser.add_argument("--real-golden", type=Path, default=REAL_GOLDEN)
    parser.add_argument("--out", type=Path, default=ROOT / "eval")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "eval-indexes")
    args = parser.parse_args(argv)

    if args.real_pdf is None:
        if args.update_readme:
            print("error: --update-readme needs the real set (--real-pdf PATH)", file=sys.stderr)
            return 2
        if args.real_golden.is_file():
            print(
                f"note: {args.real_golden.name} found but no --real-pdf given; real set skipped",
                file=sys.stderr,
            )
    else:
        for path in (args.real_pdf, args.real_golden):
            if not path.is_file():
                print(f"error: {path} not found", file=sys.stderr)
                return 2

    from app.config import get_settings
    from app.index.embedder import get_embedder
    from app.index.store import DocumentStore
    from app.qa.llm import OllamaLLM

    llm = OllamaLLM()
    if not llm.is_available():
        print(
            f"error: Ollama is not running at {llm.url} or '{llm.name}' is not pulled "
            f"(run: ollama serve && ollama pull {llm.name})",
            file=sys.stderr,
        )
        return 3
    store = DocumentStore(args.data_dir, get_embedder(get_settings().embed_model))
    report = run(
        store,
        llm,
        args.golden,
        args.pdf,
        args.out,
        sweep=args.sweep or args.apply,
        real_pdf=args.real_pdf,
        real_golden=args.real_golden,
    )
    print_report(report)
    if args.apply:
        apply_threshold(report["threshold"]["value"])
        print(f"CONFIDENCE_THRESHOLD default set to {report['threshold']['value']:.2f}")
    if args.update_readme:
        update_readme(report)
    print(f"wrote {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
