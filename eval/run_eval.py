"""Evaluate retrieval, answers, citations and refusals on the golden set.

    python -m eval.run_eval                 # metrics at the configured threshold
    python -m eval.run_eval --sweep         # + threshold sweep 0.20..0.60, picks the best
    python -m eval.run_eval --sweep --apply --update-readme

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
FIXTURE = ROOT / "tests" / "fixtures" / "handbook.pdf"
SWEEP = [round(0.20 + 0.05 * i, 2) for i in range(9)]  # 0.20 .. 0.60
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


@dataclass
class Metrics:
    threshold: float
    hit_at_5: float
    answer_accuracy: float
    refusal_accuracy: float
    false_refusal_rate: float
    citation_precision: float | None
    combined: float
    latency_p50_ms: int
    latency_p95_ms: int
    answer_accuracy_by_kind: dict[str, float] = field(default_factory=dict)


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


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
            )
        )
    return records


def refused_at(r: Record, threshold: float) -> bool:
    return r.top_score < threshold or r.llm_refused


def keywords_present(answer: str, keywords: list[str]) -> bool:
    """Every keyword must appear; "a|b" accepts either spelling (e.g. "two weeks|2 weeks")."""
    text = answer.lower()
    return all(any(alt.strip().lower() in text for alt in kw.split("|")) for kw in keywords)


def correct_at(r: Record, threshold: float) -> bool:
    return not refused_at(r, threshold) and keywords_present(r.llm_answer, r.gold_keywords)


def latency_at(r: Record, threshold: float) -> int:
    return r.retrieval_ms if r.top_score < threshold else r.answer_ms


def _mean(values: list[bool | float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(pct * len(ordered) + 0.5) - 1))]


def compute_metrics(records: list[Record], threshold: float) -> Metrics:
    ans = [r for r in records if r.answerable]
    unans = [r for r in records if not r.answerable]
    answered = [r for r in ans if not refused_at(r, threshold)]
    precision = (
        _mean([bool(r.cited_pages) and set(r.cited_pages) <= set(r.gold_pages) for r in answered])
        if answered
        else None
    )
    answer_acc = _mean([correct_at(r, threshold) for r in ans])
    refusal_acc = _mean([refused_at(r, threshold) for r in unans])
    latencies = [latency_at(r, threshold) for r in records]
    kinds = sorted({r.kind for r in ans})
    return Metrics(
        threshold=threshold,
        hit_at_5=_mean([any(set(p) & set(r.gold_pages) for p in r.top5_pages) for r in ans]),
        answer_accuracy=answer_acc,
        refusal_accuracy=refusal_acc,
        false_refusal_rate=_mean([refused_at(r, threshold) for r in ans]),
        citation_precision=precision,
        combined=round((answer_acc + refusal_acc) / 2, 4),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        answer_accuracy_by_kind={
            kind: _mean([correct_at(r, threshold) for r in ans if r.kind == kind]) for kind in kinds
        },
    )


def pick_threshold(sweep: list[Metrics]) -> Metrics:
    """Best (answer + refusal accuracy) / 2; ties go to the lower false-refusal rate."""
    return max(sweep, key=lambda m: (m.combined, -m.false_refusal_rate, -m.threshold))


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


def run(
    store,
    llm,
    golden_path: Path = GOLDEN,
    pdf_path: Path = FIXTURE,
    out_dir: Path | None = ROOT / "eval",
    sweep: bool = False,
    threshold: float | None = None,
) -> dict:
    from app.config import get_settings

    settings = get_settings()
    threshold = settings.confidence_threshold if threshold is None else threshold
    manifest = store.ingest(pdf_path)
    items = load_golden(golden_path)

    started = time.perf_counter()
    records = run_questions(store, llm, manifest.doc_id, items, settings.top_k)
    report = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "document": {
            "filename": manifest.filename,
            "pages": manifest.pages,
            "chunks": manifest.chunks,
        },
        "llm_model": llm.name,
        "embed_model": manifest.embed_model,
        "chunk_size": manifest.chunk_size,
        "overlap": manifest.overlap,
        "top_k": settings.top_k,
        "questions": {"total": len(records), "answerable": sum(r.answerable for r in records)},
        "machine": machine_info(),
        "run_seconds": round(time.perf_counter() - started, 1),
        "configured": asdict(compute_metrics(records, threshold)),
        "top_score": score_summary(records),
        "records": [asdict(r) for r in records],
    }
    if sweep:
        table = [compute_metrics(records, t) for t in SWEEP]
        report["sweep"] = [asdict(m) for m in table]
        report["best"] = asdict(pick_threshold(table))
    if out_dir is not None:
        write_results(report, Path(out_dir))
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def summary_table(report: dict) -> str:
    m = report.get("best") or report["configured"]
    label = "chosen by sweep" if "best" in report else "configured"
    rows = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Confidence threshold ({label}) | {m['threshold']:.2f} |",
        f"| Retrieval hit@5 | {_pct(m['hit_at_5'])} |",
        f"| Answer accuracy | {_pct(m['answer_accuracy'])} |",
        f"| Refusal accuracy (unanswerable) | {_pct(m['refusal_accuracy'])} |",
        f"| False-refusal rate (answerable) | {_pct(m['false_refusal_rate'])} |",
        f"| Citation precision | {_pct(m['citation_precision'])} |",
        f"| Latency p50 / p95 | {m['latency_p50_ms']} ms / {m['latency_p95_ms']} ms |",
    ]
    q = report["questions"]
    rows.append(
        f"\n{q['total']} questions ({q['answerable']} answerable) on "
        f"`{report['document']['filename']}` · {report['llm_model']} · {report['embed_model']}"
        f" · {report['date'][:10]}"
    )
    return "\n".join(rows)


def render_markdown(report: dict) -> str:
    lines = ["# Evaluation results", "", summary_table(report), ""]
    s = report["top_score"]
    lines += [
        "## Top retrieval score (what the confidence gate sees)",
        "",
        "| Questions | n | min | median | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for group in ("answerable", "unanswerable"):
        g = s.get(group) or {}
        if g:
            lines.append(f"| {group} | {g['n']} | {g['min']} | {g['median']} | {g['max']} |")
    if "sweep" in report:
        lines += [
            "",
            "## Threshold sweep",
            "",
            "| Threshold | Answer acc | Refusal acc | False refusals | (ans+ref)/2"
            " | Citation prec. |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for m in report["sweep"]:
            mark = " **(chosen)**" if m["threshold"] == report["best"]["threshold"] else ""
            lines.append(
                f"| {m['threshold']:.2f}{mark} | {_pct(m['answer_accuracy'])} | "
                f"{_pct(m['refusal_accuracy'])} | {_pct(m['false_refusal_rate'])} | "
                f"{_pct(m['combined'])} | {_pct(m['citation_precision'])} |"
            )
    by_kind = (report.get("best") or report["configured"])["answer_accuracy_by_kind"]
    lines += ["", "## Answer accuracy by question kind", "", "| Kind | Accuracy |", "| --- | --- |"]
    lines += [f"| {kind} | {_pct(v)} |" for kind, v in by_kind.items()]
    t = (report.get("best") or report["configured"])["threshold"]
    lines += [
        "",
        f"## Every question (at threshold {t:.2f})",
        "",
        "| id | kind | top score | refused | correct | cited | answer |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["records"]:
        rec = Record(**r)
        answer = rec.llm_answer.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {rec.id} | {rec.kind} | {rec.top_score:.3f} | {refused_at(rec, t)} | "
            f"{correct_at(rec, t) if rec.answerable else '-'} | "
            f"{','.join(map(str, rec.cited_pages)) or '-'} | {answer[:120]} |"
        )
    m = report["machine"]
    lines += [
        "",
        "## Setup",
        "",
        f"- Chunks: {report['chunk_size']} chars, {report['overlap']} overlap,"
        f" top_k {report['top_k']}",
        f"- Machine: {m['platform']}, {m['processor']}, {m['cpu_count']} CPUs"
        + (f", {m['ram_gb']} GB RAM" if "ram_gb" in m else ""),
        f"- Python {m['python']}, run time {report['run_seconds']} s",
        "",
    ]
    return "\n".join(lines)


def write_results(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(report), encoding="utf-8")


def apply_threshold(value: float, config_path: Path = ROOT / "app" / "config.py") -> None:
    """Make the swept threshold the default in app/config.py."""
    source = config_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(confidence_threshold: float = )[0-9.]+", rf"\g<1>{value:.2f}", source
    )
    if count != 1:
        raise RuntimeError("could not find confidence_threshold in config.py")
    config_path.write_text(updated, encoding="utf-8")


def update_readme(report: dict, readme: Path = ROOT / "README.md") -> None:
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise RuntimeError("README.md has no eval markers")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    link = "Full results: [eval/results.md](eval/results.md)"
    block = f"{README_START}\n{summary_table(report)}\n\n{link}\n{README_END}"
    readme.write_text(head + block + tail, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="sweep thresholds 0.20..0.60")
    parser.add_argument("--apply", action="store_true", help="write the chosen threshold to config")
    parser.add_argument("--update-readme", action="store_true", help="refresh the README table")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--pdf", type=Path, default=FIXTURE)
    parser.add_argument("--out", type=Path, default=ROOT / "eval")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "eval-indexes")
    args = parser.parse_args(argv)

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
    report = run(store, llm, args.golden, args.pdf, args.out, sweep=args.sweep or args.apply)
    print(summary_table(report))
    if args.apply:
        apply_threshold(report["best"]["threshold"])
        print(f"CONFIDENCE_THRESHOLD default set to {report['best']['threshold']:.2f}")
    if args.update_readme:
        update_readme(report)
    print(f"wrote {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
