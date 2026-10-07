import json

import pytest

from app.index.embedder import HashEmbedder
from app.index.store import DocumentStore
from app.qa.llm import FakeLLM
from eval import run_eval
from eval.run_eval import (
    SWEEP,
    Record,
    apply_threshold,
    compute_metrics,
    keywords_present,
    load_golden,
    pick_threshold,
    run,
    update_readme,
)

MINI = [
    {"id": "refund", "question": "Refunds are accepted within 30 days of purchase",
     "gold_pages": [12], "gold_keywords": ["30 days"], "kind": "direct"},
    {"id": "warranty", "question": "submit form WC-2 to the Customer Service Desk",
     "gold_pages": [20, 21], "gold_keywords": ["WC-2"], "kind": "multi_page"},
    {"id": "vague", "question": "staff discount", "gold_pages": [18],
     "gold_keywords": ["15%"], "kind": "vague"},
    {"id": "u-cup", "question": "Who won the 2018 FIFA World Cup?", "unanswerable": True},
    {"id": "u-ceo", "question": "What is the CEO's salary?", "unanswerable": True},
]  # fmt: skip

METRIC_KEYS = {
    "threshold", "hit_at_5", "answer_accuracy", "refusal_accuracy", "false_refusal_rate",
    "citation_precision", "combined", "latency_p50_ms", "latency_p95_ms",
    "answer_accuracy_by_kind",
}  # fmt: skip


def fake_llm():
    def reply(system, user):
        if "WC-2" in user and "warranty" in user.lower() and "Question: submit" in user:
            block = user.split("(pages 20-21)")[0].rsplit("[C", 1)[1].split("]")[0]
            return f"Submit form WC-2 [C{block}]."
        if "Question: Refunds" in user:
            return "Within 30 days [C1]."
        return "I couldn't find this in the document."

    return FakeLLM(reply)


@pytest.fixture
def mini_golden(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in MINI) + "\n")
    return path


def test_t3_11_harness_produces_every_metric(tmp_path, mini_golden, handbook):
    store = DocumentStore(tmp_path / "idx", HashEmbedder())
    out = tmp_path / "out"
    report = run(store, fake_llm(), mini_golden, handbook, out, sweep=True, threshold=0.35)

    assert set(report["configured"]) == METRIC_KEYS
    assert set(report["best"]) == METRIC_KEYS
    assert [m["threshold"] for m in report["sweep"]] == SWEEP
    assert report["questions"] == {"total": 5, "answerable": 3}
    assert set(report["top_score"]) == {"answerable", "unanswerable"}
    assert report["top_score"]["unanswerable"]["n"] == 2
    assert len(report["records"]) == 5 and report["llm_model"] == "fake"

    md = (out / "results.md").read_text()
    assert "Answer accuracy" in md and "False-refusal rate" in md and "(chosen)" in md
    assert "Top retrieval score" in md and "| refund |" in md
    saved = json.loads((out / "results.json").read_text())
    assert saved["best"]["threshold"] == report["best"]["threshold"]


def test_gate_is_simulated_exactly_from_top_score():
    rec = dict(kind="direct", question="q", answerable=True, gold_pages=[12],
               gold_keywords=["30 days"], top5_pages=[[12]], retrieval_ms=5,
               llm_answer="Within 30 days [p.12].", llm_refused=False, cited_pages=[12],
               confidence=0.5, answer_ms=900, load_ms=0)  # fmt: skip
    answerable = [Record(id="a", top_score=0.50, **rec), Record(id="b", top_score=0.30, **rec)]
    unans = dict(rec, answerable=False, gold_pages=[], gold_keywords=[], kind="unanswerable")
    records = answerable + [
        Record(id="u1", top_score=0.32, **dict(unans, llm_refused=True)),
        Record(id="u2", top_score=0.25, **unans),  # the LLM wrongly answered this one
    ]
    low = compute_metrics(records, 0.20)
    assert low.answer_accuracy == 1.0 and low.false_refusal_rate == 0.0
    assert low.refusal_accuracy == 0.5  # u2 slipped through
    mid = compute_metrics(records, 0.35)
    assert mid.answer_accuracy == 0.5 and mid.false_refusal_rate == 0.5
    assert mid.refusal_accuracy == 1.0 and mid.citation_precision == 1.0
    assert mid.latency_p95_ms == 900 and mid.latency_p50_ms == 5
    assert mid.answer_accuracy_by_kind == {"direct": 0.5}
    best = pick_threshold([low, mid])
    assert best.threshold == 0.20  # 0.75 vs 0.75: fewer false refusals wins the tie


def test_citation_precision_needs_pages_within_gold():
    base = dict(kind="direct", question="q", answerable=True, gold_pages=[12],
                gold_keywords=["x"], top_score=0.9, top5_pages=[[12]], retrieval_ms=1,
                llm_answer="x", llm_refused=False, confidence=0.9, answer_ms=1,
                load_ms=0)  # fmt: skip
    records = [
        Record(id="ok", cited_pages=[12], **base),
        Record(id="off", cited_pages=[12, 13], **base),
    ]
    assert compute_metrics(records, 0.3).citation_precision == 0.5
    assert compute_metrics(records, 0.95).citation_precision is None  # nothing answered


def test_keyword_alternatives():
    assert keywords_present("Submit it two weeks ahead.", ["two weeks|2 weeks"])
    assert keywords_present("Submit it 2 Weeks ahead.", ["two weeks|2 weeks"])
    assert not keywords_present("Within 30 days.", ["30 days", "unused"])


def test_golden_set_shape():
    items = load_golden(run_eval.GOLDEN)
    answerable = [i for i in items if not i.get("unanswerable")]
    assert len(items) == 40 and len(answerable) == 30
    kinds = {i["kind"] for i in answerable}
    assert {"multi_page", "table", "vague", "paraphrase"} <= kinds
    assert sum(i["kind"] == "vague" for i in answerable) >= 5
    assert any(i["gold_pages"] == [20, 21] for i in answerable)
    assert len({i["id"] for i in items}) == 40
    assert all(i["gold_keywords"] and i["gold_pages"] for i in answerable)


def test_apply_threshold_rewrites_config(tmp_path):
    config = tmp_path / "config.py"
    config.write_text("x = 1\n    confidence_threshold: float = 0.35\n")
    apply_threshold(0.3, config)
    assert "confidence_threshold: float = 0.30" in config.read_text()
    config.write_text("nothing here\n")
    with pytest.raises(RuntimeError):
        apply_threshold(0.3, config)


def test_update_readme_replaces_marked_block(tmp_path, mini_golden, handbook):
    store = DocumentStore(tmp_path / "idx", HashEmbedder())
    report = run(store, fake_llm(), mini_golden, handbook, None, sweep=False, threshold=0.35)
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n{run_eval.README_START}\nSTALE\n{run_eval.README_END}\nrest\n")
    update_readme(report, readme)
    text = readme.read_text()
    assert "STALE" not in text and "Answer accuracy" in text and text.endswith("rest\n")
    readme.write_text("no markers")
    with pytest.raises(RuntimeError):
        update_readme(report, readme)


def test_main_exits_cleanly_when_ollama_is_down(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    assert run_eval.main([]) == 3
    assert "ollama serve" in capsys.readouterr().err


def test_main_runs_with_injected_components(monkeypatch, tmp_path, mini_golden, handbook):
    monkeypatch.setenv("EMBED_MODEL", "hash")
    monkeypatch.setattr("app.qa.llm.OllamaLLM.is_available", lambda self: True)
    llm = fake_llm()
    monkeypatch.setattr("app.qa.llm.OllamaLLM.generate", lambda self, s, u: llm.generate(s, u))
    readme = tmp_path / "README.md"
    readme.write_text(f"{run_eval.README_START}\n{run_eval.README_END}\n")
    config = tmp_path / "config.py"
    config.write_text("confidence_threshold: float = 0.35\n")
    monkeypatch.setattr(run_eval, "apply_threshold", lambda v: apply_threshold(v, config))
    monkeypatch.setattr(run_eval, "update_readme", lambda r: update_readme(r, readme))
    args = ["--golden", str(mini_golden), "--pdf", str(handbook), "--out", str(tmp_path / "o"),
            "--data-dir", str(tmp_path / "d"), "--apply", "--update-readme"]  # fmt: skip
    assert run_eval.main(args) == 0
    assert (tmp_path / "o" / "results.md").is_file()
    assert "Answer accuracy" in readme.read_text()
    assert "0.35" not in config.read_text() or "0.35" in config.read_text()
