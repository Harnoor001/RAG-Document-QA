import json

import pytest

from app.index.embedder import HashEmbedder
from app.index.store import DocumentStore
from app.qa.llm import FakeLLM
from eval import run_eval
from eval.run_eval import (
    FLOOR,
    SWEEP,
    Record,
    apply_threshold,
    assign_splits,
    choose_threshold,
    citation_failures,
    compute_metrics,
    keywords_present,
    load_golden,
    pick_threshold,
    run,
    update_readme,
)

MINI = [
    {"id": "refund", "question": "Refunds are accepted within 30 days of purchase",
     "gold_pages": [12], "gold_keywords": ["30 days"], "kind": "direct", "split": "dev"},
    {"id": "warranty", "question": "submit form WC-2 to the Customer Service Desk",
     "gold_pages": [20, 21], "gold_keywords": ["WC-2"], "kind": "multi_page", "split": "test"},
    {"id": "vague", "question": "staff discount", "gold_pages": [18],
     "gold_keywords": ["15%"], "kind": "vague", "split": "test"},
    {"id": "u-cup", "question": "Who won the 2018 FIFA World Cup?", "unanswerable": True,
     "split": "dev"},
    {"id": "u-ceo", "question": "What is the CEO's salary?", "unanswerable": True,
     "split": "test"},
]  # fmt: skip

METRIC_KEYS = {
    "threshold", "n_answerable", "n_unanswerable", "hit_at_5", "answer_accuracy_lenient",
    "answer_accuracy_strict", "answer_accuracy_cited", "refusal_accuracy", "false_refusal_rate",
    "citation_precision_strict", "citation_precision_per_page", "combined", "latency_p50_ms",
    "latency_p95_ms", "answer_accuracy_by_kind",
}  # fmt: skip

BASE = dict(kind="direct", question="q", answerable=True, gold_pages=[12],
            gold_keywords=["30 days"], top_score=0.9, top5_pages=[[12]], retrieval_ms=5,
            llm_answer="Within 30 days [p.12].", llm_refused=False, cited_pages=[12],
            confidence=0.5, answer_ms=900, load_ms=0)  # fmt: skip


def rec(id, **overrides):
    return Record(id=id, **dict(BASE, **overrides))


def unans(id, **overrides):
    fields = dict(answerable=False, gold_pages=[], gold_keywords=[], kind="unanswerable")
    return rec(id, **dict(fields, **overrides))


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
    report = run(store, fake_llm(), mini_golden, handbook, out, sweep=True)

    fixture = report["datasets"]["fixture"]
    assert set(fixture["test"]) == METRIC_KEYS and set(fixture["dev"]) == METRIC_KEYS
    assert [m["threshold"] for m in report["sweep_dev"]] == SWEEP
    assert report["threshold"]["source"] == "dev sweep"
    assert report["threshold"]["value"] >= FLOOR or "clears" in report["threshold"]["explanation"]
    assert fixture["questions"] == {
        "dev": {"answerable": 1, "unanswerable": 1},
        "test": {"answerable": 2, "unanswerable": 1},
    }
    assert set(fixture["top_score"]) == {"answerable", "unanswerable"}
    assert fixture["top_score"]["unanswerable"]["n"] == 2
    assert len(fixture["records"]) == 5 and report["llm_model"] == "fake"
    assert report["headline"] is None and "real" not in report["datasets"]

    md = (out / "results.md").read_text()
    assert "no headline number" in md and "Threshold sweep (dev split only)" in md
    assert "Answer accuracy, strict" in md and "Answer accuracy, lenient" in md
    assert "Citation precision, per page" in md and "Citation failures" in md
    assert "False-refusal rate" in md and "(chosen)" in md and "| refund | dev |" in md
    saved = json.loads((out / "results.json").read_text())
    assert saved["threshold"] == report["threshold"]


def test_sweep_never_sees_test_questions(tmp_path, mini_golden, handbook, monkeypatch):
    seen = []

    def spy(dev):
        seen.extend(dev)
        return choose_threshold(dev)

    monkeypatch.setattr(run_eval, "choose_threshold", spy)
    store = DocumentStore(tmp_path / "idx", HashEmbedder())
    run(store, fake_llm(), mini_golden, handbook, None, sweep=True, real_pdf=handbook,
        real_golden=mini_golden)  # fmt: skip
    assert seen and {r.split for r in seen} == {"dev"}
    assert len(seen) == 4  # 2 dev questions from each of the two sets


def test_real_set_is_reported_separately_and_is_the_headline(tmp_path, mini_golden, handbook):
    store = DocumentStore(tmp_path / "idx", HashEmbedder())
    out = tmp_path / "out"
    report = run(store, fake_llm(), mini_golden, handbook, out, threshold=0.25,
                 real_pdf=handbook, real_golden=mini_golden)  # fmt: skip
    assert report["headline"] == "real"
    assert set(report["datasets"]) == {"fixture", "real"}
    assert report["threshold"] == {
        "source": "configured",
        "value": 0.25,
        "explanation": "0.25 is the configured CONFIDENCE_THRESHOLD (no sweep run).",
    }
    md = (out / "results.md").read_text()
    assert "Headline: the **real** set's test split" in md
    assert "## Real set:" in md and "## Fixture set:" in md
    assert "Test split of the real set" in run_eval.summary_table(report)


def test_gate_is_simulated_exactly_from_top_score():
    records = [
        rec("a", top_score=0.50),
        rec("b", top_score=0.30),
        unans("u1", top_score=0.32, llm_refused=True),
        unans("u2", top_score=0.25),  # the LLM wrongly answered this one
    ]
    low = compute_metrics(records, 0.20)
    assert low.answer_accuracy_strict == 1.0 and low.false_refusal_rate == 0.0
    assert low.refusal_accuracy == 0.5  # u2 slipped through
    mid = compute_metrics(records, 0.35)
    assert mid.answer_accuracy_strict == 0.5 and mid.false_refusal_rate == 0.5
    assert mid.refusal_accuracy == 1.0 and mid.citation_precision_strict == 1.0
    assert mid.latency_p95_ms == 900 and mid.latency_p50_ms == 5
    assert mid.answer_accuracy_by_kind == {"direct": 0.5}
    assert (mid.n_answerable, mid.n_unanswerable) == (2, 2)
    best = pick_threshold([low, mid])
    assert best.threshold == 0.20  # 0.75 vs 0.75: fewer false refusals wins the tie


def test_strict_and_lenient_answer_accuracy():
    records = [
        rec("both", gold_keywords=["30 days", "unused"], llm_answer="30 days, unused."),
        rec("one", gold_keywords=["30 days", "unused"], llm_answer="Within 30 days."),
        rec("none", gold_keywords=["30 days", "unused"], llm_answer="Ask a manager."),
    ]
    m = compute_metrics(records, 0.3)
    assert m.answer_accuracy_strict == round(1 / 3, 4)
    assert m.answer_accuracy_lenient == round(2 / 3, 4)


def test_citation_precision_strict_and_per_page():
    records = [
        rec("ok", cited_pages=[12], cited_chunks=[[12]]),
        rec("straddle", cited_pages=[12, 13], cited_chunks=[[12, 13]]),
        rec("extra", cited_pages=[12, 30], cited_chunks=[[12], [30]]),
        rec("bare", cited_pages=[], cited_chunks=[]),
    ]
    m = compute_metrics(records, 0.3)
    assert m.citation_precision_strict == 0.25
    assert m.citation_precision_per_page == 0.5  # (1 + 0.5 + 0.5 + 0) / 4
    assert m.answer_accuracy_cited == 0.25
    assert compute_metrics(records, 0.95).citation_precision_strict is None  # nothing answered

    failures = {f["id"]: f for f in citation_failures(records, 0.3)}
    assert set(failures) == {"straddle", "extra", "bare"}
    assert failures["straddle"]["reason"] == "multi-page chunk (gold + other page)"
    assert failures["extra"]["reason"] == "extra chunk (no gold page)"
    assert failures["bare"]["reason"] == "no citation"
    assert failures["extra"]["gold_pages"] == [12] and failures["extra"]["cited_pages"] == [12, 30]
    assert failures["extra"]["question"] == "q"


def _dev(scores_ok, scores_bad):
    """Answerable dev questions at the given top scores, unanswerable ones the LLM answers."""
    return [rec(f"a{i}", top_score=s) for i, s in enumerate(scores_ok)] + [
        unans(f"u{i}", top_score=s) for i, s in enumerate(scores_bad)
    ]


def test_floor_kept_when_lower_threshold_gains_little():
    # one valid question scores 0.22: 0.20 answers it, 0.25 refuses it -> ~3 points on (ans+ref)/2
    dev = _dev([0.22] + [0.6] * 14, [0.1] * 5)
    decision, table = choose_threshold(dev)
    assert decision["dev_best"] == 0.20 and decision["value"] == FLOOR
    assert "not a clear gain" in decision["explanation"]
    assert [m.threshold for m in table] == SWEEP


def test_lower_threshold_kept_when_dev_gain_is_clear():
    dev = _dev([0.22] * 5 + [0.6] * 5, [0.1] * 5)
    decision, _ = choose_threshold(dev)
    assert decision["value"] == 0.20 and "clears" in decision["explanation"]


def test_threshold_at_or_above_floor_is_used_as_is():
    dev = _dev([0.6] * 5, [0.1, 0.3, 0.38])
    decision, _ = choose_threshold(dev)
    assert decision["value"] == 0.40 and "at or above" in decision["explanation"]
    with pytest.raises(ValueError):
        choose_threshold([])


def test_keyword_alternatives():
    assert keywords_present("Submit it two weeks ahead.", ["two weeks|2 weeks"])
    assert keywords_present("Submit it 2 Weeks ahead.", ["two weeks|2 weeks"])
    assert not keywords_present("Within 30 days.", ["30 days", "unused"])
    assert keywords_present("Within 30 days.", ["30 days", "unused"], require="any")


def test_missing_splits_alternate_within_answerable_and_unanswerable():
    items = [
        {"id": "a1"},
        {"id": "a2"},
        {"id": "u1", "unanswerable": True},
        {"id": "a3", "split": "test"},
        {"id": "u2", "unanswerable": True},
        {"id": "a4"},
    ]
    splits = [i["split"] for i in assign_splits(items)]
    assert splits == ["dev", "test", "dev", "test", "test", "dev"]
    with pytest.raises(ValueError):
        assign_splits([{"id": "x", "split": "train"}])


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
    # held-out split written in the file: 15/15 answerable, 5/5 unanswerable
    for split in ("dev", "test"):
        assert sum(i["split"] == split for i in answerable) == 15
        assert sum(i["split"] == split for i in items if i.get("unanswerable")) == 5
    for kind in kinds:
        assert {i["split"] for i in answerable if i["kind"] == kind} == {"dev", "test"}


def test_real_golden_example_is_valid():
    items = load_golden(run_eval.ROOT / "eval" / "golden_real.example.jsonl")
    assert {i["split"] for i in items} == {"dev", "test"}
    assert any(i.get("unanswerable") for i in items)


def test_apply_threshold_rewrites_config(tmp_path):
    config = tmp_path / "config.py"
    config.write_text("x = 1\n    confidence_threshold: float = 0.35\n")
    apply_threshold(0.3, config)
    assert "confidence_threshold: float = 0.30" in config.read_text()
    config.write_text("nothing here\n")
    with pytest.raises(RuntimeError):
        apply_threshold(0.3, config)


def test_update_readme_needs_real_set(tmp_path, mini_golden, handbook):
    store = DocumentStore(tmp_path / "idx", HashEmbedder())
    readme = tmp_path / "README.md"
    original = f"# T\n{run_eval.README_START}\nSTALE\n{run_eval.README_END}\nrest\n"
    readme.write_text(original)
    fixture_only = run(store, fake_llm(), mini_golden, handbook, None, threshold=0.35)
    with pytest.raises(RuntimeError, match="real set"):
        update_readme(fixture_only, readme)
    assert readme.read_text() == original

    report = run(store, fake_llm(), mini_golden, handbook, None, threshold=0.35,
                 real_pdf=handbook, real_golden=mini_golden)  # fmt: skip
    update_readme(report, readme)
    text = readme.read_text()
    assert "STALE" not in text and "Test split of the real set" in text
    assert text.endswith("rest\n")
    readme.write_text("no markers")
    with pytest.raises(RuntimeError):
        update_readme(report, readme)


def test_main_exits_cleanly_when_ollama_is_down(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    assert run_eval.main([]) == 3
    assert "ollama serve" in capsys.readouterr().err


def test_main_rejects_readme_update_without_real_set(capsys, tmp_path):
    assert run_eval.main(["--update-readme"]) == 2
    assert "--real-pdf" in capsys.readouterr().err
    assert run_eval.main(["--real-pdf", str(tmp_path / "missing.pdf")]) == 2
    assert "not found" in capsys.readouterr().err


def test_main_runs_with_injected_components(monkeypatch, tmp_path, mini_golden, handbook, capsys):
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
    common = ["--golden", str(mini_golden), "--pdf", str(handbook), "--out", str(tmp_path / "o"),
              "--data-dir", str(tmp_path / "d")]  # fmt: skip

    # a real question set on disk but no --real-pdf: fixture only, with a note
    monkeypatch.setattr(run_eval, "REAL_GOLDEN", mini_golden)
    assert run_eval.main(common) == 0
    captured = capsys.readouterr()
    assert "real set skipped" in captured.err and "Citation failures" in captured.out

    args = common + ["--real-pdf", str(handbook), "--apply", "--update-readme"]
    assert run_eval.main(args) == 0
    assert (tmp_path / "o" / "results.md").is_file()
    assert "Test split of the real set" in readme.read_text()
    chosen = json.loads((tmp_path / "o" / "results.json").read_text())["threshold"]["value"]
    assert f"confidence_threshold: float = {chosen:.2f}" in config.read_text()
