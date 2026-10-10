"""T3.13: the evaluation harness runs end to end on the real stack and writes its report."""

import pytest

pytestmark = pytest.mark.integration


def test_t3_13_eval_writes_results(ollama, minilm_store, handbook, tmp_path):
    from eval.run_eval import FLOOR, print_report, run

    store, _ = minilm_store
    report = run(store, ollama, pdf_path=handbook, out_dir=tmp_path, sweep=True)
    assert (tmp_path / "results.md").is_file() and (tmp_path / "results.json").is_file()
    fixture = report["datasets"]["fixture"]
    assert fixture["questions"] == {
        "dev": {"answerable": 15, "unanswerable": 5},
        "test": {"answerable": 15, "unanswerable": 5},
    }
    t = report["threshold"]
    assert t["source"] == "dev sweep" and t["dev_questions"] == 20
    assert t["value"] >= FLOOR or t["dev_best_combined"] - t["floor_combined"] >= 0.05
    print_report(report)
