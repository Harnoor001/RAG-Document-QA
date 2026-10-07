"""T3.13: the evaluation harness runs end to end on the real stack and writes its report."""

import pytest

pytestmark = pytest.mark.integration


def test_t3_13_eval_writes_results(ollama, minilm_store, handbook, tmp_path):
    from eval.run_eval import run

    store, _ = minilm_store
    report = run(store, ollama, pdf_path=handbook, out_dir=tmp_path, sweep=True)
    assert (tmp_path / "results.md").is_file() and (tmp_path / "results.json").is_file()
    assert report["questions"] == {"total": 40, "answerable": 30}
    best = report["best"]
    print(
        f"T3.13 threshold {best['threshold']:.2f}: hit@5 {best['hit_at_5']:.1%}, "
        f"answer acc {best['answer_accuracy']:.1%}, refusal acc {best['refusal_accuracy']:.1%}, "
        f"false refusals {best['false_refusal_rate']:.1%}"
    )
