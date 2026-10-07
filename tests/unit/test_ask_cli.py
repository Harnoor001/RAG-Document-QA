import pytest

from app.ask import main
from app.index.embedder import HashEmbedder
from app.index.store import DocumentStore
from app.qa.llm import FakeLLM, LLMUnavailableError

REFUND = "Refunds are accepted within 30 days of purchase as long as the item is unused."


@pytest.fixture
def doc(monkeypatch, tmp_path, handbook):
    monkeypatch.setenv("EMBED_MODEL", "hash")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.25")  # hash scores run lower than MiniLM's
    store = DocumentStore(tmp_path, HashEmbedder())
    return tmp_path, store.ingest(handbook).doc_id


def test_cli_prints_answer_confidence_and_pages(doc, capsys):
    data_dir, doc_id = doc
    llm = FakeLLM("Within 30 days if unused [C1].")
    code = main([doc_id[:12], REFUND, "--data-dir", str(data_dir)], llm=llm)
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "Within 30 days if unused [p.12]."
    assert lines[1].startswith("confidence 0.") and "1 source · local LLM (fake)" in lines[1]
    assert lines[2] == "cited: p.12"


def test_cli_refusal(doc, capsys):
    data_dir, doc_id = doc
    llm = FakeLLM("never")
    assert (
        main([doc_id, "Who won the 2018 FIFA World Cup?", "--data-dir", str(data_dir)], llm=llm)
        == 0
    )
    out = capsys.readouterr().out
    assert "couldn't find this" in out and "refused" in out
    assert llm.calls == []


def test_cli_unknown_document(doc, capsys):
    data_dir, _ = doc
    assert main(["ffff", "q", "--data-dir", str(data_dir)], llm=FakeLLM()) == 2
    assert "no indexed document" in capsys.readouterr().err


def test_cli_ollama_down_is_one_line_error(doc, capsys):
    data_dir, doc_id = doc

    class DownLLM(FakeLLM):
        def generate(self, system, user):
            raise LLMUnavailableError("Cannot reach Ollama at http://localhost:11434.")

    assert main([doc_id, REFUND, "--data-dir", str(data_dir)], llm=DownLLM()) == 3
    err = capsys.readouterr().err
    assert err.strip() == "error: Cannot reach Ollama at http://localhost:11434."
