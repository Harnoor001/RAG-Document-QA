import pytest


@pytest.fixture(scope="session")
def minilm_store(tmp_path_factory, handbook):
    """The handbook indexed with the real MiniLM model (downloaded on first use)."""
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
    from app.index.embedder import MiniLMEmbedder
    from app.index.store import DocumentStore

    embedder = MiniLMEmbedder()
    try:
        assert embedder.dim == 384  # loads the weights (downloads them on first run)
    except Exception as exc:
        pytest.skip(f"cannot load {embedder.name} (no cached weights and no network?): {exc}")
    store = DocumentStore(tmp_path_factory.mktemp("idx"), embedder)
    return store, store.ingest(handbook)


@pytest.fixture(scope="session")
def ollama():
    from app.qa.llm import OllamaLLM

    llm = OllamaLLM()
    if not llm.is_available():
        pytest.skip(
            f"Ollama is not running at {llm.url} or model '{llm.name}' is not pulled "
            f"(run: ollama serve && ollama pull {llm.name})"
        )
    return llm


@pytest.fixture(scope="session")
def ask(ollama, minilm_store):  # check Ollama first: it is the cheaper probe
    from app.qa.answerer import Answerer
    from app.qa.retriever import Retriever

    store, manifest = minilm_store
    answerer = Answerer(Retriever(store), ollama)
    return lambda question: answerer.answer(manifest.doc_id, question)
