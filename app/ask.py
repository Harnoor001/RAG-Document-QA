"""python -m app.ask <doc_id> "<question>" -> a cited answer from the local LLM."""

from __future__ import annotations

import argparse
import sys

from app.index.store import DocumentNotFoundError, DocumentStore
from app.qa.answerer import Answer, Answerer
from app.qa.llm import LLM, LLMUnavailableError, OllamaLLM
from app.qa.prompt import format_pages
from app.qa.retriever import Retriever


def render(answer: Answer) -> str:
    lines = [answer.answer]
    if answer.refused:
        lines.append(f"confidence {answer.confidence:.2f} · refused · {answer.latency_ms} ms")
    else:
        noun = "source" if answer.sources == 1 else "sources"
        lines.append(
            f"confidence {answer.confidence:.2f} · {answer.sources} {noun} · local LLM"
            f" ({answer.model}) · {answer.latency_ms} ms"
        )
        pages = ", ".join(f"p.{format_pages(c.pages)}" for c in answer.citations)
        lines.append(f"cited: {pages}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, llm: LLM | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ask", description=__doc__)
    parser.add_argument("doc_id", help="document id, or a unique prefix of it (e.g. 12 chars)")
    parser.add_argument("question")
    parser.add_argument("--data-dir", help="index folder (default: DATA_DIR setting)")
    parser.add_argument("-k", "--top-k", type=int, help="chunks to retrieve (default: TOP_K)")
    args = parser.parse_args(argv)

    store = DocumentStore(data_dir=args.data_dir)
    try:
        doc_id = store.resolve_id(args.doc_id)
        answerer = Answerer(Retriever(store), llm or OllamaLLM())
        answer = answerer.answer(doc_id, args.question, args.top_k)
    except DocumentNotFoundError as exc:
        print(f"error: no indexed document matches {exc}", file=sys.stderr)
        return 2
    except LLMUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(render(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
