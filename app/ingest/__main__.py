"""python -m app.ingest <pdf> -> "<filename> · <N> pages · <M> chunks indexed · <doc_id[:12]>"."""

from __future__ import annotations

import argparse
import sys

from app.index.store import DocumentStore
from app.ingest.loader import InvalidPDFError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingest", description=__doc__)
    parser.add_argument("pdf", help="path to a PDF file")
    parser.add_argument("--data-dir", help="index folder (default: DATA_DIR setting)")
    args = parser.parse_args(argv)

    try:
        manifest = DocumentStore(data_dir=args.data_dir).ingest(args.pdf)
    except (FileNotFoundError, InvalidPDFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"{manifest.filename} · {manifest.pages} pages · {manifest.chunks} chunks indexed"
        f" · {manifest.doc_id[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
