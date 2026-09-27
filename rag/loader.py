"""Load the committed corpus into Document objects.

Mismatches between the text files and the metadata raise rather than warn: a
silently dropped document shows up much later as a retrieval miss that looks
like a ranking bug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str
    source: str
    publish_date: str | None = None
    author: str | None = None
    url: str | None = None


def load_documents(corpus_dir: Path, metadata_path: Path) -> list[Document]:
    """Read every .txt in corpus_dir, pairing it with its metadata entry.

    Returns documents sorted by doc_id, so chunk ids are stable across runs.
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    raw_text = metadata_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"metadata file could not be parsed as JSON: {metadata_path} ({exc})"
        ) from exc

    metadata = parsed.get("documents") if isinstance(parsed, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError(
            f'metadata file must contain a top-level "documents" object: {metadata_path}'
        )

    text_files = {p.stem: p for p in corpus_dir.glob("*.txt")}

    orphans = sorted(set(text_files) - set(metadata))
    if orphans:
        raise ValueError(
            f"text files with no metadata entry: {', '.join(orphans)}"
        )
    ghosts = sorted(set(metadata) - set(text_files))
    if ghosts:
        raise ValueError(
            f"metadata entries with no text file: {', '.join(ghosts)}"
        )

    documents: list[Document] = []
    for doc_id in sorted(text_files):
        text = text_files[doc_id].read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"document is empty: {doc_id}")
        record = metadata[doc_id]
        if not isinstance(record, dict):
            raise ValueError(
                f"metadata entry '{doc_id}' in {metadata_path} must be an "
                f"object, got {type(record).__name__}"
            )
        try:
            title = record["title"]
            source = record["source"]
        except KeyError as exc:
            field = exc.args[0]
            raise ValueError(
                f"metadata entry '{doc_id}' in {metadata_path} is missing "
                f"required field '{field}'"
            ) from exc
        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                title=title,
                source=source,
                publish_date=record.get("publish_date"),
                author=record.get("author"),
                url=record.get("url"),
            )
        )
    return documents
