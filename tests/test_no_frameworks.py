"""The project's central claim, enforced.

A reviewer's first question is whether "no frameworks" is actually true. This
test answers it, and stops a future phase from quietly importing a wrapper.
"""

import re
from pathlib import Path

import pytest

FORBIDDEN = [
    "langchain",
    "llama_index",
    "llamaindex",
    "sentence_transformers",
    "haystack",
    "chromadb",
    "faiss",
    "pinecone",
    "sklearn",
    "bs4",
    "requests",
    "dotenv",
]

SOURCE_DIRS = ["rag", "scripts", "tests"]


def _python_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return [
        path
        for directory in SOURCE_DIRS
        for path in (root / directory).rglob("*.py")
    ]


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_forbidden_package_is_not_imported(forbidden: str):
    pattern = re.compile(
        rf"^\s*(?:import\s+{forbidden}|from\s+{forbidden}[\s.])", re.MULTILINE
    )
    offenders = [
        str(path)
        for path in _python_files()
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"{forbidden} imported in: {offenders}"


def test_there_are_python_files_to_check():
    # Guards against the glob silently matching nothing and the suite
    # passing vacuously.
    assert len(_python_files()) > 10
