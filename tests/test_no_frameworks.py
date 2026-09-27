"""The project's central claim, enforced.

A reviewer's first question is whether "no frameworks" is actually true. This
test answers it, and stops a future phase from quietly importing a wrapper.
"""

import re
import tomllib
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

SOURCE_DIRS = ["rag", "scripts", "tests", "evaluation"]


def _forbidden_import_pattern(forbidden: str) -> re.Pattern:
    """Match `import X...` or `from X... import` for a forbidden package X.

    A trailing `[\\w.]*` lets X be a prefix of the actual module path, so
    `from langchain_core.prompts import Y` and `import langchain_community`
    are caught, not just a bare `import langchain`. The old pattern required
    whitespace or a dot immediately after the name on the `from` branch only,
    which let exactly that (the most common real-world import form) through.
    """
    return re.compile(rf"^\s*(?:import|from)\s+{forbidden}[\w.]*", re.MULTILINE)


def _python_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return [
        path
        for directory in SOURCE_DIRS
        for path in (root / directory).rglob("*.py")
    ]


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_forbidden_package_is_not_imported(forbidden: str):
    pattern = _forbidden_import_pattern(forbidden)
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


# --- the import-form gap the old pattern missed ------------------------------


@pytest.mark.parametrize(
    "forbidden, snippet",
    [
        ("langchain", "from langchain_core.prompts import PromptTemplate"),
        ("langchain", "import langchain_community"),
        ("llama_index", "from llama_index_core import VectorStoreIndex"),
    ],
)
def test_package_suffixed_import_forms_are_caught(forbidden: str, snippet: str):
    assert _forbidden_import_pattern(forbidden).search(snippet)


def test_hyphenated_hub_identifier_is_not_flagged():
    # This exact string is required (it's the embedding model id) and must
    # never be mistaken for an import of the forbidden `sentence_transformers`
    # package: it's a hyphen, not the underscore the forbidden name uses, and
    # it never follows an `import`/`from` keyword.
    snippet = 'embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"'
    assert not _forbidden_import_pattern("sentence_transformers").search(snippet)


# --- pyproject.toml dependencies ---------------------------------------------


def _pyproject_dependency_names() -> set[str]:
    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    names = set()
    for spec in specs:
        match = re.match(r"[A-Za-z0-9_.-]+", spec)
        if match:
            names.add(match.group(0).lower().replace("-", "_"))
    return names


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_forbidden_package_is_not_a_declared_dependency(forbidden: str):
    names = _pyproject_dependency_names()
    assert forbidden not in names, (
        f"{forbidden} is declared as a dependency in pyproject.toml"
    )


def test_pyproject_has_dependencies_to_check():
    # Guards against a parsing mistake silently checking an empty set.
    assert len(_pyproject_dependency_names()) >= 4
