"""Enforce that a bare `trace.note()` is never used for a silent fallback.

`evaluation/benchmark.py` hard-fails a run whose trace contains the word
"degraded", so a technique that silently falls back can never be measured as
"this technique does not help". That contract used to rely on an author
remembering to spell the word correctly inside a plain `.note()` call, and it
was missed five times by four authors, including in code whose own docstring
promised it.

`Trace.degraded` (see rag/trace.py) now writes the sentinel mechanically, so
the fix is structural rather than a reminder: this test walks every module
under rag/ with `ast`, finds every `.note(` call made on something named
`trace` or ending in `.trace`, and fails unless the enclosing `module:
function` is on the allowlist below with a written reason. A bare `.note()`
call added for a fallback fails this test immediately and must be justified
by name -- converted to `.degraded()`, or added to the allowlist with a
reason explaining why it is a fact, not a fallback.

Deliberately no "failure-shaped wording" heuristic (matching on "failed",
"could not", etc.): wording heuristics rot as vocabulary drifts. The
allowlist is exact strings, checked byte for byte.
"""

from __future__ import annotations

import ast
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent.parent / "rag"

# Each entry is "relative/path.py:qualified.function.name" -> a reason the
# note() call there is a fact worth reporting, not a fallback that needs the
# "degraded" sentinel. Keep this small and exact.
ALLOWLIST: dict[str, str] = {
    "generation.py:generate_answer": (
        "There is no fallback here: trace.answer simply stays None, and that "
        "absence is already the structural signal a reader or the benchmark "
        "needs. The cache-hit note is informational in the same way."
    ),
    "pipeline.py:ask": (
        "'retrieval only: no LLM configured' and the filter-matched-nothing "
        "note are both facts about how this run was invoked or what the "
        "filter did, not a fallback to a less-restricted default -- both "
        "leave the pipeline doing exactly what was asked. The unconditional "
        "filter-exclusion-count note added for the RAPTOR demo is the same "
        "kind of fact."
    ),
    "query_construction.py:build_filter": (
        "When some proposed topics are valid and applied, the note about the "
        "dropped ones is visibility into what was trimmed, not a fallback -- "
        "the filter is smaller than proposed, not absent. (When *no* proposed "
        "topic survives, that path already calls trace.degraded instead.)"
    ),
}


def _qualname(stack: list[str]) -> str:
    return ".".join(stack)


class _NoteCallFinder(ast.NodeVisitor):
    """Records every `<trace-like>.note(...)` call, tagged by enclosing scope."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.hits: list[tuple[str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "note":
            receiver = func.value
            is_trace_like = (
                isinstance(receiver, ast.Name) and receiver.id == "trace"
            ) or (isinstance(receiver, ast.Attribute) and receiver.attr == "trace")
            if is_trace_like:
                self.hits.append((_qualname(self.stack), node.lineno))
        self.generic_visit(node)


def _find_note_calls() -> list[tuple[str, str, int]]:
    """(relative_path, qualified_function_name, lineno) for every match."""
    found = []
    for path in sorted(RAG_DIR.rglob("*.py")):
        rel = path.relative_to(RAG_DIR).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        finder = _NoteCallFinder()
        finder.visit(tree)
        for qualname, lineno in finder.hits:
            found.append((rel, qualname, lineno))
    return found


def test_every_bare_note_call_on_a_trace_is_allowlisted():
    violations = [
        f"{rel}:{qualname} (line {lineno})"
        for rel, qualname, lineno in _find_note_calls()
        if f"{rel}:{qualname}" not in ALLOWLIST
    ]
    assert not violations, (
        "trace.note() used outside the allowlist -- if this is a silent "
        "fallback, use trace.degraded(what, fallback) instead; if it is "
        "genuinely just a fact, add it to ALLOWLIST in "
        f"tests/test_degradation_notes.py with a reason:\n"
        + "\n".join(violations)
    )


def test_the_allowlist_has_no_stale_entries():
    # Keeps the allowlist exact and small, per the module docstring: an entry
    # for a note() call site that no longer exists (e.g. converted to
    # .degraded()) must be removed, not left to accumulate.
    present = {f"{rel}:{qualname}" for rel, qualname, _ in _find_note_calls()}
    stale = sorted(set(ALLOWLIST) - present)
    assert not stale, f"allowlist entries with no matching note() call: {stale}"


def test_there_are_note_calls_to_check():
    # Guards against the walk silently matching nothing.
    assert _find_note_calls()
