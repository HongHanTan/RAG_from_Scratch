# Phase 7: Full Gold Set and Final Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the gold set from 10 questions to 30, including the cross-document questions the current set cannot express, and re-run the complete benchmark.

**Architecture:** The gold schema generalises from one document per question to several, because RAPTOR's entire claim is about answers that span documents and the present schema cannot state one. The 20 new questions are written and committed with a *pre-registered prediction* of which technique each should favour, before any of them is measured.

**Tech Stack:** Python 3.14, NumPy, pytest. No new dependencies.

## Global Constraints

- **No RAG framework.** `langchain`, `llama_index`, `sentence_transformers`, `sklearn`, `bs4`, `requests`, `dotenv` are installed and MUST NOT be imported. `tests/test_no_frameworks.py` scans `rag/`, `scripts/`, `tests/`, `evaluation/` and `pyproject.toml`.
- **Allowed third-party:** `numpy`, `torch`, `transformers`, `google.genai`, `pytest`, `fastapi`, `uvicorn`.
- **Determinism.** Temperature 0, seeded clustering, stable sorts.
- **Test-driven.** Write the test, run it, watch it fail for the expected reason, then implement.
- **Every unit test runs offline** unless marked `@pytest.mark.slow` or `@pytest.mark.live`.
- **Platform is Windows.** `pathlib`, explicit `encoding="utf-8"`.
- **Never set `PYTHONIOENCODING`.**
- **Do not rebuild or overwrite anything in `data/`.** The three index files are the baseline every published number rests on. Phase 7 adds questions, not documents.
- **Commit after every task.** Messages end with a blank line then exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## The methodological hazard, and what this plan does about it

The spec says to choose the 30 questions "with knowledge of what every technique does". That is deliberate — a gold set that cannot distinguish the techniques measures nothing — but it is also exactly how a benchmark gets fitted to the result you want. The author knows RAPTOR lost in Phase 5 and has an interest in it not losing.

Four rules, and the plan is built so that breaking one is visible in the git history:

1. **All 20 questions are written and committed before anything is measured.** Tasks 2 and 3 are forbidden from running the benchmark. The commit timestamps are the evidence.
2. **Each question carries a pre-registered `expects` field** naming the technique it is predicted to favour, written at authoring time.
3. **No question may be edited or removed after its result is seen.** Phase 4 set this precedent: `longcontext-position` was left misclassified rather than relabelled after the router missed it, because changing labels after seeing which one hurt is fitting labels to the metric.
4. **The prediction is scored and reported.** Task 4 reports how often the predicted technique actually won. If the predictions are mostly wrong, that is the finding, and it says the author's model of these techniques is worse than assumed — which is worth more than a flattering table.

## What the earlier phases provide

| Thing | Signature |
|---|---|
| `evaluation.gold.GoldQuestion` | frozen: `id`, `question`, `doc_id`, `quotes`, `why`, `spans: tuple[tuple[int,int], ...]` |
| `evaluation.gold.load_gold(path, documents)` | resolves each quote to a char span; raises `ValueError` naming the question id on a missing, ambiguous or non-string quote |
| `evaluation.spans.chunks_overlapping(doc_id, start, end, chunks)` | `-> set[str]`; half-open; a synthetic node (`char_start == -1`) never matches |
| `evaluation.spans.relevant_chunk_ids(question, chunks)` | unions `chunks_overlapping` over `question.spans`, all within `question.doc_id` |
| `evaluation.metrics` | `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `doc_precision_at_k(retrieved_doc_ids, gold_doc_id, k)` |
| `evaluation.benchmark` | `build_parser()`, `score_strategy(strategy, gold, store, embedder, llm, config, k, route=False, rerank=False) -> StrategyScore`, `format_table`, `DOC_PRECISION_K`; `benchmark.py:237-243` builds `retrieved_doc_ids` and calls `doc_precision_at_k` |
| `evaluation.routing_eval` | `routing_eval.py:98` looks up `store.doc_meta[question.doc_id]` for the question's topic |
| `rag.pipeline.ask(...)` | `(question, store, embedder, llm, config, k=None, strategy="direct", strategy_options=None, generate=True, route=False, construct=False, semantic_prompt=False, rerank=False)` |

Corpus: 38 documents, 5,116 chunks. Indexes: `data/index.npz` (flat), `data/index-multirep.npz` (38), `data/index-raptor.npz` (5,846, levels 0/1/2/3). Suite: 712 passed, 15 deselected.

Present gold set: 10 questions, one document each, in `evaluation/gold.json`:

```json
{"id": "colbert-maxsim",
 "question": "How does ColBERT score a document against a query?",
 "doc_id": "colbert",
 "quotes": ["every query embedding interacts with all document embeddings via a MaxSim operator, which computes maximum similarity"],
 "why": "Direct vocabulary match. Baseline case plain retrieval should handle."}
```

## File Structure

```
evaluation/
  gold.py        MODIFY: several documents per question; `expects`
  spans.py       MODIFY: relevant_chunk_ids unions across documents
  metrics.py     MODIFY: doc_precision_at_k takes a set of gold documents
  benchmark.py   MODIFY: pass the set; --predictions report
  routing_eval.py MODIFY: union of topics
  gold.json      MODIFY: 10 -> 30 questions
tests/           test_gold.py, test_spans.py, test_metrics.py, test_benchmark.py
README.md        MODIFY: final table
```

---

### Task 1: A gold question may name several documents

**Files:**
- Modify: `evaluation/gold.py`, `evaluation/spans.py`, `evaluation/metrics.py`, `evaluation/benchmark.py`, `evaluation/routing_eval.py`
- Test: `tests/test_gold.py`, `tests/test_spans.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `GoldSpan` frozen dataclass: `doc_id: str`, `char_start: int`, `char_end: int`
  - `GoldQuestion.sources: tuple[str, ...]` — every gold document, in file order
  - `GoldQuestion.spans: tuple[GoldSpan, ...]` — now carries its document
  - `GoldQuestion.doc_id` — **kept**, returns `sources[0]`, so single-document call sites are unchanged
  - `GoldQuestion.expects: str` — the pre-registered prediction; `""` when absent
  - `doc_precision_at_k(retrieved_doc_ids: list[str], gold_doc_ids: set[str] | str, k: int) -> float`

**The safety property for this whole task: the ten existing questions must score byte-identically afterwards.** If they move, every number published in Phases 3–6 becomes incomparable and the "final table" is measuring a schema change rather than the techniques. Step 6 checks this directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gold.py`:

```python
def test_a_question_may_name_several_documents(tmp_path):
    # RAPTOR's whole claim is about answers that span documents. The old
    # schema could not state such a question, so the Phase 5 gold set
    # contained none and RAPTOR was judged only on cases it is not for.
    docs = _docs()
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "what do these share?",
            "sources": [
                {"doc_id": "alpha", "quotes": ["measures the angle"]},
                {"doc_id": "beta", "quotes": ["Rank fusion sums"]},
            ],
            "why": "spans two documents",
        }
    ])
    gold = load_gold(path, docs)
    assert gold[0].sources == ("alpha", "beta")
    assert {s.doc_id for s in gold[0].spans} == {"alpha", "beta"}


def test_each_span_reproduces_its_own_quote(tmp_path):
    docs = _docs()
    path = _write(tmp_path, [
        {
            "id": "q1", "question": "q",
            "sources": [
                {"doc_id": "alpha", "quotes": ["measures the angle"]},
                {"doc_id": "beta", "quotes": ["Rank fusion sums"]},
            ],
            "why": "w",
        }
    ])
    by_id = {d.doc_id: d for d in docs}
    for span in load_gold(path, docs)[0].spans:
        assert by_id[span.doc_id].text[span.char_start:span.char_end] in (
            "measures the angle", "Rank fusion sums"
        )


def test_the_old_single_document_form_still_loads(tmp_path):
    # Every existing question uses it, and rewriting them all would risk
    # changing the numbers they produce.
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha",
         "quotes": ["measures the angle"], "why": "w"}
    ])
    gold = load_gold(path, _docs())
    assert gold[0].sources == ("alpha",)
    assert gold[0].doc_id == "alpha"


def test_doc_id_returns_the_first_source(tmp_path):
    # Kept so single-document call sites, and routing_eval's topic lookup,
    # keep working unchanged.
    path = _write(tmp_path, [
        {"id": "q1", "question": "q",
         "sources": [{"doc_id": "beta", "quotes": ["Rank fusion sums"]},
                     {"doc_id": "alpha", "quotes": ["measures the angle"]}],
         "why": "w"}
    ])
    assert load_gold(path, _docs())[0].doc_id == "beta"


def test_mixing_both_forms_in_one_question_is_an_error(tmp_path):
    # Ambiguous: which is the gold document? Better to refuse than guess.
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha", "quotes": ["angle"],
         "sources": [{"doc_id": "beta", "quotes": ["Rank fusion sums"]}],
         "why": "w"}
    ])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_an_empty_sources_list_is_an_error(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "sources": [], "why": "w"}
    ])
    with pytest.raises(ValueError, match="non-empty"):
        load_gold(path, _docs())


def test_a_bad_quote_in_the_second_source_names_the_question(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q",
         "sources": [{"doc_id": "alpha", "quotes": ["measures the angle"]},
                     {"doc_id": "beta", "quotes": ["not present at all"]}],
         "why": "w"}
    ])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_expects_is_loaded_when_present(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha",
         "quotes": ["measures the angle"], "why": "w", "expects": "raptor"}
    ])
    assert load_gold(path, _docs())[0].expects == "raptor"


def test_expects_defaults_to_empty(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "q", "doc_id": "alpha",
         "quotes": ["measures the angle"], "why": "w"}
    ])
    assert load_gold(path, _docs())[0].expects == ""
```

Append to `tests/test_spans.py`:

```python
def test_relevant_chunks_span_every_gold_document():
    from evaluation.gold import GoldQuestion, GoldSpan

    question = GoldQuestion(
        id="q", question="q", sources=("alpha", "beta"),
        quotes=("a", "b"), why="w", expects="",
        spans=(GoldSpan("alpha", 0, 50), GoldSpan("beta", 0, 50)),
    )
    assert relevant_chunk_ids(question, _chunks()) == {"a:0", "b:0"}


def test_a_span_only_matches_chunks_of_its_own_document():
    # The doc_id travels with the span, so a range that would overlap a
    # chunk in another paper must not credit it.
    from evaluation.gold import GoldQuestion, GoldSpan

    question = GoldQuestion(
        id="q", question="q", sources=("alpha",), quotes=("a",), why="w",
        expects="", spans=(GoldSpan("alpha", 0, 50),),
    )
    assert relevant_chunk_ids(question, _chunks()) == {"a:0"}
```

Append to `tests/test_metrics.py`:

```python
def test_doc_precision_counts_any_gold_document():
    # A cross-document question has several right answers; crediting only
    # one of them would score a correct retrieval as a miss.
    assert doc_precision_at_k(
        ["alpha", "beta", "gamma", "alpha"], {"alpha", "beta"}, 4
    ) == pytest.approx(0.75)


def test_doc_precision_still_accepts_a_bare_string():
    # Keeps every existing single-document call site working unchanged.
    assert doc_precision_at_k(["alpha", "beta"], "alpha", 2) == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_gold.py tests/test_spans.py tests/test_metrics.py -q`
Expected: `ImportError: cannot import name 'GoldSpan'`, `TypeError` on the `sources=` constructor argument, and the `doc_precision_at_k` set case scoring wrongly or raising.

- [ ] **Step 3: Generalise `evaluation/gold.py`**

Add above `GoldQuestion`:

```python
@dataclass(frozen=True)
class GoldSpan:
    """One answering passage, and the document it lives in.

    The document travels with the span because a question may now have
    answers in several papers, and a bare (start, end) pair would silently
    credit a chunk at the same offset in the wrong one.
    """

    doc_id: str
    char_start: int
    char_end: int
```

Change `GoldQuestion` to:

```python
@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    sources: tuple[str, ...]
    """Every document holding an answer, in the order the file lists them."""
    quotes: tuple[str, ...]
    why: str
    spans: tuple[GoldSpan, ...]
    expects: str = ""
    """The technique this question was predicted to favour, recorded when it
    was written and before anything was measured. Empty for the original ten,
    which predate the practice. Task 4 scores these predictions: a gold set
    written by someone who knows what each technique does is a gold set that
    can be fitted to the answer, and reporting how often the prediction held
    is what makes that risk visible instead of hidden.
    """

    @property
    def doc_id(self) -> str:
        """The first gold document.

        Kept so single-document call sites -- `routing_eval`'s topic lookup
        among them -- keep working without a sweep, and so the ten original
        questions behave exactly as before.
        """
        return self.sources[0]
```

In `load_gold`, accept both forms. For each entry:

```python
        if "sources" in entry and ("doc_id" in entry or "quotes" in entry):
            raise ValueError(
                f"question {qid}: use either 'sources' or 'doc_id'/'quotes', "
                "not both -- which document is the gold one is otherwise "
                "ambiguous"
            )
        if "sources" in entry:
            sources = entry["sources"]
            if not isinstance(sources, list) or not sources:
                raise ValueError(f"question {qid}: 'sources' must be a non-empty list")
        else:
            sources = [{"doc_id": entry.get("doc_id"), "quotes": entry.get("quotes")}]
```

Then resolve each source's quotes with the existing per-quote logic — missing,
ambiguous and non-string quotes must still raise a `ValueError` naming `qid`
— and build `GoldSpan(doc_id, start, end)` for each. Set
`sources=tuple(s["doc_id"] for s in sources)`, `quotes` to the flattened
tuple of every quote, and `expects=entry.get("expects", "")`.

- [ ] **Step 4: Generalise `relevant_chunk_ids` in `evaluation/spans.py`**

```python
def relevant_chunk_ids(question: GoldQuestion, chunks: list[Chunk]) -> set[str]:
    """The chunks that count as a correct retrieval for this question.

    Any answering passage counts, in any of the question's gold documents.
    Each span carries its own `doc_id`, so a span is only ever matched
    against chunks of the paper it came from.
    """
    relevant: set[str] = set()
    for span in question.spans:
        relevant |= chunks_overlapping(
            span.doc_id, span.char_start, span.char_end, chunks
        )
    return relevant
```

- [ ] **Step 5: Generalise `doc_precision_at_k` and its callers**

In `evaluation/metrics.py`, accept either a set or a bare string:

```python
def doc_precision_at_k(
    retrieved_doc_ids: list[str], gold_doc_ids: set[str] | str, k: int
) -> float:
    """Fraction of the top k results that come from a document holding the answer.

    A cross-document question has several right documents, so membership is
    tested against a set. A bare string is still accepted, which keeps every
    single-document caller working unchanged.
    """
    gold = {gold_doc_ids} if isinstance(gold_doc_ids, str) else set(gold_doc_ids)
    top = retrieved_doc_ids[:k]
    if not top:
        return 0.0
    return sum(1 for doc_id in top if doc_id in gold) / len(top)
```

In `evaluation/benchmark.py:243`, pass `set(question.sources)` instead of
`question.doc_id`. In `evaluation/routing_eval.py:98`, look up the topic for
every source and treat the router as correct when it picks any of them —
keep the existing behaviour for a single-source question, and say in your
summary what you changed.

- [ ] **Step 6: Prove the existing ten questions did not move**

This is the point of the task. Run the benchmark on the unchanged 10-question
gold set and compare against the committed Phase 6a baseline:

```
python -m evaluation.benchmark --index flat
```

Expected, exactly:

| Strategy | Recall@20 | MRR@20 | nDCG@20 | DocPrec@5 |
|---|---:|---:|---:|---:|
| hyde | 0.558 | 0.358 | 0.315 | 0.660 |
| rag-fusion | 0.392 | 0.140 | 0.184 | 0.560 |
| step-back | 0.383 | 0.136 | 0.170 | 0.400 |
| multi-query | 0.375 | 0.190 | 0.208 | 0.400 |
| decomposition | 0.333 | 0.194 | 0.187 | 0.600 |
| direct | 0.325 | 0.103 | 0.137 | 0.560 |

If any cell differs, STOP and report it. A schema change that moves the
numbers makes the final table incomparable with everything before it.

- [ ] **Step 7: Run the full suite and commit**

Run: `python -m pytest -q` (712 + the new tests).

```bash
git add evaluation tests/
git commit -m "feat: a gold question may name several documents

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Ten cross-document questions

**Files:**
- Modify: `evaluation/gold.json`
- Test: `tests/test_gold_set_content.py` (new)

**You must not run the benchmark in this task.** The questions are pre-registered: written, predicted and committed before anything is measured. Running it here and then adjusting a question would defeat the whole arrangement, and the commit order is the evidence that it did not happen.

These are the questions the old schema could not express, and their absence is why Phase 5's RAPTOR verdict was unfair: every existing question has one answering passage in one paper, which is the case flat top-k already wins.

- [ ] **Step 1: Write the content test first**

Create `tests/test_gold_set_content.py`:

```python
"""Properties the gold set itself must hold.

Separate from test_gold.py, which tests the loader. This tests the data.
"""

from pathlib import Path

import pytest

from evaluation.gold import load_gold
from rag.__main__ import load_config
from rag.loader import load_documents

GOLD = Path(__file__).resolve().parent.parent / "evaluation" / "gold.json"


@pytest.fixture(scope="module")
def gold():
    config = load_config()
    return load_gold(GOLD, load_documents(config.corpus_dir, config.metadata_path))


def test_every_quote_resolves(gold):
    # load_gold raises on a quote that is missing or ambiguous, so reaching
    # here at all means all 30 resolved. This asserts the fixture ran.
    assert gold


def test_ids_are_unique(gold):
    ids = [q.id for q in gold]
    assert len(ids) == len(set(ids))


def test_every_question_has_a_reason(gold):
    assert all(q.why.strip() for q in gold)


def test_every_span_is_non_empty(gold):
    for q in gold:
        for span in q.spans:
            assert span.char_end > span.char_start, f"{q.id} has an empty span"


def test_there_are_cross_document_questions(gold):
    # Their absence is why Phase 5 could not judge RAPTOR fairly.
    multi = [q for q in gold if len(q.sources) > 1]
    assert len(multi) >= 10, f"only {len(multi)} cross-document questions"


def test_cross_document_questions_name_distinct_documents(gold):
    for q in gold:
        assert len(set(q.sources)) == len(q.sources), f"{q.id} repeats a document"


def test_new_questions_carry_a_prediction(gold):
    # Written before measuring. The original ten predate the practice and
    # are exempt by id, not by silence.
    original = {
        "colbert-maxsim", "raptor-clustering", "dpr-encoder", "selfrag-tokens",
        "hyde-problem", "rag-memory", "crag-quality", "longcontext-position",
        "colbertv2-tradeoff", "cot-limits",
    }
    missing = [q.id for q in gold if q.id not in original and not q.expects]
    assert not missing, f"no 'expects' prediction on: {missing}"
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python -m pytest tests/test_gold_set_content.py -q`
Expected: `test_there_are_cross_document_questions` fails with "only 0 cross-document questions".

- [ ] **Step 3: Write the ten questions**

Add ten entries to `evaluation/gold.json` using the `sources` form, each
naming **two or three** documents. Requirements for each:

- The question must be answerable *only* by combining the documents — if one
  paper answers it alone, it is not a cross-document question and belongs in
  Task 3.
- Every quote must be an **exact, unique substring** of its document.
  `load_gold` raises on a quote that is missing or appears twice; run it
  after every few additions rather than writing all ten and debugging at the
  end.
- `why` states what the question is testing.
- `expects` is the pre-registered prediction, one of: `raptor`, `multirep`,
  `hyde`, `multi-query`, `rag-fusion`, `step-back`, `decomposition`,
  `rerank`, `direct`.

Draw on the corpus's real themes. It holds 38 papers across
`retrieval-models`, `rag-systems`, `prompting-reasoning`,
`evaluation-benchmarks` and `foundations`, so genuine cross-document
questions exist — for example, what several retrieval papers share in how
they handle a query's relationship to a passage, or how different papers
justify the same evaluation choice. Read the documents; do not invent a
theme the corpus does not support.

Shape:

```json
{
  "id": "late-interaction-tradeoff",
  "question": "What storage cost do late-interaction retrievers accept, and how do they justify it?",
  "sources": [
    {"doc_id": "colbert", "quotes": ["<exact quote>"]},
    {"doc_id": "colbertv2", "quotes": ["<exact quote>"]}
  ],
  "why": "Neither paper states both halves; the answer needs both.",
  "expects": "raptor"
}
```

- [ ] **Step 4: Validate every quote resolves**

```
python -c "
from pathlib import Path
from rag.__main__ import load_config
from rag.loader import load_documents
from evaluation.gold import load_gold
cfg = load_config()
gold = load_gold(Path('evaluation/gold.json'), load_documents(cfg.corpus_dir, cfg.metadata_path))
multi = [q for q in gold if len(q.sources) > 1]
print(f'{len(gold)} questions, {len(multi)} cross-document')
for q in multi:
    print(f'  {q.id:32s} {len(q.spans)} spans over {q.sources} -> expects {q.expects}')
"
```

Every quote must resolve. If `load_gold` raises, fix the quote — do not
loosen the loader.

- [ ] **Step 5: Check the spans actually reach chunks**

A span that resolves but overlaps no chunk would make the question
unanswerable and silently drag every strategy's score down:

```
python -c "
from pathlib import Path
from rag.__main__ import load_config
from rag.loader import load_documents
from rag.pipeline import load_index
from evaluation.gold import load_gold
from evaluation.spans import relevant_chunk_ids
cfg = load_config()
gold = load_gold(Path('evaluation/gold.json'), load_documents(cfg.corpus_dir, cfg.metadata_path))
store = load_index(cfg)
for q in gold:
    n = len(relevant_chunk_ids(q, store.chunks))
    flag = '  <-- UNREACHABLE' if n == 0 else ''
    print(f'{q.id:32s} {n:3d} relevant chunks{flag}')
"
```

Report any question with 0 relevant chunks and fix its quotes.

- [ ] **Step 6: Run the tests and commit**

Run: `python -m pytest -q`

```bash
git add evaluation/gold.json tests/test_gold_set_content.py
git commit -m "test: ten cross-document gold questions, predictions pre-registered

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Ten technique-targeted single-document questions

**Files:**
- Modify: `evaluation/gold.json`

**You must not run the benchmark in this task either.**

The spec asks for questions "whose phrasing rewards HyDE" and "with date and source constraints that exercise query construction". These are single-document questions, chosen to separate techniques that the present ten do not separate.

- [ ] **Step 1: Write the ten questions**

Add ten entries in the existing single-document form (`doc_id` + `quotes`),
each with `why` and a pre-registered `expects`. Cover at least:

- **Three phrased to reward HyDE** — a question whose wording shares little
  vocabulary with the passage that answers it, so a hypothetical answer
  document embeds closer to the passage than the question does. HyDE was the
  strongest technique in Phases 2–6 and the present set has one such case.
- **Three with a date or author constraint** that `--construct` can turn
  into a metadata filter, e.g. naming a period or an author the corpus
  metadata records. Check `data/metadata.json` for what is actually there
  rather than inventing a constraint the filter cannot express.
- **Two multi-hop** questions whose answer needs two passages in the *same*
  paper, which is what decomposition is for.
- **Two plain vocabulary-match** questions, as baseline cases plain
  retrieval should win. A gold set with no easy questions cannot show a
  technique making things worse.

Same rules as Task 2: exact unique quotes, `load_gold` must not raise, and
`expects` is written now, not after measuring.

- [ ] **Step 2: Validate**

Run the two validation commands from Task 2, Steps 4 and 5. Expect 30
questions total and no unreachable ones. Report the counts.

- [ ] **Step 3: Confirm the mix**

```
python -c "
from collections import Counter
from pathlib import Path
from rag.__main__ import load_config
from rag.loader import load_documents
from evaluation.gold import load_gold
cfg = load_config()
gold = load_gold(Path('evaluation/gold.json'), load_documents(cfg.corpus_dir, cfg.metadata_path))
print('total:', len(gold))
print('cross-document:', sum(1 for q in gold if len(q.sources) > 1))
print('predictions:', dict(Counter(q.expects for q in gold)))
print('documents covered:', len({s for q in gold for s in q.sources}), 'of 38')
"
```

Report all four numbers. Thirty questions over 38 documents will not cover
every document, and that is fine — say what the coverage is rather than
padding it.

- [ ] **Step 4: Run the tests and commit**

Run: `python -m pytest -q`

```bash
git add evaluation/gold.json
git commit -m "test: ten technique-targeted gold questions, predictions pre-registered

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The final benchmark

**Files:**
- Modify: `evaluation/benchmark.py`, `README.md`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `python -m evaluation.benchmark --predictions` — reports how often each question's `expects` technique actually scored best on it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark.py`:

```python
def test_benchmark_accepts_predictions():
    from evaluation.benchmark import build_parser

    assert build_parser().parse_args(["--predictions"]).predictions is True


def test_predictions_defaults_off():
    from evaluation.benchmark import build_parser

    assert build_parser().parse_args([]).predictions is False


def test_prediction_scoring_counts_a_hit():
    from evaluation.benchmark import score_predictions

    # question -> {strategy: recall}; the prediction is "hyde"
    per_question = {"q1": {"hyde": 0.9, "direct": 0.1}}
    expects = {"q1": "hyde"}
    hits, total = score_predictions(per_question, expects)
    assert (hits, total) == (1, 1)


def test_prediction_scoring_counts_a_miss():
    from evaluation.benchmark import score_predictions

    per_question = {"q1": {"hyde": 0.1, "direct": 0.9}}
    hits, total = score_predictions(per_question, {"q1": "hyde"})
    assert (hits, total) == (0, 1)


def test_questions_without_a_prediction_are_not_counted():
    from evaluation.benchmark import score_predictions

    per_question = {"q1": {"hyde": 0.9}}
    assert score_predictions(per_question, {"q1": ""}) == (0, 0)


def test_a_tie_does_not_count_as_a_hit():
    # If every strategy scores the same the prediction told us nothing, and
    # counting it as correct would inflate the scorecard.
    from evaluation.benchmark import score_predictions

    per_question = {"q1": {"hyde": 0.5, "direct": 0.5}}
    assert score_predictions(per_question, {"q1": "hyde"}) == (0, 1)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_benchmark.py -q`
Expected: `ImportError: cannot import name 'score_predictions'`.

- [ ] **Step 3: Implement**

Add to `evaluation/benchmark.py`:

```python
def score_predictions(
    per_question: dict[str, dict[str, float]], expects: dict[str, str]
) -> tuple[int, int]:
    """How often the pre-registered prediction named the winning strategy.

    A question whose strategies all tie is counted as a miss, not a hit: the
    prediction distinguished nothing there, and crediting it would inflate
    the scorecard exactly where it is least informative.
    """
    hits = total = 0
    for qid, scores in per_question.items():
        predicted = expects.get(qid, "")
        if not predicted or not scores:
            continue
        total += 1
        best = max(scores.values())
        if best > min(scores.values()) and scores.get(predicted, -1.0) == best:
            hits += 1
    return hits, total
```

Add `--predictions` to `build_parser`, collect per-question recall per
strategy during the sweep, and print the scorecard under the table.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 5: Run the complete final benchmark**

Every variant, on 30 questions. This is a long run; use the background and
be patient rather than cutting it short. Gemini's free tier is 15
requests/minute and 500/day, and the 20 new questions have **no cached query
rewrites**, so this will make real API calls — roughly 20 questions x 5
LLM-using strategies, plus decomposition's extra calls. If a 429 cannot be
absorbed, stop and report rather than letting a strategy degrade; the
benchmark hard-fails on the `degraded` sentinel by design.

```
python -m evaluation.benchmark --index flat --predictions
python -m evaluation.benchmark --index flat --rerank
python -m evaluation.benchmark --index raptor
python -m evaluation.benchmark --index multirep
python -m evaluation.benchmark --index flat --route
```

- [ ] **Step 6: Re-measure the document-level table**

Multi-representation cannot be scored by span-based recall, so the
document-level comparison from Phase 5 has to be redone on 30 questions.
Adapt the Phase 5 command, treating a question as a hit when **any** of its
gold documents is retrieved:

```
python -c "
from pathlib import Path
from rag.__main__ import load_config
from rag.embedding import Embedder
from rag.loader import load_documents
from rag.store import VectorStore
from evaluation.gold import load_gold
cfg = load_config()
docs = load_documents(cfg.corpus_dir, cfg.metadata_path)
gold = load_gold(Path('evaluation/gold.json'), docs)
emb = Embedder(cfg.embedding_model, max_length=cfg.max_seq_tokens)
stores = {'flat': VectorStore.load(cfg.index_path),
          'multirep': VectorStore.load(Path('data/index-multirep.npz')),
          'raptor': VectorStore.load(Path('data/index-raptor.npz'))}
def rank(store, q, k=20):
    seen = []
    for h in store.search(emb.encode([q.question]), k=k)[0]:
        d = h.chunk.doc_id
        if d not in seen and not d.startswith('raptor:'):
            seen.append(d)
    hits = [seen.index(s) + 1 for s in q.sources if s in seen]
    return min(hits) if hits else None
for name, st in stores.items():
    r = [rank(st, q) for q in gold]
    n = len(r)
    print(f'{name:10s} {len(st):6d} nodes  DocHit@1={sum(1 for v in r if v==1)/n:.3f} '
          f'DocHit@5={sum(1 for v in r if v and v<=5)/n:.3f} '
          f'DocMRR={sum((1/v) if v else 0 for v in r)/n:.3f}')
"
```

- [ ] **Step 7: Write the final README section**

Replace the benchmark tables with the 30-question results. Required content:

- The full table, all six strategies, flat index.
- Reranked, RAPTOR and multi-representation tables.
- The document-level table for multi-representation.
- **How the 10-question and 30-question numbers differ.** Where a Phase 2–6
  conclusion no longer holds, say so directly and name the old claim. This
  is the most valuable thing in the section: it says how much a 10-question
  benchmark could be trusted.
- **Whether RAPTOR's verdict changed** now that cross-document questions
  exist. Phase 5 said RAPTOR loses and that the gold set contained no
  question it was built for. Report what the answer is, either way.
- **The prediction scorecard**, and what it says. A low score means the
  author's model of these techniques was wrong, which is a real finding and
  must not be buried.
- **The gold set's remaining limits**: 30 questions is still small, written
  by one author, and the `expects` predictions are that author's too.

Tick **Phase 7** in the roadmap.

- [ ] **Step 8: Commit**

```bash
git add evaluation/benchmark.py README.md tests/
git commit -m "feat: final benchmark on the 30-question gold set

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 7 Definition of Done

- [ ] `python -m pytest` passes offline
- [ ] The gold set holds 30 questions, at least 10 cross-document
- [ ] Every quote resolves and every question reaches at least one chunk
- [ ] The original ten score byte-identically to the Phase 6a baseline
- [ ] Every new question carries an `expects` prediction, committed before measuring
- [ ] No question was edited after its result was seen
- [ ] The prediction scorecard is reported whatever it says
- [ ] RAPTOR's verdict is re-reported against cross-document questions
- [ ] Differences from the 10-question numbers are stated, not smoothed
- [ ] Phase 7 ticked in the roadmap
- [ ] Working tree clean
