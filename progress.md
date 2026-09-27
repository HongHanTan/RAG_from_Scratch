# Phase 1 — Core Pipeline: Progress

Branch: `phase-1-core-pipeline`
Plan: [docs/superpowers/plans/2026-09-27-phase-1-core-pipeline.md](docs/superpowers/plans/2026-09-27-phase-1-core-pipeline.md)
Spec: [docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md](docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md)

Status key: ⬜ not started · 🔄 in progress · ✅ complete

| # | Task | Status | Commits | Notes |
|---|------|--------|---------|-------|
| 1 | Scaffolding, Config, test fixtures | ✅ | `1ae1bad` | 9 tests pass; review clean |
| 2 | HTML text extraction | ✅ | `77f3c41`..`b497946` | 31 tests pass; 3 review rounds, all findings fixed |
| 3 | Corpus fetch script + fetch corpus | ✅ | `cdd5aee`, `ae41c4f` | 38 docs, all titles verified; 46 tests |
| 4 | Document loader | ✅ | `e6943af`, `e138637`, `f209fad` | 16 tests; all malformed-metadata shapes now actionable |
| 5 | Token-aware chunking | ✅ | `f72b505` | 18 tests; 5,116 chunks, offsets independently verified |
| 6 | Embeddings | ✅ | `dfde3c1` | 9 fast + 4 slow; review clean, no findings |
| 7 | Similarity and top-k | ✅ | `7fd1286` | 16 tests; review approved, 2 Minor |
| 8 | Vector store | ✅ | `1937bf1` | 13 tests; 6-18 ms search over 5,116 chunks; review approved |
| 9 | Trace | ✅ | `673a407` | 8 tests; review approved |
| 10 | Gemini client (cache + retry) | ✅ | `31a1fba`, `920eea4` | 16 tests; live call + cache verified |
| 11 | Prompt template and generation | ✅ | `ff64a4f` | 16 tests; grounding verified against off-corpus question |
| 12 | Pipeline | ✅ | `a093547` | 16 tests; full index built, end-to-end ask() works |
| 13 | CLI | ✅ | `bf41261`, `461c227` | 18 tests; Windows encoding crash fixed |
| 14 | README + no-frameworks guard | ✅ | `adc41d3`, `9f9c811`, `536dfb1` | 211 total; Quick start run end to end |
| — | Final whole-branch review | ✅ | `82d347a`..`858497d` | 5 Important + minors, all fixed |

**Phase 1 complete.** 246 tests passing, plus 4 slow tests against the real model.
Final review found 5 Important issues; all are fixed.

Working system: 38 documents → 5,116 chunks → 9.0 MB index; search 2–4 ms;
grounded answers with citations; `--no-llm` runs with no API key.

```
python -m rag index
python -m rag ask "How does ColBERT score a document?" --trace
```

Detail on findings, decisions and deviations:
[docs/superpowers/phase-1-notes.md](docs/superpowers/phase-1-notes.md)

---

# Phase 2 — Query Translation: Progress

Branch: `phase-2-query-translation` (stacked on `phase-1-core-pipeline`; neither merged to `main` yet)
Plan: [docs/superpowers/plans/2026-09-27-phase-2-query-translation.md](docs/superpowers/plans/2026-09-27-phase-2-query-translation.md)

| # | Task | Status | Commits | Notes |
|---|------|--------|---------|-------|
| 1 | Trace depth, score labels, translation steps | ⬜ | | fixes 3 deferred Phase 1 findings |
| 2 | Reciprocal rank fusion + best-score merge | ⬜ | | |
| 3 | Strategy protocol, context, direct | ⬜ | | |
| 4 | Multi-query | ⬜ | | |
| 5 | RAG-Fusion | ⬜ | | |
| 6 | Step-back | ⬜ | | |
| 7 | HyDE | ⬜ | | |
| 8 | Decomposition (recursive + independent) | ⬜ | | largest task |
| 9 | CLI `--strategy`, rendering, README | ⬜ | | |
| — | Final whole-branch review | ⬜ | | |
