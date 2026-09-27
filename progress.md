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
| 4 | Document loader | 🔄 | | |
| 5 | Token-aware chunking | ⬜ | | |
| 6 | Embeddings | ⬜ | | downloads model |
| 7 | Similarity and top-k | ⬜ | | |
| 8 | Vector store | ⬜ | | |
| 9 | Trace | ⬜ | | |
| 10 | Gemini client (cache + retry) | ⬜ | | needs API key to verify model id |
| 11 | Prompt template and generation | ⬜ | | |
| 12 | Pipeline | ⬜ | | |
| 13 | CLI | ⬜ | | builds real index |
| 14 | README + no-frameworks guard | ⬜ | | |
| — | Final whole-branch review | ⬜ | | |

## Minor findings deferred to final review

- **Task 1, Minor** — `Config` implementation is a verbatim transcription of the plan's
  reference code. Acceptable (the plan supplied working code), noted for the record.
- **Task 3, Minor** — `save_metadata` in `scripts/fetch_corpus.py` uses a single
  `write_text` (truncate-then-write), not a temp-file + `Path.replace` swap. A kill
  inside that window leaves `data/metadata.json` truncated, which makes `json.loads`
  raise on the next run — arguably worse than the missing entry the incremental write
  was added to prevent. Not a regression (the pre-fix code had the same single-shot
  write, just once instead of per-document). Its docstring also overclaims by saying
  "atomically-enough". Three-line fix: write `.tmp`, then `Path.replace`. Worth doing
  before Phase 4 re-runs the fetch to add 2024 papers.

## Resolved during execution

- **Task 2, Critical** — unclosed dropped tags (`<head>` without `</head>`, unclosed
  `<math>`) left the drop-depth counter stuck, silently discarding the rest of the
  document. Confirmed empirically: an omitted `</head>` returned an empty string.
  Would have put truncated papers into the committed corpus, invisibly. Fixed with
  stack-based drop tracking plus a recovery re-parse (`c89edde`).
- **Task 2, Important (round 2)** — the recovery re-parse demoted an unclosed tag by
  *name*, document-wide, so one malformed element exposed every well-formed element of
  the same name. Confirmed: `<script>A</script><p>keep</p><script>B` leaked the
  JavaScript `A`; the `<style>` variant leaked raw CSS into the text. Fixed by
  demoting per *occurrence* rather than per name (`b497946`).
- **Task 3, Important x2** — `metadata.json` was written only after the whole fetch
  loop, so an interrupted run left `.txt` files with no metadata entry, which Task 4's
  loader treats as a hard error. Proven real by a test that failed against the original
  code. Fixed with incremental per-document writes (`ae41c4f`). Same round added 6 tests
  for `main()`, which previously had none.
- **Task 2, Important** — `nav` and `aside` were neither dropped nor treated as block
  elements, so sidebar chrome would splice inline into paragraph text. Added to the
  dropped set (`c89edde`). `header`/`footer` deliberately left in place: in academic
  HTML they carry title and authors.

## Decisions and deviations

- **Corpus date split is thin.** Of 38 documents, 36 are pre-2024 and only 2 (`raptor`,
  `crag`) are 2024. Phase 4 demos `publish_date < 2024`, which technically works but
  filters out 36 of 38 — a weak demonstration. Cheap fix when Phase 4 arrives: add two
  or three 2024–2025 papers. Not worth a refetch now.
- **Publish dates are approximate.** They were written from domain knowledge, not read
  from arXiv, so some may be revision dates rather than original submission dates.
  Harmless for the date-filter demo, which only needs a field that partitions the
  corpus, but they are not authoritative metadata.
- **`splade.txt` opens with a ~53-char leaked LaTeX fragment.** Verified isolated: the
  reviewer grepped all 38 files for `\command{`, `itemjoin`, `inlinelist`, `[label=`,
  plus HTML/JS/MathML markers, and found no other occurrence anywhere. Not a systemic
  extraction defect and not fixable via `DROPPED_TAGS` (it is literal text, not markup),
  so left as-is.
