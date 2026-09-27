# Phase 1 — Core Pipeline: Progress

Branch: `phase-1-core-pipeline`
Plan: [docs/superpowers/plans/2026-09-27-phase-1-core-pipeline.md](docs/superpowers/plans/2026-09-27-phase-1-core-pipeline.md)
Spec: [docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md](docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md)

Status key: ⬜ not started · 🔄 in progress · ✅ complete

| # | Task | Status | Commits | Notes |
|---|------|--------|---------|-------|
| 1 | Scaffolding, Config, test fixtures | ✅ | `1ae1bad` | 9 tests pass; review clean |
| 2 | HTML text extraction | ✅ | `77f3c41`..`b497946` | 31 tests pass; 3 review rounds, all findings fixed |
| 3 | Corpus fetch script + fetch corpus | 🔄 | | needs network |
| 4 | Document loader | ⬜ | | |
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
- **Task 2, Important** — `nav` and `aside` were neither dropped nor treated as block
  elements, so sidebar chrome would splice inline into paragraph text. Added to the
  dropped set (`c89edde`). `header`/`footer` deliberately left in place: in academic
  HTML they carry title and authors.

## Decisions and deviations

_None yet._
