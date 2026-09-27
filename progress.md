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
| 9 | Trace | ✅ | `673a407` | 8 tests; JSON contract verified |
| 10 | Gemini client (cache + retry) | ✅ | `31a1fba`, `920eea4` | 16 tests; live call + cache verified |
| 11 | Prompt template and generation | 🔄 | | |
| 12 | Pipeline | ⬜ | | |
| 13 | CLI | ⬜ | | builds real index |
| 14 | README + no-frameworks guard | ⬜ | | |
| — | Final whole-branch review | ⬜ | | |

## Minor findings deferred to final review

- **Task 1, Minor** — `Config` implementation is a verbatim transcription of the plan's
  reference code. Acceptable (the plan supplied working code), noted for the record.
- **Task 5, Minor (downgraded from Important, with evidence)** — the reviewer flagged
  that `chunk_document` trusts every tokenizer offset, so a zero-width or `(0,0)` offset
  would silently corrupt `char_start`/`char_end`. Sound reasoning, but measured against
  the real corpus it does not occur: 0 zero-width offsets, 0 `(0,0)` offsets and 0
  non-monotonic offsets across 766,278 tokens. Left unguarded; worth a cheap assertion
  if Phase 4 adds documents or the embedding model changes.
- **Task 5, Minor** — `window_bounds` duplicates the size/overlap validation in
  `Config.__post_init__`. Deliberate (the function is meant to be self-contained and
  exhaustively testable) but the two could drift.
- **Note on verification method** — the controller's initial "0 offset mismatches" check
  was tautological: `chunk.text` is assigned as `doc.text[char_start:char_end]`, so
  re-deriving that slice could never fail. Caught by the Task 5 reviewer. Replaced with
  a real check (token-surface correspondence and monotonicity over 766k tokens), which
  found zero genuine offset errors.
- **Task 8, Minor** — `str(data["chunks"])` pulls the JSON blob out of a 0-d NumPy
  array; `.item()` would be more idiomatic. Verified correct in practice: the reviewer
  round-tripped a 2 MB non-ASCII payload byte-for-byte with no truncation.
- **Plan error (mine), Task 9** — the plan's prose said "Expected: 9 passed" but the
  test code it specified contains 8 functions. No test was dropped; the count in the
  plan was simply wrong. Worth knowing before trusting the other per-task counts.
- **Task 7, Minor** — `cosine_similarity` re-normalises defensively, which is not
  bit-exact for input that is already unit-length float32: max abs difference ~6e-8, so
  a vector's self-similarity returns `1.0000001` rather than exactly `1.0`. Harmless at
  the tolerances used, but any future code that asserts an exact `1.0` will fail.
- **Task 7, Minor** — no test covers a 1-D `matrix` argument (only a 1-D query).
  `atleast_2d` treats it as a single item, matching convention, but it is untested.
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
- **Task 4, Important** — `rag/loader.py` raised raw `JSONDecodeError`/`KeyError` for a
  malformed `data/metadata.json`, naming neither the file nor the bad document. Against
  the loader's own design intent of failing strictly but comprehensibly. Originated in
  the plan's reference code. Fixed with actionable `ValueError`s (`e138637`).
- **Task 4, Important (round 2)** — a metadata *entry* whose value was not a mapping
  escaped the round-1 fix, since `except KeyError` does not catch `TypeError`. All three
  shapes (string, null, list) raised raw `TypeError` naming neither file nor document.
  Fixed with an `isinstance` guard (`f209fad`); verified all three now raise `ValueError`
  naming both. The loader thread is closed: every malformed-metadata shape is actionable.
- **Task 2, Important** — `nav` and `aside` were neither dropped nor treated as block
  elements, so sidebar chrome would splice inline into paragraph text. Added to the
  dropped set (`c89edde`). `header`/`footer` deliberately left in place: in academic
  HTML they carry title and authors.

## Decisions and deviations

- **API key works despite its unusual `AQ.` prefix.** Google AI Studio keys normally
  start with `AIza`; this one authenticated fine and lists 61 models. Stored in `.env`,
  which is gitignored — confirmed the key string appears in no tracked file.
- **Default model changed to `gemini-3.5-flash-lite`** (`920eea4`), replacing the
  unverified `gemini-2.0-flash` the plan guessed. Verified present in the account
  listing and confirmed with a live call returning `'pong'`, cached on repeat. It is
  genuinely the newest stable flash-lite tier (2.5 -> 3.1 -> 3.5; the 3.8 entries are
  text-to-speech). Pinned to an exact version rather than `gemini-flash-lite-latest`,
  because the spec requires reproducible benchmark numbers.

- **Brute-force search is comfortably fast at this scale**, which is the claim the README
  makes. Measured over the real 5,116-chunk index: 18.3 ms; over a synthetic matrix of
  the same shape: 6.3 ms. Index file is 9.0 MB. No approximate index needed or wanted
  here; the O(n) limit is documented rather than hidden.

- **Retrieval quality sanity check (Task 6).** Embedded 853 chunks spanning all 38
  documents and queried three of the plan's demo questions. "How does ColBERT score a
  document?" returns the ColBERT paper's own text at 0.55/0.53 — real semantic matching.
  But **the corpus has no dedicated source explaining reciprocal rank fusion**, the
  plan's headline demo question; it returns reranking papers at ~0.37. The plan already
  expected hits from `rag_survey`/`rankgpt`, so this is consistent rather than broken,
  but Phase 3's gold set should not assume a strong RRF answer exists. Consider adding
  a source that covers RRF directly when Phase 4 re-runs the fetch.
- **Similarity magnitudes run low** (~0.3 for a good match, ~0.55 for an excellent one).
  Normal for all-MiniLM-L6-v2 on long technical prose; absolute values matter less than
  ranking. Worth remembering before treating a 0.4 score as a weak result.

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
