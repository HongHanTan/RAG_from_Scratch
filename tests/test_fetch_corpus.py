import json
import urllib.error

import pytest

import scripts.fetch_corpus as fetch_corpus
from scripts.fetch_corpus import (
    MIN_CHARS,
    SOURCES,
    Source,
    fetch_text,
    main,
    metadata_record,
    source_url,
)

LONG_TEXT = "x" * (MIN_CHARS + 100)
SHORT_TEXT = "too short"


def test_arxiv_sources_resolve_to_ar5iv_html():
    src = Source(
        doc_id="raptor",
        kind="arxiv",
        ref="2401.18059",
        title="RAPTOR",
        author="Sarthi et al.",
        publish_date="2024-01-31",
    )
    assert source_url(src) == "https://ar5iv.labs.arxiv.org/html/2401.18059"


def test_web_sources_use_their_ref_verbatim():
    src = Source(
        doc_id="blog",
        kind="web",
        ref="https://example.invalid/post",
        title="A Post",
        author="Someone",
        publish_date="2024-03-01",
    )
    assert source_url(src) == "https://example.invalid/post"


def test_unknown_kind_is_rejected():
    src = Source("x", "carrier-pigeon", "ref", "T", "A", "2024-01-01")
    with pytest.raises(ValueError, match="carrier-pigeon"):
        source_url(src)


def test_metadata_record_has_every_field_the_loader_expects():
    src = Source(
        doc_id="raptor",
        kind="arxiv",
        ref="2401.18059",
        title="RAPTOR",
        author="Sarthi et al.",
        publish_date="2024-01-31",
    )
    record = metadata_record(src)
    assert record == {
        "title": "RAPTOR",
        "source": "arxiv",
        "publish_date": "2024-01-31",
        "author": "Sarthi et al.",
        "url": "https://ar5iv.labs.arxiv.org/html/2401.18059",
    }


def test_fetch_text_strips_markup_via_the_injected_opener():
    class FakeResponse:
        def read(self):
            return b"<html><body><p>Hello</p><script>x</script></body></html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_opener(request, timeout=0):
        return FakeResponse()

    assert fetch_text("https://example.invalid", opener=fake_opener) == "Hello"


def test_doc_ids_are_unique():
    ids = [s.doc_id for s in SOURCES]
    assert len(ids) == len(set(ids))


def test_doc_ids_are_safe_filenames():
    import re
    for s in SOURCES:
        assert re.fullmatch(r"[a-z0-9_]+", s.doc_id), s.doc_id


def test_corpus_is_large_enough_for_raptor_clustering():
    assert len(SOURCES) >= 35


def test_publish_dates_are_iso_and_span_the_2024_boundary():
    import datetime
    dates = [datetime.date.fromisoformat(s.publish_date) for s in SOURCES]
    assert any(d.year < 2024 for d in dates)
    assert any(d.year >= 2024 for d in dates)


# --- main() CLI loop -------------------------------------------------------


def _read_documents(metadata_path):
    return json.loads(metadata_path.read_text(encoding="utf-8"))["documents"]


def test_skip_if_exists_does_not_refetch_but_writes_metadata(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"

    doc_id = SOURCES[0].doc_id
    existing_path = corpus_dir / f"{doc_id}.txt"
    existing_path.write_text("PRE-EXISTING CONTENT", encoding="utf-8")

    def fake_fetch_text(url):
        raise AssertionError("fetch_text should not be called for an existing file")

    monkeypatch.setattr(fetch_corpus, "fetch_text", fake_fetch_text)

    rc = main([
        "--corpus-dir", str(corpus_dir),
        "--metadata", str(metadata_path),
        "--only", doc_id,
        "--delay", "0",
    ])

    assert rc == 0
    assert existing_path.read_text(encoding="utf-8") == "PRE-EXISTING CONTENT"
    documents = _read_documents(metadata_path)
    assert doc_id in documents
    assert documents[doc_id] == metadata_record(SOURCES[0])


def test_force_refetches_existing_file(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"

    doc_id = SOURCES[0].doc_id
    existing_path = corpus_dir / f"{doc_id}.txt"
    existing_path.write_text("STALE CONTENT", encoding="utf-8")

    monkeypatch.setattr(fetch_corpus, "fetch_text", lambda url: LONG_TEXT)

    rc = main([
        "--corpus-dir", str(corpus_dir),
        "--metadata", str(metadata_path),
        "--only", doc_id,
        "--force",
        "--delay", "0",
    ])

    assert rc == 0
    assert existing_path.read_text(encoding="utf-8") == LONG_TEXT
    documents = _read_documents(metadata_path)
    assert documents[doc_id] == metadata_record(SOURCES[0])


def test_only_fetches_the_requested_document(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"

    doc_id = SOURCES[1].doc_id
    calls = []

    def fake_fetch_text(url):
        calls.append(url)
        return LONG_TEXT

    monkeypatch.setattr(fetch_corpus, "fetch_text", fake_fetch_text)

    rc = main([
        "--corpus-dir", str(corpus_dir),
        "--metadata", str(metadata_path),
        "--only", doc_id,
        "--delay", "0",
    ])

    assert rc == 0
    assert calls == [source_url(SOURCES[1])]
    txt_files = sorted(p.name for p in corpus_dir.glob("*.txt"))
    assert txt_files == [f"{doc_id}.txt"]
    documents = _read_documents(metadata_path)
    assert list(documents.keys()) == [doc_id]


def test_failing_fetch_leaves_no_file_and_no_metadata_and_nonzero_exit(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"

    doc_id = SOURCES[0].doc_id

    def fake_fetch_text(url):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(fetch_corpus, "fetch_text", fake_fetch_text)

    rc = main([
        "--corpus-dir", str(corpus_dir),
        "--metadata", str(metadata_path),
        "--only", doc_id,
        "--delay", "0",
    ])

    assert rc == 1
    assert not (corpus_dir / f"{doc_id}.txt").exists()
    if metadata_path.is_file():
        assert doc_id not in _read_documents(metadata_path)


def test_text_below_min_chars_is_rejected_like_a_failure(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"

    doc_id = SOURCES[0].doc_id

    monkeypatch.setattr(fetch_corpus, "fetch_text", lambda url: SHORT_TEXT)

    rc = main([
        "--corpus-dir", str(corpus_dir),
        "--metadata", str(metadata_path),
        "--only", doc_id,
        "--delay", "0",
    ])

    assert rc == 1
    assert not (corpus_dir / f"{doc_id}.txt").exists()
    if metadata_path.is_file():
        assert doc_id not in _read_documents(metadata_path)


def test_interruption_partway_through_leaves_metadata_consistent_with_disk(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"

    doc_ids = [SOURCES[0].doc_id, SOURCES[1].doc_id, SOURCES[2].doc_id]
    calls = []

    def fake_fetch_text(url):
        calls.append(url)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        return LONG_TEXT

    monkeypatch.setattr(fetch_corpus, "fetch_text", fake_fetch_text)

    argv = [
        "--corpus-dir", str(corpus_dir),
        "--metadata", str(metadata_path),
        "--delay", "0",
    ]
    for doc_id in doc_ids:
        argv += ["--only", doc_id]

    with pytest.raises(KeyboardInterrupt):
        main(argv)

    # Exactly the first two documents made it to disk before the interrupt...
    txt_files = sorted(p.name for p in corpus_dir.glob("*.txt"))
    assert txt_files == sorted([f"{doc_ids[0]}.txt", f"{doc_ids[1]}.txt"])

    # ...and metadata.json must describe exactly those, not the interrupted third.
    documents = _read_documents(metadata_path)
    assert set(documents.keys()) == {doc_ids[0], doc_ids[1]}


def test_save_metadata_leaves_no_temporary_file(tmp_path):
    from scripts.fetch_corpus import save_metadata

    target = tmp_path / "metadata.json"
    save_metadata(target, {"a": {"title": "A"}})
    assert [p.name for p in tmp_path.iterdir()] == ["metadata.json"]


def test_save_metadata_does_not_destroy_the_old_file_on_failure(tmp_path):
    import json as _json

    from scripts.fetch_corpus import save_metadata

    target = tmp_path / "metadata.json"
    save_metadata(target, {"a": {"title": "A"}})
    good = target.read_text(encoding="utf-8")

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        save_metadata(target, {"a": Unserialisable()})
    assert target.read_text(encoding="utf-8") == good
    assert _json.loads(good)["documents"]["a"]["title"] == "A"
