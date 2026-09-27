import pytest

from scripts.fetch_corpus import (
    SOURCES,
    Source,
    fetch_text,
    metadata_record,
    source_url,
)


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
