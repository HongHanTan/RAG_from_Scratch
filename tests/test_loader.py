import json

import pytest

from rag.config import Config
from rag.loader import Document, load_documents


def test_loads_every_document(tiny_corpus: Config):
    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert [d.doc_id for d in docs] == ["alpha", "beta"]


def test_attaches_metadata_to_text(tiny_corpus: Config):
    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    alpha = docs[0]
    assert isinstance(alpha, Document)
    assert alpha.title == "Cosine Similarity"
    assert alpha.publish_date == "2023-05-01"
    assert alpha.author == "A. Author"
    assert alpha.source == "notes"
    assert "Cosine similarity" in alpha.text


def test_result_is_sorted_by_doc_id_for_reproducible_chunk_ids(tiny_corpus: Config):
    (tiny_corpus.corpus_dir / "aardvark.txt").write_text("Zebra text.", encoding="utf-8")
    meta = json.loads(tiny_corpus.metadata_path.read_text(encoding="utf-8"))
    meta["documents"]["aardvark"] = {
        "title": "Aardvark", "source": "notes",
        "publish_date": "2022-01-01", "author": "C. Author", "url": None,
    }
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")

    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert [d.doc_id for d in docs] == ["aardvark", "alpha", "beta"]


def test_text_file_without_metadata_is_an_error(tiny_corpus: Config):
    (tiny_corpus.corpus_dir / "orphan.txt").write_text("No metadata.", encoding="utf-8")
    with pytest.raises(ValueError, match="orphan"):
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)


def test_metadata_without_text_file_is_an_error(tiny_corpus: Config):
    meta = json.loads(tiny_corpus.metadata_path.read_text(encoding="utf-8"))
    meta["documents"]["ghost"] = {
        "title": "Ghost", "source": "notes",
        "publish_date": "2022-01-01", "author": None, "url": None,
    }
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="ghost"):
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)


def test_empty_text_file_is_an_error(tiny_corpus: Config):
    (tiny_corpus.corpus_dir / "alpha.txt").write_text("   \n  ", encoding="utf-8")
    with pytest.raises(ValueError, match="alpha"):
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)


def test_missing_corpus_directory_is_an_error(tiny_corpus: Config):
    with pytest.raises(FileNotFoundError):
        load_documents(tiny_corpus.corpus_dir / "nope", tiny_corpus.metadata_path)


def test_optional_metadata_fields_default_to_none(tiny_corpus: Config):
    meta = {"documents": {"alpha": {"title": "T", "source": "notes"}, }}
    (tiny_corpus.corpus_dir / "beta.txt").unlink()
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    docs = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert docs[0].publish_date is None
    assert docs[0].author is None
    assert docs[0].url is None


def test_invalid_json_metadata_raises_actionable_value_error(tiny_corpus: Config):
    tiny_corpus.metadata_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON") as exc_info:
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    message = str(exc_info.value)
    assert str(tiny_corpus.metadata_path) in message
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_missing_documents_key_raises_actionable_value_error(tiny_corpus: Config):
    tiny_corpus.metadata_path.write_text(json.dumps({"not_documents": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="documents") as exc_info:
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert str(tiny_corpus.metadata_path) in str(exc_info.value)


def test_documents_key_that_is_not_a_mapping_raises_actionable_value_error(tiny_corpus: Config):
    tiny_corpus.metadata_path.write_text(json.dumps({"documents": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="documents") as exc_info:
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    assert str(tiny_corpus.metadata_path) in str(exc_info.value)


def test_metadata_entry_missing_title_raises_actionable_value_error(tiny_corpus: Config):
    meta = json.loads(tiny_corpus.metadata_path.read_text(encoding="utf-8"))
    del meta["documents"]["alpha"]["title"]
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="alpha") as exc_info:
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    message = str(exc_info.value)
    assert str(tiny_corpus.metadata_path) in message
    assert "title" in message


def test_metadata_entry_missing_source_raises_actionable_value_error(tiny_corpus: Config):
    meta = json.loads(tiny_corpus.metadata_path.read_text(encoding="utf-8"))
    del meta["documents"]["beta"]["source"]
    tiny_corpus.metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="beta") as exc_info:
        load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    message = str(exc_info.value)
    assert str(tiny_corpus.metadata_path) in message
    assert "source" in message
