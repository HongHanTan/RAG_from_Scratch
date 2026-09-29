from pathlib import Path

from rag.config import Config, load_env_file


def test_defaults_match_the_spec():
    cfg = Config()
    assert cfg.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.llm_model == "gemini-3.5-flash-lite"
    assert cfg.chunk_tokens == 200
    assert cfg.chunk_overlap == 50
    assert cfg.max_seq_tokens == 256
    assert cfg.top_k == 5


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    import pytest
    with pytest.raises(ValueError, match="overlap"):
        Config(chunk_tokens=100, chunk_overlap=100)


def test_chunk_size_must_fit_the_model_window():
    import pytest
    with pytest.raises(ValueError, match="max_seq_tokens"):
        Config(chunk_tokens=300, max_seq_tokens=256)


def test_from_env_reads_the_api_key():
    cfg = Config.from_env(env={"GOOGLE_API_KEY": "abc123"})
    assert cfg.api_key == "abc123"


def test_from_env_leaves_api_key_none_when_unset():
    cfg = Config.from_env(env={})
    assert cfg.api_key is None


def test_overrides_beat_the_environment():
    cfg = Config.from_env(env={"GOOGLE_API_KEY": "abc123"}, top_k=9)
    assert cfg.top_k == 9
    assert cfg.api_key == "abc123"


def test_load_env_file_parses_pairs_and_ignores_comments(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "\n"
        "GOOGLE_API_KEY=secret\n"
        "QUOTED=\"with spaces\"\n"
        "  SPACED  =  padded  \n",
        encoding="utf-8",
    )
    assert load_env_file(p) == {
        "GOOGLE_API_KEY": "secret",
        "QUOTED": "with spaces",
        "SPACED": "padded",
    }


def test_load_env_file_returns_empty_when_missing(tmp_path: Path):
    assert load_env_file(tmp_path / "nope.env") == {}


def test_env_file_is_a_fallback_not_an_override(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text("GOOGLE_API_KEY=from-file\n", encoding="utf-8")
    cfg = Config.from_env(env={"GOOGLE_API_KEY": "from-env"}, env_file=p)
    assert cfg.api_key == "from-env"


def test_retrieval_depth_defaults_above_top_k():
    # Each query retrieves deeper than the final answer needs, so fusion has
    # material to work with. If they were equal, RAG-Fusion would fuse five
    # 5-item lists down to 5 and could not differ much from a plain union.
    cfg = Config()
    assert cfg.retrieval_depth == 20
    assert cfg.retrieval_depth >= cfg.top_k


def test_retrieval_depth_below_top_k_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="retrieval_depth"):
        Config(retrieval_depth=3, top_k=5)


def test_raptor_defaults():
    cfg = Config()
    assert cfg.raptor_max_depth == 3
    assert cfg.raptor_cluster_size == 8


def test_raptor_max_depth_must_be_positive():
    import pytest
    with pytest.raises(ValueError, match="raptor_max_depth"):
        Config(raptor_max_depth=0)


def test_rerank_depth_default():
    assert Config().rerank_depth == 50


def test_rerank_depth_must_be_at_least_top_k():
    import pytest
    with pytest.raises(ValueError, match="rerank_depth"):
        Config(rerank_depth=3, top_k=5)
