import pytest
from fastapi.testclient import TestClient

from rag.dashboard.app import create_app
from rag.pipeline import build_index
from tests.conftest import FakeEmbedder, FakeLLM


@pytest.fixture
def client(tiny_corpus):
    store = build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    app = create_app(store, FakeEmbedder(), FakeLLM("an answer"), tiny_corpus)
    return TestClient(app)


# --- meta ---------------------------------------------------------------------

def test_meta_reports_the_index(client):
    body = client.get("/api/meta").json()
    assert body["chunks"] > 0
    assert body["dim"] == 8
    assert "direct" in body["strategies"]


# --- ask ----------------------------------------------------------------------

def test_ask_returns_a_trace(client):
    body = client.get("/api/ask", params={"q": "cosine"}).json()
    assert body["question"] == "cosine"
    assert body["retrieved"]


def test_ask_includes_the_score_kind(client):
    # The chunk table shows raw scores; cosine and maxsim are different
    # quantities and the panel must not present them as one.
    body = client.get("/api/ask", params={"q": "cosine"}).json()
    assert all("score_kind" in r for r in body["retrieved"])


def test_ask_includes_stage_timings(client):
    body = client.get("/api/ask", params={"q": "cosine"}).json()
    assert body["timings"]
    assert all({"name", "ms"} <= set(t) for t in body["timings"])


def test_ask_includes_a_projection_for_every_chunk(client):
    body = client.get("/api/ask", params={"q": "cosine"}).json()
    assert len(body["projection"]["chunks"]) == len(body["retrieved"])
    assert len(body["projection"]["query"]) == 2


def test_ask_honours_k(client):
    body = client.get("/api/ask", params={"q": "cosine", "k": 2}).json()
    assert len(body["retrieved"]) <= 2


def test_ask_honours_the_strategy(client):
    body = client.get("/api/ask", params={"q": "cosine", "strategy": "multi-query"}).json()
    assert body["strategy"] == "multi-query"


def test_ask_exposes_the_translation_trace(client):
    body = client.get("/api/ask", params={"q": "cosine", "strategy": "multi-query"}).json()
    assert "translation" in body


def test_an_unknown_strategy_is_a_400_not_a_500(client):
    response = client.get("/api/ask", params={"q": "cosine", "strategy": "nope"})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_an_empty_question_is_a_400(client):
    assert client.get("/api/ask", params={"q": "   "}).status_code == 400


# --- compare ------------------------------------------------------------------

def test_compare_returns_one_result_per_strategy(client):
    body = client.get(
        "/api/compare", params={"q": "cosine", "strategies": "direct,multi-query"}
    ).json()
    assert [r["strategy"] for r in body["results"]] == ["direct", "multi-query"]


def test_compare_rejects_an_unknown_strategy(client):
    response = client.get(
        "/api/compare", params={"q": "cosine", "strategies": "direct,nope"}
    )
    assert response.status_code == 400


def test_compare_defaults_to_every_strategy(client):
    body = client.get("/api/compare", params={"q": "cosine"}).json()
    assert len(body["results"]) >= 6


# --- read-only ----------------------------------------------------------------

@pytest.mark.parametrize("method", ["post", "put", "delete", "patch"])
def test_the_api_is_read_only(client, method):
    # The dashboard inspects; it must not offer a way to mutate an index.
    response = getattr(client, method)("/api/ask", params={"q": "x"})
    assert response.status_code == 405


def test_the_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
