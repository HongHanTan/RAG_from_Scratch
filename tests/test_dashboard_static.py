import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "rag" / "dashboard" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["index.html", "style.css", "app.js"])
def test_the_file_exists(name):
    assert (STATIC / name).is_file()


@pytest.mark.parametrize("panel", [
    "chunk-table", "strategy-comparison", "translation-trace",
    "stage-timings", "vector-projection",
])
def test_every_spec_panel_is_present(panel):
    assert panel in _read("index.html"), f"missing panel: {panel}"


def test_no_script_is_fetched_from_the_network():
    # A framework loaded from a CDN is still a framework wrapper, and the
    # project's premise is that none of this is one. This also keeps the
    # dashboard working offline.
    html = _read("index.html")
    remote = re.findall(r'<(?:script|link)[^>]+(?:src|href)\s*=\s*["\']https?://[^"\']+', html)
    assert not remote, f"remote asset(s): {remote}"


def test_no_front_end_framework_is_referenced():
    combined = (_read("index.html") + _read("app.js")).lower()
    for framework in ("react", "vue", "angular", "svelte", "streamlit", "jquery"):
        assert framework not in combined, f"references {framework}"


def test_each_panel_has_its_own_loading_state():
    # The LLM-heavy strategies take seconds; one shared spinner would block
    # the whole page on the slowest panel.
    html = _read("index.html")
    assert html.count("data-loading") >= 5


def test_the_projection_is_labelled_as_a_projection():
    # 384 dimensions do not fit in 2. The label stops a reader treating the
    # scatter as the embedding space.
    html = _read("index.html").lower()
    assert "projection" in html
    assert "not the space itself" in html


def test_the_page_calls_the_two_endpoints():
    js = _read("app.js")
    assert "/api/ask" in js
    assert "/api/compare" in js


def test_comparison_is_fetched_separately_from_ask():
    # If compare were awaited inside the same fetch as ask, the fast panels
    # would wait on the slow one, which is what the spec rules out.
    js = _read("app.js")
    assert js.count("fetch(") >= 2
