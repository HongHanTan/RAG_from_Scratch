from scripts.html_text import HTMLTextExtractor, html_to_text, normalize_whitespace


def test_normalize_collapses_runs_of_spaces():
    assert normalize_whitespace("a   b\t\tc") == "a b c"


def test_normalize_keeps_paragraph_breaks_but_collapses_bigger_gaps():
    assert normalize_whitespace("one\n\n\n\n\ntwo") == "one\n\ntwo"


def test_normalize_strips_leading_and_trailing_space():
    assert normalize_whitespace("  hello  ") == "hello"


def test_extracts_visible_text():
    html = "<html><body><p>Hello world</p></body></html>"
    assert html_to_text(html) == "Hello world"


def test_drops_script_and_style_content():
    html = (
        "<html><head><style>p { color: red }</style></head>"
        "<body><p>Keep me</p><script>alert('no')</script></body></html>"
    )
    assert html_to_text(html) == "Keep me"


def test_block_elements_become_paragraph_breaks():
    html = "<p>First para</p><p>Second para</p>"
    assert html_to_text(html) == "First para\n\nSecond para"


def test_inline_elements_do_not_split_words():
    html = "<p>re<em>trie</em>val</p>"
    assert html_to_text(html) == "retrieval"


def test_headings_are_kept_as_their_own_paragraphs():
    html = "<h2>Method</h2><p>We propose</p>"
    assert html_to_text(html) == "Method\n\nWe propose"


def test_entities_are_decoded():
    assert html_to_text("<p>alpha &amp; beta &lt;tag&gt;</p>") == "alpha & beta <tag>"


def test_br_becomes_a_single_newline_not_a_paragraph_break():
    assert html_to_text("<p>line one<br>line two</p>") == "line one\nline two"


def test_empty_input_yields_empty_string():
    assert html_to_text("") == ""


def test_omitted_head_close_tag_does_not_lose_body():
    # </head> is legal to omit in HTML5. A <body> start tag must implicitly
    # close a still-open <head> so the rest of the document is not dropped.
    html = "<html><head><title>T</title><body><p>Keep me</p></body></html>"
    assert html_to_text(html) == "Keep me"


def test_unclosed_math_recovers_rest_of_document():
    # A never-closed <math> element used to leave the drop-counter stuck
    # above zero, silently discarding everything after it. The recovery
    # pass must surface the trailing text instead of losing it.
    html = "<p>before</p><math><mi>x</mi><p>after</p>"
    result = html_to_text(html)
    assert "before" in result
    assert "after" in result
    assert result != "before"


def test_stray_close_tag_with_no_open_is_ignored():
    html = "<p>One</p></style><p>Two</p>"
    assert html_to_text(html) == "One\n\nTwo"


def test_self_closing_dropped_tag_does_not_swallow_rest():
    html = "<p>a</p><math/><p>b</p>"
    assert html_to_text(html) == "a\n\nb"


def test_nav_and_aside_are_dropped():
    html = "<nav>Menu</nav><p>Real content</p><aside>Sidebar</aside>"
    assert html_to_text(html) == "Real content"


def test_unclosed_attribute_reports_tags_left_open():
    extractor = HTMLTextExtractor()
    extractor.feed("<p>before</p><math><mi>x</mi>")
    extractor.close()
    assert extractor.unclosed == ["math"]


def test_unclosed_attribute_is_empty_when_all_tags_close():
    extractor = HTMLTextExtractor()
    extractor.feed("<p>before</p><math><mi>x</mi></math>")
    extractor.close()
    assert extractor.unclosed == []
