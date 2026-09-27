from scripts.html_text import html_to_text, normalize_whitespace


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
