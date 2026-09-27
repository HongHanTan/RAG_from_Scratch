"""Turn HTML into clean plain text using only the standard library.

BeautifulSoup would be one line, but it is outside this project's dependency
budget and hides exactly the kind of mechanic the project exists to show.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Content inside these never belongs in the text.
DROPPED_TAGS = frozenset(
    {"script", "style", "noscript", "head", "svg", "math", "nav", "aside"}
)

# These force a paragraph break around their content.
BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "header", "footer", "li", "tr",
        "blockquote", "pre", "figure", "figcaption", "table",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
)

PARAGRAPH_BREAK = "\n\n"


def normalize_whitespace(text: str) -> str:
    """Collapse spaces and runs of blank lines, preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)          # spaces/tabs, but not newlines
    text = re.sub(r" *\n *", "\n", text)           # trim around newlines
    text = re.sub(r"\n{3,}", PARAGRAPH_BREAK, text)  # 3+ newlines -> exactly one break
    return text.strip()


class HTMLTextExtractor(HTMLParser):
    """Collects visible text, inserting breaks at block boundaries.

    Dropped-tag content is tracked with a stack of tag *names* rather than a
    bare depth counter. ``html.parser.HTMLParser`` is a pure tokenizer: it
    never infers an omitted end tag, so an unclosed dropped element (e.g. a
    stray ``<math>`` with no ``</math>``) would otherwise leave a counter
    stuck above zero and silently discard the rest of the document. With a
    stack, an end tag unwinds to its most recent matching open tag (ignoring
    it entirely if it never appears on the stack, which preserves the
    behaviour for a stray close tag with no open), and whatever is still on
    the stack when parsing ends is recorded on ``unclosed`` so the caller can
    react to it.
    """

    def __init__(self, dropped_tags: frozenset[str] = DROPPED_TAGS) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._dropped_tags = dropped_tags
        self._drop_stack: list[str] = []
        self.unclosed: list[str] = []

    def _unwind_to(self, tag: str) -> bool:
        """Pop the stack up to and including the last occurrence of tag.

        Returns True if tag was found (and the stack was unwound), False if
        it was not on the stack at all (in which case the stack is left
        untouched -- this is what makes a stray close tag a no-op).
        """
        for index in range(len(self._drop_stack) - 1, -1, -1):
            if self._drop_stack[index] == tag:
                del self._drop_stack[index:]
                return True
        return False

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "body":
            # HTML5 permits omitting </head>; a <body> start implicitly
            # closes any still-open head so the rest of the document is
            # not mistaken for head content.
            self._unwind_to("head")
        if tag in self._dropped_tags:
            self._drop_stack.append(tag)
        elif tag == "br":
            self._parts.append("\n")
        elif tag in BLOCK_TAGS:
            self._parts.append(PARAGRAPH_BREAK)

    def handle_endtag(self, tag: str) -> None:
        if self._unwind_to(tag):
            return
        if tag in BLOCK_TAGS:
            self._parts.append(PARAGRAPH_BREAK)

    def handle_data(self, data: str) -> None:
        if not self._drop_stack:
            self._parts.append(data)

    def close(self) -> None:
        super().close()
        self.unclosed = list(self._drop_stack)

    def text(self) -> str:
        return normalize_whitespace("".join(self._parts))


def html_to_text(html: str) -> str:
    """Extract normalised plain text from an HTML document.

    If a dropped tag (e.g. ``<math>``) is left unclosed, the tokenizer has no
    way to know where its content should have ended, and everything after it
    would otherwise be discarded. Rather than raise -- which would cost the
    caller an entire document over one malformed tag -- this does exactly one
    recovery pass: it re-parses with those specific tag names demoted out of
    the dropped set, so their content (and everything after them) is
    recovered as visible text instead of silently vanishing.
    """
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    extractor.close()
    if not extractor.unclosed:
        return extractor.text()

    recovered_dropped_tags = DROPPED_TAGS - set(extractor.unclosed)
    recovery = HTMLTextExtractor(dropped_tags=recovered_dropped_tags)
    recovery.feed(html)
    recovery.close()
    return recovery.text()
