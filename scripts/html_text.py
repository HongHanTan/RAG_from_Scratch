"""Turn HTML into clean plain text using only the standard library.

BeautifulSoup would be one line, but it is outside this project's dependency
budget and hides exactly the kind of mechanic the project exists to show.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Content inside these never belongs in the text.
DROPPED_TAGS = frozenset({"script", "style", "noscript", "head", "svg", "math"})

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
    """Collects visible text, inserting breaks at block boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in DROPPED_TAGS:
            self._drop_depth += 1
        elif tag == "br":
            self._parts.append("\n")
        elif tag in BLOCK_TAGS:
            self._parts.append(PARAGRAPH_BREAK)

    def handle_endtag(self, tag: str) -> None:
        if tag in DROPPED_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
        elif tag in BLOCK_TAGS:
            self._parts.append(PARAGRAPH_BREAK)

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return normalize_whitespace("".join(self._parts))


def html_to_text(html: str) -> str:
    """Extract normalised plain text from an HTML document."""
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    extractor.close()
    return extractor.text()
