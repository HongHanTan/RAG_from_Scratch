"""Atomic file writes.

A direct write truncates the target before writing it, so an interruption
leaves a truncated file. That is worse than no file at all: the next run fails
on a parse error rather than on a clean absence, and the parse error points at
the data rather than at the interruption that caused it.

Writing to a temporary file beside the target and then renaming avoids that.
`Path.replace` is atomic when both paths are on the same filesystem, which is
why the temporary lives in the target's own directory rather than in /tmp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a temporary file, then atomically move it into place.

    `write` receives the temporary path and must write the whole file. If it
    raises, the temporary is removed and any existing target is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        write(tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
