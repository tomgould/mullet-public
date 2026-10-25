"""Model text made safe to print or commit: no terminal escape sequences and no control characters.

Planner, worker and reviewer replies reach the terminal (`mullet job show`,
the ladder report) and commit messages. An escape sequence in them could clear
the screen, set the window title or hide text, and a newline in a one-line
field could fake a section header in the approval view, so everything printed
from a model goes through `one_line` (a single-line field) or `strip_control`
(text whose line breaks matter, such as a diff or a quoted claim).
"""

from __future__ import annotations

import re
from typing import Any

# A whole escape sequence goes, not just its ESC: CSI (`ESC [ ... final`), the
# string types OSC, DCS, SOS, PM and APC (up to BEL or ST, or the end of the
# text), other short escapes, and the 8-bit CSI.
_ESCAPES = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b[\]PX^_][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x1b[ -/]*[0-~]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
)
# C0 and C1 controls except tab and newline, and the bidi controls that can
# make text read differently from how it is stored.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f‎‏‪-‮⁦-⁩]")


def strip_control(text: str) -> str:
    """`text` without escape sequences or control characters; newlines and tabs stay, carriage returns go."""
    return _CONTROL.sub("", _ESCAPES.sub("", text))


def one_line(text: Any, limit: int | None = None) -> str:
    """`text` sanitised and flattened to one line of single spaces, cut to `limit` characters with "..."."""
    flat = " ".join(strip_control(str(text)).split())
    if limit is not None and len(flat) > limit:
        return flat[:limit].rstrip() + "..."
    return flat
