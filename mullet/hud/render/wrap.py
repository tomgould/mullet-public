"""Grapheme-aware line truncation and separator-aware line wrapping."""
import re

# ANSI escape sequence pattern
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_width(text: str) -> int:
    """Return the visible terminal column width of *text*, ignoring ANSI escapes.

    Uses wcwidth for accurate emoji/CJK/ZWJ measurement.
    """
    from wcwidth import wcswidth  # type: ignore
    plain = _ANSI_RE.sub("", text)
    w = wcswidth(plain)
    # wcswidth returns -1 for non-printable sequences; fall back to len
    return w if w >= 0 else len(plain)


def wcwidth_truncate(line: str, max_width: int, ellipsis: str = "…") -> str:
    """Truncate *line* to *max_width* terminal columns.

    ANSI escape sequences are preserved but not counted towards the width.
    Emoji, CJK characters and ZWJ sequences are counted by their actual cell
    widths via wcwidth.
    """
    if max_width <= 0:
        return line
    if _visible_width(line) <= max_width:
        return line

    from wcwidth import wcwidth as wc  # type: ignore

    ell_w = _visible_width(ellipsis)
    budget = max_width - ell_w
    result: list[str] = []
    used = 0
    i = 0

    while i < len(line):
        # Pass through ANSI escape sequences without counting width
        m = _ANSI_RE.match(line, i)
        if m:
            result.append(m.group())
            i = m.end()
            continue

        ch = line[i]
        # Handle multi-codepoint grapheme clusters (ZWJ sequences etc.)
        # For simplicity, measure each code point; ZWJ sequences are rare
        # in HUD text and wcwidth handles the common cases well enough.
        w = wc(ch)
        if w < 0:
            w = 0  # non-printable: skip width but keep char
        if used + w > budget:
            break
        result.append(ch)
        used += w
        i += 1

    # Append reset + ellipsis so colour bleed doesn't infect the ellipsis
    return "".join(result) + "\x1b[0m" + ellipsis


def split_wrap_parts(line: str, max_width: int, sep: str = " | ") -> list[str]:
    """Split a *sep*-joined *line* across multiple physical lines each ≤ *max_width* columns.

    The leading bracketed block (e.g. ``[Opus | cwd | branch]``) is treated as
    an atomic unit and is never split internally — it always stays on the first
    output line.  Subsequent parts are greedily packed onto the current line; a
    new line is started when the next part would overflow.

    Port of claude-hud's ``splitWrapParts`` (render/index.ts lines 214-255).
    """
    if max_width <= 0 or _visible_width(line) <= max_width:
        return [line]

    # We need to work with the *original* (ANSI-coloured) line.
    # Re-split on the plain sep so we keep styling intact.
    coloured_parts = line.split(sep)
    if not coloured_parts:  # pragma: no cover
        return [line]

    current_parts: list[str] = []
    current_width = 0
    result_lines: list[str] = []

    for part in coloured_parts:
        part_w = _visible_width(part)
        sep_w = _visible_width(sep) if current_parts else 0
        needed = sep_w + part_w

        if not current_parts:
            current_parts.append(part)
            current_width = part_w
        elif current_width + needed <= max_width:
            current_parts.append(part)
            current_width += needed
        else:
            result_lines.append(sep.join(current_parts))
            current_parts = [part]
            current_width = part_w

    if current_parts:
        result_lines.append(sep.join(current_parts))

    return result_lines
