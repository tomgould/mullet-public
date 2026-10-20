"""Deterministic, non-LLM diff-risk classifier.

Delegate LLMs occasionally self-report "completed successfully" while having
silently corrupted a literal, flattened curly quotes to ASCII, changed a
test's expected value, or hallucinated an extra call argument. Those
self-reports are not a trustworthy signal on their own.

This module provides a cheap, deterministic (regex/difflib only — no LLM
call, zero cost, fast) heuristic classifier that flags suspicious hunks in a
diff so callers get a `risk_flags` list to inspect instead of blindly
trusting `completed: true`.
"""

import re
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class RiskFlag:
    file: str
    line: int          # line number in the new file (1-indexed)
    category: str      # "literal_change" | "quote_flattening" | "new_control_flow" |
                        # "assertion_on_literal" | "call_args_changed" |
                        # "comment_changed" | "test_assertion_changed"
    old_snippet: str
    new_snippet: str
    reason: str         # short human-readable explanation


_STRING_LITERAL_RE = re.compile(
    r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|`((?:[^`\\]|\\.)*)`"
)
_NUMERIC_LITERAL_RE = re.compile(r"\b\d+\.?\d*\b")

_CURLY_QUOTES = "‘’“”"
_STRAIGHT_QUOTES = "'\""

_CONTROL_FLOW_TOKENS = ["if (", "if(", "while (", "while(", "&&", "||", "??"]
# Bare ternary "?" — exclude optional chaining ("?.") and optional-param /
# type-suffix ("?:") uses, which are not control flow.
_TERNARY_RE = re.compile(r"\?(?![.:])")

_CALL_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*\(([^()]*)\)")
_CAST_RE = re.compile(r"\bas\s+[A-Za-z_$][\w$.<>\[\],\s]*$")
_TYPE_ANNOTATION_RE = re.compile(r":\s*[A-Za-z_$][\w$.<>\[\]]*")

_TEST_ASSERTION_NAMES = {
    "toBe", "toEqual", "toHaveLength", "toMatch", "toContain", "expect",
}


def _extract_string_literals(line: str) -> list[tuple[int, int, str]]:
    """Return (start, end, content) for each quoted string literal, in order."""
    result = []
    for m in _STRING_LITERAL_RE.finditer(line):
        content = next(g for g in m.groups() if g is not None)
        result.append((m.start(), m.end(), content))
    return result


def _mask_spans(line: str, spans: list[tuple[int, int, str]]) -> str:
    """Blank out the given character spans (used to hide string literal contents
    before scanning for numeric literals, so digits inside strings aren't
    mistaken for standalone numeric literals)."""
    chars = list(line)
    for start, end, _ in spans:
        for idx in range(start, end):
            chars[idx] = " "
    return "".join(chars)


def _check_literal_change(old_line: str, new_line: str) -> str | None:
    old_strings = [c for _, _, c in _extract_string_literals(old_line)]
    new_strings = [c for _, _, c in _extract_string_literals(new_line)]
    for old_val, new_val in zip(old_strings, new_strings):
        if old_val != new_val:
            return f"string literal changed: {old_val!r} -> {new_val!r}"

    old_masked = _mask_spans(old_line, _extract_string_literals(old_line))
    new_masked = _mask_spans(new_line, _extract_string_literals(new_line))
    old_nums = _NUMERIC_LITERAL_RE.findall(old_masked)
    new_nums = _NUMERIC_LITERAL_RE.findall(new_masked)
    for old_val, new_val in zip(old_nums, new_nums):
        if old_val != new_val:
            return f"numeric literal changed: {old_val} -> {new_val}"

    return None


def _check_quote_flattening(old_line: str, new_line: str) -> bool:
    for i, ch in enumerate(old_line):
        if ch not in _CURLY_QUOTES:
            continue
        window = range(max(0, i - 2), min(len(new_line), i + 3))
        for j in window:
            if new_line[j] in _STRAIGHT_QUOTES:
                return True
    return False


def _check_new_control_flow(old_line: str, new_line: str) -> tuple[bool, str | None]:
    for token in _CONTROL_FLOW_TOKENS:
        if token in new_line and token not in old_line:
            return True, token

    old_ternary = len(_TERNARY_RE.findall(old_line))
    new_ternary = len(_TERNARY_RE.findall(new_line))
    if new_ternary > old_ternary:
        return True, "?"

    return False, None


def _check_assertion_on_literal(old_line: str, new_line: str) -> bool:
    pattern = re.compile(r"['\"`\]}]!")
    for m in pattern.finditer(new_line):
        idx = m.end() - 1  # index of the "!" itself
        if idx >= len(old_line) or old_line[idx] != "!":
            return True
    return False


def _normalize_args(args: str) -> str:
    args = _CAST_RE.sub("", args)
    args = _TYPE_ANNOTATION_RE.sub("", args)
    return re.sub(r"\s+", "", args)


def _check_call_args(old_line: str, new_line: str) -> list[tuple[str, str]]:
    old_calls = _CALL_RE.findall(old_line)
    new_calls = _CALL_RE.findall(new_line)
    is_test_line = "assert" in new_line or any(
        name in _TEST_ASSERTION_NAMES for name, _ in new_calls
    )

    seen_categories: dict[str, str] = {}
    for (old_name, old_args), (new_name, new_args) in zip(old_calls, new_calls):
        if old_name != new_name:
            continue
        if _normalize_args(old_args) == _normalize_args(new_args):
            continue
        reason = (
            f"Argument(s) to {new_name}(...) changed: "
            f"{old_args.strip()!r} -> {new_args.strip()!r}"
        )
        if new_name in _TEST_ASSERTION_NAMES or is_test_line:
            seen_categories.setdefault("test_assertion_changed", reason)
        else:
            seen_categories.setdefault("call_args_changed", reason)

    return list(seen_categories.items())


def _extract_comment(line: str) -> str | None:
    idx = line.find("//")
    if idx != -1:
        return line[idx + 2:].strip()
    m = re.search(r"/\*(.*?)\*/", line)
    if m:
        return m.group(1).strip()
    return None


def _check_comment_changed(old_line: str, new_line: str) -> bool:
    old_comment = _extract_comment(old_line)
    new_comment = _extract_comment(new_line)
    if old_comment is not None and new_comment is not None:
        return old_comment != new_comment
    return False


def _classify_line_pair(
    old_line: str, new_line: str, filename: str, lineno: int
) -> list[RiskFlag]:
    if old_line == new_line:
        return []

    flags: list[RiskFlag] = []

    def add(category: str, reason: str) -> None:
        flags.append(
            RiskFlag(
                file=filename,
                line=lineno,
                category=category,
                old_snippet=old_line,
                new_snippet=new_line,
                reason=reason,
            )
        )

    literal_reason = _check_literal_change(old_line, new_line)
    if literal_reason:
        add("literal_change", literal_reason)

    if _check_quote_flattening(old_line, new_line):
        add("quote_flattening", "curly quote flattened to straight ASCII quote")

    cf_hit, cf_token = _check_new_control_flow(old_line, new_line)
    if cf_hit:
        add("new_control_flow", f"new control-flow token introduced: {cf_token!r}")

    if _check_assertion_on_literal(old_line, new_line):
        add(
            "assertion_on_literal",
            "non-null assertion (!) applied directly to a literal",
        )

    if _check_comment_changed(old_line, new_line):
        add("comment_changed", "comment text changed")

    for category, reason in _check_call_args(old_line, new_line):
        add(category, reason)

    return flags


def classify_diff(
    old_content: str, new_content: str, filename: str = ""
) -> list[RiskFlag]:
    """Classify a diff between old_content and new_content into RiskFlags.

    Only "replace" opcodes (an old line region replaced by a new line
    region) are considered — these are where a delegate's edit is most
    likely to have silently changed semantics rather than purely added or
    removed lines. Pure insertions/deletions are not scored.
    """
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    matcher = SequenceMatcher(None, old_lines, new_lines)

    flags: list[RiskFlag] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        for old_idx, new_idx in zip(range(i1, i2), range(j1, j2)):
            old_line = old_lines[old_idx]
            new_line = new_lines[new_idx]
            flags.extend(
                _classify_line_pair(old_line, new_line, filename, new_idx + 1)
            )

    return flags


def classify_git_diff(cwd: str, filepath: str) -> list[RiskFlag]:
    """Classify the working-tree diff of filepath against HEAD, in the git repo at cwd.

    Best-effort: a missing HEAD entry (newly-created file) yields old_content
    "" (SequenceMatcher then reports pure inserts, which are never scored, so
    brand-new files never spuriously flag). A missing working-tree file
    (deleted) yields new_content "".
    """
    old_content = ""
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{filepath}"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            old_content = proc.stdout
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        old_content = ""

    new_content = ""
    try:
        file_path = Path(cwd) / filepath
        if file_path.exists():
            new_content = file_path.read_text()
    except (OSError, FileNotFoundError):
        new_content = ""

    return classify_diff(old_content, new_content, filepath)
