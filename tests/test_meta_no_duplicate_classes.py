"""CI guard: every test class within a file must have a unique name.

Pytest silently picks only the last definition when classes are duplicated,
hiding tests that were "above" the duplicate. We hit this in May 2026
(see commit ``4792b38``) — ``TestStoryFormatting`` was defined twice in
``tests/test_ralph_integration.py`` and three ``test_next_story_*`` tests
had been silently dropped for some unknown number of weeks. Surfaced only
by enabling ruff F811.

This test parses every ``tests/**/test_*.py`` (skipping ``_archive``) and
asserts every class name is unique within its file.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent


def _test_files() -> list[Path]:
    """Return every collectable ``test_*.py`` under the tests/ tree."""
    return [
        p for p in sorted(_TESTS_DIR.rglob("test_*.py"))
        if "_archive" not in p.parts
    ]


@pytest.mark.parametrize("path", _test_files(), ids=lambda p: str(p.relative_to(_TESTS_DIR)))
def test_no_duplicate_test_class_names(path: Path) -> None:
    """No two ``class TestX`` statements at module level share a name."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        pytest.skip(f"unparseable test file: {e}")
        return

    class_names = [
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    ]
    counts = Counter(class_names)
    duplicates = {n: c for n, c in counts.items() if c > 1}
    assert not duplicates, (
        f"{path}: duplicate class names — {duplicates}. "
        f"Pytest will silently drop earlier definitions, hiding tests."
    )
