"""Tests for mullet.harness.skills.frontmatter."""

from __future__ import annotations

import pytest

from mullet.harness.skills.frontmatter import ParseError, parse_frontmatter

pytestmark = pytest.mark.harness


def test_parses_basic_frontmatter() -> None:
    text = "---\nname: foo\ndescription: bar\n---\nbody"
    fm, body = parse_frontmatter(text)
    assert fm == {"name": "foo", "description": "bar"}
    assert body == "body"


def test_no_frontmatter_returns_empty_dict_and_full_body() -> None:
    text = "# Just markdown\nNo frontmatter here.\n"
    fm, body = parse_frontmatter(text)
    assert fm == {}
    assert body == text


def test_handles_block_scalar_pipe() -> None:
    text = "---\nname: |\n  multi\n  line\n---\nbody"
    fm, body = parse_frontmatter(text)
    assert fm["name"] == "multi\nline"
    assert body == "body"


def test_handles_block_scalar_gt() -> None:
    text = "---\ndescription: >\n  rolling\n  text\n---\nbody"
    fm, body = parse_frontmatter(text)
    assert fm["description"].startswith("rolling text")
    assert body == "body"


def test_handles_quoted_strings() -> None:
    text = "---\nsingle: 'hi there'\ndouble: \"hi there\"\n---\nbody"
    fm, body = parse_frontmatter(text)
    assert fm["single"] == "hi there"
    assert fm["double"] == "hi there"


def test_malformed_yaml_raises_parse_error() -> None:
    text = "---\nfoo: bar\n- item\n---\nbody"
    with pytest.raises(ParseError):
        parse_frontmatter(text, source="bad.md")


def test_empty_frontmatter_block() -> None:
    text = "---\n---\nbody"
    fm, body = parse_frontmatter(text)
    assert fm == {}
    assert body == "body"


def test_tools_list() -> None:
    text = "---\ntools: [Bash, Read]\n---\nbody"
    fm, _ = parse_frontmatter(text)
    assert fm["tools"] == ["Bash", "Read"]
