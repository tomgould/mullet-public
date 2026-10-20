"""Tests for url_safety, skill discovery, and bedrock_stream pure helpers."""

import json

from mullet.core.tools.url_safety import is_safe_url
from mullet.core.skills.discovery import SkillDiscovery
from mullet.core.providers.bedrock_stream import BedrockStreamEvent, BedrockStreamHandler


# ---------------------------------------------------------------------------
# is_safe_url
# ---------------------------------------------------------------------------

class TestIsSafeUrl:
    def test_https_public_domain_is_safe(self):
        assert is_safe_url("https://example.com") is True

    def test_http_public_domain_is_safe(self):
        assert is_safe_url("http://example.com/api") is True

    def test_file_scheme_blocked(self):
        assert is_safe_url("file:///etc/passwd") is False

    def test_ftp_scheme_blocked(self):
        assert is_safe_url("ftp://example.com") is False

    def test_gopher_scheme_blocked(self):
        assert is_safe_url("gopher://example.com") is False

    def test_dict_scheme_blocked(self):
        assert is_safe_url("dict://example.com") is False

    def test_localhost_blocked(self):
        assert is_safe_url("http://127.0.0.1/api") is False

    def test_loopback_ipv6_bracket_notation_blocked(self):
        # IPv6 in URLs requires bracket notation for urlparse to parse hostname correctly
        assert is_safe_url("http://[::1]/path") is False

    def test_private_class_a_blocked(self):
        assert is_safe_url("http://10.0.0.1/resource") is False

    def test_private_class_b_blocked(self):
        assert is_safe_url("http://172.16.0.1/") is False

    def test_private_class_c_blocked(self):
        assert is_safe_url("http://192.168.1.100/") is False

    def test_empty_string_returns_bool(self):
        # Empty string has no blocked scheme or private IP — returns True (no explicit block)
        result = is_safe_url("")
        assert isinstance(result, bool)

    def test_domain_name_not_blocked_as_private(self):
        # Domain names (not IPs) go through — SSRF check is IP-based
        assert is_safe_url("https://api.openai.com/v1/chat") is True

    def test_returns_bool(self):
        assert isinstance(is_safe_url("https://example.com"), bool)


# ---------------------------------------------------------------------------
# SkillDiscovery
# ---------------------------------------------------------------------------

class TestSkillDiscoveryDiscover:
    def setup_method(self):
        self.disc = SkillDiscovery()

    def test_empty_dir_returns_empty_list(self, tmp_path):
        result = self.disc.discover(str(tmp_path))
        assert result == []

    def test_finds_skill_md(self, tmp_path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\n")
        result = self.disc.discover(str(tmp_path))
        assert len(result) == 1

    def test_skill_name_is_dir_name(self, tmp_path):
        skill_dir = tmp_path / "code_review"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Code Review\n")
        result = self.disc.discover(str(tmp_path))
        assert result[0]["name"] == "code_review"

    def test_skill_path_is_correct(self, tmp_path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text("# Skill\n")
        result = self.disc.discover(str(tmp_path))
        assert result[0]["path"] == str(skill_path)

    def test_multiple_skills_found(self, tmp_path):
        for name in ["skill_a", "skill_b", "skill_c"]:
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"# {name}\n")
        result = self.disc.discover(str(tmp_path))
        assert len(result) == 3

    def test_non_skill_dirs_ignored(self, tmp_path):
        (tmp_path / "README.md").write_text("docs")
        result = self.disc.discover(str(tmp_path))
        assert result == []


class TestSkillDiscoveryExtractTriggers:
    def setup_method(self):
        self.disc = SkillDiscovery()

    def test_no_triggers_line_returns_empty(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nSome description.\n")
        result = self.disc._extract_triggers(str(skill_file))
        assert result == []

    def test_extracts_single_trigger(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("Triggers: /review\n")
        result = self.disc._extract_triggers(str(skill_file))
        assert "/review" in result

    def test_extracts_multiple_triggers(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("Triggers: /review, /check, /lint\n")
        result = self.disc._extract_triggers(str(skill_file))
        assert "/review" in result
        assert "/check" in result
        assert "/lint" in result

    def test_strips_whitespace_around_triggers(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("Triggers: /review ,  /check  \n")
        result = self.disc._extract_triggers(str(skill_file))
        assert "/review" in result
        assert "/check" in result


# ---------------------------------------------------------------------------
# BedrockStreamEvent dataclass
# ---------------------------------------------------------------------------

class TestBedrockStreamEvent:
    def test_type_stored(self):
        e = BedrockStreamEvent(type="content_block_start")
        assert e.type == "content_block_start"

    def test_default_text_empty(self):
        e = BedrockStreamEvent(type="message_stop")
        assert e.text == ""

    def test_text_stored(self):
        e = BedrockStreamEvent(type="content_block_delta", text="hello")
        assert e.text == "hello"

    def test_default_events_none(self):
        e = BedrockStreamEvent(type="x")
        assert e.start_event is None
        assert e.delta_event is None
        assert e.stop_event is None


# ---------------------------------------------------------------------------
# BedrockStreamHandler.parse_stream
# ---------------------------------------------------------------------------

def _make_event(payload: dict) -> dict:
    """Wrap a payload dict as a Bedrock chunk event."""
    return {"chunk": {"bytes": json.dumps(payload).encode("utf-8")}}


class TestBedrockStreamHandlerParseStream:
    def setup_method(self):
        self.handler = BedrockStreamHandler()

    def test_content_block_start_parsed(self):
        events = [_make_event({"content_block_start": {"index": 0, "content_block": {"type": "text"}}})]
        result = list(self.handler.parse_stream(events))
        assert len(result) == 1
        assert result[0].type == "content_block_start"

    def test_content_block_delta_parsed(self):
        payload = {"content_block_delta": {"index": 0, "delta": {"type": "text_delta", "text": "Hello"}}}
        events = [_make_event(payload)]
        result = list(self.handler.parse_stream(events))
        assert len(result) == 1
        assert result[0].type == "content_block_delta"
        assert result[0].text == "Hello"

    def test_message_stop_parsed(self):
        events = [_make_event({"message_stop": {"stop_reason": "end_turn"}})]
        result = list(self.handler.parse_stream(events))
        assert len(result) == 1
        assert result[0].type == "message_stop"

    def test_invalid_json_skipped(self):
        bad_event = {"chunk": {"bytes": b"not json at all"}}
        events = [bad_event]
        result = list(self.handler.parse_stream(events))
        assert result == []

    def test_non_chunk_events_skipped(self):
        events = [{"other_key": "value"}]
        result = list(self.handler.parse_stream(events))
        assert result == []

    def test_multiple_events_in_stream(self):
        events = [
            _make_event({"content_block_start": {"content_block": {}}}),
            _make_event({"content_block_delta": {"index": 0, "delta": {"text": "Hello"}}}),
            _make_event({"message_stop": {"stop_reason": "end_turn"}}),
        ]
        result = list(self.handler.parse_stream(events))
        assert len(result) == 3

    def test_empty_stream_returns_nothing(self):
        result = list(self.handler.parse_stream([]))
        assert result == []

    def test_delta_text_extracted(self):
        payload = {"content_block_delta": {"index": 0, "delta": {"type": "text_delta", "text": "World"}}}
        result = list(self.handler.parse_stream([_make_event(payload)]))
        assert result[0].text == "World"
