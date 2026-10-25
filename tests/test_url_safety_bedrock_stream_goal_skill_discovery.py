"""Tests for is_safe_url, BedrockStreamHandler, and SkillDiscovery."""

import json

from mullet.core.tools.url_safety import is_safe_url
from mullet.core.providers.bedrock_stream import BedrockStreamEvent, BedrockStreamHandler
from mullet.core.skills.discovery import SkillDiscovery


# ---------------------------------------------------------------------------
# is_safe_url
# ---------------------------------------------------------------------------

class TestIsSafeUrl:
    def test_https_public_domain_is_safe(self):
        assert is_safe_url("https://example.com") is True

    def test_http_public_domain_is_safe(self):
        assert is_safe_url("http://example.com") is True

    def test_file_scheme_is_unsafe(self):
        assert is_safe_url("file:///etc/passwd") is False

    def test_ftp_scheme_is_unsafe(self):
        assert is_safe_url("ftp://example.com") is False

    def test_gopher_scheme_is_unsafe(self):
        assert is_safe_url("gopher://example.com") is False

    def test_dict_scheme_is_unsafe(self):
        assert is_safe_url("dict://example.com") is False

    def test_tftp_scheme_is_unsafe(self):
        assert is_safe_url("tftp://example.com") is False

    def test_loopback_ip_is_unsafe(self):
        assert is_safe_url("http://127.0.0.1") is False

    def test_private_ip_192_is_unsafe(self):
        assert is_safe_url("http://192.168.1.1") is False

    def test_private_ip_10_is_unsafe(self):
        assert is_safe_url("http://10.0.0.1") is False

    def test_public_domain_with_path_is_safe(self):
        assert is_safe_url("https://api.example.com/v1/data") is True

    def test_returns_bool(self):
        assert isinstance(is_safe_url("https://example.com"), bool)


# ---------------------------------------------------------------------------
# BedrockStreamEvent
# ---------------------------------------------------------------------------

class TestBedrockStreamEvent:
    def test_type_stored(self):
        e = BedrockStreamEvent(type="content_block_start")
        assert e.type == "content_block_start"

    def test_text_default_empty(self):
        e = BedrockStreamEvent(type="content_block_delta")
        assert e.text == ""

    def test_text_stored(self):
        e = BedrockStreamEvent(type="content_block_delta", text="hello")
        assert e.text == "hello"

    def test_start_event_default_none(self):
        e = BedrockStreamEvent(type="content_block_start")
        assert e.start_event is None

    def test_stop_event_default_none(self):
        e = BedrockStreamEvent(type="message_stop")
        assert e.stop_event is None


# ---------------------------------------------------------------------------
# BedrockStreamHandler
# ---------------------------------------------------------------------------

def _make_chunk(payload: dict) -> dict:
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


class TestBedrockStreamHandler:
    def setup_method(self):
        self.handler = BedrockStreamHandler()

    def test_empty_stream_yields_nothing(self):
        result = list(self.handler.parse_stream(iter([])))
        assert result == []

    def test_no_chunk_key_yields_nothing(self):
        result = list(self.handler.parse_stream(iter([{"other": "value"}])))
        assert result == []

    def test_invalid_json_bytes_skipped(self):
        events = [{"chunk": {"bytes": b"not json"}}]
        result = list(self.handler.parse_stream(iter(events)))
        assert result == []

    def test_content_block_start_type(self):
        events = [_make_chunk({"content_block_start": {"type": "text"}})]
        result = list(self.handler.parse_stream(iter(events)))
        assert result[0].type == "content_block_start"

    def test_content_block_start_event_stored(self):
        payload = {"content_block_start": {"type": "text"}}
        events = [_make_chunk(payload)]
        result = list(self.handler.parse_stream(iter(events)))
        assert result[0].start_event == {"type": "text"}

    def test_content_block_delta_type(self):
        events = [_make_chunk({"content_block_delta": {"delta": {"text": "hello"}}})]
        result = list(self.handler.parse_stream(iter(events)))
        assert result[0].type == "content_block_delta"

    def test_content_block_delta_text(self):
        events = [_make_chunk({"content_block_delta": {"delta": {"text": "world"}}})]
        result = list(self.handler.parse_stream(iter(events)))
        assert result[0].text == "world"

    def test_message_stop_type(self):
        events = [_make_chunk({"message_stop": {"stop_reason": "end_turn"}})]
        result = list(self.handler.parse_stream(iter(events)))
        assert result[0].type == "message_stop"

    def test_message_stop_event_stored(self):
        payload = {"message_stop": {"stop_reason": "end_turn"}}
        events = [_make_chunk(payload)]
        result = list(self.handler.parse_stream(iter(events)))
        assert result[0].stop_event == {"stop_reason": "end_turn"}

    def test_multiple_events_yielded(self):
        events = [
            _make_chunk({"content_block_start": {}}),
            _make_chunk({"content_block_delta": {"delta": {"text": "text"}}}),
            _make_chunk({"message_stop": {}}),
        ]
        result = list(self.handler.parse_stream(iter(events)))
        assert len(result) == 3

    def test_yields_stream_event_type(self):
        events = [_make_chunk({"content_block_start": {}})]
        result = list(self.handler.parse_stream(iter(events)))
        assert isinstance(result[0], BedrockStreamEvent)


# ---------------------------------------------------------------------------
# SkillDiscovery
# ---------------------------------------------------------------------------

class TestSkillDiscovery:
    def test_discover_empty_dir_returns_empty(self, tmp_path):
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        assert result == []

    def test_discover_returns_list(self, tmp_path):
        sd = SkillDiscovery()
        assert isinstance(sd.discover(str(tmp_path)), list)

    def test_discover_finds_skill_md(self, tmp_path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Triggers: deploy, release\n")
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        assert len(result) == 1

    def test_skill_name_is_dir_name(self, tmp_path):
        skill_dir = tmp_path / "my_deploy_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Triggers: deploy\n")
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        assert result[0]["name"] == "my_deploy_skill"

    def test_skill_has_path_key(self, tmp_path):
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Triggers: test\n")
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        assert "path" in result[0]

    def test_skill_path_ends_with_skill_md(self, tmp_path):
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Triggers: test\n")
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        assert result[0]["path"].endswith("SKILL.md")

    def test_triggers_extracted(self, tmp_path):
        skill_dir = tmp_path / "deploy_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Triggers: deploy, release, push\n")
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        triggers = result[0]["triggers"]
        assert "deploy" in triggers
        assert "release" in triggers
        assert "push" in triggers

    def test_no_triggers_line_returns_empty_triggers(self, tmp_path):
        skill_dir = tmp_path / "no_trigger_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\n\nNo triggers line here.\n")
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        assert result[0]["triggers"] == []

    def test_multiple_skills_discovered(self, tmp_path):
        for name in ("skill_a", "skill_b", "skill_c"):
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text("Triggers: trigger\n")
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        assert len(result) == 3

    def test_non_skill_dir_not_included(self, tmp_path):
        (tmp_path / "not_a_skill").mkdir()
        (tmp_path / "not_a_skill" / "README.md").write_text("# readme")
        sd = SkillDiscovery()
        result = sd.discover(str(tmp_path))
        assert len(result) == 0
