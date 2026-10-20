"""Tests for mullet.cli.routing_cmd helpers and mullet.core.tools.image_gen schema/local backend."""

import json

from mullet.cli.routing_cmd import _rate_str, _provider_col, run as routing_run
from mullet.core.tools.image_gen import (
    IMAGE_GEN_SCHEMA,
    image_gen_handler,
)


# ---------------------------------------------------------------------------
# _rate_str
# ---------------------------------------------------------------------------

class TestRateStr:
    def test_returns_string(self):
        assert isinstance(_rate_str(0.95), str)

    def test_contains_percent_sign(self):
        result = _rate_str(0.5)
        assert "%" in result

    def test_zero_rate(self):
        result = _rate_str(0.0)
        assert "%" in result

    def test_full_rate(self):
        result = _rate_str(1.0)
        assert "%" in result

    def test_high_rate_contains_value(self):
        result = _rate_str(0.95)
        # 95.0% should appear in output (may include ANSI codes)
        assert "95.0" in result

    def test_low_rate_contains_value(self):
        result = _rate_str(0.30)
        assert "30.0" in result


# ---------------------------------------------------------------------------
# _provider_col
# ---------------------------------------------------------------------------

class TestProviderCol:
    def test_returns_string(self):
        assert isinstance(_provider_col("kimi"), str)

    def test_provider_name_in_output(self):
        result = _provider_col("gemini")
        assert "gemini" in result

    def test_empty_provider(self):
        result = _provider_col("")
        assert isinstance(result, str)

    def test_long_provider_name(self):
        result = _provider_col("very-long-provider-name")
        assert "very-long-provider-name" in result


# ---------------------------------------------------------------------------
# routing_cmd.run — dispatch / unknown action
# ---------------------------------------------------------------------------

class TestRoutingCmdRun:
    def test_no_args_returns_zero(self):
        assert routing_run([]) == 0

    def test_unknown_action_returns_nonzero(self):
        assert routing_run(["nonexistent_action_xyz"]) != 0


# ---------------------------------------------------------------------------
# IMAGE_GEN_SCHEMA
# ---------------------------------------------------------------------------

class TestImageGenSchema:
    def test_has_description(self):
        assert "description" in IMAGE_GEN_SCHEMA
        assert len(IMAGE_GEN_SCHEMA["description"]) > 0

    def test_has_parameters(self):
        assert "parameters" in IMAGE_GEN_SCHEMA

    def test_prompt_is_required(self):
        required = IMAGE_GEN_SCHEMA["parameters"]["required"]
        assert "prompt" in required

    def test_size_enum_has_valid_sizes(self):
        size_enum = IMAGE_GEN_SCHEMA["parameters"]["properties"]["size"]["enum"]
        assert "1024x1024" in size_enum

    def test_backend_enum_has_openai(self):
        backend_enum = IMAGE_GEN_SCHEMA["parameters"]["properties"]["backend"]["enum"]
        assert "openai" in backend_enum

    def test_backend_enum_has_local(self):
        backend_enum = IMAGE_GEN_SCHEMA["parameters"]["properties"]["backend"]["enum"]
        assert "local" in backend_enum

    def test_num_images_has_max(self):
        props = IMAGE_GEN_SCHEMA["parameters"]["properties"]
        assert "maximum" in props["num_images"]
        assert props["num_images"]["maximum"] >= 1


# ---------------------------------------------------------------------------
# image_gen_handler — local backend (no API key needed)
# ---------------------------------------------------------------------------

class TestImageGenHandlerLocal:
    def test_local_backend_returns_json_string(self, tmp_path):
        result = image_gen_handler(
            prompt="a red cat",
            backend="local",
            output_dir=str(tmp_path / "images"),
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_local_backend_success_false(self, tmp_path):
        result = image_gen_handler(
            prompt="a blue sky",
            backend="local",
            output_dir=str(tmp_path / "images"),
        )
        parsed = json.loads(result)
        assert parsed["success"] is False

    def test_local_backend_error_message_set(self, tmp_path):
        result = image_gen_handler(
            prompt="something",
            backend="local",
            output_dir=str(tmp_path / "images"),
        )
        parsed = json.loads(result)
        assert parsed["error"] != ""

    def test_local_backend_prompt_echoed(self, tmp_path):
        result = image_gen_handler(
            prompt="my special prompt",
            backend="local",
            output_dir=str(tmp_path / "images"),
        )
        parsed = json.loads(result)
        assert parsed["prompt"] == "my special prompt"

    def test_local_backend_no_images_generated(self, tmp_path):
        result = image_gen_handler(
            prompt="anything",
            backend="local",
            output_dir=str(tmp_path / "images"),
        )
        parsed = json.loads(result)
        assert parsed["images"] == []

    def test_openai_without_key_success_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = image_gen_handler(
            prompt="test prompt",
            backend="openai",
            output_dir=str(tmp_path / "images"),
        )
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "OPENAI_API_KEY" in parsed["error"]

    def test_fal_without_key_success_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        result = image_gen_handler(
            prompt="test prompt",
            backend="fal",
            output_dir=str(tmp_path / "images"),
        )
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "FAL_KEY" in parsed["error"]
