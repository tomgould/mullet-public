"""Tests for image_gen schema/no-key paths and error_log pure helpers."""

import json
import logging
from logging.handlers import RotatingFileHandler

from mullet.core.tools.image_gen import (
    image_gen_handler,
    IMAGE_GEN_SCHEMA,
)
from mullet.core.agent.error_log import (
    get_error_log_path,
    setup_error_log,
)


# ---------------------------------------------------------------------------
# IMAGE_GEN_SCHEMA
# ---------------------------------------------------------------------------

class TestImageGenSchema:
    def test_has_description(self):
        assert "description" in IMAGE_GEN_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in IMAGE_GEN_SCHEMA

    def test_prompt_required(self):
        assert "prompt" in IMAGE_GEN_SCHEMA["parameters"]["required"]

    def test_prompt_param_exists(self):
        props = IMAGE_GEN_SCHEMA["parameters"]["properties"]
        assert "prompt" in props

    def test_size_param_has_enum(self):
        props = IMAGE_GEN_SCHEMA["parameters"]["properties"]
        assert "enum" in props["size"]

    def test_backend_enum_has_openai(self):
        enum = IMAGE_GEN_SCHEMA["parameters"]["properties"]["backend"]["enum"]
        assert "openai" in enum

    def test_backend_enum_has_fal(self):
        enum = IMAGE_GEN_SCHEMA["parameters"]["properties"]["backend"]["enum"]
        assert "fal" in enum

    def test_backend_enum_has_local(self):
        enum = IMAGE_GEN_SCHEMA["parameters"]["properties"]["backend"]["enum"]
        assert "local" in enum

    def test_num_images_has_max(self):
        props = IMAGE_GEN_SCHEMA["parameters"]["properties"]
        assert "maximum" in props["num_images"]


# ---------------------------------------------------------------------------
# image_gen_handler — no-network paths
# ---------------------------------------------------------------------------

class TestImageGenHandlerNoKey:
    def _run(self, **kwargs):
        return json.loads(image_gen_handler(**kwargs))

    def test_local_backend_returns_failure(self, tmp_path):
        result = self._run(prompt="A cat", backend="local", output_dir=str(tmp_path))
        assert result["success"] is False

    def test_local_backend_error_mentions_not_implemented(self, tmp_path):
        result = self._run(prompt="A cat", backend="local", output_dir=str(tmp_path))
        assert "not yet implemented" in result["error"].lower() or "not" in result["error"].lower()

    def test_openai_no_key_returns_failure(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = self._run(prompt="A cat", backend="openai", output_dir=str(tmp_path))
        assert result["success"] is False

    def test_openai_no_key_error_mentions_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = self._run(prompt="A cat", backend="openai", output_dir=str(tmp_path))
        assert "OPENAI_API_KEY" in result["error"]

    def test_fal_no_key_returns_failure(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FAL_KEY", raising=False)
        result = self._run(prompt="A cat", backend="fal", output_dir=str(tmp_path))
        assert result["success"] is False

    def test_fal_no_key_error_mentions_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FAL_KEY", raising=False)
        result = self._run(prompt="A cat", backend="fal", output_dir=str(tmp_path))
        assert "FAL_KEY" in result["error"]

    def test_returns_json_string(self, tmp_path):
        raw = image_gen_handler(prompt="test", backend="local", output_dir=str(tmp_path))
        assert isinstance(raw, str)
        data = json.loads(raw)
        assert "success" in data

    def test_prompt_in_result(self, tmp_path):
        result = self._run(prompt="beautiful sunset", backend="local", output_dir=str(tmp_path))
        assert result["prompt"] == "beautiful sunset"

    def test_backend_in_result(self, tmp_path):
        result = self._run(prompt="test", backend="local", output_dir=str(tmp_path))
        assert result["backend"] == "local"

    def test_images_list_in_result(self, tmp_path):
        result = self._run(prompt="test", backend="local", output_dir=str(tmp_path))
        assert "images" in result
        assert isinstance(result["images"], list)

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "generated"
        self._run(prompt="test", backend="local", output_dir=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# get_error_log_path
# ---------------------------------------------------------------------------

class TestGetErrorLogPath:
    def test_returns_string(self):
        assert isinstance(get_error_log_path(), str)

    def test_ends_with_errors_log(self):
        assert get_error_log_path().endswith("errors.log")

    def test_contains_mullet(self):
        assert ".mullet" in get_error_log_path()


# ---------------------------------------------------------------------------
# setup_error_log
# ---------------------------------------------------------------------------

class TestSetupErrorLog:
    def test_does_not_raise(self):
        setup_error_log()  # should not raise

    def test_adds_rotating_handler(self):
        setup_error_log()
        root_logger = logging.getLogger()
        has_rotating = any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers)
        assert has_rotating is True

    def test_idempotent(self):
        setup_error_log()
        count_before = sum(
            1 for h in logging.getLogger().handlers
            if isinstance(h, RotatingFileHandler) and "errors.log" in h.baseFilename
        )
        setup_error_log()
        count_after = sum(
            1 for h in logging.getLogger().handlers
            if isinstance(h, RotatingFileHandler) and "errors.log" in h.baseFilename
        )
        assert count_after == count_before
