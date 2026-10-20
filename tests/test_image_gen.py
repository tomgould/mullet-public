"""Tests for mullet.core.tools.image_gen — AI image generation tool."""

import json
import os
from unittest import mock
from unittest.mock import patch, MagicMock
import httpx
from pathlib import Path

from mullet.core.tools.image_gen import IMAGE_GEN_SCHEMA, image_gen_handler, register, _save_image


def test_schema_has_prompt_required():
    assert "prompt" in IMAGE_GEN_SCHEMA["parameters"]["required"]


def test_register_with_mock_registry():
    mock_registry = mock.Mock()
    register(mock_registry)
    mock_registry.register.assert_called_once_with(
        name="image_gen",
        func=image_gen_handler,
        schema=IMAGE_GEN_SCHEMA,
        toolset="creative",
    )


def test_local_backend_not_implemented():
    result = image_gen_handler(prompt="test prompt", backend="local")
    response = json.loads(result)
    assert not response["success"]
    assert "not yet implemented" in response.get("error", response.get("message", "")).lower() or "local" in result.lower()


def test_openai_no_api_key():
    with mock.patch.dict(os.environ, {}, clear=True):
        result = image_gen_handler(prompt="test prompt", backend="openai")
        response = json.loads(result)
        assert not response["success"]


def test_fal_no_api_key():
    with mock.patch.dict(os.environ, {}, clear=True):
        result = image_gen_handler(prompt="test prompt", backend="fal")
        response = json.loads(result)
        assert not response["success"]


def test_unknown_backend():
    result = image_gen_handler(prompt="test", backend="nonexistent")
    response = json.loads(result)
    assert not response["success"]


def test_missing_prompt():
    result = image_gen_handler(prompt="")
    response = json.loads(result)
    assert not response["success"]


class TestImageGenMissingLines:
    def test_save_image(self, tmp_path):
        mock_response = MagicMock()
        mock_response.content = b"fakepng"
        with patch("httpx.get", return_value=mock_response):
            filepath = _save_image("http://example.com/img.png", tmp_path, "hash123")
            assert filepath.exists()
            assert filepath.read_bytes() == b"fakepng"
            assert filepath.suffix == ".png"

    def test_openai_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"url": "http://example.com/img.png"}]}
            mock_post.return_value = mock_response
            with patch("mullet.core.tools.image_gen._save_image", return_value=tmp_path / "img.png"):
                result = image_gen_handler(prompt="a cat", backend="openai")
                response = json.loads(result)
                assert response["success"] is True
                assert len(response["images"]) == 1

    def test_openai_http_status_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError("err", request=MagicMock(), response=MagicMock(text="bad"))
            result = image_gen_handler(prompt="a cat", backend="openai")
            response = json.loads(result)
            assert not response["success"]
            assert "OpenAI API error" in response["error"]

    def test_openai_request_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = httpx.RequestError("conn failed")
            result = image_gen_handler(prompt="a cat", backend="openai")
            response = json.loads(result)
            assert not response["success"]
            assert "OpenAI API request failed" in response["error"]

    def test_fal_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "fake-key")
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"images": [{"url": "http://example.com/img.png"}]}
            mock_post.return_value = mock_response
            with patch("mullet.core.tools.image_gen._save_image", return_value=tmp_path / "img.png"):
                result = image_gen_handler(prompt="a cat", backend="fal", size="1024x1024")
                response = json.loads(result)
                assert response["success"] is True
                assert len(response["images"]) == 1

    def test_fal_http_status_error(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "fake-key")
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError("err", request=MagicMock(), response=MagicMock(text="bad"))
            result = image_gen_handler(prompt="a cat", backend="fal")
            response = json.loads(result)
            assert not response["success"]
            assert "FAL.ai API error" in response["error"]

    def test_fal_request_error(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "fake-key")
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = httpx.RequestError("conn failed")
            result = image_gen_handler(prompt="a cat", backend="fal")
            response = json.loads(result)
            assert not response["success"]
            assert "FAL.ai API request failed" in response["error"]
