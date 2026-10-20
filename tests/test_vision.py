"""Tests for the vision tool — image encoding, validation, analysis."""

import json
import pytest
from mullet.core.tools.vision import analyze_image, _encode_image


@pytest.fixture
def png_image(tmp_path):
    """Create a minimal valid PNG file."""
    # Minimal 1x1 red pixel PNG
    import struct
    import zlib
    def make_png():
        sig = b'\x89PNG\r\n\x1a\n'
        # IHDR
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        # IDAT
        raw = zlib.compress(b'\x00\xff\x00\x00')
        idat_crc = zlib.crc32(b'IDAT' + raw) & 0xFFFFFFFF
        idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
        # IEND
        iend_crc = zlib.crc32(b'IEND') & 0xFFFFFFFF
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return sig + ihdr + idat + iend

    path = tmp_path / "test.png"
    path.write_bytes(make_png())
    return path


@pytest.fixture
def jpeg_image(tmp_path):
    """Create a minimal JPEG-like file."""
    path = tmp_path / "test.jpg"
    # JPEG files start with FF D8 FF
    path.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)
    return path


# --- Encoding tests ---

def test_encode_png(png_image):
    data_url, error = _encode_image(str(png_image))
    assert data_url is not None
    assert error == ""
    assert data_url.startswith("data:image/png;base64,")


def test_encode_jpeg(jpeg_image):
    data_url, error = _encode_image(str(jpeg_image))
    assert data_url is not None
    assert data_url.startswith("data:image/jpeg;base64,")


def test_encode_missing_file():
    data_url, error = _encode_image("/nonexistent/image.png")
    assert data_url is None
    assert "not found" in error.lower()


def test_encode_unsupported_format(tmp_path):
    path = tmp_path / "image.bmp"
    path.write_bytes(b'BM' + b'\x00' * 100)
    data_url, error = _encode_image(str(path))
    assert data_url is None
    assert "unsupported" in error.lower()


def test_encode_empty_file(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b'')
    data_url, error = _encode_image(str(path))
    assert data_url is None
    assert "empty" in error.lower()


def test_encode_too_large(tmp_path):
    path = tmp_path / "huge.png"
    # Write a file just over the limit
    path.write_bytes(b'\x89PNG' + b'\x00' * (11 * 1024 * 1024))
    data_url, error = _encode_image(str(path))
    assert data_url is None
    assert "too large" in error.lower()


# --- Tool function tests ---

def test_analyze_image_success(png_image):
    result = json.loads(analyze_image(path=str(png_image), question="What is this?"))
    assert result["type"] == "image_analysis"
    assert result["image_data_url"].startswith("data:image/png")
    assert result["question"] == "What is this?"


def test_analyze_image_default_question(png_image):
    result = json.loads(analyze_image(path=str(png_image)))
    assert "Describe" in result["question"]


def test_analyze_image_missing_path():
    result = json.loads(analyze_image())
    assert "error" in result


def test_analyze_image_bad_path():
    result = json.loads(analyze_image(path="/nonexistent.png"))
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_analyze_image_unsupported_format(tmp_path):
    path = tmp_path / "file.pdf"
    path.write_bytes(b'%PDF' + b'\x00' * 100)
    result = json.loads(analyze_image(path=str(path)))
    assert "error" in result
    assert "unsupported" in result["error"].lower()
