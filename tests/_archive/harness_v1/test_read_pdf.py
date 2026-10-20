"""Tests for the ReadPDF harness tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from mullet.harness.tools.read_pdf import SCHEMA, execute, register


class TestSchema:
    def test_schema_valid_jsonschema(self):
        assert SCHEMA["name"] == "ReadPDF"
        props = SCHEMA["input_schema"]["properties"]
        assert "file_path" in props
        assert "pages" in props
        assert SCHEMA["input_schema"]["required"] == ["file_path"]


class TestExecute:
    def test_execute_with_missing_pypdf_returns_error_with_hint(self):
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf":
                raise ModuleNotFoundError("No module named 'pypdf'")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = execute("foo.pdf")
        assert "error" in result
        assert "pip install pypdf" in result["error"]

    def test_execute_with_real_pdf_extracts_text(self, tmp_path: Path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("%PDF-1.4\n")  # create file so isfile passes
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello PDF"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = execute(str(pdf_path))
        assert result["text"] == "Hello PDF"
        assert result["page_count"] == 1
        assert result["pages_extracted"] == [1]

    def test_execute_invalid_path_returns_error(self):
        result = execute("/nonexistent/path/to/file.pdf")
        assert "error" in result
        assert "File not found" in result["error"]

    def test_execute_pages_range(self, tmp_path: Path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("%PDF-1.4\n")
        mock_pages = [MagicMock() for _ in range(5)]
        for i, mp in enumerate(mock_pages):
            mp.extract_text.return_value = f"Page {i + 1}"
        mock_reader = MagicMock()
        mock_reader.pages = mock_pages

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = execute(str(pdf_path), pages="2-4")
        assert result["page_count"] == 5
        assert result["pages_extracted"] == [2, 3, 4]
        assert "Page 2" in result["text"]
        assert "Page 4" in result["text"]
        assert "Page 1" not in result["text"]


class TestRegister:
    def test_register_adds_tool(self):
        registry = MagicMock()
        register(registry)
        registry.register.assert_called_once()
        call = registry.register.call_args
        assert call.kwargs["name"] == "ReadPDF"
        assert call.kwargs["toolset"] == "documents"
