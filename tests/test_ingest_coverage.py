"""Coverage tests for mullet.knowledge.engine.ingest.

Targets the largest uncovered blocks in ingest.py:
- 52-57: validate_chunk
- 72-85: build_context_prefix
- 104-120: _decode_email_header
- 123-135: _strip_html
- 138-174: _detect_charset
- 177-225: _extract_email_body
- 248-264: DocumentIngestor.__init__ (GPU OOM fallback)
- 276-286: parse_document
- 310-386: chunk_elements (semantic/recursive/title, validation, dedup)
- 405-484: create_embeddings (OOM cascade, CPU fallback)
- 499-517: ingest_document
- 532-620: ingest_mbox
- 635-719: ingest_mbox_chunks_only
- 721-752: _split_email_content
- 786-929: ingest_directory (pipeline/serial, resume, exclude)
- 1037-1151: ingest_code (pipeline/serial, stale removal, state)
- 1247-1295: build_file_manifest
- 1306-1315: save_file_manifest
- 1328-1337: load_file_manifest
- 1354-1367: diff_file_manifest
"""
from __future__ import annotations

import mailbox
from email.message import EmailMessage, Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.knowledge.engine import ingest as ingest_module
from mullet.knowledge.engine.ingest import (
    DocumentIngestor,
    _decode_email_header,
    _detect_charset,
    _extract_email_body,
    _strip_html,
    build_context_prefix,
    build_file_manifest,
    diff_file_manifest,
    ingest_code,
    ingest_directory,
    load_file_manifest,
    save_file_manifest,
    validate_chunk,
)


# Text long enough to pass MIN_CHUNK_CHARS (50) validation.
_LONG_TEXT = "This is a sufficiently long chunk of text that will pass the minimum character validation requirement."


def _make_ingestor(mock_embedder):
    """Create a DocumentIngestor with a mocked embedder using patch."""
    with patch("mullet.knowledge.engine.model_cache.get_model_cache") as mock_cache:
        mock_cache.return_value.get_embedder.return_value = mock_embedder
        with patch("mullet.knowledge.engine.ingest.torch.cuda.is_available", return_value=False):
            return DocumentIngestor()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestValidateChunk:
    def test_too_short(self):
        ok, reason = validate_chunk("hi", min_chars=10)
        assert ok is False
        assert reason == "too_short"

    def test_no_alphanumeric(self):
        ok, reason = validate_chunk("   !@#$%^&*()   ", min_chars=5)
        assert ok is False
        assert reason == "no_alphanumeric"

    def test_valid(self):
        ok, reason = validate_chunk("Hello world", min_chars=5)
        assert ok is True
        assert reason is None


class TestBuildContextPrefix:
    def test_empty(self):
        assert build_context_prefix({}) == ""

    def test_source_file_only(self):
        assert build_context_prefix({"source_file": "doc.pdf"}) == "Document: doc.pdf. "

    def test_page_only(self):
        assert build_context_prefix({"page_number": 3}) == "Page 3. "

    def test_both(self):
        result = build_context_prefix({"source_file": "doc.pdf", "page_number": 2})
        assert result == "Document: doc.pdf. Page 2. "


class TestDecodeEmailHeader:
    def test_empty(self):
        assert _decode_email_header("") == ""

    def test_none(self):
        assert _decode_email_header(None) == ""

    def test_plain_ascii(self):
        assert _decode_email_header("Hello") == "Hello"

    def test_encoded_word(self):
        encoded = "=?utf-8?b?SGVsbG8gV29ybGQ=?="
        assert _decode_email_header(encoded) == "Hello World"


class TestStripHtml:
    def test_basic(self):
        html = "<html><body><p>Hello</p></body></html>"
        assert _strip_html(html) == "Hello"

    def test_removes_script(self):
        html = "<script>alert(1)</script><p>text</p>"
        assert "alert" not in _strip_html(html)
        assert "text" in _strip_html(html)

    def test_normalizes_whitespace(self):
        html = "<p>a</p>\n\n\n\n<p>b</p>"
        text = _strip_html(html)
        assert "\n\n" in text


class TestDetectCharset:
    def test_declared_valid(self):
        payload = "hello".encode("utf-8")
        assert _detect_charset(payload, "utf-8") == "utf-8"

    def test_declared_invalid_falls_back(self):
        payload = "hello".encode("utf-8")
        result = _detect_charset(payload, "not-a-charset")
        assert result in ("utf-8", "ascii")

    def test_no_declared_uses_chardet_or_utf8(self):
        payload = "hello".encode("utf-8")
        result = _detect_charset(payload, None)
        assert result in ("utf-8", "ascii")


class TestExtractEmailBody:
    def test_plain_text_single_part(self):
        msg = EmailMessage()
        msg.set_content("Hello world")
        assert "Hello world" in _extract_email_body(msg)

    def test_html_single_part(self):
        msg = EmailMessage()
        msg.set_content("<p>Hello world</p>", subtype="html")
        body = _extract_email_body(msg)
        assert "Hello world" in body
        assert "<p>" not in body

    def test_multipart_plain_and_html(self):
        msg = EmailMessage()
        msg.make_alternative()
        msg.add_alternative("Hello plain", subtype="plain")
        msg.add_alternative("<p>Hello html</p>", subtype="html")
        body = _extract_email_body(msg)
        assert "Hello plain" in body

    def test_attachment_skipped(self):
        msg = Message()
        msg.add_header("Content-Type", "multipart/mixed")
        att = Message()
        att.add_header("Content-Type", "text/plain")
        att.add_header("Content-Disposition", "attachment; filename=\"test.txt\"")
        att.set_payload("file content")
        msg.attach(att)
        body_part = Message()
        body_part.add_header("Content-Type", "text/plain")
        body_part.set_payload("body text")
        msg.attach(body_part)
        body = _extract_email_body(msg)
        assert "body text" in body
        assert "file content" not in body


# ---------------------------------------------------------------------------
# DocumentIngestor
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.encode.return_value = MagicMock()
    emb.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
    emb.device = "cpu"
    emb.to.return_value = emb
    return emb


@pytest.fixture(autouse=True)
def _reset_global_batch_size(monkeypatch):
    """Reset global batch size cache before each test."""
    monkeypatch.setattr(ingest_module, "_global_effective_batch_size", None)


@pytest.fixture
def mock_ingestor(mock_embedder):
    """Return a DocumentIngestor with a mocked embedder."""
    return _make_ingestor(mock_embedder)


class TestDocumentIngestorInit:
    def test_cpu_fallback_on_cuda_oom(self):
        cache = MagicMock()
        emb = MagicMock()
        cache.get_embedder.side_effect = [
            RuntimeError("CUDA out of memory"),
            emb,
        ]
        with patch("mullet.knowledge.engine.model_cache.get_model_cache", return_value=cache):
            with patch("mullet.knowledge.engine.ingest.torch.cuda.is_available", return_value=True):
                ingestor = DocumentIngestor()
        assert ingestor.embedder is emb
        assert cache.get_embedder.call_count == 2

    def test_non_cuda_runtime_error_reraises(self):
        cache = MagicMock()
        cache.get_embedder.side_effect = RuntimeError("some other error")
        with patch("mullet.knowledge.engine.model_cache.get_model_cache", return_value=cache):
            with patch("mullet.knowledge.engine.ingest.torch.cuda.is_available", return_value=True):
                with pytest.raises(RuntimeError, match="some other error"):
                    DocumentIngestor()


class TestParseDocument:
    def test_file_not_found(self, mock_ingestor):
        with pytest.raises(FileNotFoundError):
            mock_ingestor.parse_document("/nonexistent/file.txt")

    def test_success(self, mock_ingestor, tmp_path, monkeypatch):
        fake_elem = MagicMock()
        fake_elem.__str__ = lambda self: _LONG_TEXT
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.partition", lambda *a, **k: [fake_elem]
        )
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = mock_ingestor.parse_document(f)
        assert len(result) == 1


class TestChunkElements:
    def test_title_strategy_default(self, mock_ingestor, monkeypatch):
        fake_chunk = MagicMock()
        fake_chunk.__str__ = lambda self: _LONG_TEXT
        fake_chunk.metadata.to_dict.return_value = {"page_number": 1}
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.chunk_by_title",
            lambda *a, **k: [fake_chunk],
        )
        elements = [fake_chunk]
        chunks = mock_ingestor.chunk_elements(elements, source_id="doc.txt")
        assert len(chunks) == 1
        assert chunks[0]["content"] == _LONG_TEXT
        assert chunks[0]["metadata"]["page_number"] == 1

    def test_semantic_strategy(self, mock_ingestor):
        with patch("mullet.knowledge.engine.chunking.chunk_by_semantic") as mock_semantic:
            mock_semantic.return_value = [_LONG_TEXT, _LONG_TEXT + " second"]
            chunks = mock_ingestor.chunk_elements(
                [], source_id="doc.txt", chunking_config={"strategy": "semantic"}
            )
            assert len(chunks) == 2
            assert chunks[0]["content"] == _LONG_TEXT

    def test_recursive_strategy(self, mock_ingestor):
        with patch("mullet.knowledge.engine.chunking.chunk_by_recursive") as mock_recursive:
            mock_recursive.return_value = [_LONG_TEXT]
            chunks = mock_ingestor.chunk_elements(
                [MagicMock()], source_id="doc.txt", chunking_config={"strategy": "recursive"}
            )
            assert len(chunks) == 1
            assert chunks[0]["content"] == _LONG_TEXT

    def test_skips_too_short(self, mock_ingestor, monkeypatch):
        fake_chunk = MagicMock()
        fake_chunk.__str__ = lambda self: "hi"
        fake_chunk.metadata.to_dict.return_value = {}
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.chunk_by_title",
            lambda *a, **k: [fake_chunk],
        )
        chunks = mock_ingestor.chunk_elements([fake_chunk])
        assert len(chunks) == 0

    def test_skips_no_alphanumeric(self, mock_ingestor, monkeypatch):
        fake_chunk = MagicMock()
        fake_chunk.__str__ = lambda self: "!!!!!!"
        fake_chunk.metadata.to_dict.return_value = {}
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.chunk_by_title",
            lambda *a, **k: [fake_chunk],
        )
        chunks = mock_ingestor.chunk_elements([fake_chunk])
        assert len(chunks) == 0

    def test_deduplicates_by_content_hash(self, mock_ingestor, monkeypatch):
        fake_chunk = MagicMock()
        fake_chunk.__str__ = lambda self: _LONG_TEXT
        fake_chunk.metadata.to_dict.return_value = {}
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.chunk_by_title",
            lambda *a, **k: [fake_chunk, fake_chunk],
        )
        chunks = mock_ingestor.chunk_elements([fake_chunk, fake_chunk])
        assert len(chunks) == 1


class TestCreateEmbeddings:
    def test_empty_chunks(self, mock_ingestor):
        assert mock_ingestor.create_embeddings([]) == []

    def test_context_strings_length_mismatch(self, mock_ingestor):
        with pytest.raises(ValueError, match="context_strings length"):
            mock_ingestor.create_embeddings(
                [{"content": "a"}], context_strings=["a", "b"]
            )

    def test_basic_encode(self, mock_ingestor):
        chunks = [{"content": "hello"}]
        result = mock_ingestor.create_embeddings(chunks)
        assert result == [[0.1, 0.2, 0.3]]

    def test_passage_prefix(self, mock_ingestor):
        mock_ingestor.passage_prefix = "passage: "
        chunks = [{"content": "hello"}]
        mock_ingestor.create_embeddings(chunks)
        args, _ = mock_ingestor.embedder.encode.call_args
        assert args[0][0] == "passage: hello"

    def test_context_strings(self, mock_ingestor):
        chunks = [{"content": "hello"}, {"content": "world"}]
        mock_ingestor.create_embeddings(chunks, context_strings=["ctx1 ", "ctx2 "])
        args, _ = mock_ingestor.embedder.encode.call_args
        assert args[0][0] == "ctx1 hello"
        assert args[0][1] == "ctx2 world"

    def test_gpu_oom_cascade_falls_back_to_cpu(self, mock_ingestor, monkeypatch):
        emb = mock_ingestor.embedder
        emb.encode.side_effect = [
            RuntimeError("CUDA out of memory"),
            MagicMock(tolist=lambda: [[0.4, 0.5]]),
        ]
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.torch.cuda.is_available", lambda: True
        )
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest._clear_gpu_memory", lambda: None
        )
        monkeypatch.setenv("TROVE_EMBED_BATCH_SIZE", "64")
        chunks = [{"content": "x"}]
        result = mock_ingestor.create_embeddings(chunks)
        assert result == [[0.4, 0.5]]

    def test_gpu_oom_all_sizes_exhausted_cpu_fallback(self, mock_ingestor, monkeypatch):
        emb = mock_ingestor.embedder
        def oom_always(*a, **k):
            raise RuntimeError("CUDA out of memory")
        emb.encode.side_effect = oom_always
        cpu_emb = MagicMock()
        cpu_emb.encode.return_value = MagicMock(tolist=lambda: [[0.9]])
        cpu_emb.to.return_value = cpu_emb
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.torch.cuda.is_available", lambda: True
        )
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest._clear_gpu_memory", lambda: None
        )
        monkeypatch.setenv("TROVE_EMBED_BATCH_SIZE", "32")
        emb.to.return_value = cpu_emb
        chunks = [{"content": "x"}]
        result = mock_ingestor.create_embeddings(chunks)
        assert result == [[0.9]]

    def test_encode_holds_inference_lock(self, mock_ingestor):
        """Regression: create_embeddings must serialize .encode() through
        ModelCache.inference_lock(), not call it unguarded — see
        model_cache.py's inference_lock() docstring for why."""
        mock_lock = MagicMock()
        mock_cache = MagicMock()
        mock_cache.inference_lock.return_value = mock_lock

        with patch(
            "mullet.knowledge.engine.model_cache.get_model_cache",
            return_value=mock_cache,
        ):
            mock_ingestor.create_embeddings([{"content": "hello"}])

        mock_cache.inference_lock.assert_called_once()
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


class TestIngestDocument:
    def test_full_pipeline(self, mock_ingestor, tmp_path, monkeypatch):
        f = tmp_path / "doc.txt"
        f.write_text("hello world")
        fake_elem = MagicMock()
        fake_elem.__str__ = lambda self: _LONG_TEXT
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.partition", lambda *a, **k: [fake_elem]
        )
        fake_chunk = MagicMock()
        fake_chunk.__str__ = lambda self: _LONG_TEXT
        fake_chunk.metadata.to_dict.return_value = {}
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.chunk_by_title",
            lambda *a, **k: [fake_chunk],
        )
        chunks, embeddings = mock_ingestor.ingest_document(f)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source_file"] == "doc.txt"
        assert embeddings == [[0.1, 0.2, 0.3]]

    def test_contextualize(self, mock_ingestor, tmp_path, monkeypatch):
        f = tmp_path / "doc.txt"
        f.write_text("hello world")
        fake_elem = MagicMock()
        fake_elem.__str__ = lambda self: _LONG_TEXT
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.partition", lambda *a, **k: [fake_elem]
        )
        fake_chunk = MagicMock()
        fake_chunk.__str__ = lambda self: _LONG_TEXT
        fake_chunk.metadata.to_dict.return_value = {}
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.chunk_by_title",
            lambda *a, **k: [fake_chunk],
        )
        chunks, embeddings = mock_ingestor.ingest_document(f, contextualize=True)
        assert chunks[0]["metadata"].get("contextualized") is True


# ---------------------------------------------------------------------------
# Mbox ingestion
# ---------------------------------------------------------------------------

def _make_mbox(path: Path, messages: list[EmailMessage]):
    mbox = mailbox.mbox(str(path))
    mbox.lock()
    for msg in messages:
        mbox.add(msg)
    mbox.unlock()
    mbox.close()


class TestIngestMbox:
    def test_file_not_found(self, mock_ingestor):
        with pytest.raises(FileNotFoundError):
            mock_ingestor.ingest_mbox("/nonexistent.mbox")

    def test_basic_email(self, mock_ingestor, tmp_path):
        msg = EmailMessage()
        msg["Subject"] = "Test Subject"
        msg["From"] = "sender@example.com"
        msg["To"] = "receiver@example.com"
        msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0000"
        msg.set_content("This is the email body.")
        mbox_path = tmp_path / "test.mbox"
        _make_mbox(mbox_path, [msg])

        chunks, embeddings = mock_ingestor.ingest_mbox(mbox_path)
        assert len(chunks) >= 1
        assert chunks[0]["metadata"]["email_subject"] == "Test Subject"
        assert chunks[0]["metadata"]["email_from"] == "sender@example.com"
        assert chunks[0]["metadata"]["email_to"] == "receiver@example.com"
        assert embeddings == [[0.1, 0.2, 0.3]]

    def test_skips_empty_body(self, mock_ingestor, tmp_path):
        msg = EmailMessage()
        msg["Subject"] = "Empty"
        msg.set_content("x")
        mbox_path = tmp_path / "test.mbox"
        _make_mbox(mbox_path, [msg])

        chunks, embeddings = mock_ingestor.ingest_mbox(mbox_path)
        assert len(chunks) == 0
        assert embeddings == []

    def test_bad_date_uses_raw(self, mock_ingestor, tmp_path):
        msg = Message()
        msg["Subject"] = "Bad Date"
        msg["Date"] = "not-a-date"
        msg.set_payload("Body text here for the email")
        mbox_path = tmp_path / "test.mbox"
        _make_mbox(mbox_path, [msg])

        chunks, embeddings = mock_ingestor.ingest_mbox(mbox_path)
        assert len(chunks) >= 1
        assert chunks[0]["metadata"]["email_date"] == "not-a-date"


class TestIngestMboxChunksOnly:
    def test_returns_chunks_without_embeddings(self, mock_ingestor, tmp_path):
        msg = EmailMessage()
        msg["Subject"] = "Chunks Only"
        msg.set_content("This is the email body for chunks only.")
        mbox_path = tmp_path / "test.mbox"
        _make_mbox(mbox_path, [msg])

        chunks = mock_ingestor.ingest_mbox_chunks_only(mbox_path)
        assert len(chunks) >= 1
        assert "id" in chunks[0]
        assert "content" in chunks[0]
        assert "metadata" in chunks[0]

    def test_file_not_found(self, mock_ingestor):
        with pytest.raises(FileNotFoundError):
            mock_ingestor.ingest_mbox_chunks_only("/nonexistent.mbox")


class TestSplitEmailContent:
    def test_short_email_no_split(self, mock_ingestor):
        content = "Short email"
        result = mock_ingestor._split_email_content(content, "Subject: test")
        assert result == [content]

    def test_long_email_splits(self, mock_ingestor):
        header = "Subject: test"
        body = "A" * 3000
        content = f"{header}\n\n{body}"
        result = mock_ingestor._split_email_content(content, header)
        assert len(result) > 1
        assert all(header in r or "[continued]" in r for r in result)

    def test_overlap_guard(self, mock_ingestor):
        mock_ingestor.chunk_max_chars = 100
        mock_ingestor.chunk_overlap = 90
        content = "A" * 300
        result = mock_ingestor._split_email_content(content, "Subj")
        assert len(result) > 1


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------

class TestIngestDirectory:
    def test_not_a_directory(self, mock_ingestor):
        with pytest.raises(NotADirectoryError):
            ingest_directory("/nonexistent/dir", ingestor=mock_ingestor)

    def _mock_partition_and_chunk(self, monkeypatch):
        """Mock partition and chunk_by_title for ingest_directory tests."""
        fake_elem = MagicMock()
        fake_elem.__str__ = lambda self: _LONG_TEXT
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.partition", lambda *a, **k: [fake_elem]
        )
        fake_chunk = MagicMock()
        fake_chunk.__str__ = lambda self: _LONG_TEXT
        fake_chunk.metadata.to_dict.return_value = {}
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.chunk_by_title",
            lambda *a, **k: [fake_chunk],
        )

    def test_serial_mode(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("hello world")
        self._mock_partition_and_chunk(monkeypatch)
        chunks, embeddings = ingest_directory(docs, ingestor=mock_ingestor, recursive=False)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source_file"] == "a.txt"

    def test_mbox_in_serial_mode(self, mock_ingestor, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        msg = EmailMessage()
        msg["Subject"] = "Test"
        msg.set_content("Email body here.")
        mbox_path = docs / "mail.mbox"
        _make_mbox(mbox_path, [msg])

        chunks, embeddings = ingest_directory(docs, ingestor=mock_ingestor, recursive=False)
        assert len(chunks) >= 1
        assert any(c["metadata"].get("is_email") for c in chunks)

    def test_pipeline_mode(self, mock_ingestor, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("hello")
        searcher = MagicMock()
        searcher.collection = MagicMock()
        pipeline_inst = MagicMock()
        pipeline_inst.run_documents.return_value = {
            "total_chunks_indexed": 1,
            "total_files_processed": 1,
            "flush_count": 1,
            "elapsed_seconds": 0.1,
            "chunks_per_second": 10.0,
            "failed_files": [],
        }
        with patch("mullet.knowledge.engine.pipeline.IngestionPipeline", return_value=pipeline_inst):
            chunks, embeddings = ingest_directory(
                docs, ingestor=mock_ingestor, pipeline_searcher=searcher, recursive=False
            )
        assert chunks == []
        assert embeddings == []
        pipeline_inst.run_documents.assert_called_once()

    def test_resume_from_checkpoint(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("hello")
        (docs / "b.txt").write_text("world")
        self._mock_partition_and_chunk(monkeypatch)
        from mullet.knowledge.engine.ingest_progress import IngestionCheckpoint

        checkpoint = IngestionCheckpoint(
            client_name="c",
            processed_files=[str(docs / "a.txt")],
            total_files=[str(docs / "a.txt"), str(docs / "b.txt")],
        )
        chunks, embeddings = ingest_directory(
            docs, ingestor=mock_ingestor, resume_from=checkpoint, recursive=False
        )
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source_file"] == "b.txt"

    def test_exclude_patterns(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "skip.txt").write_text("skip me")
        (docs / "keep.txt").write_text("keep me")
        self._mock_partition_and_chunk(monkeypatch)
        chunks, embeddings = ingest_directory(
            docs,
            ingestor=mock_ingestor,
            recursive=False,
            exclude_patterns=["*skip.txt"],
        )
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source_file"] == "keep.txt"

    def test_exclude_patterns_prunes_traversal(self, mock_ingestor, tmp_path, monkeypatch):
        """Same pruning guarantee as build_file_manifest, for the recursive
        os.walk branch of ingest_directory (used by the reindex pipeline).
        """
        docs = tmp_path / "docs"
        docs.mkdir()
        sub = docs / "sub"
        sub.mkdir()
        (sub / "skip.txt").write_text("skip me")
        (docs / "keep.txt").write_text("keep me")
        self._mock_partition_and_chunk(monkeypatch)

        visited_roots = []
        real_walk = ingest_module.os.walk

        def spy_walk(top, *a, **k):
            for root, dirs, names in real_walk(top, *a, **k):
                visited_roots.append(root)
                yield root, dirs, names

        monkeypatch.setattr(ingest_module.os, "walk", spy_walk)

        chunks, embeddings = ingest_directory(
            docs,
            ingestor=mock_ingestor,
            recursive=True,
            exclude_patterns=["**/sub/**"],
        )
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source_file"] == "keep.txt"
        assert str(sub) not in visited_roots

    def test_per_file_error_continues(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "good.txt").write_text("good")
        (docs / "bad.txt").write_text("bad")
        calls = []

        def fake_parse(path, *a, **k):
            calls.append(Path(path).name)
            if "bad" in str(path):
                raise RuntimeError("parse fail")
            return [MagicMock(__str__=lambda self: _LONG_TEXT)]

        monkeypatch.setattr("mullet.knowledge.engine.ingest.partition", fake_parse)
        fake_chunk = MagicMock()
        fake_chunk.__str__ = lambda self: _LONG_TEXT
        fake_chunk.metadata.to_dict.return_value = {}
        monkeypatch.setattr(
            "mullet.knowledge.engine.ingest.chunk_by_title",
            lambda *a, **k: [fake_chunk],
        )
        chunks, embeddings = ingest_directory(
            docs, ingestor=mock_ingestor, recursive=False
        )
        assert len(calls) == 2
        assert len(chunks) == 1

    def test_progress_callback(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("hello")
        self._mock_partition_and_chunk(monkeypatch)
        progress_calls = []
        ingest_directory(
            docs,
            ingestor=mock_ingestor,
            recursive=False,
            progress_callback=lambda p: progress_calls.append(p),
        )
        assert len(progress_calls) >= 1

    def test_non_recursive(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        sub = docs / "sub"
        sub.mkdir()
        (docs / "root.txt").write_text("root")
        (sub / "nested.txt").write_text("nested")
        self._mock_partition_and_chunk(monkeypatch)
        chunks, embeddings = ingest_directory(docs, ingestor=mock_ingestor, recursive=False)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source_file"] == "root.txt"

    def test_recursive(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        sub = docs / "sub"
        sub.mkdir()
        (docs / "root.txt").write_text("root")
        (sub / "nested.txt").write_text("nested")
        self._mock_partition_and_chunk(monkeypatch)
        chunks, embeddings = ingest_directory(docs, ingestor=mock_ingestor, recursive=True)
        assert len(chunks) == 2

    def test_symlink_cycle_guard(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        sub = docs / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("hello")
        try:
            (sub / "loop").symlink_to(docs)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")
        self._mock_partition_and_chunk(monkeypatch)
        chunks, embeddings = ingest_directory(docs, ingestor=mock_ingestor, recursive=True)
        assert len(chunks) == 1

    def test_broken_symlink_skipped(self, mock_ingestor, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        try:
            (docs / "broken").symlink_to("/nonexistent/target.txt")
        except OSError:
            pytest.skip("Symlinks not supported on this platform")
        chunks, embeddings = ingest_directory(docs, ingestor=mock_ingestor, recursive=False)
        assert len(chunks) == 0

    def test_filenames_without_extension(self, mock_ingestor, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "Makefile").write_text("build:")
        self._mock_partition_and_chunk(monkeypatch)
        chunks, embeddings = ingest_directory(
            docs, ingestor=mock_ingestor, recursive=False, filenames=("Makefile",)
        )
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source_file"] == "Makefile"


# ---------------------------------------------------------------------------
# ingest_code
# ---------------------------------------------------------------------------

class TestIngestCode:
    def _patch_code_deps(self, chunks=None):
        chunks = chunks or []
        mock_chunker_cls = MagicMock()
        mock_chunker = MagicMock()
        mock_chunker.detect_repositories.return_value = {}
        mock_chunker.collect_files.return_value = [Path("/fake/file.py")]
        mock_chunker.chunk_files.return_value = chunks
        mock_chunker_cls.return_value = mock_chunker

        mock_state_cls = MagicMock()
        mock_state = MagicMock()
        mock_state.state = {}
        mock_state.get_changes.return_value = (
            [Path("/fake/file.py")],
            [],
            [],
        )
        mock_state.remove_file.return_value = []
        mock_state.get_chunk_ids_for_file.return_value = []
        mock_state_cls.return_value = mock_state

        patchers = [
            patch("mullet.knowledge.code.CodeChunker", mock_chunker_cls),
            patch("mullet.knowledge.code.CodeIndexState", mock_state_cls),
            patch(
                "mullet.knowledge.code.compute_chunk_id",
                lambda c: f"id-{c.get('name', 'x')}",
            ),
        ]
        return patchers, mock_chunker, mock_state

    def test_full_rebuild_pipeline_mode(self, mock_ingestor):
        chunks = [
            {"name": "foo", "content": _LONG_TEXT, "metadata": {"source_file": "file.py"}},
        ]
        patchers, _mock_chunker, _mock_state = self._patch_code_deps(chunks=chunks)
        pipeline_inst = MagicMock()
        pipeline_inst.run_code.return_value = {
            "total_chunks_indexed": 1,
            "flush_count": 1,
            "elapsed_seconds": 0.1,
            "chunks_per_second": 10.0,
        }
        patchers.append(patch("mullet.knowledge.engine.pipeline.IngestionPipeline", return_value=pipeline_inst))

        src = Path("/fake/src")
        searcher = MagicMock()
        searcher.collection = MagicMock()
        searcher.collection.get.return_value = {"ids": []}
        code_config = {"extensions": [".py"], "incremental": False}

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            result_chunks, result_embs = ingest_code(
                code_paths=[src],
                code_config=code_config,
                ingestor=mock_ingestor,
                rebuild=True,
                pipeline_searcher=searcher,
            )
        assert len(result_chunks) == 1
        pipeline_inst.run_code.assert_called_once()

    def test_incremental_no_changes(self, mock_ingestor, tmp_path):
        patchers, _mock_chunker, mock_state = self._patch_code_deps(chunks=[])
        mock_state.get_changes.return_value = ([], [], [])
        src = tmp_path / "src"
        src.mkdir()
        index_path = tmp_path / ".index"
        index_path.mkdir()
        code_config = {"extensions": [".py"], "incremental": True}

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            result_chunks, result_embs = ingest_code(
                code_paths=[src],
                code_config=code_config,
                ingestor=mock_ingestor,
                index_path=index_path,
                rebuild=False,
            )
        assert result_chunks == []
        assert result_embs == []

    def test_incremental_with_deletions_only(self, mock_ingestor, tmp_path):
        patchers, _mock_chunker, mock_state = self._patch_code_deps(chunks=[])
        mock_state.get_changes.return_value = (
            [],
            [],
            [Path("/fake/deleted.py")],
        )
        mock_state.remove_file.return_value = ["old-chunk-1"]
        src = tmp_path / "src"
        src.mkdir()
        index_path = tmp_path / ".index"
        index_path.mkdir()
        code_config = {"extensions": [".py"], "incremental": True}

        searcher = MagicMock()
        searcher.collection = MagicMock()
        searcher.collection.get.return_value = {"ids": ["old-chunk-1"]}

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            result_chunks, result_embs = ingest_code(
                code_paths=[src],
                code_config=code_config,
                ingestor=mock_ingestor,
                index_path=index_path,
                rebuild=False,
                pipeline_searcher=searcher,
            )
        assert result_chunks == []
        searcher.collection.delete.assert_called_once_with(ids=["old-chunk-1"])

    def test_serial_mode(self, mock_ingestor, tmp_path):
        chunks = [
            {"name": "foo", "content": _LONG_TEXT, "metadata": {"source_file": "file.py"}},
        ]
        patchers, _mock_chunker, _mock_state = self._patch_code_deps(chunks=chunks)
        src = tmp_path / "src"
        src.mkdir()
        code_config = {"extensions": [".py"], "incremental": False}

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            result_chunks, result_embs = ingest_code(
                code_paths=[src],
                code_config=code_config,
                ingestor=mock_ingestor,
                rebuild=True,
            )
        assert len(result_chunks) == 1
        assert result_embs == [[0.1, 0.2, 0.3]]

    def test_stale_chunk_removal_failure_logged(self, mock_ingestor, tmp_path):
        chunks = [
            {"name": "foo", "content": _LONG_TEXT, "metadata": {"source_file": "file.py"}},
        ]
        _patchers, _mock_chunker, _mock_state = self._patch_code_deps(chunks=chunks)
        mock_state = MagicMock()
        mock_state.get_changes.return_value = (
            [Path("/fake/file.py")],
            [],
            [Path("/fake/deleted.py")],
        )
        mock_state.remove_file.return_value = ["stale-1"]
        mock_state.get_chunk_ids_for_file.return_value = ["stale-2"]
        mock_state.state = {}
        mock_state_cls = MagicMock(return_value=mock_state)
        mock_chunker_cls = MagicMock()
        mock_chunker = MagicMock()
        mock_chunker.detect_repositories.return_value = {}
        mock_chunker.collect_files.return_value = [Path("/fake/file.py")]
        mock_chunker.chunk_files.return_value = chunks
        mock_chunker_cls.return_value = mock_chunker

        searcher = MagicMock()
        searcher.collection = MagicMock()
        searcher.collection.get.side_effect = RuntimeError("chroma down")

        src = tmp_path / "src"
        src.mkdir()
        index_path = tmp_path / ".index"
        index_path.mkdir()
        code_config = {"extensions": [".py"], "incremental": True}

        with patch("mullet.knowledge.code.CodeChunker", mock_chunker_cls), \
             patch("mullet.knowledge.code.CodeIndexState", mock_state_cls), \
             patch("mullet.knowledge.code.compute_chunk_id", lambda c: f"id-{c.get('name', 'x')}"):
            result_chunks, result_embs = ingest_code(
                code_paths=[src],
                code_config=code_config,
                ingestor=mock_ingestor,
                index_path=index_path,
                rebuild=False,
                pipeline_searcher=searcher,
            )
        assert len(result_chunks) == 1


# ---------------------------------------------------------------------------
# File manifest helpers
# ---------------------------------------------------------------------------

class TestBuildFileManifest:
    def test_empty_dir(self, tmp_path):
        manifest = build_file_manifest(tmp_path, (".py",))
        assert manifest == {}

    def test_matches_extension(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        manifest = build_file_manifest(tmp_path, (".py",), recursive=False)
        assert len(manifest) == 1
        assert "a.py" in str(list(manifest.keys())[0])

    def test_excludes_hidden(self, tmp_path):
        (tmp_path / ".hidden.py").write_text("a")
        manifest = build_file_manifest(tmp_path, (".py",), recursive=False)
        assert len(manifest) == 0

    def test_exclude_patterns(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.py").write_text("a")
        manifest = build_file_manifest(
            tmp_path, (".py",), exclude_patterns=["**/sub/**"]
        )
        assert len(manifest) == 0

    def test_exclude_patterns_prunes_traversal(self, tmp_path, monkeypatch):
        """Excluded directories must never be descended into by os.walk —
        not just filtered out of the result after a full traversal. A repo
        with a huge excluded dir (vendor/, node_modules/, a data mirror)
        should never even be listed, or indexing pays the walk cost anyway.
        """
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.py").write_text("a")
        (tmp_path / "keep.py").write_text("b")

        visited_roots = []
        real_walk = ingest_module.os.walk

        def spy_walk(top, *a, **k):
            for root, dirs, names in real_walk(top, *a, **k):
                visited_roots.append(root)
                yield root, dirs, names

        monkeypatch.setattr(ingest_module.os, "walk", spy_walk)

        manifest = build_file_manifest(
            tmp_path, (".py",), exclude_patterns=["**/sub/**"]
        )

        assert len(manifest) == 1
        assert str(sub) not in visited_roots

    def test_filenames(self, tmp_path):
        (tmp_path / "Makefile").write_text("build")
        manifest = build_file_manifest(
            tmp_path, (".py",), recursive=False, filenames=("Makefile",)
        )
        assert len(manifest) == 1

    def test_non_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.py").write_text("a")
        (sub / "nested.py").write_text("b")
        manifest = build_file_manifest(tmp_path, (".py",), recursive=False)
        assert len(manifest) == 1

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.py").write_text("a")
        (sub / "nested.py").write_text("b")
        manifest = build_file_manifest(tmp_path, (".py",), recursive=True)
        assert len(manifest) == 2

    def test_broken_symlink_skipped(self, tmp_path):
        try:
            (tmp_path / "broken").symlink_to("/nonexistent")
        except OSError:
            pytest.skip("Symlinks not supported")
        manifest = build_file_manifest(tmp_path, (".py",), recursive=False)
        assert len(manifest) == 0

    def test_symlink_cycle_guard(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        try:
            (sub / "loop").symlink_to(tmp_path)
        except OSError:
            pytest.skip("Symlinks not supported")
        (tmp_path / "a.py").write_text("a")
        manifest = build_file_manifest(tmp_path, (".py",), recursive=True)
        assert len(manifest) == 1


class TestDiffFileManifest:
    def test_empty_both(self):
        new, mod, deleted = diff_file_manifest({}, {})
        assert new == []
        assert mod == []
        assert deleted == []

    def test_new_files(self):
        current = {"/a.py": {"mtime": 1, "size": 10}}
        indexed = {}
        new, mod, deleted = diff_file_manifest(current, indexed)
        assert new == ["/a.py"]

    def test_deleted_files(self):
        current = {}
        indexed = {"/a.py": {"mtime": 1, "size": 10}}
        new, mod, deleted = diff_file_manifest(current, indexed)
        assert deleted == ["/a.py"]

    def test_modified_files(self):
        current = {"/a.py": {"mtime": 2, "size": 10}}
        indexed = {"/a.py": {"mtime": 1, "size": 10}}
        new, mod, deleted = diff_file_manifest(current, indexed)
        assert mod == ["/a.py"]

    def test_unchanged_files(self):
        current = {"/a.py": {"mtime": 1, "size": 10}}
        indexed = {"/a.py": {"mtime": 1, "size": 10}}
        new, mod, deleted = diff_file_manifest(current, indexed)
        assert new == mod == deleted == []


class TestSaveLoadFileManifest:
    def test_roundtrip(self, tmp_path):
        client = MagicMock()
        client.index_path = tmp_path / "index"
        client.machine_name = "test-client"
        manifest = {"/a.py": {"mtime": 1.0, "size": 10}}
        save_file_manifest(client, manifest)
        loaded = load_file_manifest(client)
        assert loaded == manifest

    def test_load_missing_returns_empty(self, tmp_path):
        client = MagicMock()
        client.index_path = tmp_path / "index"
        assert load_file_manifest(client) == {}

    def test_load_bad_json_returns_empty(self, tmp_path):
        client = MagicMock()
        client.index_path = tmp_path / "index"
        client.index_path.mkdir(parents=True, exist_ok=True)
        manifest_path = client.index_path / "file_manifest.json"
        manifest_path.write_text("not json")
        assert load_file_manifest(client) == {}
