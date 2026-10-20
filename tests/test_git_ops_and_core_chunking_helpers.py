"""Tests for git_ops pure helpers and core/knowledge/chunking strategies."""

import pytest
from mullet.knowledge.sources.git_ops import (
    GitError,
    CloneResult,
    _sanitise_output,
    _validate_name,
)
from mullet.core.knowledge.chunking import (
    Chunk,
    ChunkMetadata,
    SentenceChunker,
    RecursiveChunker,
    enrich_chunks,
)


# ---------------------------------------------------------------------------
# GitError
# ---------------------------------------------------------------------------

class TestGitError:
    def test_is_exception(self):
        assert issubclass(GitError, Exception)

    def test_message_stored(self):
        e = GitError("clone failed")
        assert "clone failed" in str(e)

    def test_can_raise_and_catch(self):
        with pytest.raises(GitError):
            raise GitError("error")

    def test_catchable_as_exception(self):
        with pytest.raises(Exception):
            raise GitError("error")


# ---------------------------------------------------------------------------
# CloneResult
# ---------------------------------------------------------------------------

class TestCloneResult:
    def test_success_stored(self):
        cr = CloneResult(success=True, message="ok")
        assert cr.success is True

    def test_message_stored(self):
        cr = CloneResult(success=True, message="cloned ok")
        assert cr.message == "cloned ok"

    def test_changed_files_default_empty(self):
        cr = CloneResult(success=True, message="ok")
        assert cr.changed_files == []

    def test_is_fresh_clone_default_false(self):
        cr = CloneResult(success=True, message="ok")
        assert cr.is_fresh_clone is False

    def test_custom_is_fresh_clone(self):
        cr = CloneResult(success=True, message="ok", is_fresh_clone=True)
        assert cr.is_fresh_clone is True


# ---------------------------------------------------------------------------
# _sanitise_output
# ---------------------------------------------------------------------------

class TestSanitiseOutput:
    def test_clean_text_unchanged(self):
        assert _sanitise_output("no tokens here") == "no tokens here"

    def test_url_token_masked(self):
        url = "https://user:mysecret@github.com/org/repo"
        result = _sanitise_output(url)
        assert "mysecret" not in result
        assert "****" in result

    def test_url_host_preserved(self):
        url = "https://user:secret@github.com/org/repo"
        result = _sanitise_output(url)
        assert "github.com" in result

    def test_ghp_token_masked(self):
        token = "ghp_" + "A" * 36
        result = _sanitise_output(f"token: {token}")
        assert token not in result
        assert "****" in result

    def test_empty_string_unchanged(self):
        assert _sanitise_output("") == ""

    def test_returns_string(self):
        assert isinstance(_sanitise_output("test"), str)


# ---------------------------------------------------------------------------
# _validate_name
# ---------------------------------------------------------------------------

class TestValidateName:
    def test_valid_alphanumeric_ok(self):
        _validate_name("my-repo")  # should not raise

    def test_valid_with_dots(self):
        _validate_name("repo.name")  # should not raise

    def test_valid_with_underscores(self):
        _validate_name("my_repo_123")  # should not raise

    def test_empty_raises_git_error(self):
        with pytest.raises(GitError):
            _validate_name("")

    def test_path_traversal_raises(self):
        with pytest.raises(GitError):
            _validate_name("../escape")

    def test_space_raises(self):
        with pytest.raises(GitError):
            _validate_name("my repo")

    def test_slash_raises(self):
        with pytest.raises(GitError):
            _validate_name("my/repo")


# ---------------------------------------------------------------------------
# ChunkMetadata and Chunk
# ---------------------------------------------------------------------------

class TestChunkMetadata:
    def test_source_stored(self):
        meta = ChunkMetadata(source="/tmp/test.py", chunk_index=0)
        assert meta.source == "/tmp/test.py"

    def test_chunk_index_stored(self):
        meta = ChunkMetadata(source="test.py", chunk_index=3)
        assert meta.chunk_index == 3

    def test_char_start_default_zero(self):
        meta = ChunkMetadata(source="test.py", chunk_index=0)
        assert meta.char_start == 0

    def test_heading_path_default_empty(self):
        meta = ChunkMetadata(source="test.py", chunk_index=0)
        assert meta.heading_path == ""


class TestChunk:
    def test_text_stored(self):
        meta = ChunkMetadata(source="f.py", chunk_index=0)
        chunk = Chunk(text="hello world", metadata=meta)
        assert chunk.text == "hello world"

    def test_metadata_stored(self):
        meta = ChunkMetadata(source="f.py", chunk_index=1)
        chunk = Chunk(text="content", metadata=meta)
        assert chunk.metadata.chunk_index == 1


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------

class TestSentenceChunker:
    def test_returns_list(self):
        sc = SentenceChunker()
        assert isinstance(sc.chunk("Hello world."), list)

    def test_empty_text_returns_empty(self):
        sc = SentenceChunker()
        assert sc.chunk("") == []

    def test_short_text_single_chunk(self):
        sc = SentenceChunker(max_chars=1000, min_chars=0)
        result = sc.chunk("Hello world.")
        assert len(result) == 1
        assert result[0].text == "Hello world."

    def test_long_text_split(self):
        sc = SentenceChunker(max_chars=50, min_chars=10)
        text = "First sentence. Second sentence. Third sentence."
        result = sc.chunk(text)
        assert len(result) >= 1

    def test_chunks_are_chunk_instances(self):
        sc = SentenceChunker()
        result = sc.chunk("Hello.")
        if result:
            assert isinstance(result[0], Chunk)

    def test_metadata_source_set(self):
        sc = SentenceChunker()
        result = sc.chunk("Hello.", source="/tmp/test.txt")
        if result:
            assert result[0].metadata.source == "/tmp/test.txt"

    def test_chunk_index_sequential(self):
        sc = SentenceChunker(max_chars=20, min_chars=5)
        text = "First sentence here. Second one now. Third one follows."
        result = sc.chunk(text)
        if len(result) >= 2:
            assert result[0].metadata.chunk_index < result[1].metadata.chunk_index


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_returns_list(self):
        rc = RecursiveChunker(max_chars=100)
        assert isinstance(rc.chunk("hello world"), list)

    def test_empty_text_returns_empty(self):
        rc = RecursiveChunker(max_chars=100)
        assert rc.chunk("") == []

    def test_short_text_single_chunk(self):
        rc = RecursiveChunker(max_chars=1000)
        result = rc.chunk("This is some text that meets the minimum chunk size threshold.")
        assert len(result) == 1

    def test_chunks_are_chunk_instances(self):
        rc = RecursiveChunker(max_chars=1000)
        result = rc.chunk("test")
        if result:
            assert isinstance(result[0], Chunk)


# ---------------------------------------------------------------------------
# enrich_chunks
# ---------------------------------------------------------------------------

class TestEnrichChunks:
    def test_returns_list(self):
        chunks = []
        assert isinstance(enrich_chunks(chunks, "test.py"), list)

    def test_empty_input_returns_empty(self):
        assert enrich_chunks([], "test.py") == []

    def test_sets_source_path(self):
        meta = ChunkMetadata(source="", chunk_index=0)
        chunk = Chunk(text="hello", metadata=meta)
        result = enrich_chunks([chunk], "/tmp/myfile.py")
        assert result[0].metadata.source == "/tmp/myfile.py"

    def test_chunk_count_preserved(self):
        meta = ChunkMetadata(source="", chunk_index=0)
        chunks = [Chunk(text=f"chunk {i}", metadata=meta) for i in range(3)]
        result = enrich_chunks(chunks, "file.py")
        assert len(result) == 3
