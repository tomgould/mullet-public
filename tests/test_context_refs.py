"""Tests for context reference parsing and resolution."""

import subprocess
from unittest.mock import patch, MagicMock

from mullet.core.agent.context_refs import (
    ContextReference,
    _expand_file,
    _expand_folder,
    _expand_git_diff,
    _expand_git_log,
    _expand_reference,
    _expand_url,
    _is_binary,
    _resolve_path,
    parse_context_references,
    resolve_context_references,
)


# --- Parsing tests ---

def test_parse_file_reference():
    refs = parse_context_references("look at @file:src/main.py")
    assert len(refs) == 1
    assert refs[0].kind == "file"
    assert refs[0].target == "src/main.py"


def test_parse_file_with_line_range():
    refs = parse_context_references("check @file:app.py:10-20")
    assert len(refs) == 1
    assert refs[0].kind == "file"
    assert refs[0].target == "app.py"
    assert refs[0].line_start == 10
    assert refs[0].line_end == 20


def test_parse_folder_reference():
    refs = parse_context_references("list @folder:src/components")
    assert len(refs) == 1
    assert refs[0].kind == "folder"
    assert refs[0].target == "src/components"


def test_parse_diff():
    refs = parse_context_references("review @diff please")
    assert len(refs) == 1
    assert refs[0].kind == "diff"


def test_parse_staged():
    refs = parse_context_references("check @staged changes")
    assert len(refs) == 1
    assert refs[0].kind == "staged"


def test_parse_git_log():
    refs = parse_context_references("show @git:5 recent commits")
    assert len(refs) == 1
    assert refs[0].kind == "git"
    assert refs[0].target == "5"


def test_parse_url():
    refs = parse_context_references("read @url:https://example.com/api")
    assert len(refs) == 1
    assert refs[0].kind == "url"
    assert refs[0].target == "https://example.com/api"


def test_parse_multiple_refs():
    refs = parse_context_references("compare @file:old.py and @file:new.py with @diff")
    assert len(refs) == 3
    assert refs[0].kind == "file"
    assert refs[1].kind == "file"
    assert refs[2].kind == "diff"


def test_parse_no_refs():
    refs = parse_context_references("just a normal message")
    assert refs == []


def test_parse_email_not_matched():
    """Email addresses should NOT be parsed as references."""
    refs = parse_context_references("send to user@example.com")
    assert refs == []


def test_parse_trailing_punctuation_stripped():
    refs = parse_context_references("look at @file:main.py.")
    assert len(refs) == 1
    assert refs[0].target == "main.py"


# --- Resolution tests ---

def test_resolve_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.py").write_text("print('hello')\n")
    result = resolve_context_references("review @file:hello.py", cwd=str(tmp_path))
    assert result.expanded
    assert "hello" in result.message
    assert "```python" in result.message
    assert result.injected_tokens > 0


def test_resolve_file_line_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lines = "\n".join(f"line {i}" for i in range(1, 21))
    (tmp_path / "big.txt").write_text(lines)
    result = resolve_context_references("show @file:big.txt:5-8", cwd=str(tmp_path))
    assert result.expanded
    assert "line 5" in result.message
    assert "line 8" in result.message
    assert "line 1\n" not in result.message


def test_resolve_file_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = resolve_context_references("show @file:missing.py", cwd=str(tmp_path))
    assert not result.expanded
    assert len(result.warnings) >= 1
    assert "not found" in result.warnings[0].lower()


def test_resolve_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x=1\n")
    (tmp_path / "src" / "utils.py").write_text("y=2\n")
    result = resolve_context_references("list @folder:src", cwd=str(tmp_path))
    assert result.expanded
    assert "app.py" in result.message
    assert "utils.py" in result.message


def test_resolve_diff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "a.py").write_text("old\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "a.py").write_text("new\n")
    result = resolve_context_references("review @diff", cwd=str(tmp_path))
    assert result.expanded
    assert "new" in result.message


def test_resolve_staged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "a.py").write_text("initial\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "a.py").write_text("staged change\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    result = resolve_context_references("check @staged", cwd=str(tmp_path))
    assert result.expanded
    assert "staged change" in result.message


def test_resolve_git_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "a.py").write_text("x\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "test commit"], cwd=tmp_path, capture_output=True)
    result = resolve_context_references("show @git:1", cwd=str(tmp_path))
    assert result.expanded
    assert "test commit" in result.message


def test_resolve_blocked_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = resolve_context_references("show @file:~/.ssh/id_rsa", cwd=str(tmp_path))
    assert not result.expanded
    assert len(result.warnings) >= 1
    assert "blocked" in result.warnings[0].lower()


def test_resolve_binary_file_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    result = resolve_context_references("show @file:image.png", cwd=str(tmp_path))
    assert not result.expanded
    assert any("binary" in w.lower() for w in result.warnings)


def test_resolve_strips_refs_from_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("code\n")
    result = resolve_context_references("review @file:a.py please", cwd=str(tmp_path))
    assert "@file" not in result.message
    assert "review" in result.message
    assert "please" in result.message


def test_token_budget_blocking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Create a large file
    (tmp_path / "big.txt").write_text("x " * 100_000)
    result = resolve_context_references(
        "read @file:big.txt",
        cwd=str(tmp_path),
        context_length=1000,  # Very small context
    )
    assert result.blocked


def test_large_file_rejected(tmp_path, monkeypatch):
    """Files over 500KB should be rejected."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "huge.sql").write_text("x" * 600_000)
    result = resolve_context_references("load @file:huge.sql", cwd=str(tmp_path))
    assert not result.expanded
    assert any("too large" in w.lower() for w in result.warnings)


def test_no_refs_passthrough():
    result = resolve_context_references("just a normal task")
    assert result.message == "just a normal task"
    assert not result.expanded
    assert result.references == []


class TestContextRefsMissingLines:
    """Tests covering previously uncovered lines in context_refs.py."""

    # Lines 146-148: injection finding triggers sanitize + category warning
    def test_injection_finding_sanitizes_and_warns(self, tmp_path):
        (tmp_path / "hello.py").write_text("print('hello')")
        with patch("mullet.core.agent.injection_guard.scan_content", return_value=[{"category": "prompt_injection"}]) as mock_scan, \
             patch("mullet.core.agent.injection_guard.sanitize_content", return_value=("sanitized", [])) as mock_sanitize:
            result = resolve_context_references("review @file:hello.py", cwd=str(tmp_path))
            assert result.expanded
            assert any("injection patterns" in w.lower() for w in result.warnings)
            assert any("prompt_injection" in w for w in result.warnings)
            mock_scan.assert_called_once()
            mock_sanitize.assert_called_once()

    # Line 164: injected_tokens > soft_limit warning
    def test_soft_limit_warning_appended(self, tmp_path):
        (tmp_path / "medium.txt").write_text("x" * 1200)
        result = resolve_context_references("read @file:medium.txt", cwd=str(tmp_path), context_length=1000)
        assert result.expanded
        assert any("large context injection" in w.lower() for w in result.warnings)

    # Lines 198-200: unknown ref kind
    def test_expand_reference_unknown_kind(self):
        ref = ContextReference(raw="@unknown:x", kind="unknown", target="x")
        content, error = _expand_reference(ref, "/tmp")
        assert content is None
        assert "unknown reference kind" in error.lower()

    # Line 211: target exists but is a directory
    def test_expand_file_target_is_directory(self, tmp_path):
        (tmp_path / "adir").mkdir()
        ref = ContextReference(raw="@file:adir", kind="file", target="adir")
        content, error = _expand_file(ref, str(tmp_path))
        assert content is None
        assert "not a file" in error.lower()

    # Lines 221-222: stat OSError is swallowed, read proceeds
    def test_expand_file_stat_oserror(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        with patch("mullet.core.agent.context_refs._resolve_path") as mock_resolve, \
             patch("mullet.core.agent.context_refs._is_binary", return_value=False):
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.is_file.return_value = True
            mock_path.stat.side_effect = OSError("stat fail")
            mock_path.read_text.return_value = "hello"
            mock_path.suffix = ".txt"
            mock_resolve.return_value = mock_path
            ref = ContextReference(raw="@file:foo.txt", kind="file", target="foo.txt")
            content, error = _expand_file(ref, str(tmp_path))
            assert content is not None
            assert "hello" in content
            assert error is None

    # Lines 226-227: read_text raises → Read error
    def test_expand_file_read_text_error(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hi")
        ref = ContextReference(raw="@file:foo.txt", kind="file", target="foo.txt")
        with patch("pathlib.Path.read_text", side_effect=IOError("read fail")):
            content, error = _expand_file(ref, str(tmp_path))
            assert content is None
            assert "read error" in error.lower()

    # Line 245: blocked folder path
    def test_expand_folder_blocked_path(self):
        ref = ContextReference(raw="@folder:~/.ssh", kind="folder", target="~/.ssh")
        content, error = _expand_folder(ref, "/tmp")
        assert content is None
        assert "path blocked" in error.lower()

    # Line 247: folder does not exist
    def test_expand_folder_not_found(self, tmp_path):
        ref = ContextReference(raw="@folder:nonexistent", kind="folder", target="nonexistent")
        content, error = _expand_folder(ref, str(tmp_path))
        assert content is None
        assert "directory not found" in error.lower()

    # Lines 258-259, 264: folder walk hits MAX_FOLDER_ENTRIES
    def test_expand_folder_max_entries(self, tmp_path):
        (tmp_path / "src").mkdir()
        ref = ContextReference(raw="@folder:src", kind="folder", target="src")
        many_files = [f"file{i}.py" for i in range(250)]
        with patch("mullet.core.agent.context_refs.os.walk", return_value=[(str(tmp_path / "src"), [], many_files)]):
            content, error = _expand_folder(ref, str(tmp_path))
            assert content is not None
            assert "... and more" in content
            assert error is None

    # Lines 279-282: git diff non-zero, "not a git repository"
    def test_expand_git_diff_not_git_repo(self):
        ref = ContextReference(raw="@diff", kind="diff", target="")
        fake = MagicMock(returncode=1, stderr="fatal: not a git repository")
        with patch("mullet.core.agent.context_refs.subprocess.run", return_value=fake):
            content, error = _expand_git_diff(ref, "/tmp")
            assert content is None
            assert "not a git repository" in error.lower()

    # Lines 279-282: git diff non-zero, other stderr
    def test_expand_git_diff_generic_error(self):
        ref = ContextReference(raw="@diff", kind="diff", target="")
        fake = MagicMock(returncode=1, stderr="some generic error\nline two")
        with patch("mullet.core.agent.context_refs.subprocess.run", return_value=fake):
            content, error = _expand_git_diff(ref, "/tmp")
            assert content is None
            assert "git diff failed" in error.lower()

    # Line 285: git diff empty output
    def test_expand_git_diff_empty(self):
        ref = ContextReference(raw="@diff", kind="diff", target="")
        fake = MagicMock(returncode=0, stdout="   \n")
        with patch("mullet.core.agent.context_refs.subprocess.run", return_value=fake):
            content, error = _expand_git_diff(ref, "/tmp")
            assert "no changes" in content.lower()
            assert error is None

    # Line 287: git diff output truncated
    def test_expand_git_diff_truncated(self):
        ref = ContextReference(raw="@diff", kind="diff", target="")
        fake = MagicMock(returncode=0, stdout="a" * 40_000)
        with patch("mullet.core.agent.context_refs.subprocess.run", return_value=fake):
            content, error = _expand_git_diff(ref, "/tmp")
            assert "[truncated" in content
            assert error is None

    # Lines 289-290: git diff TimeoutExpired
    def test_expand_git_diff_timeout(self):
        ref = ContextReference(raw="@diff", kind="diff", target="")
        with patch("mullet.core.agent.context_refs.subprocess.run", side_effect=subprocess.TimeoutExpired("git diff", 30)):
            content, error = _expand_git_diff(ref, "/tmp")
            assert content is None
            assert "timed out" in error.lower()

    # Lines 291-292: git diff FileNotFoundError
    def test_expand_git_diff_git_not_found(self):
        ref = ContextReference(raw="@diff", kind="diff", target="")
        with patch("mullet.core.agent.context_refs.subprocess.run", side_effect=FileNotFoundError("git")):
            content, error = _expand_git_diff(ref, "/tmp")
            assert content is None
            assert "git not found" in error.lower()

    # Lines 299-300: git log non-integer target
    def test_expand_git_log_invalid_count(self):
        ref = ContextReference(raw="@git:abc", kind="git", target="abc")
        content, error = _expand_git_log(ref, "/tmp")
        assert content is None
        assert "invalid git count" in error.lower()

    # Lines 308-311: git log non-zero, "not a git repository"
    def test_expand_git_log_not_git_repo(self):
        ref = ContextReference(raw="@git:3", kind="git", target="3")
        fake = MagicMock(returncode=1, stderr="fatal: not a git repository")
        with patch("mullet.core.agent.context_refs.subprocess.run", return_value=fake):
            content, error = _expand_git_log(ref, "/tmp")
            assert content is None
            assert "not a git repository" in error.lower()

    # Lines 308-311: git log non-zero, other stderr
    def test_expand_git_log_generic_error(self):
        ref = ContextReference(raw="@git:3", kind="git", target="3")
        fake = MagicMock(returncode=1, stderr="some error")
        with patch("mullet.core.agent.context_refs.subprocess.run", return_value=fake):
            content, error = _expand_git_log(ref, "/tmp")
            assert content is None
            assert "git log failed" in error.lower()

    # Line 314: git log empty output
    def test_expand_git_log_empty(self):
        ref = ContextReference(raw="@git:3", kind="git", target="3")
        fake = MagicMock(returncode=0, stdout="   ")
        with patch("mullet.core.agent.context_refs.subprocess.run", return_value=fake):
            content, error = _expand_git_log(ref, "/tmp")
            assert "no commits found" in content.lower()
            assert error is None

    # Line 316: git log output truncated
    def test_expand_git_log_truncated(self):
        ref = ContextReference(raw="@git:3", kind="git", target="3")
        fake = MagicMock(returncode=0, stdout="a" * 40_000)
        with patch("mullet.core.agent.context_refs.subprocess.run", return_value=fake):
            content, error = _expand_git_log(ref, "/tmp")
            assert "[truncated" in content
            assert error is None

    # Lines 318-319: git log TimeoutExpired
    def test_expand_git_log_timeout(self):
        ref = ContextReference(raw="@git:3", kind="git", target="3")
        with patch("mullet.core.agent.context_refs.subprocess.run", side_effect=subprocess.TimeoutExpired("git log", 30)):
            content, error = _expand_git_log(ref, "/tmp")
            assert content is None
            assert "timed out" in error.lower()

    # Lines 320-321: git log FileNotFoundError
    def test_expand_git_log_git_not_found(self):
        ref = ContextReference(raw="@git:3", kind="git", target="3")
        with patch("mullet.core.agent.context_refs.subprocess.run", side_effect=FileNotFoundError("git")):
            content, error = _expand_git_log(ref, "/tmp")
            assert content is None
            assert "git not found" in error.lower()

    # Lines 326-337: expand URL success
    def test_expand_url_success(self):
        ref = ContextReference(raw="@url:https://example.com", kind="url", target="https://example.com")
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"Hello World"
        with patch("mullet.core.tools.url_safety.safe_urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            mock_urlopen.return_value.__exit__.return_value = False
            content, error = _expand_url(ref)
            assert "Hello World" in content
            assert error is None

    # Lines 335-336: expand URL truncated
    def test_expand_url_truncated(self):
        ref = ContextReference(raw="@url:https://example.com", kind="url", target="https://example.com")
        mock_resp = MagicMock()
        mock_resp.read.return_value = ("x" * 25_000).encode()
        with patch("mullet.core.tools.url_safety.safe_urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            mock_urlopen.return_value.__exit__.return_value = False
            content, error = _expand_url(ref)
            assert "[truncated" in content
            assert error is None

    # Lines 338-339: expand URL exception
    def test_expand_url_exception(self):
        ref = ContextReference(raw="@url:https://example.com", kind="url", target="https://example.com")
        with patch("mullet.core.tools.url_safety.safe_urlopen", side_effect=IOError("network error")):
            content, error = _expand_url(ref)
            assert content is None
            assert "url fetch failed" in error.lower()

    # Line 354: resolve_path blocked dir
    def test_resolve_path_blocked_dir(self):
        result = _resolve_path("~/.ssh/id_rsa", "/tmp")
        assert result is None

    # Line 357: resolve_path .env variant
    def test_resolve_path_env_file(self):
        result = _resolve_path(".env.production", "/tmp")
        assert result is None

    # Lines 378-380: _is_binary open exception
    def test_is_binary_open_exception(self, tmp_path, caplog):
        (tmp_path / "foo.txt").write_text("text")
        with patch("builtins.open", side_effect=OSError("cannot open")):
            result = _is_binary(tmp_path / "foo.txt")
            assert result is True
            assert "failed to check file" in caplog.text.lower()


def test_expand_reference_url_kind(tmp_path):
    """Line 199: _expand_reference dispatches url kind to _expand_url."""
    from mullet.core.agent.context_refs import _expand_reference, ContextReference
    from unittest.mock import patch, MagicMock

    ref = ContextReference(raw="@url:example.com", kind="url", target="example.com")
    fake_resp = MagicMock()
    fake_resp.read.return_value = b"hello world"
    fake_resp.__enter__ = lambda s: s
    fake_resp.__exit__ = MagicMock(return_value=False)

    with patch("mullet.core.tools.url_safety.safe_urlopen", return_value=fake_resp):
        content, error = _expand_reference(ref, str(tmp_path))
    assert error is None
    assert "example.com" in content


def test_expand_url_no_scheme_prepends_https(tmp_path):
    """Line 329: URL without http:// prefix gets https:// prepended."""
    from mullet.core.agent.context_refs import _expand_url, ContextReference
    from unittest.mock import patch, MagicMock

    ref = ContextReference(raw="@url:example.com", kind="url", target="example.com")
    fake_resp = MagicMock()
    fake_resp.read.return_value = b"content"
    fake_resp.__enter__ = lambda s: s
    fake_resp.__exit__ = MagicMock(return_value=False)

    captured_urls = []

    def fake_urlopen(req, timeout=None):
        captured_urls.append(req.full_url)
        return fake_resp

    with patch("mullet.core.tools.url_safety.safe_urlopen", side_effect=fake_urlopen):
        content, error = _expand_url(ref)
    assert error is None
    assert any("https://example.com" in u for u in captured_urls)


def test_resolve_path_blocked_filename():
    """Line 354: filename in _BLOCKED_FILES returns None."""
    from mullet.core.agent.context_refs import _resolve_path
    # id_rsa is in _BLOCKED_FILES
    result = _resolve_path("id_rsa", "/tmp")
    assert result is None


class TestExpandUrlBlocksNonPublic:
    def test_loopback_url_is_blocked_without_a_fetch(self):
        ref = ContextReference(raw="@url:http://127.0.0.1/x", kind="url", target="http://127.0.0.1/x")
        with patch("mullet.core.tools.url_safety.safe_urlopen") as opener:
            content, error = _expand_url(ref)
        assert content is None
        assert "blocked" in error.lower()
        opener.assert_not_called()

    def test_name_resolving_to_private_address_is_blocked(self, monkeypatch):
        monkeypatch.setattr("mullet.core.tools.url_safety._resolve", lambda host: {"10.0.0.5"})
        ref = ContextReference(raw="@url:evil.test", kind="url", target="evil.test")
        content, error = _expand_url(ref)
        assert content is None
        assert "blocked" in error.lower()
