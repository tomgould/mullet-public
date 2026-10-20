"""Tests for mcp_handlers._sanitize_paths — path traversal guard."""

from pathlib import Path


from mullet.server.mcp_handlers import _sanitize_paths


def test_clean_paths_inside_target(tmp_path):
    """Normal relative file paths inside target_dir → pass through."""
    target = tmp_path / "repo"
    target.mkdir()
    (target / "a.py").write_text("# a")
    (target / "sub").mkdir()
    (target / "sub" / "b.py").write_text("# b")

    safe_files, safe_target, err = _sanitize_paths(["a.py", "sub/b.py"], str(target))
    assert err is None
    assert sorted(safe_files) == ["a.py", "sub/b.py"]
    assert Path(safe_target) == target


def test_absolute_paths_inside_target_ok(tmp_path):
    """Absolute paths that resolve inside target_dir are accepted."""
    target = tmp_path / "repo"
    target.mkdir()
    file_path = target / "a.py"
    file_path.write_text("# a")

    # Absolute path as input — resolves to inside target
    safe_files, _target, err = _sanitize_paths([str(file_path)], str(target))
    # The sanitizer joins target+path; since absolute paths are handled by Path(),
    # the resolved path should land inside target
    assert err is None


def test_traversal_blocked(tmp_path):
    """../ escape is rejected."""
    target = tmp_path / "repo"
    target.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("# malicious")

    safe_files, safe_target, err = _sanitize_paths(["../outside.py"], str(target))
    assert err is not None
    assert "Path traversal detected" in err
    assert safe_files == []


def test_absolute_outside_target_blocked(tmp_path):
    """Absolute path outside target_dir is rejected."""
    target = tmp_path / "repo"
    target.mkdir()
    outside = tmp_path / "other" / "secret.py"
    outside.parent.mkdir()
    outside.write_text("secret")

    safe_files, _target, err = _sanitize_paths([str(outside)], str(target))
    assert err is not None
    assert "Path traversal detected" in err


def test_empty_files_list(tmp_path):
    """Empty files list returns empty safe_files, no error."""
    target = tmp_path / "repo"
    target.mkdir()
    safe_files, safe_target, err = _sanitize_paths([], str(target))
    assert err is None
    assert safe_files == []
    assert Path(safe_target) == target


def test_nonexistent_target_dir_still_sanitizes(tmp_path):
    """Non-existent target_dir: Path.resolve() returns a resolved string anyway."""
    target = str(tmp_path / "nope")
    safe_files, safe_target, err = _sanitize_paths(["a.py"], target)
    # Relative path 'a.py' resolves under target even if target doesn't exist
    # The function should either succeed or return a clear error — not crash
    assert isinstance(safe_files, list)
    assert isinstance(safe_target, str)


def test_symlink_traversal_blocked(tmp_path):
    """Symlink pointing outside target_dir is resolved and rejected."""
    target = tmp_path / "repo"
    target.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("secret")
    link = target / "evil.py"
    link.symlink_to(outside)

    safe_files, _target, err = _sanitize_paths(["evil.py"], str(target))
    # When resolved, the symlink points outside target → blocked
    assert err is not None
    assert "Path traversal detected" in err
