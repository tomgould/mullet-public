import os
from pathlib import Path

import pytest

from mullet.harness.sessions.project_hash import project_hash


@pytest.mark.harness
class TestProjectHash:
    def test_returns_12_hex_chars(self, tmp_path: Path) -> None:
        result = project_hash(tmp_path)
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self, tmp_path: Path) -> None:
        h1 = project_hash(tmp_path)
        h2 = project_hash(tmp_path)
        assert h1 == h2

    def test_distinguishes_paths(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert project_hash(a) != project_hash(b)

    def test_resolves_symlink(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        assert project_hash(real) == project_hash(link)

    def test_default_cwd(self) -> None:
        result = project_hash()
        assert result == project_hash(Path.cwd())

    def test_handles_nonexistent_path(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        result = project_hash(nonexistent)
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    @pytest.mark.skipif(os.getenv("HOME") is None, reason="HOME not set")
    def test_expands_tilde(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        foo = tmp_path / "foo"
        foo.mkdir()
        assert project_hash("~/foo") == project_hash(foo)
