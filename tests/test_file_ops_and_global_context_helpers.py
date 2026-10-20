"""Tests for file_ops and global_context pure helpers."""

import json
from pathlib import Path


from mullet.core.tools.file_ops import (
    batch_rename,
    batch_copy,
    batch_move,
    FILE_OPS_SCHEMA,
    file_ops_handler,
)
from mullet.server.global_context import GlobalSource, GlobalContextManager


# ---------------------------------------------------------------------------
# batch_rename
# ---------------------------------------------------------------------------

class TestBatchRename:
    def test_renames_matching_file(self, tmp_path):
        (tmp_path / "foo_test.py").write_text("content")
        result = batch_rename(str(tmp_path), "foo_", "bar_")
        assert len(result) == 1
        assert (tmp_path / "bar_test.py").exists()

    def test_returns_old_and_new_paths(self, tmp_path):
        (tmp_path / "old_file.txt").write_text("x")
        result = batch_rename(str(tmp_path), "old_", "new_")
        old_path, new_path = result[0]
        assert "old_file.txt" in old_path
        assert "new_file.txt" in new_path

    def test_no_match_returns_empty(self, tmp_path):
        (tmp_path / "no_match.txt").write_text("x")
        result = batch_rename(str(tmp_path), "xyz_", "abc_")
        assert result == []

    def test_multiple_files_renamed(self, tmp_path):
        (tmp_path / "test_a.py").write_text("a")
        (tmp_path / "test_b.py").write_text("b")
        result = batch_rename(str(tmp_path), "test_", "spec_")
        assert len(result) == 2

    def test_non_matching_files_untouched(self, tmp_path):
        (tmp_path / "keep.txt").write_text("x")
        (tmp_path / "rename_me.txt").write_text("y")
        batch_rename(str(tmp_path), "rename_", "done_")
        assert (tmp_path / "keep.txt").exists()

    def test_returns_list_of_tuples(self, tmp_path):
        (tmp_path / "pfx_file.txt").write_text("x")
        result = batch_rename(str(tmp_path), "pfx_", "new_")
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) for r in result)


# ---------------------------------------------------------------------------
# batch_copy
# ---------------------------------------------------------------------------

class TestBatchCopy:
    def test_copies_matching_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        dest = tmp_path / "dest"
        result = batch_copy(str(src / "*.txt"), str(dest))
        assert len(result) == 1
        assert (dest / "a.txt").exists()

    def test_original_files_preserved(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        dest = tmp_path / "dest"
        batch_copy(str(src / "*.txt"), str(dest))
        assert (src / "a.txt").exists()

    def test_dest_dir_created(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "x.txt").write_text("x")
        dest = tmp_path / "new_dest"
        batch_copy(str(src / "*.txt"), str(dest))
        assert dest.exists()

    def test_returns_list_of_tuples(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        dest = tmp_path / "dest"
        result = batch_copy(str(src / "*.txt"), str(dest))
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) for r in result)

    def test_no_match_returns_empty(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        result = batch_copy(str(src / "*.xyz"), str(dest))
        assert result == []

    def test_multiple_files_copied(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("a")
        (src / "b.py").write_text("b")
        dest = tmp_path / "dest"
        result = batch_copy(str(src / "*.py"), str(dest))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# batch_move
# ---------------------------------------------------------------------------

class TestBatchMove:
    def test_moves_matching_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        dest = tmp_path / "dest"
        result = batch_move(str(src / "*.txt"), str(dest))
        assert len(result) == 1
        assert (dest / "a.txt").exists()

    def test_original_files_removed_after_move(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        dest = tmp_path / "dest"
        batch_move(str(src / "*.txt"), str(dest))
        assert not (src / "a.txt").exists()

    def test_dest_dir_created(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "x.txt").write_text("x")
        dest = tmp_path / "new_dest"
        batch_move(str(src / "*.txt"), str(dest))
        assert dest.exists()

    def test_returns_list_of_tuples(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        dest = tmp_path / "dest"
        result = batch_move(str(src / "*.txt"), str(dest))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# FILE_OPS_SCHEMA
# ---------------------------------------------------------------------------

class TestFileOpsSchema:
    def test_is_dict(self):
        assert isinstance(FILE_OPS_SCHEMA, dict)

    def test_has_name(self):
        assert "name" in FILE_OPS_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in FILE_OPS_SCHEMA

    def test_action_parameter_exists(self):
        props = FILE_OPS_SCHEMA["parameters"]["properties"]
        assert "action" in props

    def test_path_parameter_exists(self):
        props = FILE_OPS_SCHEMA["parameters"]["properties"]
        assert "path" in props


# ---------------------------------------------------------------------------
# file_ops_handler
# ---------------------------------------------------------------------------

class TestFileOpsHandlerMissingPath:
    def test_missing_path_returns_error(self):
        result = json.loads(file_ops_handler(action="read_file", path=""))
        assert "error" in result

    def test_unknown_action_returns_error(self):
        result = json.loads(file_ops_handler(action="unknown_action", path="/tmp"))
        assert "error" in result


class TestFileOpsHandlerReadFile:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line 1\nline 2\nline 3\n")
        result = json.loads(file_ops_handler(action="read_file", path=str(f)))
        assert "lines" in result
        assert "line 1" in result["lines"]

    def test_read_nonexistent_file(self, tmp_path):
        result = json.loads(file_ops_handler(action="read_file", path=str(tmp_path / "nope.txt")))
        assert "error" in result

    def test_read_returns_total_lines(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        result = json.loads(file_ops_handler(action="read_file", path=str(f)))
        assert result["total_lines"] == 5

    def test_read_respects_limit(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("\n".join(str(i) for i in range(100)))
        result = json.loads(file_ops_handler(action="read_file", path=str(f), limit=5))
        assert len(result["lines"]) == 5

    def test_read_respects_offset(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("first\nsecond\nthird\n")
        result = json.loads(file_ops_handler(action="read_file", path=str(f), offset=1, limit=1))
        assert result["lines"] == ["second"]


class TestFileOpsHandlerWriteFile:
    def test_write_creates_file(self, tmp_path):
        path = str(tmp_path / "new.txt")
        result = json.loads(file_ops_handler(action="write_file", path=path, content="hello"))
        assert "bytes_written" in result
        assert Path(path).read_text() == "hello"

    def test_write_returns_bytes_written(self, tmp_path):
        path = str(tmp_path / "out.txt")
        result = json.loads(file_ops_handler(action="write_file", path=path, content="abc"))
        assert result["bytes_written"] == 3

    def test_write_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "file.txt")
        file_ops_handler(action="write_file", path=path, content="data")
        assert Path(path).exists()


class TestFileOpsHandlerPatch:
    def test_patch_replaces_text(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def old_name():\n    pass\n")
        file_ops_handler(action="patch", path=str(f), old_text="old_name", new_text="new_name")
        assert "new_name" in f.read_text()

    def test_patch_old_text_not_found_error(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    pass\n")
        result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="missing", new_text="xyz"))
        assert "error" in result

    def test_patch_nonexistent_file_error(self, tmp_path):
        result = json.loads(file_ops_handler(action="patch", path=str(tmp_path / "nope.py"), old_text="x", new_text="y"))
        assert "error" in result

    def test_patch_empty_old_text_error(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("content")
        result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="", new_text="x"))
        assert "error" in result

    def test_patch_returns_status_patched(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello world")
        result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="hello", new_text="bye"))
        assert result.get("status") == "patched"


class TestFileOpsHandlerListDirectory:
    def test_list_existing_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        result = json.loads(file_ops_handler(action="list_directory", path=str(tmp_path)))
        assert "entries" in result

    def test_list_nonexistent_directory(self, tmp_path):
        result = json.loads(file_ops_handler(action="list_directory", path=str(tmp_path / "nope")))
        assert "error" in result

    def test_list_count_correct(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = json.loads(file_ops_handler(action="list_directory", path=str(tmp_path)))
        assert result["count"] == 2

    def test_entries_have_type_field(self, tmp_path):
        (tmp_path / "x.txt").write_text("x")
        result = json.loads(file_ops_handler(action="list_directory", path=str(tmp_path)))
        for entry in result["entries"]:
            assert "type" in entry


class TestFileOpsHandlerSearchFiles:
    def test_search_finds_files(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = json.loads(file_ops_handler(action="search_files", path=str(tmp_path), pattern="*.py"))
        assert result["count"] == 1

    def test_search_returns_matches(self, tmp_path):
        (tmp_path / "foo.py").write_text("x")
        result = json.loads(file_ops_handler(action="search_files", path=str(tmp_path), pattern="*.py"))
        assert any("foo.py" in m for m in result["matches"])

    def test_search_no_match_returns_zero(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        result = json.loads(file_ops_handler(action="search_files", path=str(tmp_path), pattern="*.xyz"))
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# GlobalSource
# ---------------------------------------------------------------------------

class TestGlobalSource:
    def test_fields_stored(self):
        s = GlobalSource(id="gs_1", type="github_org", config={"org": "acme"})
        assert s.id == "gs_1"
        assert s.type == "github_org"
        assert s.config["org"] == "acme"

    def test_default_status_pending(self):
        s = GlobalSource(id="gs_1", type="web_crawl")
        assert s.status == "pending"

    def test_default_last_sync_zero(self):
        s = GlobalSource(id="gs_1", type="confluence")
        assert s.last_sync == 0

    def test_default_schedule(self):
        s = GlobalSource(id="gs_1", type="jira")
        assert "2" in s.schedule  # default is 2am


# ---------------------------------------------------------------------------
# GlobalContextManager
# ---------------------------------------------------------------------------

class TestGlobalContextManagerEmpty:
    def test_list_sources_empty(self):
        gcm = GlobalContextManager()
        assert gcm.list_sources() == []

    def test_build_global_context_empty(self):
        gcm = GlobalContextManager()
        assert gcm.build_global_context("query") == ""


class TestGlobalContextManagerAddSource:
    def test_add_source_returns_global_source(self):
        gcm = GlobalContextManager()
        source = gcm.add_source("github_org", {"org": "acme"})
        assert isinstance(source, GlobalSource)

    def test_add_source_has_id(self):
        gcm = GlobalContextManager()
        source = gcm.add_source("github_org", {})
        assert source.id.startswith("gs_")

    def test_add_source_type_stored(self):
        gcm = GlobalContextManager()
        source = gcm.add_source("web_crawl", {"url": "https://example.com"})
        assert source.type == "web_crawl"

    def test_list_sources_after_add(self):
        gcm = GlobalContextManager()
        gcm.add_source("confluence", {})
        assert len(gcm.list_sources()) == 1

    def test_list_sources_has_id_key(self):
        gcm = GlobalContextManager()
        gcm.add_source("jira", {})
        entry = gcm.list_sources()[0]
        assert "id" in entry

    def test_list_sources_has_type_key(self):
        gcm = GlobalContextManager()
        gcm.add_source("jira", {})
        entry = gcm.list_sources()[0]
        assert "type" in entry

    def test_multiple_sources_listed(self):
        gcm = GlobalContextManager()
        gcm.add_source("github_org", {})
        gcm.add_source("web_crawl", {})
        assert len(gcm.list_sources()) == 2


class TestGlobalContextManagerRemoveSource:
    def test_remove_source_removes_it(self):
        gcm = GlobalContextManager()
        s = gcm.add_source("github_org", {})
        gcm.remove_source(s.id)
        assert gcm.list_sources() == []

    def test_remove_unknown_id_no_error(self):
        gcm = GlobalContextManager()
        gcm.remove_source("nonexistent")  # should not raise


class TestGlobalContextManagerBuildContext:
    def test_build_context_returns_string(self):
        gcm = GlobalContextManager()
        result = gcm.build_global_context("any query")
        assert isinstance(result, str)

    def test_build_context_with_ready_source(self):
        gcm = GlobalContextManager()
        s = gcm.add_source("github_org", {"org": "acme"})
        s.status = "ready"
        result = gcm.build_global_context("query")
        assert len(result) > 0

    def test_build_context_pending_source_excluded(self):
        gcm = GlobalContextManager()
        gcm.add_source("web_crawl", {"url": "https://example.com"})
        # status is 'pending' by default
        result = gcm.build_global_context("query")
        assert result == ""

    def test_build_context_contains_global_context_header(self):
        gcm = GlobalContextManager()
        s = gcm.add_source("confluence", {"org": "acme"})
        s.status = "ready"
        result = gcm.build_global_context("query")
        assert "Global Context" in result
