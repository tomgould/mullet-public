from mullet.core.tools.patch import parse_unified_diff, apply_hunk, PatchParser, Hunk

def test_parse_unified_diff_single_hunk():
    patch_text = """--- a/file1.txt
+++ b/file1.txt
@@ -1,3 +1,4 @@
 line 1
-line 2
+new line 2
 line 3
"""
    parsed = parse_unified_diff(patch_text)
    assert len(parsed) == 1
    old_file, new_file, hunks = parsed[0]
    assert old_file == "file1.txt"
    assert new_file == "file1.txt"
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.old_start == 1
    assert hunk.old_count == 3
    assert hunk.new_start == 1
    assert hunk.new_count == 4
    assert hunk.lines == [' line 1', '-line 2', '+new line 2', ' line 3']

def test_parse_unified_diff_multiple_hunks():
    patch_text = """--- a/file1.txt
+++ b/file1.txt
@@ -1,2 +1,3 @@
 line 1
+new line 2
 line 2
@@ -4,2 +5,2 @@
 line 4
 line 5
"""
    parsed = parse_unified_diff(patch_text)
    assert len(parsed) == 1
    _, _, hunks = parsed[0]
    assert len(hunks) == 2

def test_parse_unified_diff_new_file():
    patch_text = """--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1,2 @@
+new line 1
+new line 2
"""
    parsed = parse_unified_diff(patch_text)
    assert len(parsed) == 1
    old_file, new_file, hunks = parsed[0]
    assert old_file == "/dev/null"
    assert new_file == "new_file.txt"
    assert len(hunks) == 1

def test_parse_unified_diff_deleted_file():
    patch_text = """--- a/old_file.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-line 1
-line 2
"""
    parsed = parse_unified_diff(patch_text)
    assert len(parsed) == 1
    old_file, new_file, hunks = parsed[0]
    assert old_file == "old_file.txt"
    assert new_file == "/dev/null"
    assert len(hunks) == 1

def test_parse_unified_diff_multiple_files():
    patch_text = """--- a/file1.txt
+++ b/file1.txt
@@ -1 +1 @@
-a
+b
--- a/file2.txt
+++ b/file2.txt
@@ -1 +1 @@
-c
+d
"""
    parsed = parse_unified_diff(patch_text)
    assert len(parsed) == 2
    assert parsed[0][0] == "file1.txt"
    assert parsed[1][0] == "file2.txt"

def test_apply_hunk_exact_match():
    file_lines = ["line 1\n", "line 2\n", "line 3\n"]
    hunk = Hunk(old_start=2, old_count=1, new_start=2, new_count=1, lines=['-line 2', '+new line 2'])
    result = apply_hunk(file_lines, hunk)
    assert result.applied is True
    assert "".join(file_lines) == "line 1\nnew line 2\nline 3\n"

def test_apply_hunk_fuzzy_match(tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("line 1\noriginal line 2\nline 3\n")
    
    parser = PatchParser(offset_tolerance=1)
    
    # Simulate a small change around "original line 2"
    patch_text = """--- a/test_file.txt
+++ b/test_file.txt
@@ -1,3 +1,3 @@
 line 1
-original line 2
+modified line 2
 line 3
"""
    result = parser.apply_patch(patch_text, tmp_path)
    assert result.success is True
    assert (tmp_path / "test_file.txt").read_text() == "line 1\nmodified line 2\nline 3\n"

def test_apply_hunk_rejected():
    file_lines = ["line 1\n", "wrong line\n", "line 3\n"]
    hunk = Hunk(old_start=2, old_count=1, new_start=2, new_count=1, lines=['-line 2', '+new line 2'])
    result = apply_hunk(file_lines, hunk)
    assert result.applied is False
    assert result.reason == "Context mismatch — no matching location found"

def test_patchparser_apply_new_file(tmp_path):
    parser = PatchParser()
    patch_text = """--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1,2 @@
+hello
+world
"""
    result = parser.apply_patch(patch_text, tmp_path)
    assert result.success is True
    assert "new_file.txt" in result.files_created
    assert (tmp_path / "new_file.txt").read_text() == "hello\nworld\n"

def test_patchparser_apply_modified_file(tmp_path):
    file_path = tmp_path / "existing_file.txt"
    file_path.write_text("line 1\nline 2\nline 3\n")
    parser = PatchParser()
    patch_text = """--- a/existing_file.txt
+++ b/existing_file.txt
@@ -1,3 +1,3 @@
 line 1
-line 2
+modified line 2
 line 3
"""
    result = parser.apply_patch(patch_text, tmp_path)
    assert result.success is True
    assert "existing_file.txt" in result.files_modified
    assert (tmp_path / "existing_file.txt").read_text() == "line 1\nmodified line 2\nline 3\n"

def test_patchparser_apply_deleted_file(tmp_path):
    file_path = tmp_path / "delete_me.txt"
    file_path.write_text("line 1\nline 2\n")
    parser = PatchParser()
    patch_text = """--- a/delete_me.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-line 1
-line 2
"""
    result = parser.apply_patch(patch_text, tmp_path)
    assert result.success is True
    assert "delete_me.txt" in result.files_deleted
    assert not file_path.exists()

def test_patchparser_empty_patch(tmp_path):
    parser = PatchParser()
    patch_text = ""
    result = parser.apply_patch(patch_text, tmp_path)
    assert result.success is False
    assert "No valid diffs found in patch text" in result.errors

def test_patchparser_rejected_hunks(tmp_path):
    file_path = tmp_path / "file_to_patch.txt"
    file_path.write_text("line 1\nline 2\nline 3\n")
    parser = PatchParser()
    patch_text = """--- a/file_to_patch.txt
+++ b/file_to_patch.txt
@@ -1,3 +1,3 @@
 line 1
-WRONG LINE 2
+new line 2
 line 3
"""
    result = parser.apply_patch(patch_text, tmp_path)
    assert result.success is False
    assert result.hunks_rejected == 1
    assert "Context mismatch" in result.errors[0]

def test_patchparser_multiple_files_in_one_patch(tmp_path):
    file1_path = tmp_path / "file1.txt"
    file1_path.write_text("A\nB\nC\n")
    file2_path = tmp_path / "file2.txt"
    file2_path.write_text("X\nY\nZ\n")

    parser = PatchParser()
    patch_text = """--- a/file1.txt
+++ b/file1.txt
@@ -1,3 +1,3 @@
 A
-B
+BB
 C
--- a/file2.txt
+++ b/file2.txt
@@ -1,3 +1,3 @@
 X
-Y
+YY
 Z
"""
    result = parser.apply_patch(patch_text, tmp_path)
    assert result.success is True
    assert "file1.txt" in result.files_modified
    assert "file2.txt" in result.files_modified
    assert (tmp_path / "file1.txt").read_text() == "A\nBB\nC\n"
    assert (tmp_path / "file2.txt").read_text() == "X\nYY\nZ\n"

def test_patchparser_fuzzy_matching_with_offset(tmp_path):
    """Test that a hunk with a wrong line number still matches via offset tolerance."""
    file_path = tmp_path / "fuzzy.txt"
    # File has target at line 4, but hunk header will say line 2
    file_path.write_text("extra1\nextra2\nline A\nline B\nline C\n")

    parser = PatchParser(offset_tolerance=3)
    # Hunk says @@ -2,3 but the real content starts at line 3
    patch_text = """--- a/fuzzy.txt
+++ b/fuzzy.txt
@@ -2,3 +2,3 @@
 line A
-line B
+line REPLACED
 line C
"""
    result = parser.apply_patch(patch_text, tmp_path)
    assert result.success is True
    assert "line REPLACED" in (tmp_path / "fuzzy.txt").read_text()

class TestPatchMissingLines:
    def test_parse_skip_lines_before_header(self):
        patch_text = """some random header
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old
+new
"""
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1

    def test_parse_no_header_eof(self):
        patch_text = "just a random line\nanother random line"
        parsed = parse_unified_diff(patch_text)
        assert parsed == []

    def test_parse_no_new_file_after_old(self):
        patch_text = """--- a/file.txt
random line without plus header
--- a/file2.txt
+++ b/file2.txt
@@ -1 +1 @@
-old
+new
"""
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1
        assert parsed[0][0] == "file2.txt"

    def test_parse_malformed_hunk_line(self):
        patch_text = """--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
 line 1
-bad
malformed line
 line 2
"""
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1
        _, _, hunks = parsed[0]
        assert len(hunks) == 1
        assert len(hunks[0].lines) == 2

    def test_patchparser_path_traversal_new_file(self, tmp_path):
        parser = PatchParser()
        patch_text = """--- /dev/null
+++ /etc/passwd
@@ -0,0 +1 @@
+evil
"""
        result = parser.apply_patch(patch_text, tmp_path)
        assert result.success is False
        assert any("Path traversal blocked" in e for e in result.errors)

    def test_patchparser_path_traversal_delete(self, tmp_path):
        parser = PatchParser()
        patch_text = """--- /etc/passwd
+++ /dev/null
@@ -1 +0,0 @@
-evil
"""
        result = parser.apply_patch(patch_text, tmp_path)
        assert result.success is False
        assert any("Path traversal blocked" in e for e in result.errors)

    def test_patchparser_path_traversal_modify(self, tmp_path):
        parser = PatchParser()
        patch_text = """--- /etc/passwd
+++ /etc/passwd
@@ -1 +1 @@
-old
+new
"""
        result = parser.apply_patch(patch_text, tmp_path)
        assert result.success is False
        assert any("Path traversal blocked" in e for e in result.errors)

    def test_patchparser_file_not_found(self, tmp_path):
        parser = PatchParser()
        patch_text = """--- a/nonexistent.txt
+++ b/nonexistent.txt
@@ -1 +1 @@
-old
+new
"""
        result = parser.apply_patch(patch_text, tmp_path)
        assert result.success is False
        assert any("File not found" in e for e in result.errors)

    def test_patch_handler_empty(self):
        from mullet.core.tools.patch import patch_handler
        import json
        result = patch_handler(patch_text="")
        data = json.loads(result)
        assert "error" in data

    def test_patch_handler_success(self, tmp_path):
        from mullet.core.tools.patch import patch_handler
        import json
        file_path = tmp_path / "file.txt"
        file_path.write_text("old\n")
        patch_text = """--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old
+new
"""
        result = patch_handler(patch_text=patch_text, base_dir=str(tmp_path))
        data = json.loads(result)
        assert data["success"] is True
        assert "file.txt" in data["files_modified"]

    def test_parse_no_newline_at_end_of_file(self):
        patch_text = "--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n line 1\n line 2\n\\ No newline at end of file\n"
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1
        _, _, hunks = parsed[0]
        assert len(hunks) == 1
        assert len(hunks[0].lines) == 2

    def test_register_function(self):
        class MockRegistry:
            def __init__(self):
                self.registered_tools = {}
            def register(self, name, func, schema, toolset="default"):
                self.registered_tools[name] = {"schema": schema, "func": func, "toolset": toolset}
        from mullet.core.tools.patch import register, PATCH_SCHEMA, patch_handler
        mock_registry = MockRegistry()
        register(mock_registry)
        assert "patch" in mock_registry.registered_tools
        assert mock_registry.registered_tools["patch"]["schema"] == PATCH_SCHEMA
        assert mock_registry.registered_tools["patch"]["func"] == patch_handler
        assert mock_registry.registered_tools["patch"]["toolset"] == "file"
