"""Tests for pool_metadata, project_detect, and llm_extraction pure helpers."""


from mullet.knowledge.engine.pool_metadata import (
    read_metadata,
    ensure_created,
    mark_searched,
    mark_indexed,
    _metadata_path,
    _now_iso,
)
from mullet.core.agent.project_detect import (
    ProjectInfo,
    detect_project,
    format_project_context,
    _file_contains,
)
from mullet.knowledge.engine.llm_extraction import (
    CitationMap,
    build_citation_map,
    parse_citations_from_response,
    _build_context_parts,
    build_rag_prompt,
    build_rag_prompt_with_citations,
    categorize_chunks,
)


# ---------------------------------------------------------------------------
# _now_iso
# ---------------------------------------------------------------------------

class TestNowIso:
    def test_returns_string(self):
        assert isinstance(_now_iso(), str)

    def test_contains_t_separator(self):
        result = _now_iso()
        assert "T" in result

    def test_contains_timezone_marker(self):
        result = _now_iso()
        assert "+" in result or "Z" in result or result.endswith("+00:00")


# ---------------------------------------------------------------------------
# _metadata_path
# ---------------------------------------------------------------------------

class TestMetadataPath:
    def test_returns_path_inside_dir(self, tmp_path):
        result = _metadata_path(tmp_path)
        assert result.parent == tmp_path

    def test_filename_is_json(self, tmp_path):
        result = _metadata_path(tmp_path)
        assert result.suffix == ".json"

    def test_filename_contains_metadata(self, tmp_path):
        result = _metadata_path(tmp_path)
        assert "metadata" in result.name


# ---------------------------------------------------------------------------
# read_metadata
# ---------------------------------------------------------------------------

class TestReadMetadata:
    def test_missing_dir_returns_empty(self, tmp_path):
        result = read_metadata(tmp_path / "nonexistent")
        assert result == {}

    def test_missing_file_returns_empty(self, tmp_path):
        result = read_metadata(tmp_path)
        assert result == {}

    def test_reads_existing_metadata(self, tmp_path):
        (tmp_path / ".pool_metadata.json").write_text('{"created_at": "2024-01-01T00:00:00+00:00"}')
        result = read_metadata(tmp_path)
        assert result["created_at"] == "2024-01-01T00:00:00+00:00"

    def test_malformed_json_returns_empty(self, tmp_path):
        (tmp_path / ".pool_metadata.json").write_text("{bad json}")
        result = read_metadata(tmp_path)
        assert result == {}

    def test_returns_dict(self, tmp_path):
        assert isinstance(read_metadata(tmp_path), dict)


# ---------------------------------------------------------------------------
# ensure_created
# ---------------------------------------------------------------------------

class TestEnsureCreated:
    def test_creates_created_at(self, tmp_path):
        ensure_created(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data

    def test_idempotent(self, tmp_path):
        ensure_created(tmp_path)
        first = read_metadata(tmp_path)["created_at"]
        ensure_created(tmp_path)
        second = read_metadata(tmp_path)["created_at"]
        assert first == second

    def test_created_at_is_string(self, tmp_path):
        ensure_created(tmp_path)
        data = read_metadata(tmp_path)
        assert isinstance(data["created_at"], str)


# ---------------------------------------------------------------------------
# mark_searched
# ---------------------------------------------------------------------------

class TestMarkSearched:
    def test_sets_last_searched_at(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert "last_searched_at" in data

    def test_increments_search_count(self, tmp_path):
        mark_searched(tmp_path)
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert data["search_count"] == 2

    def test_initial_search_count_one(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert data["search_count"] == 1

    def test_sets_created_at_if_missing(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data


# ---------------------------------------------------------------------------
# mark_indexed
# ---------------------------------------------------------------------------

class TestMarkIndexed:
    def test_sets_last_indexed_at(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert "last_indexed_at" in data

    def test_sets_created_at_if_missing(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data

    def test_last_indexed_at_is_string(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert isinstance(data["last_indexed_at"], str)


# ---------------------------------------------------------------------------
# ProjectInfo
# ---------------------------------------------------------------------------

class TestProjectInfo:
    def test_default_language_unknown(self):
        info = ProjectInfo()
        assert info.language == "unknown"

    def test_default_framework_empty(self):
        info = ProjectInfo()
        assert info.framework == ""

    def test_default_test_command_empty(self):
        info = ProjectInfo()
        assert info.test_command == ""

    def test_default_has_git_false(self):
        info = ProjectInfo()
        assert info.has_git is False

    def test_default_entry_points_empty(self):
        info = ProjectInfo()
        assert info.entry_points == []

    def test_explicit_values(self):
        info = ProjectInfo(language="python", framework="django", has_git=True)
        assert info.language == "python"
        assert info.framework == "django"
        assert info.has_git is True


# ---------------------------------------------------------------------------
# _file_contains
# ---------------------------------------------------------------------------

class TestFileContains:
    def test_true_when_content_present(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("fastapi>=0.100.0\nuvicorn")
        assert _file_contains(f, "fastapi") is True

    def test_false_when_content_absent(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests\nurllib3")
        assert _file_contains(f, "fastapi") is False

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "req.txt"
        f.write_text("Django==4.2")
        assert _file_contains(f, "django") is True

    def test_missing_file_returns_false(self, tmp_path):
        assert _file_contains(tmp_path / "missing.txt", "anything") is False


# ---------------------------------------------------------------------------
# detect_project
# ---------------------------------------------------------------------------

class TestDetectProject:
    def test_returns_project_info(self, tmp_path):
        result = detect_project(str(tmp_path))
        assert isinstance(result, ProjectInfo)

    def test_empty_dir_unknown_language(self, tmp_path):
        info = detect_project(str(tmp_path))
        assert info.language == "unknown"

    def test_python_via_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests\n")
        info = detect_project(str(tmp_path))
        assert info.language == "python"
        assert info.package_manager == "pip"

    def test_python_via_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'\n")
        info = detect_project(str(tmp_path))
        assert info.language == "python"

    def test_django_framework_detected(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("django\n")
        (tmp_path / "manage.py").write_text("#!/usr/bin/env python\n")
        info = detect_project(str(tmp_path))
        assert info.framework == "django"

    def test_fastapi_framework_detected(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi>=0.100\nuvicorn\n")
        info = detect_project(str(tmp_path))
        assert info.framework == "fastapi"

    def test_javascript_via_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "myapp", "scripts": {}}')
        info = detect_project(str(tmp_path))
        assert info.language == "javascript"
        assert info.package_manager == "npm"

    def test_typescript_via_tsconfig(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}')
        info = detect_project(str(tmp_path))
        assert info.language == "typescript"

    def test_yarn_detected(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        (tmp_path / "yarn.lock").write_text("# yarn lockfile v1\n")
        info = detect_project(str(tmp_path))
        assert info.package_manager == "yarn"

    def test_rust_via_cargo(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "myapp"\n')
        info = detect_project(str(tmp_path))
        assert info.language == "rust"
        assert info.package_manager == "cargo"

    def test_go_via_go_mod(self, tmp_path):
        (tmp_path / "go.mod").write_text("module myapp\n\ngo 1.21\n")
        info = detect_project(str(tmp_path))
        assert info.language == "go"

    def test_git_detected(self, tmp_path):
        (tmp_path / ".git").mkdir()
        info = detect_project(str(tmp_path))
        assert info.has_git is True

    def test_docker_via_dockerfile(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.10\n")
        info = detect_project(str(tmp_path))
        assert info.has_docker is True

    def test_ci_via_github_workflows(self, tmp_path):
        wf = tmp_path / ".github" / "workflows"
        wf.mkdir(parents=True)
        info = detect_project(str(tmp_path))
        assert info.has_ci is True

    def test_entry_point_detected(self, tmp_path):
        (tmp_path / "requirements.txt").touch()
        (tmp_path / "main.py").write_text("# entry\n")
        info = detect_project(str(tmp_path))
        assert "main.py" in info.entry_points

    def test_python_test_command(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("pytest\n")
        info = detect_project(str(tmp_path))
        assert "pytest" in info.test_command


# ---------------------------------------------------------------------------
# format_project_context
# ---------------------------------------------------------------------------

class TestFormatProjectContext:
    def test_unknown_language_returns_empty(self):
        info = ProjectInfo()
        assert format_project_context(info) == ""

    def test_known_language_returns_string(self):
        info = ProjectInfo(language="python")
        result = format_project_context(info)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_language(self):
        info = ProjectInfo(language="rust")
        result = format_project_context(info)
        assert "rust" in result

    def test_contains_framework_when_set(self):
        info = ProjectInfo(language="python", framework="django")
        result = format_project_context(info)
        assert "django" in result

    def test_omits_framework_when_empty(self):
        info = ProjectInfo(language="python")
        result = format_project_context(info)
        assert "Framework:" not in result

    def test_contains_test_command(self):
        info = ProjectInfo(language="python", test_command="pytest -x")
        result = format_project_context(info)
        assert "pytest -x" in result

    def test_contains_docker_when_has_docker(self):
        info = ProjectInfo(language="python", has_docker=True)
        result = format_project_context(info)
        assert "Docker" in result

    def test_contains_ci_when_has_ci(self):
        info = ProjectInfo(language="python", has_ci=True)
        result = format_project_context(info)
        assert "CI" in result


# ---------------------------------------------------------------------------
# CitationMap
# ---------------------------------------------------------------------------

class TestCitationMap:
    def test_get_returns_chunk_by_1_index(self):
        chunks = [{"content": "first"}, {"content": "second"}]
        cm = CitationMap(chunks=chunks)
        assert cm.get(1) == {"content": "first"}
        assert cm.get(2) == {"content": "second"}

    def test_get_returns_none_for_out_of_range(self):
        cm = CitationMap(chunks=[{"content": "x"}])
        assert cm.get(0) is None
        assert cm.get(2) is None

    def test_len_returns_chunk_count(self):
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        cm = CitationMap(chunks=chunks)
        assert len(cm) == 3

    def test_empty_map_len_zero(self):
        cm = CitationMap()
        assert len(cm) == 0


# ---------------------------------------------------------------------------
# build_citation_map
# ---------------------------------------------------------------------------

class TestBuildCitationMap:
    def test_returns_citation_map(self):
        chunks = [{"content": "data"}]
        result = build_citation_map(chunks)
        assert isinstance(result, CitationMap)

    def test_chunks_preserved(self):
        chunks = [{"content": "a"}, {"content": "b"}]
        cm = build_citation_map(chunks)
        assert len(cm) == 2


# ---------------------------------------------------------------------------
# parse_citations_from_response
# ---------------------------------------------------------------------------

class TestParseCitationsFromResponse:
    def test_single_citation(self):
        result = parse_citations_from_response("See [1] for details.")
        assert result == [1]

    def test_multiple_citations(self):
        result = parse_citations_from_response("From [1] and [3] we see...")
        assert result == [1, 3]

    def test_no_citations(self):
        result = parse_citations_from_response("No references here.")
        assert result == []

    def test_returns_list_of_ints(self):
        result = parse_citations_from_response("[2] is the source")
        assert all(isinstance(x, int) for x in result)

    def test_duplicate_citations_included(self):
        result = parse_citations_from_response("[1] and [1] again")
        assert result == [1, 1]


# ---------------------------------------------------------------------------
# _build_context_parts
# ---------------------------------------------------------------------------

class TestBuildContextParts:
    def _make_chunk(self, content, source="file.py"):
        return {"content": content, "metadata": {"source_file": source}}

    def test_returns_parts_list(self):
        chunks = [self._make_chunk("hello world")]
        parts, _, _ = _build_context_parts(chunks, max_context_chars=10000)
        assert isinstance(parts, list)

    def test_includes_content(self):
        chunks = [self._make_chunk("important content here")]
        parts, _, _ = _build_context_parts(chunks, max_context_chars=10000)
        assert any("important content" in p for p in parts)

    def test_truncates_at_budget(self):
        chunks = [self._make_chunk("x" * 100) for _ in range(10)]
        parts, total_chars, _ = _build_context_parts(chunks, max_context_chars=200)
        assert total_chars <= 200

    def test_citations_mode_uses_bracket_format(self):
        chunks = [self._make_chunk("data")]
        parts, _, _ = _build_context_parts(chunks, max_context_chars=10000, citations=True)
        assert any(p.startswith("[1]") for p in parts)

    def test_non_citations_mode_uses_source_format(self):
        chunks = [self._make_chunk("data")]
        parts, _, _ = _build_context_parts(chunks, max_context_chars=10000, citations=False)
        assert any("[Source 1:" in p for p in parts)

    def test_start_index_respected(self):
        chunks = [self._make_chunk("data")]
        parts, _, _ = _build_context_parts(chunks, max_context_chars=10000, start_index=5, citations=True)
        assert any(p.startswith("[5]") for p in parts)

    def test_empty_chunks_returns_empty(self):
        parts, total, _ = _build_context_parts([], max_context_chars=10000)
        assert parts == []
        assert total == 0


# ---------------------------------------------------------------------------
# build_rag_prompt
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:
    def _make_chunk(self, content):
        return {"content": content, "metadata": {"source_file": "doc.txt"}}

    def test_returns_string(self):
        result = build_rag_prompt("What is Python?", [self._make_chunk("Python is...")])
        assert isinstance(result, str)

    def test_contains_question(self):
        result = build_rag_prompt("What is Python?", [self._make_chunk("...")])
        assert "What is Python?" in result

    def test_contains_chunk_content(self):
        result = build_rag_prompt("q", [self._make_chunk("some important data")])
        assert "some important data" in result

    def test_empty_chunks_no_crash(self):
        result = build_rag_prompt("question", [])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_rag_prompt_with_citations
# ---------------------------------------------------------------------------

class TestBuildRagPromptWithCitations:
    def _make_chunk(self, content):
        return {"content": content, "metadata": {"source_file": "doc.txt"}}

    def test_returns_string(self):
        result = build_rag_prompt_with_citations("Q?", [self._make_chunk("data")])
        assert isinstance(result, str)

    def test_contains_question(self):
        result = build_rag_prompt_with_citations("What is X?", [self._make_chunk("X is...")])
        assert "What is X?" in result

    def test_contains_citation_instruction(self):
        result = build_rag_prompt_with_citations("q", [self._make_chunk("content")])
        assert "cite" in result.lower() or "[1]" in result or "citations" in result.lower()


# ---------------------------------------------------------------------------
# categorize_chunks
# ---------------------------------------------------------------------------

class TestCategorizeChunks:
    def _make_chunk(self, chunk_type):
        return {"content": "text", "metadata": {"chunk_type": chunk_type}}

    def test_returns_dict(self):
        result = categorize_chunks([])
        assert isinstance(result, dict)

    def test_has_code_key(self):
        result = categorize_chunks([])
        assert "code" in result

    def test_has_documentation_key(self):
        result = categorize_chunks([])
        assert "documentation" in result

    def test_code_chunks_grouped(self):
        chunks = [self._make_chunk("code"), self._make_chunk("code")]
        result = categorize_chunks(chunks)
        assert len(result["code"]) == 2

    def test_unknown_type_goes_to_other(self):
        chunks = [self._make_chunk("unknown_type")]
        result = categorize_chunks(chunks)
        assert len(result["other"]) == 1

    def test_mixed_chunks_sorted(self):
        chunks = [
            self._make_chunk("code"),
            self._make_chunk("documentation"),
            self._make_chunk("conversation"),
            self._make_chunk("xyz_unknown"),
        ]
        result = categorize_chunks(chunks)
        assert len(result["code"]) == 1
        assert len(result["documentation"]) == 1
        assert len(result["conversation"]) == 1
        assert len(result["other"]) == 1
