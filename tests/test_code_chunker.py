"""Tests for mullet.knowledge.code.chunker.CodeChunker configuration methods."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mullet.knowledge.code.chunker import CodeChunker, _process_file_worker


class TestGetExtensionsForLanguages:
    def test_no_config_returns_all_supported(self) -> None:
        chunker = CodeChunker()
        exts = chunker.get_extensions_for_languages()
        assert isinstance(exts, set)
        assert ".py" in exts

    def test_explicit_language_config(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "python": {"extensions": [".py", ".pyw"]},
                "js": {"extensions": [".js", ".mjs"]},
            }
        })
        exts = chunker.get_extensions_for_languages()
        assert ".py" in exts
        assert ".pyw" in exts
        assert ".js" in exts
        assert ".mjs" in exts

    def test_extensions_without_dot_get_dot_prefix(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "custom": {"extensions": ["mycustom"]}
            }
        })
        exts = chunker.get_extensions_for_languages()
        assert ".mycustom" in exts
        assert "mycustom" not in exts

    def test_extensions_lowercased(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "lang": {"extensions": [".PY", ".JS"]}
            }
        })
        exts = chunker.get_extensions_for_languages()
        assert ".py" in exts
        assert ".js" in exts

    def test_non_dict_lang_config_skipped(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "python": "invalid_config"  # Not a dict
            }
        })
        # Should not raise; falls back to all supported
        exts = chunker.get_extensions_for_languages()
        assert ".py" in exts

    def test_lang_config_without_extensions_uses_all_supported(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "python": {"enabled": True}  # No extensions key
            }
        })
        exts = chunker.get_extensions_for_languages()
        # Falls back to all supported since no explicit extensions
        assert ".py" in exts


class TestGetExcludePatterns:
    def test_empty_config_returns_empty_list(self) -> None:
        chunker = CodeChunker()
        patterns = chunker.get_exclude_patterns()
        assert patterns == []

    def test_exclude_patterns_from_languages(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "python": {
                    "extensions": [".py"],
                    "exclude": ["**/migrations/**", "**/__pycache__/**"],
                }
            }
        })
        patterns = chunker.get_exclude_patterns()
        assert "**/migrations/**" in patterns
        assert "**/__pycache__/**" in patterns

    def test_exclude_patterns_merged_across_languages(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "python": {"exclude": ["**/*.pyc"]},
                "js": {"exclude": ["**/node_modules/**"]},
            }
        })
        patterns = chunker.get_exclude_patterns()
        assert "**/*.pyc" in patterns
        assert "**/node_modules/**" in patterns

    def test_language_without_exclude_ignored(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "python": {"extensions": [".py"]}
            }
        })
        patterns = chunker.get_exclude_patterns()
        assert patterns == []

    def test_non_dict_lang_config_skipped(self) -> None:
        chunker = CodeChunker(config={
            "languages": {
                "python": "not a dict"
            }
        })
        patterns = chunker.get_exclude_patterns()
        assert patterns == []


class TestCodeChunkerInit:
    def test_default_init(self) -> None:
        chunker = CodeChunker()
        assert chunker.config == {}

    def test_config_stored(self) -> None:
        cfg = {"parallel_workers": 4}
        chunker = CodeChunker(config=cfg)
        assert chunker.config["parallel_workers"] == 4

    def test_parser_created(self) -> None:
        chunker = CodeChunker()
        assert chunker.parser is not None


class TestChunkFilesSequential:
    def test_empty_file_list_returns_empty(self) -> None:
        chunker = CodeChunker()
        result = chunker.chunk_files([], parallel=False)
        assert result == []

    def test_chunks_python_file(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("def hello():\n    return 'hi'\n")
        chunker = CodeChunker()
        result = chunker.chunk_files([f], parallel=False)
        assert len(result) > 0
        # Chunks are dicts with at least content
        assert isinstance(result[0], dict)
        assert "content" in result[0]

    def test_unsupported_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "file.unknownxyz"
        f.write_text("some content")
        chunker = CodeChunker()
        result = chunker.chunk_files([f], parallel=False)
        assert result == []


class TestProcessFileWorker:
    def test_success_returns_chunk_dicts(self):
        mock_chunk = MagicMock()
        mock_chunk.to_dict.return_value = {"content": "def hello(): pass"}
        mock_parser = MagicMock()
        mock_parser.parse_file.return_value = [mock_chunk]
        with patch("mullet.knowledge.code.parser.CodeParser", return_value=mock_parser):
            result = _process_file_worker(("test.py", None, {}))
        assert result == [{"content": "def hello(): pass"}]

    def test_success_with_repo_info(self):
        mock_chunk = MagicMock()
        mock_chunk.to_dict.return_value = {"content": "x"}
        mock_parser = MagicMock()
        mock_parser.parse_file.return_value = [mock_chunk]
        with patch("mullet.knowledge.code.parser.CodeParser", return_value=mock_parser):
            result = _process_file_worker(("test.py", {
                "repo_root": "/repo",
                "repo_name": "myrepo",
                "remote_url": "https://git",
                "current_branch": "main",
            }, {}))
        assert result == [{"content": "x"}]

    def test_exception_returns_empty_list(self):
        with patch("mullet.knowledge.code.parser.CodeParser", side_effect=RuntimeError("boom")):
            result = _process_file_worker(("test.py", None, {}))
        assert result == []


class TestCollectFiles:
    def test_extends_top_level_excludes(self, tmp_path):
        chunker = CodeChunker(config={"exclude_patterns": ["*.log", "*.tmp"]})
        with patch("mullet.knowledge.code.chunker.walk_code_files") as mock_walk:
            mock_walk.return_value = iter([tmp_path / "a.py"])
            result = chunker.collect_files([tmp_path])
            assert result == [tmp_path / "a.py"]
            call_args, _ = mock_walk.call_args
            exclude_patterns = call_args[2]
            assert "*.log" in exclude_patterns
            assert "*.tmp" in exclude_patterns


class TestDetectRepositories:
    def test_detect_repos_false_returns_empty(self):
        chunker = CodeChunker(config={"git": {"detect_repos": False}})
        result = chunker.detect_repositories([Path("/fake")])
        assert result == []

    def test_detect_repos_true_calls_detect_repos(self, tmp_path):
        chunker = CodeChunker(config={"git": {"detect_repos": True}})
        with patch("mullet.knowledge.code.chunker.detect_repos") as mock_detect:
            mock_detect.return_value = []
            result = chunker.detect_repositories([tmp_path])
            assert result == []
            mock_detect.assert_called_once_with(tmp_path)


class TestChunkFilesParallelTrigger:
    def test_triggers_chunk_parallel(self):
        chunker = CodeChunker(config={"parallel_workers": 4})
        files = [Path(f"/tmp/f{i}.py") for i in range(11)]
        with patch.object(chunker, "_chunk_parallel", return_value=[]) as mock_parallel:
            chunker.chunk_files(files, parallel=True)
            mock_parallel.assert_called_once()

    def test_does_not_trigger_parallel_when_small(self):
        chunker = CodeChunker(config={"parallel_workers": 4})
        files = [Path(f"/tmp/f{i}.py") for i in range(10)]
        with patch.object(chunker, "_chunk_sequential", return_value=[]) as mock_seq:
            chunker.chunk_files(files, parallel=True)
            mock_seq.assert_called_once()


class TestChunkParallel:
    def test_process_pool_success(self):
        chunker = CodeChunker()
        files = [Path("/tmp/f.py")]

        mock_future = MagicMock()
        mock_future.result.return_value = [{"content": "chunk"}]

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        with patch("mullet.knowledge.code.chunker.ProcessPoolExecutor", return_value=mock_executor):
            with patch("mullet.knowledge.code.chunker.as_completed", return_value=[mock_future]):
                result = chunker._chunk_parallel(files, [], 2)
        assert result == [{"content": "chunk"}]

    def test_process_pool_future_exception(self):
        chunker = CodeChunker()
        files = [Path("/tmp/f.py")]

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("boom")

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        with patch("mullet.knowledge.code.chunker.ProcessPoolExecutor", return_value=mock_executor):
            with patch("mullet.knowledge.code.chunker.as_completed", return_value=[mock_future]):
                result = chunker._chunk_parallel(files, [], 2)
        assert result == []

    def test_fallback_to_threads_on_process_pool_failure(self):
        chunker = CodeChunker()
        files = [Path("/tmp/f.py")]
        with patch("mullet.knowledge.code.chunker.ProcessPoolExecutor", side_effect=RuntimeError("fail")):
            with patch.object(chunker, "_chunk_parallel_threads", return_value=[{"content": "fb"}]) as mock_threads:
                result = chunker._chunk_parallel(files, [], 2)
        assert result == [{"content": "fb"}]
        mock_threads.assert_called_once_with(files, [], 2)


class TestChunkParallelThreads:
    def test_thread_pool_success(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def hello():\n    return 'hi'\n")
        chunker = CodeChunker()
        result = chunker._chunk_parallel_threads([f], [], 2)
        assert len(result) > 0
        assert isinstance(result[0], dict)
        assert "content" in result[0]

    def test_thread_pool_future_exception(self):
        chunker = CodeChunker()
        files = [Path("/tmp/f.py")]

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("boom")

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        with patch("mullet.knowledge.code.chunker.ThreadPoolExecutor", return_value=mock_executor):
            with patch("mullet.knowledge.code.chunker.as_completed", return_value=[mock_future]):
                result = chunker._chunk_parallel_threads(files, [], 2)
        assert result == []


class TestChunkDirectoryAndPaths:
    def test_chunk_directory_delegates_to_chunk_paths(self, tmp_path):
        chunker = CodeChunker()
        with patch.object(chunker, "chunk_paths", return_value=[{"content": "x"}]) as mock_paths:
            result = chunker.chunk_directory(tmp_path)
            assert result == [{"content": "x"}]
            mock_paths.assert_called_once_with([tmp_path])

    def test_chunk_paths_with_directory_and_file(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        f = tmp_path / "standalone.py"
        f.write_text("def x(): pass\n")

        chunker = CodeChunker()
        with patch.object(chunker, "detect_repositories", return_value=[]) as mock_detect:
            with patch.object(chunker, "collect_files", return_value=[]) as mock_collect:
                with patch.object(chunker, "chunk_files", return_value=[]) as mock_chunk:
                    chunker.chunk_paths([d, f])
                    mock_detect.assert_called_once_with([d])
                    mock_collect.assert_called_once_with([d], repos=[])
                    mock_chunk.assert_called_once_with([f], [])

    def test_chunk_paths_real_python_file(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        f = d / "mod.py"
        f.write_text("def hello():\n    return 'hi'\n")
        chunker = CodeChunker()
        result = chunker.chunk_paths([d])
        assert len(result) > 0
        assert isinstance(result[0], dict)
        assert "content" in result[0]


class TestProcessFileWorkerMore:
    def test_worker_logs_warning_on_exception(self, caplog):
        with patch("mullet.knowledge.code.parser.CodeParser", side_effect=RuntimeError("parse fail")):
            result = _process_file_worker(("test.py", None, {}))
        assert result == []
        assert "Worker failed on" in caplog.text

    def test_worker_with_partial_repo_info(self):
        mock_chunk = MagicMock()
        mock_chunk.to_dict.return_value = {"content": "x"}
        mock_parser = MagicMock()
        mock_parser.parse_file.return_value = [mock_chunk]
        with patch("mullet.knowledge.code.parser.CodeParser", return_value=mock_parser):
            result = _process_file_worker(("test.py", {
                "repo_root": "/repo",
                "repo_name": "myrepo",
            }, {}))
        assert result == [{"content": "x"}]

    def test_worker_converts_path_str_to_path_object(self):
        mock_chunk = MagicMock()
        mock_chunk.to_dict.return_value = {"content": "x"}
        mock_parser = MagicMock()
        mock_parser.parse_file.return_value = [mock_chunk]
        with patch("mullet.knowledge.code.parser.CodeParser", return_value=mock_parser):
            result = _process_file_worker(("some/file.py", None, {}))
        assert result == [{"content": "x"}]
        call_args, _ = mock_parser.parse_file.call_args
        assert isinstance(call_args[0], Path)
        assert call_args[0] == Path("some/file.py")


class TestCollectFilesAdditional:
    def test_warns_on_nonexistent_path(self):
        chunker = CodeChunker()
        with patch("mullet.knowledge.code.chunker.logger") as mock_logger:
            result = chunker.collect_files([Path("/nonexistent_path_xyz")])
            assert result == []
            mock_logger.warning.assert_called_once()

    def test_empty_top_level_excludes_does_not_extend(self, tmp_path):
        chunker = CodeChunker(config={"exclude_patterns": []})
        with patch("mullet.knowledge.code.chunker.walk_code_files") as mock_walk:
            mock_walk.return_value = iter([tmp_path / "a.py"])
            result = chunker.collect_files([tmp_path])
            call_args, _ = mock_walk.call_args
            exclude_patterns = call_args[2]
            assert exclude_patterns == []


class TestDetectReposAdditional:
    def test_detect_repos_true_skips_missing_paths(self):
        chunker = CodeChunker(config={"git": {"detect_repos": True}})
        with patch("mullet.knowledge.code.chunker.detect_repos") as mock_detect:
            result = chunker.detect_repositories([Path("/nonexistent")])
            assert result == []
            mock_detect.assert_not_called()


class TestChunkFilesAdditional:
    def test_parallel_false_uses_sequential(self):
        chunker = CodeChunker(config={"parallel_workers": 4})
        files = [Path(f"/tmp/f{i}.py") for i in range(11)]
        with patch.object(chunker, "_chunk_sequential", return_value=[]) as mock_seq:
            chunker.chunk_files(files, parallel=False)
            mock_seq.assert_called_once()


class TestChunkSequential:
    def test_empty_list(self):
        chunker = CodeChunker()
        result = chunker._chunk_sequential([], [])
        assert result == []

    def test_real_python_file(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def hello():\n    return 'hi'\n")
        chunker = CodeChunker()
        result = chunker._chunk_sequential([f], [])
        assert len(result) > 0
        assert isinstance(result[0], dict)
        assert "content" in result[0]

    def test_with_repos_calls_find_repo_for_file(self):
        chunker = CodeChunker()
        mock_chunk = MagicMock()
        mock_chunk.to_dict.return_value = {"content": "x"}
        chunker.parser = MagicMock()
        chunker.parser.parse_file.return_value = [mock_chunk]
        mock_repo = MagicMock()
        with patch("mullet.knowledge.code.git_utils.find_repo_for_file", return_value=mock_repo) as mock_find:
            result = chunker._chunk_sequential([Path("/tmp/f.py")], [mock_repo])
            mock_find.assert_called_once_with(Path("/tmp/f.py"), [mock_repo])
            chunker.parser.parse_file.assert_called_once_with(Path("/tmp/f.py"), mock_repo)
            assert result == [{"content": "x"}]


class TestChunkParallelAdditional:
    def test_builds_repo_dict_for_repos(self):
        chunker = CodeChunker()
        files = [Path("/tmp/f.py")]
        mock_repo = MagicMock()
        mock_repo.repo_root = Path("/repo")
        mock_repo.repo_name = "myrepo"
        mock_repo.remote_url = "https://git"
        mock_repo.current_branch = "main"

        mock_future = MagicMock()
        mock_future.result.return_value = [{"content": "chunk"}]

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        with patch("mullet.knowledge.code.git_utils.find_repo_for_file", return_value=mock_repo):
            with patch("mullet.knowledge.code.chunker.ProcessPoolExecutor", return_value=mock_executor):
                with patch("mullet.knowledge.code.chunker.as_completed", return_value=[mock_future]):
                    result = chunker._chunk_parallel(files, [mock_repo], 2)
        assert result == [{"content": "chunk"}]
        call_args, _ = mock_executor.submit.call_args
        work_item = call_args[1]
        assert work_item[1] == {
            "repo_root": str(mock_repo.repo_root),
            "repo_name": "myrepo",
            "remote_url": "https://git",
            "current_branch": "main",
        }

    def test_multiple_files_all_submitted(self):
        chunker = CodeChunker()
        files = [Path("/tmp/a.py"), Path("/tmp/b.py")]

        mock_future = MagicMock()
        mock_future.result.return_value = [{"content": "c"}]

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        with patch("mullet.knowledge.code.chunker.ProcessPoolExecutor", return_value=mock_executor):
            with patch("mullet.knowledge.code.chunker.as_completed", return_value=[mock_future, mock_future]):
                result = chunker._chunk_parallel(files, [], 2)
        assert result == [{"content": "c"}, {"content": "c"}]
        assert mock_executor.submit.call_count == 2

    def test_process_pool_outer_exception_fallback(self):
        chunker = CodeChunker()
        files = [Path("/tmp/f.py")]
        with patch("mullet.knowledge.code.chunker.ProcessPoolExecutor", side_effect=OSError("pool fail")):
            with patch.object(chunker, "_chunk_parallel_threads", return_value=[{"content": "fb"}]) as mock_threads:
                result = chunker._chunk_parallel(files, [], 2)
        assert result == [{"content": "fb"}]
        mock_threads.assert_called_once_with(files, [], 2)


class TestChunkParallelThreadsAdditional:
    def test_with_repos_calls_find_repo(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def hello():\n    return 'hi'\n")
        chunker = CodeChunker()
        mock_repo = MagicMock()
        with patch("mullet.knowledge.code.git_utils.find_repo_for_file", return_value=mock_repo) as mock_find:
            result = chunker._chunk_parallel_threads([f], [mock_repo], 2)
            mock_find.assert_called_once_with(f, [mock_repo])
        assert len(result) > 0
        assert isinstance(result[0], dict)

    def test_multiple_real_files(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("def a(): pass\n")
        f2 = tmp_path / "b.py"
        f2.write_text("def b(): pass\n")
        chunker = CodeChunker()
        result = chunker._chunk_parallel_threads([f1, f2], [], 2)
        assert len(result) >= 2

    def test_future_exception_continues(self):
        chunker = CodeChunker()
        files = [Path("/tmp/f.py")]

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("boom")

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        with patch("mullet.knowledge.code.chunker.ThreadPoolExecutor", return_value=mock_executor):
            with patch("mullet.knowledge.code.chunker.as_completed", return_value=[mock_future]):
                result = chunker._chunk_parallel_threads(files, [], 2)
        assert result == []


class TestChunkDirectoryAdditional:
    def test_real_directory(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        f = d / "mod.py"
        f.write_text("def hello(): pass\n")
        chunker = CodeChunker()
        result = chunker.chunk_directory(d)
        assert len(result) > 0
        assert isinstance(result[0], dict)


class TestChunkPathsAdditional:
    def test_only_files_no_directories(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("def a(): pass\n")
        f2 = tmp_path / "b.py"
        f2.write_text("def b(): pass\n")
        chunker = CodeChunker()
        result = chunker.chunk_paths([f1, f2])
        assert len(result) >= 2

    def test_empty_directory_returns_empty(self, tmp_path):
        d = tmp_path / "empty_src"
        d.mkdir()
        chunker = CodeChunker()
        result = chunker.chunk_paths([d])
        assert result == []

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        d = tmp_path / "missing"
        chunker = CodeChunker()
        result = chunker.chunk_paths([d])
        assert result == []


class TestDetectReposLogsDebug:
    """Cover line 168: logger.debug called for each detected repo."""

    def test_detect_repos_logs_debug_per_repo(self, tmp_path):
        """When repos are returned, logger.debug is called for each one (line 168)."""
        mock_repo = MagicMock()
        mock_repo.repo_name = "testrepo"
        mock_repo.repo_root = tmp_path

        chunker = CodeChunker(config={"git": {"detect_repos": True}})
        with patch("mullet.knowledge.code.chunker.detect_repos", return_value=[mock_repo]):
            with patch("mullet.knowledge.code.chunker.logger") as mock_logger:
                result = chunker.detect_repositories([tmp_path])

        assert result == [mock_repo]
        # logger.debug should be called once for the single repo (line 168)
        mock_logger.debug.assert_called_once()
        debug_call_args = mock_logger.debug.call_args[0][0]
        assert "testrepo" in debug_call_args

    def test_detect_repos_no_debug_when_empty(self, tmp_path):
        """When no repos are found, logger.debug is never called (loop body skipped)."""
        chunker = CodeChunker(config={"git": {"detect_repos": True}})
        with patch("mullet.knowledge.code.chunker.detect_repos", return_value=[]):
            with patch("mullet.knowledge.code.chunker.logger") as mock_logger:
                result = chunker.detect_repositories([tmp_path])

        assert result == []
        mock_logger.debug.assert_not_called()


class TestChunkParallelProgressLog:
    """Cover line 263: progress logger.info fires when done % 5000 == 0."""

    def test_progress_logged_at_5000_files(self):
        """Submit exactly 5000 futures so done reaches 5000 and triggers line 263."""
        chunker = CodeChunker()
        n = 5000
        files = [Path(f"/tmp/f{i}.py") for i in range(n)]

        # Each future returns an empty list so we accumulate no real chunks
        mock_future = MagicMock()
        mock_future.result.return_value = []

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        futures_list = [mock_future] * n

        with patch("mullet.knowledge.code.chunker.ProcessPoolExecutor", return_value=mock_executor):
            with patch("mullet.knowledge.code.chunker.as_completed", return_value=iter(futures_list)):
                with patch("mullet.knowledge.code.chunker.logger") as mock_logger:
                    result = chunker._chunk_parallel(files, [], 2)

        assert result == []
        # The progress line should have been called at least once (done==5000)
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("5000" in c for c in info_calls), f"Expected progress log at 5000; calls={info_calls}"


class TestChunkParallelThreadsProgressLog:
    """Cover line 304: progress logger.info fires when done % 5000 == 0."""

    def test_progress_logged_at_5000_files(self):
        """Submit exactly 5000 futures so done reaches 5000 and triggers line 304."""
        chunker = CodeChunker()
        n = 5000
        files = [Path(f"/tmp/f{i}.py") for i in range(n)]

        mock_future = MagicMock()
        mock_future.result.return_value = []

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None

        futures_list = [mock_future] * n

        with patch("mullet.knowledge.code.chunker.ThreadPoolExecutor", return_value=mock_executor):
            with patch("mullet.knowledge.code.chunker.as_completed", return_value=iter(futures_list)):
                with patch("mullet.knowledge.code.chunker.logger") as mock_logger:
                    result = chunker._chunk_parallel_threads(files, [], 2)

        assert result == []
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("5000" in c for c in info_calls), f"Expected progress log at 5000; calls={info_calls}"
