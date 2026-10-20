"""Regression tests for ingest_code rebuild=True deletion behavior.

Bug (2026-05-08): when rebuild=True and pipeline_searcher was provided, the
Chroma collection was never cleared. upsert with same IDs was a no-op, so
chunk counts stayed identical after --rebuild-code. Fixed by adding explicit
collection.delete in the rebuild branch.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_pipeline_searcher(existing_ids: list[str]) -> MagicMock:
    searcher = MagicMock()
    searcher.collection.count.return_value = len(existing_ids)
    searcher.collection.get.return_value = {"ids": existing_ids}
    return searcher


def _make_typed_pipeline_searcher(
    code_ids: list[str], doc_ids: list[str]
) -> MagicMock:
    """Mock whose collection.get honors the where={"chunk_type": "code"} filter.

    Returns code_ids when the rebuild path filters by chunk_type='code',
    returns code_ids+doc_ids otherwise. Lets a test assert that the delete
    was scoped — not a bulk wipe of every chunk.
    """
    searcher = MagicMock()
    searcher.collection.count.return_value = len(code_ids) + len(doc_ids)

    def _get(*args, **kwargs):
        where = kwargs.get("where") or {}
        if where == {"chunk_type": "code"}:
            return {"ids": list(code_ids)}
        return {"ids": list(code_ids) + list(doc_ids)}

    searcher.collection.get.side_effect = _get
    return searcher


def _patch_code_chunker(tmp_path: Path):
    """Context manager that patches CodeChunker to avoid real tree-sitter calls."""
    mock_chunker_cls = MagicMock()
    mock_chunker = MagicMock()
    mock_chunker.detect_repositories.return_value = {}
    mock_chunker.collect_files.return_value = []
    mock_chunker.chunk_files.return_value = []
    mock_chunker_cls.return_value = mock_chunker

    mock_state_cls = MagicMock()
    mock_state = MagicMock()
    mock_state_cls.return_value = mock_state

    return (
        patch("mullet.knowledge.code.CodeChunker", mock_chunker_cls),
        patch("mullet.knowledge.code.CodeIndexState", mock_state_cls),
        patch("mullet.knowledge.code.compute_chunk_id", side_effect=lambda c: f"id-{c.get('name', 'x')}"),
        patch("mullet.knowledge.engine.ingest.DocumentIngestor", MagicMock),
    )


def test_rebuild_clears_existing_chunks_from_collection(tmp_path):
    """rebuild=True must delete all existing Chroma IDs before re-indexing."""
    from mullet.knowledge.engine.ingest import ingest_code

    src = tmp_path / "src"
    src.mkdir()
    pipeline_searcher = _make_pipeline_searcher(["chunk-1", "chunk-2", "chunk-3"])
    code_config = {"extensions": [".py"], "incremental": False}

    patches = _patch_code_chunker(tmp_path)
    with patches[0], patches[1], patches[2], patches[3]:
        try:
            ingest_code(
                code_paths=[src],
                code_config=code_config,
                pipeline_searcher=pipeline_searcher,
                rebuild=True,
            )
        except Exception:
            pass  # OK — focus is on whether delete was called

    # The fix: collection.delete must have been called with all existing IDs
    pipeline_searcher.collection.delete.assert_called()
    all_deleted_ids = []
    for c in pipeline_searcher.collection.delete.call_args_list:
        ids = c[1].get("ids") or (c[0][0] if c[0] else [])
        all_deleted_ids.extend(ids)
    assert set(all_deleted_ids) == {"chunk-1", "chunk-2", "chunk-3"}, (
        f"Expected all 3 IDs deleted, got: {all_deleted_ids!r}"
    )


def test_rebuild_empty_collection_skips_delete(tmp_path):
    """rebuild=True with empty collection must not call collection.delete."""
    from mullet.knowledge.engine.ingest import ingest_code

    src = tmp_path / "src"
    src.mkdir()
    pipeline_searcher = _make_pipeline_searcher([])
    code_config = {"extensions": [".py"], "incremental": False}

    patches = _patch_code_chunker(tmp_path)
    with patches[0], patches[1], patches[2], patches[3]:
        try:
            ingest_code(
                code_paths=[src],
                code_config=code_config,
                pipeline_searcher=pipeline_searcher,
                rebuild=True,
            )
        except Exception:
            pass

    pipeline_searcher.collection.delete.assert_not_called()


def test_rebuild_preserves_doc_chunks(tmp_path):
    """rebuild=True must scope the delete to chunk_type='code'.

    Regression for the 2026-05-18 bug: when ``rebuild_index`` called
    ``ingest_client_documents`` followed by ``ingest_client_code_files(rebuild=True)``,
    the second step wiped *all* chunks in the collection — including the
    document chunks the first step had just added. The fix is to filter the
    delete to ``where={"chunk_type": "code"}``. This test fails (delete is
    called with doc IDs) if that filter is removed.
    """
    from mullet.knowledge.engine.ingest import ingest_code

    src = tmp_path / "src"
    src.mkdir()
    pipeline_searcher = _make_typed_pipeline_searcher(
        code_ids=["code-1", "code-2"],
        doc_ids=["doc-1", "doc-2"],
    )
    code_config = {"extensions": [".py"], "incremental": False}

    patches = _patch_code_chunker(tmp_path)
    with patches[0], patches[1], patches[2], patches[3]:
        try:
            ingest_code(
                code_paths=[src],
                code_config=code_config,
                pipeline_searcher=pipeline_searcher,
                rebuild=True,
            )
        except Exception:
            pass  # OK — focus is on what got deleted

    # collection.get must have been called with the chunk_type filter
    where_calls = [
        c for c in pipeline_searcher.collection.get.call_args_list
        if (c.kwargs.get("where") or {}) == {"chunk_type": "code"}
    ]
    assert where_calls, (
        "rebuild path must call collection.get(where={'chunk_type': 'code'}) "
        "so it only deletes code chunks. Without this filter, ingest_code "
        "wipes doc chunks ingest_client_documents just added."
    )

    # And the delete must NOT have been called with any doc IDs.
    all_deleted_ids = []
    for c in pipeline_searcher.collection.delete.call_args_list:
        ids = c.kwargs.get("ids") or (c.args[0] if c.args else [])
        all_deleted_ids.extend(ids)
    deleted = set(all_deleted_ids)
    assert "doc-1" not in deleted and "doc-2" not in deleted, (
        f"doc chunks must survive rebuild=True. Deleted IDs: {deleted!r}"
    )
    assert {"code-1", "code-2"}.issubset(deleted), (
        f"code chunks must be deleted on rebuild=True. Deleted IDs: {deleted!r}"
    )


def test_incremental_mode_skips_bulk_delete(tmp_path):
    """Incremental mode (rebuild=False) must not call collection.get to list all IDs."""
    from mullet.knowledge.engine.ingest import ingest_code

    src = tmp_path / "src"
    src.mkdir()
    pipeline_searcher = _make_pipeline_searcher(["chunk-old"])
    code_config = {"extensions": [".py"], "incremental": True}
    index_path = tmp_path / ".index"
    index_path.mkdir()

    patches = _patch_code_chunker(tmp_path)
    with patches[0], patches[1], patches[2], patches[3]:
        try:
            ingest_code(
                code_paths=[src],
                code_config=code_config,
                pipeline_searcher=pipeline_searcher,
                index_path=index_path,
                rebuild=False,
            )
        except Exception:
            pass

    # The rebuild-specific path calls collection.get to enumerate ALL IDs.
    # In incremental mode this must NOT happen.
    pipeline_searcher.collection.get.assert_not_called()
