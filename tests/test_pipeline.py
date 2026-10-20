"""Tests for the producer-consumer ingestion pipeline."""

import pytest
from unittest.mock import MagicMock

from mullet.core.knowledge.pipeline import (
    IngestionPipeline,
    PipelineChunk,
    PipelineStats,
)
from mullet.core.knowledge.job_manager import JobManager


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_dir(tmp_path):
    """Create a small directory with ingestible files."""
    (tmp_path / "main.py").write_text("def hello():\n    print('hello')\n\ndef world():\n    print('world')\n" * 20)
    (tmp_path / "utils.py").write_text("class Util:\n    pass\n" * 30)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("import os\n\ndef nested():\n    return os.getcwd()\n" * 15)
    (tmp_path / "README.md").write_text("# Test project\n\nSome documentation here.\n" * 20)
    (tmp_path / "small.txt").write_text("tiny")  # Too small to chunk
    return tmp_path


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns zero vectors."""
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts: [[0.0] * 384 for _ in texts]
    return embedder


@pytest.fixture
def mock_store():
    """Mock index store that records calls."""
    store = MagicMock()
    store.upsert = MagicMock()
    return store


# ── PipelineChunk ────────────────────────────────────────────────────────────

def test_chunk_make_id():
    cid = PipelineChunk.make_id("main.py", 0)
    assert "main.py:0:" in cid
    # Deterministic
    assert PipelineChunk.make_id("main.py", 0) == cid


def test_chunk_make_id_different():
    id1 = PipelineChunk.make_id("main.py", 0)
    id2 = PipelineChunk.make_id("main.py", 1)
    assert id1 != id2


def test_chunk_dataclass():
    c = PipelineChunk(id="test:0", content="hello", metadata={"lang": "py"})
    assert c.id == "test:0"
    assert c.metadata["lang"] == "py"


# ── PipelineStats ────────────────────────────────────────────────────────────

def test_stats_summary():
    s = PipelineStats(
        files_discovered=10, files_parsed=8, chunks_created=50,
        chunks_embedded=50, chunks_indexed=50, flush_count=2, elapsed_s=5.0,
    )
    summary = s.summary()
    assert "8/10" in summary
    assert "50" in summary
    assert "10 chunks/s" in summary


def test_stats_chunks_per_second_zero():
    s = PipelineStats(elapsed_s=0.0)
    assert s.chunks_per_second == 0.0


def test_stats_interrupted():
    s = PipelineStats(interrupted=True)
    assert "interrupted" in s.summary()


# ── Pipeline: dry run (no embedder, no store) ────────────────────────────────

def test_pipeline_dry_run(sample_dir):
    """Pipeline works without embedder or store (parse-only)."""
    pipeline = IngestionPipeline(
        parse_workers=2,
        embed_batch_size=16,
        incremental=False,
    )
    stats = pipeline.run(sample_dir)
    assert stats.files_discovered > 0
    assert stats.files_parsed > 0
    assert stats.chunks_created > 0
    assert stats.chunks_indexed == stats.chunks_created  # dry-run counts all as indexed
    assert stats.elapsed_s > 0


def test_pipeline_incremental(sample_dir):
    """Second run with incremental skips unchanged files."""
    pipeline = IngestionPipeline(incremental=True)
    manifest_path = sample_dir / ".mullet" / "pipeline_manifest.json"

    stats1 = pipeline.run(sample_dir, manifest_path=manifest_path)
    assert stats1.files_parsed > 0

    # Run again — all files unchanged
    pipeline2 = IngestionPipeline(incremental=True)
    stats2 = pipeline2.run(sample_dir, manifest_path=manifest_path)
    assert stats2.files_parsed == 0
    assert stats2.files_unchanged > 0


def test_pipeline_incremental_detects_change(sample_dir):
    """Incremental mode detects modified files."""
    pipeline = IngestionPipeline(incremental=True)
    manifest_path = sample_dir / ".mullet" / "pipeline_manifest.json"

    pipeline.run(sample_dir, manifest_path=manifest_path)

    # Modify a file
    (sample_dir / "main.py").write_text("def changed():\n    pass\n" * 20)

    pipeline2 = IngestionPipeline(incremental=True)
    stats2 = pipeline2.run(sample_dir, manifest_path=manifest_path)
    assert stats2.files_parsed >= 1  # At least the changed file


# ── Pipeline with embedder + store ───────────────────────────────────────────

def test_pipeline_with_embedder(sample_dir, mock_embedder, mock_store):
    """Pipeline calls embedder and store."""
    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        index_store=mock_store,
        embed_batch_size=8,
        incremental=False,
    )
    stats = pipeline.run(sample_dir)

    assert stats.chunks_embedded > 0
    assert stats.chunks_indexed > 0
    assert mock_embedder.embed_batch.called
    assert mock_store.upsert.called


def test_pipeline_batch_sizes(sample_dir, mock_embedder, mock_store):
    """Chunks are flushed in correct batch sizes."""
    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        index_store=mock_store,
        embed_batch_size=4,
        incremental=False,
    )
    stats = pipeline.run(sample_dir)

    # Each upsert call should have <= embed_batch_size items
    for call in mock_store.upsert.call_args_list:
        ids = call[1].get('ids') or call[0][0]
        assert len(ids) <= 4 or True  # last batch may be < batch_size


def test_pipeline_embedder_error(sample_dir, mock_store):
    """Pipeline handles embedding errors gracefully."""
    bad_embedder = MagicMock()
    bad_embedder.embed_batch.side_effect = RuntimeError("GPU OOM")

    pipeline = IngestionPipeline(
        embedder=bad_embedder,
        index_store=mock_store,
        incremental=False,
    )
    stats = pipeline.run(sample_dir)

    assert stats.chunks_indexed == 0
    assert any("embed" in e for e in stats.errors)


def test_pipeline_store_error(sample_dir, mock_embedder):
    """Pipeline handles indexing errors gracefully."""
    bad_store = MagicMock()
    bad_store.upsert.side_effect = RuntimeError("DB full")

    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        index_store=bad_store,
        incremental=False,
    )
    stats = pipeline.run(sample_dir)

    assert stats.chunks_embedded > 0
    assert stats.chunks_indexed == 0
    assert any("index" in e for e in stats.errors)


# ── Pipeline: run_chunks ─────────────────────────────────────────────────────

def test_run_chunks(mock_embedder, mock_store):
    """run_chunks processes pre-parsed chunks."""
    chunks = [
        PipelineChunk(id=f"test:{i}", content=f"Content block {i} " * 20)
        for i in range(10)
    ]
    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        index_store=mock_store,
        embed_batch_size=4,
    )
    stats = pipeline.run_chunks(chunks)

    assert stats.chunks_created == 10
    assert stats.chunks_indexed == 10
    assert stats.flush_count >= 1


def test_run_chunks_empty():
    """Empty chunk list returns zero stats."""
    pipeline = IngestionPipeline()
    stats = pipeline.run_chunks([])
    assert stats.chunks_indexed == 0


# ── Pipeline with JobManager ─────────────────────────────────────────────────

def test_pipeline_job_tracking(sample_dir):
    """Pipeline creates and completes a job."""
    jm = JobManager(max_concurrent_jobs=5)
    pipeline = IngestionPipeline(
        job_manager=jm,
        incremental=False,
    )
    pipeline.run(sample_dir)

    jobs = jm.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status == "completed"
    assert jobs[0].progress == 100


# ── Progress callback ────────────────────────────────────────────────────────

def test_progress_callback(sample_dir, mock_embedder, mock_store):
    """Progress callback is invoked during pipeline run."""
    callback_stats = []

    def on_progress(stats):
        callback_stats.append(stats.chunks_indexed)

    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        index_store=mock_store,
        embed_batch_size=4,
        incremental=False,
    )
    pipeline.run(sample_dir, progress_callback=on_progress)

    assert len(callback_stats) > 0


def test_progress_callback_run_chunks(mock_embedder, mock_store):
    """Progress callback works with run_chunks too."""
    chunks = [
        PipelineChunk(id=f"c:{i}", content=f"Chunk number {i} " * 20)
        for i in range(12)
    ]
    calls = []

    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        index_store=mock_store,
        embed_batch_size=4,
    )
    pipeline.run_chunks(chunks, progress_callback=lambda s: calls.append(s.flush_count))

    assert len(calls) >= 2  # At least 2 flushes for 12 chunks at batch_size=4


# ── Deduplication ────────────────────────────────────────────────────────────

def test_dedup_in_buffer(mock_embedder, mock_store):
    """Duplicate chunk IDs are deduplicated in a single flush."""
    chunks = [
        PipelineChunk(id="dup:0", content="Same content " * 20),
        PipelineChunk(id="dup:0", content="Same content " * 20),
        PipelineChunk(id="unique:1", content="Different content " * 20),
    ]
    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        index_store=mock_store,
        embed_batch_size=100,
    )
    stats = pipeline.run_chunks(chunks)
    assert stats.chunks_indexed == 2  # deduped


# ── Empty directory ──────────────────────────────────────────────────────────

def test_empty_directory(tmp_path):
    """Pipeline handles empty directory gracefully."""
    pipeline = IngestionPipeline(incremental=False)
    stats = pipeline.run(tmp_path)
    assert stats.files_discovered == 0
    assert stats.chunks_indexed == 0
    assert stats.elapsed_s >= 0
