"""Cross-pool search normalisation tests (P2-5).

Covers:
- Source-file dedup across pools (collapse duplicates from same file indexed
  in two pools, keep the highest-RRF occurrence).
- `_merged_from` audit trail on the kept chunk.
- Configurable `rrf_k` (default 60, 1-200 bounds).
- Per-pool `pool_weights` overrides (multiply RRF contribution, default 1.0).
- Chunks without `metadata.source_file` must NOT collapse.
"""

from __future__ import annotations

import types


from mullet.server import mcp_knowledge


def _make_chunk(chunk_id: str, source_file: str | None, text: str = "body") -> dict:
    """Return a plain-dict chunk resembling what HybridSearcher emits."""
    metadata = {}
    if source_file is not None:
        metadata["source_file"] = source_file
    return {
        "id": chunk_id,
        "text": text,
        "content": text,
        "score": 1.0,
        "_rerank_score": 1.0,
        "metadata": metadata,
    }


def _install_pools(monkeypatch, pool_results: dict[str, list[dict]]) -> None:
    """Wire up `_load_pool` so each pool returns the supplied result list."""
    def fake_load(name: str):
        results = pool_results.get(name, [])
        searcher = types.SimpleNamespace(
            search_with_quality=lambda **kw: types.SimpleNamespace(
                results=[dict(r) for r in results],
                quality=types.SimpleNamespace(
                    top_score=1.0,
                    avg_rerank_score=0.9,
                    timing={"total": 0.01},
                ),
            ),
        )
        client = types.SimpleNamespace(name=name)
        return client, searcher

    monkeypatch.setattr(mcp_knowledge, "_load_pool", fake_load)


def _parse_results(output: str) -> list[str]:
    """Extract the 'Source:' lines from formatted output, in order."""
    sources = []
    for line in output.splitlines():
        if line.startswith("**Source:**"):
            sources.append(line.split("**Source:**", 1)[1].strip())
    return sources


def test_dedup_identical_source_file_across_pools(monkeypatch):
    """Same source_file in two pools MUST appear exactly once.

    Fails under the current impl because chunk_id is namespaced as
    `f"{pool}:{chunk_id}"`, so two pools returning the same file produce
    two separate keys and both survive the merge.
    """
    pool_a = [_make_chunk("c1", "foo.py", "text-a")]
    pool_b = [_make_chunk("c2", "foo.py", "text-b")]
    _install_pools(monkeypatch, {"pool_a": pool_a, "pool_b": pool_b})

    out = mcp_knowledge.handle_search_multi({
        "query": "q",
        "pool_names": ["pool_a", "pool_b"],
    })

    sources = _parse_results(out)
    assert sources.count("foo.py") == 1, (
        f"Expected foo.py once after dedup, got {sources}"
    )


def test_dedup_preserves_merged_from(monkeypatch):
    """Surviving chunk MUST carry `_merged_from` with both pool names.

    Fails under the current impl because no `_merged_from` field is set
    anywhere in `handle_search_multi`.
    """
    captured_chunks = {}

    real_format = mcp_knowledge._format_results

    def capturing_format(results, *a, **kw):
        captured_chunks["results"] = list(results)
        return real_format(results, *a, **kw)

    monkeypatch.setattr(mcp_knowledge, "_format_results", capturing_format)

    pool_a = [_make_chunk("c1", "bar.py")]
    pool_b = [_make_chunk("c2", "bar.py")]
    _install_pools(monkeypatch, {"pool_a": pool_a, "pool_b": pool_b})

    mcp_knowledge.handle_search_multi({
        "query": "q",
        "pool_names": ["pool_a", "pool_b"],
    })

    results = captured_chunks.get("results", [])
    bar_chunks = [r for r in results if r.get("metadata", {}).get("source_file") == "bar.py"]
    assert len(bar_chunks) == 1, f"Expected exactly one bar.py chunk, got {len(bar_chunks)}"
    merged_from = bar_chunks[0].get("metadata", {}).get("_merged_from")
    assert merged_from is not None, "Kept chunk missing metadata._merged_from"
    assert set(merged_from) == {"pool_a", "pool_b"}, (
        f"_merged_from should list both pools, got {merged_from}"
    )


def test_dedup_keeps_highest_rrf_occurrence(monkeypatch):
    """When the same file appears at different ranks in two pools, the
    higher-ranked (rank 0) occurrence's text MUST survive.

    Fails under the current impl because no collapse happens; both
    appear, and the ordering / identity check against a single canonical
    chunk has no anchor.
    """
    captured_chunks = {}

    real_format = mcp_knowledge._format_results

    def capturing_format(results, *a, **kw):
        captured_chunks["results"] = list(results)
        return real_format(results, *a, **kw)

    monkeypatch.setattr(mcp_knowledge, "_format_results", capturing_format)

    # Pool A: target file at rank 0 (highest RRF contribution).
    pool_a = [
        _make_chunk("c1", "baz.py", "TEXT-FROM-A"),
        _make_chunk("c99", "other1.py", "t"),
    ]
    # Pool B: target file at rank 3 (rank 0..3 means 4th position).
    pool_b = [
        _make_chunk("b0", "other2.py", "t"),
        _make_chunk("b1", "other3.py", "t"),
        _make_chunk("b2", "other4.py", "t"),
        _make_chunk("c2", "baz.py", "TEXT-FROM-B"),
    ]
    _install_pools(monkeypatch, {"pool_a": pool_a, "pool_b": pool_b})

    mcp_knowledge.handle_search_multi({
        "query": "q",
        "pool_names": ["pool_a", "pool_b"],
    })

    results = captured_chunks.get("results", [])
    baz_chunks = [r for r in results if r.get("metadata", {}).get("source_file") == "baz.py"]
    assert len(baz_chunks) == 1, f"Expected baz.py collapsed to 1 entry, got {len(baz_chunks)}"
    assert baz_chunks[0].get("text") == "TEXT-FROM-A", (
        f"Higher-ranked (pool_a rank 0) text should survive, got {baz_chunks[0].get('text')!r}"
    )


def test_rrf_k_parameter_honoured(monkeypatch):
    """`rrf_k` MUST alter merge ordering. The same chunk set under
    k=60 vs k=1 should produce different top results because k shifts
    the weight of low-ranked occurrences.

    With dedup by source_file applied (required by this feature):
      k=60: A=1/61=0.01639, B=2/66=0.03030 → B wins
      k=1 : A=1/2 =0.5    , B=2/7=0.2857  → A wins

    Fails under the current impl because:
      (a) `rrf_k` arg is ignored (hardcoded 60),
      (b) no source_file dedup, so B never aggregates across pools and
          A wins at both k values; the two orderings are identical.
    """
    captured_chunks = {"k60": None, "k1": None}

    real_format = mcp_knowledge._format_results

    def capturing_factory(slot):
        def _cap(results, *a, **kw):
            captured_chunks[slot] = list(results)
            return real_format(results, *a, **kw)
        return _cap

    # Pool X: A at rank 0.
    pool_x = [_make_chunk("A", "a.py", "A")]
    # Pool Y: 5 filler chunks, then B at rank 5.
    pool_y = [
        _make_chunk("fill1", "f1.py", "f"),
        _make_chunk("fill2", "f2.py", "f"),
        _make_chunk("fill3", "f3.py", "f"),
        _make_chunk("fill4", "f4.py", "f"),
        _make_chunk("fill5", "f5.py", "f"),
        _make_chunk("B", "b.py", "B"),
    ]
    # Pool Z: B also at rank 5 (so B appears in two pools at rank 5).
    pool_z = [
        _make_chunk("fill1z", "f1z.py", "f"),
        _make_chunk("fill2z", "f2z.py", "f"),
        _make_chunk("fill3z", "f3z.py", "f"),
        _make_chunk("fill4z", "f4z.py", "f"),
        _make_chunk("fill5z", "f5z.py", "f"),
        _make_chunk("B", "b.py", "B"),
    ]
    _install_pools(monkeypatch, {"x": pool_x, "y": pool_y, "z": pool_z})

    monkeypatch.setattr(mcp_knowledge, "_format_results", capturing_factory("k60"))
    mcp_knowledge.handle_search_multi({
        "query": "q",
        "pool_names": ["x", "y", "z"],
        "rrf_k": 60,
    })
    monkeypatch.setattr(mcp_knowledge, "_format_results", capturing_factory("k1"))
    mcp_knowledge.handle_search_multi({
        "query": "q",
        "pool_names": ["x", "y", "z"],
        "rrf_k": 1,
    })

    top_k60 = captured_chunks["k60"][0].get("metadata", {}).get("source_file")
    top_k1 = captured_chunks["k1"][0].get("metadata", {}).get("source_file")
    assert top_k60 == "b.py", (
        f"With rrf_k=60 + dedup, b.py (aggregated rank-5 across 2 pools) should "
        f"outrank a.py (rank-0 in one pool); got top {top_k60!r}"
    )
    assert top_k1 == "a.py", (
        f"With rrf_k=1 + dedup, a.py (rank-0 in one pool) should outrank b.py "
        f"(aggregated rank-5); got top {top_k1!r}"
    )


def test_pool_weights_honoured(monkeypatch):
    """Weighting a pool MUST bubble its chunks up.

    Setup: pool X has A at rank 0; pool Y has B at rank 2 (behind fillers).
    Without weighting, A wins (rank-0 beats rank-2 at the same default k=60).
    With pool_weights={"y": 10.0}, pool Y's contribution is 10x — pool Y's
    rank-0 filler AND rank-2 B all outrank A.

    Math (k=60, weight_y=10):
      A (pool X, rank 0, weight 1):  1/61           = 0.01639
      B (pool Y, rank 2, weight 10): 10/63          = 0.15873
      → B outranks A.

    Without weighting (weight_y=1):
      A = 1/61 = 0.01639
      B = 1/63 = 0.01587
      → A outranks B.

    Fails under the current impl because `pool_weights` is silently
    ignored (all pools contribute weight 1.0 implicitly).
    """
    captured_chunks = {"off": None, "on": None}
    real_format = mcp_knowledge._format_results

    def capturing_factory(slot):
        def _cap(results, *a, **kw):
            captured_chunks[slot] = list(results)
            return real_format(results, *a, **kw)
        return _cap

    pool_x = [_make_chunk("A", "a.py", "A")]
    pool_y = [
        _make_chunk("pad1", "p1.py", "p"),
        _make_chunk("pad2", "p2.py", "p"),
        _make_chunk("B", "b.py", "B"),
    ]
    _install_pools(monkeypatch, {"x": pool_x, "y": pool_y})

    # Baseline — no weights. A should be top.
    monkeypatch.setattr(mcp_knowledge, "_format_results", capturing_factory("off"))
    mcp_knowledge.handle_search_multi({
        "query": "q",
        "pool_names": ["x", "y"],
    })

    # Weighted — pool y at 10x. B must outrank A.
    monkeypatch.setattr(mcp_knowledge, "_format_results", capturing_factory("on"))
    mcp_knowledge.handle_search_multi({
        "query": "q",
        "pool_names": ["x", "y"],
        "pool_weights": {"y": 10.0},
    })

    def _find_rank(chunks, src):
        for i, r in enumerate(chunks):
            if r.get("metadata", {}).get("source_file") == src:
                return i
        return -1

    off = captured_chunks["off"]
    on = captured_chunks["on"]

    # Without weighting, A must outrank B.
    assert _find_rank(off, "a.py") < _find_rank(off, "b.py"), (
        f"Baseline: a.py should outrank b.py without pool weights; "
        f"got order {[r.get('metadata', {}).get('source_file') for r in off]}"
    )
    # With weighting, B must outrank A.
    assert _find_rank(on, "b.py") < _find_rank(on, "a.py"), (
        f"Weighted pool y at 10x: b.py should outrank a.py; "
        f"got order {[r.get('metadata', {}).get('source_file') for r in on]}"
    )


def test_no_dedup_when_source_file_missing(monkeypatch):
    """Chunks without `metadata.source_file` MUST remain distinct.

    Two pools each return a chunk with no source_file metadata but
    different ids; the output must contain BOTH. Fails if dedup uses a
    bad fallback key (e.g. empty string) that collapses unrelated chunks.
    """
    captured_chunks = {}
    real_format = mcp_knowledge._format_results

    def capturing_format(results, *a, **kw):
        captured_chunks["results"] = list(results)
        return real_format(results, *a, **kw)

    monkeypatch.setattr(mcp_knowledge, "_format_results", capturing_format)

    pool_a = [_make_chunk("unique-a", source_file=None, text="AAA")]
    pool_b = [_make_chunk("unique-b", source_file=None, text="BBB")]
    _install_pools(monkeypatch, {"pool_a": pool_a, "pool_b": pool_b})

    mcp_knowledge.handle_search_multi({
        "query": "q",
        "pool_names": ["pool_a", "pool_b"],
    })

    results = captured_chunks.get("results", [])
    texts = sorted(r.get("text") for r in results)
    assert texts == ["AAA", "BBB"], (
        f"Both metadata-less chunks must survive (no accidental collapse); got texts {texts}"
    )
