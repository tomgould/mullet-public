# On-Disk BM25 Backends — Design Note

> Status: Design note — no implementation yet.  
> Target: `mullet/knowledge/engine/search/searcher.py`

## 1. Current Problem

`HybridSearcher` currently rebuilds the full BM25 index in memory on every process start:

1. **Cache hit** — loads `bm25_cache.json` + `.npz`, decodes the tokenized corpus, and reconstructs `rank_bm25.BM25Okapi`. For a 1.5 M-chunk pool this is ~200 MB on disk → ~1.5 GB RAM and takes **30–60 s**.
2. **Cache miss / stale** — fetches every document from ChromaDB, tokenizes them (ProcessPool for big corpora), then builds `BM25Okapi` from scratch. Measured at **4–8 min** for the same 1.5 M chunks.

The root cause is that `BM25Okapi` has no native serialization; it stores everything as dense Python/Numpy objects in RAM. Every child process spawned by the agent loop repeats this work.

## 2. Option A — Tantivy (Rust, via `tantivy-py`)

| Pros | Cons |
|------|------|
| True on-disk index (mmap). Query time ~1 ms. | New heavy dependency (Rust binary wheel, ~15 MB). |
| Incremental updates without full rebuild. | FFI bridging complexity; error handling across the boundary. |
| BM25 scoring is built-in and well-tested. | Schema migration required (Chroma doc → Tantivy doc). |

**Integration path**
1. Add `tantivy>=0.21` to `pyproject.toml`.
2. On pool creation, build a Tantivy index alongside ChromaDB.
3. Replace `_bm25_search` with `tantivy.IndexReader` queries.
4. Keep ChromaDB for vector search; Tantivy handles keyword search only.

## 3. Option B — Pyserini-lite (Java Lucene bridge)

| Pros | Cons |
|------|------|
| Lucene BM25 is the industry standard, extremely fast. | Requires a JVM at runtime (or pre-built JNI). |
| Supports incremental indexing and complex queries. | Heaviest option; packaging nightmare on end-user machines. |
| Pyserini already bridges Python ↔ Lucene. | Overkill unless we also need advanced retrieval (learning-to-rank, etc.). |

**Integration path**
1. Vendor a minimal Pyserini build or use `pyserini` pip package.
2. Write a Python adapter that serializes chunks into Lucene `Document` objects.
3. Query via `pyserini.search.SimpleSearcher` and map scores back to chunk IDs.

## 4. Option C — Custom pickle/mmap of `BM25Okapi` state

| Pros | Cons |
|------|------|
| Zero new dependencies. | Still Python-object based; not as compact as a native index. |
| `BM25Okapi` internals are just `np.ndarray` and `dict` — pickle handles them. | Does not support incremental updates; full re-serialization on corpus change. |
| Can be mmapped back in for copy-on-read (reduces RSS across forks). | Must pin `rank-bm25` version; internal layout is not a public API. |

**Integration path**
1. After `_rebuild_bm25_index`, pickle `bm25_index.doc_len`, `bm25_index.doc_freqs`, `bm25_index.idf`, `bm25_index.average_idf`, and the inverted index structures into `bm25_state.pkl`.
2. On load, `mmap` + `pickle.load` the state and reconstruct `BM25Okapi` without touching ChromaDB or re-tokenizing.
3. Fall back to the existing cache path if the pickle file is missing or `rank-bm25` version changed.

## 5. Recommendation

**Implement Option C first.** It eliminates the rebuild cost with no new dependencies, is the smallest diff, and buys immediate value. If profiling later shows the pickle file is still too slow or too large at multi-million-chunk scale, **then** evaluate Option A (Tantivy) as a drop-in replacement for the BM25 leg of hybrid search.

Option B should only be considered if we later need Lucene-specific features (e.g., fuzzy queries, learning-to-rank, or distributed retrieval).

## 6. Migration Path

| Step | Option C | Option A |
|------|----------|----------|
| 1. Add backend | `_save_bm25_state()` / `_load_bm25_state()` methods in `searcher.py`. | New `TantivySearcher` wrapper class. |
| 2. Config flag | `bm25_backend: "native" \| "tantivy"` in pool metadata; default `"native"`. | Same flag; `"tantivy"` opt-in per pool. |
| 3. Dual-write period | Keep existing JSON/NPZ cache for rollback. | Build Tantivy index in background; do not delete Chroma docs. |
| 4. Rollback | Delete `bm25_state.pkl`; next load falls back to JSON cache rebuild. | Switch flag back to `"native"`; ignore Tantivy directory. |
| 5. Cleanup | After one release cycle, deprecate JSON/NPZ cache format. | After validation, make `"tantivy"` the default for new pools. |

## Open Questions

1. Does `rank_bm25.BM25Okapi` guarantee stability of internal attributes across patch releases? We should vendor a thin wrapper that pins the pickle schema version.
2. For Option C, should we use `cloudpickle` (already a transitive dep via `chromadb`) to handle potential lambda captures, or stick to stdlib `pickle` for safety?
3. For Option A, how do we keep the Tantivy schema in sync when the user changes `tokenizer_config` or BM25 parameters (`k1`, `b`)?
