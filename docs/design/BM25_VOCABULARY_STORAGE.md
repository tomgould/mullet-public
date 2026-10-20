# BM25 Vocabulary Storage — Packed Numpy Format

> Status: Design note — no implementation yet.  
> Target: `mullet/knowledge/engine/search/searcher.py`

## 1. Current State

The BM25 cache stores tokenized corpus data in two files:

- `bm25_cache.json` — metadata, corpus hash, doc count, **vocabulary dict** (`term → uint32 id`), and `bm25_chunks`
- `bm25_cache.npz` — compressed sidecar with `ids` (uint32 flat) and `offsets` (uint64)

The `.npz` sidecar already reduced the tokenized-corpus portion from ~3 GB JSON to ~200 MB.  
However, the **vocabulary dict** and **chunk metadata list** remain in the JSON file. On large pools the JSON alone still exceeds 1 GB, causing:

- ~45 s load time dominated by `orjson.loads` / `json.load`
- ~2 GB RAM spike during deserialization (millions of Python string objects + dict overhead)
- Cache invalidation is skipped because loading the cache to check the hash is already too expensive

## 2. Proposed Format

Move the vocabulary out of JSON into two binary packed arrays:

- **`vocab.npy`** — sorted `np.ndarray` of object dtype holding term strings (one Python string ref per unique term)
- **`vocab_ids.npy`** — parallel `np.ndarray` of `np.int32` holding the integer ID for each term

Lookup becomes `np.searchsorted(vocab, query_term)` → index → `vocab_ids[idx]`. Binary search on a sorted string array is `O(log V)`; for vocab sizes in the low millions this is only marginally slower than a Python dict and far faster than parsing a giant JSON dict.

## 3. Data Layout

```
<pool_dir>/
  bm25_cache.json        # format version, tokenizer version, corpus hash, doc count, bm25_chunks
  bm25_cache.npz         # ids (uint32) + offsets (uint64) — unchanged
  vocab.npy              # sorted terms (object dtype)
  vocab_ids.npy          # parallel int32 IDs
```

The JSON metadata shrinks to a few KB. The vocabulary lives entirely in `.npy` files that load near-instantly via `np.load` (no text decode, no dict construction).

## 4. Migration Path

1. On first load, `_load_bm25_cache` detects the absence of `.npy` files.
2. Falls back to the old JSON vocab, decodes normally, then immediately re-encodes and writes `vocab.npy` + `vocab_ids.npy`.
3. Sets `"bm25_vocab_format": "numpy"` in `.pool_metadata.json`.
4. Next load hits the fast path.

A `mullet reindex --pool <name>` naturally writes the new format. No user action is required; old-format caches remain readable until overwritten.

## 5. Backward Compatibility

- If `vocab.npy` or `vocab_ids.npy` is missing, fall back to the legacy JSON vocab field.
- If the `bm25_vocab_format` metadata flag is unset or `"json"`, use the legacy path.
- `clear_bm25_cache` removes both `.npy` files alongside the existing `.json` and `.npz`.

## 6. Estimated Savings

For a 50 k-chunk pool:

| Component | Current (JSON) | Proposed (packed) |
|-----------|---------------|-------------------|
| Vocabulary | ~1–1.5 GB | ~120 MB (terms + ids) |
| Tokenized corpus (.npz) | ~200 MB | ~200 MB (unchanged) |
| JSON metadata | ~1–1.5 GB | ~< 1 MB |
| **Total** | **~2–3 GB** | **~200–250 MB** |

Memory: avoids the ~2 GB RAM spike from deserializing the vocab dict.  
Disk: 10–15× reduction in cache file size.

## 7. Risk Assessment

| Risk | Mitigation |
|------|------------|
| **numpy dependency** | Already required via `sentence-transformers` and `chromadb`. No new dep. |
| **Binary search slower than dict** | Negligible in practice — BM25 queries typically contain < 20 terms. |
| **Format churn** | Bump `_BM25_CACHE_FORMAT` to `3`. Old caches auto-migrate on first load. |
| **Object-array portability** | `np.ndarray` of object dtype stores Python string refs; format is stable across numpy versions and platforms. |

## Open Question

Should `bm25_chunks` (the list of `{id, content, metadata}` dicts) also move out of JSON if it grows large? Measurement needed before implementation — the vocabulary change is the highest-impact first step.
