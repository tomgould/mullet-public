# Reindexing Trawl Pools

Reindexing rebuilds the BM25 and vector indexes that power Trawl search, code
lookup, and the knowledge graph. Run it after pulling new code, adding sources,
or when search results feel stale.

## Quick start — single pool

```bash
# Reindex one pool incrementally (fast, preserves existing data)
mullet reindex --pool <name>

# Full rebuild — wipes the index and starts from scratch
mullet reindex --pool <name> --rebuild
```

## Flag reference

### Source sync

| Flag | Meaning |
|------|---------|
| `--sync-sources` | Refresh external sources before indexing. Safe to pass on every pool; see below for what it actually does. |

### Scope

| Flag | Meaning |
|------|---------|
| `--rebuild`, `-r` | Delete the existing index first, then re-ingest everything. |
| `--code-only` | Only index code files, skip documents. |
| `--rebuild-code` | Rebuild the code index from scratch (keeps document index). |
| `--no-code` | Skip the code-index pass entirely. |

### Graph

| Flag | Meaning |
|------|---------|
| `--graph` | Run knowledge-graph extraction after the doc+code pass (incremental). |
| `--graph-rebuild` | Rebuild the knowledge graph from scratch. |
| `--max-graph-chunks N` | Cap graph build at N unprocessed chunks per pool. Useful for overnight runs so one pool doesn't dominate the schedule. |

### Parallelism

| Flag | Meaning |
|------|---------|
| `--workers N` | Thread-based parallel pool workers when using `--all`. Default: 1. |
| `--processes N` | Process-based parallel workers (CUDA-isolated). Overrides `--workers` when > 1. |

### Observability

| Flag | Meaning |
|------|---------|
| `--list`, `-l` | List available pools and exit. |
| `--stats` | Show index stats for the target pool(s) and exit. |
| `--resume` | Resume from the last checkpoint instead of starting over. |

## `--sync-sources` behaviour

Pools fall into two categories:

1. **Mullet-managed GitHub sources** — created with `trawl_connect_github` and
   have a `sources.yaml`. `--sync-sources` calls the GitHub connector:
   discovers repos for the owner/org via the GitHub API, `git clone`s new repos,
   and `git fetch`s existing ones into
   `pools/<pool>/sources/github/<owner>/<repo>/`.

2. **Symlinked or local-only pools** — `documents/` is a symlink (e.g. into
   `github_auditer/data/...`) and there is no `sources.yaml`. For these pools
   `--sync-sources` is a no-op; reindex simply ingests whatever is currently on
   disk.

Because it is harmless for category-2 pools, you can safely pass
`--sync-sources` in blanket scripts.

## `--graph` (knowledge graph extraction)

Graph extraction is **hardcoded to local LLM via LM Studio only**.
The docstring in `scripts/reindex.py` explains why: *"Graph extraction never
uses API providers — it would rack up unexpected bills."*

- If no model is loaded in LM Studio, the pool prints
  `GRAPH SKIPPED: no local model loaded` and continues.
- If `llm_config.graph_model` is set for the pool, that model is used;
  otherwise the first non-embedding model visible in LM Studio is picked.
- `--graph` is additive — the regular reindex pass runs first. On an
  up-to-date pool that pass is a fast no-op, so adding `--graph` only costs
  the graph extraction time.

Do **not** combine `--graph` with `--processes > 1` for local-LLM pools;
graph extraction is one-task-at-a-time and the extra processes just sit idle.

## Overnight multi-pool workflow

The repo ships `scripts/overnight_reindex_all.sh` for unattended reindexing of
every pool. It loops smallest-first (so small pools finish early and are
searchable while large pools churn), runs `mullet reindex --pool <p>
--sync-sources` per pool, logs to `~/.mullet/logs/overnight-reindex-<timestamp>.log`,
and continues past individual failures. Exit code is 1 if any pool failed.

Flags:

| Flag | Meaning |
|------|---------|
| `--graph` | Also run graph extraction on each pool. |
| `--rebuild` | Run a full rebuild instead of incremental. |
| `--dry-run` | Print what would be done without running anything. |

Typical invocation:

```bash
nohup scripts/overnight_reindex_all.sh --graph > /dev/null 2>&1 & disown
```

Or run inside a persistent `tmux` / `screen` session.

## Troubleshooting

| Message | What it means |
|---------|---------------|
| `UP TO DATE` | The pool's on-disk mtime matches the index timestamp; nothing was re-ingested. |
| `BM25 REBUILT` | The BM25 index was regenerated. This happens on every reindex (fast). |
| `GRAPH SKIPPED: no local model loaded` | You passed `--graph` but LM Studio has no model loaded. Load a model and retry, or omit `--graph`. |
| Stale BM25 cache | If you delete documents manually without reindexing, the BM25 cache may reference missing files. Run `mullet reindex --pool <name> --rebuild` to clear it. |
| Locked Chroma | Another process (e.g. a running Claude Code session) has the Chroma DB open. Close the other session or wait for it to finish. |
