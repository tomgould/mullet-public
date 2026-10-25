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

# A bare pool name works too
mullet reindex <name> --graph
```

Pools live in the pools directory, `mullet paths pools` (`TRAWL_POOLS_DIR` overrides it). `mullet reindex --list` shows them. It works from an installed copy as well as a checkout.

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

## Web sources (crawled websites)

A pool can crawl a website on every reindex. Add a `sources.web` entry to the
pool's `config.yaml`:

```yaml
sources:
  web:
    - url: https://www.example-news.org/
      sync_on_ingest: true        # re-crawl on every `mullet reindex` (and --all)
      use_sitemap: true           # seed the queue from /sitemap.xml (+ sitemap indexes)
      allow_paths: ['^/about-news/']  # regexes exempt from the built-in skip list
      download_types: [pdf, docx] # save linked documents for indexing
      max_file_mb: 50             # skip larger downloads
      max_pages: 3000             # pages + documents written per crawl
      wait_time: 1.0              # seconds between fetches
      timeout: 60
```

- Pages are saved as Markdown in `sources/web/<domain>/`, with `title`, `url`
  and (when the page has one) `published` frontmatter. Downloads go to
  `sources/web/<domain>/files/`. List their extensions in
  `documents.extensions`, or they are not indexed.
- The crawler skips login, cart, tag, search, about, contact, privacy and
  similar paths. The `about`/`contact` rules also match prefixes like
  `/about-us-news/`, so use `allow_paths` when real content lives there.
- `use_sitemap` matters when listings paginate with JavaScript. The crawler
  only follows `href` links, so it never reaches page 2.
- Downloads are recognised by extension, `Content-Type` or the
  `Content-Disposition` filename, so extension-less links like
  `/_downloads/<hash>` work. Each document is fetched once. Delete the file
  from `files/` to force a re-download.
- A page counts as changed only when its converted text changes. Many sites
  vary their raw HTML on every request (script nonces), and the text hash
  ignores download and view counters ("205 downloads"). Unchanged pages are
  not rewritten, so they are not re-embedded.
- Only the CLI crawls. `mullet reindex --pool <p>` crawls when
  `sync_on_ingest` is set, and `--sync-sources` forces it. The MCP
  `trawl_reindex_pool` tool does not crawl or index `sources/`.

### Dates

Each document gets the best date available, stored as `recency_ts` in its
chunk metadata. In order of trust:

1. **`published`**: the page's own date, from `article:published_time` and
   similar `<meta>` tags, JSON-LD `datePublished`, or microdata
   `itemprop="datePublished"`. Listing pages that carry several dates are
   left undated.
2. **PDF creation date**, from the PDF's metadata, falling back to its
   modification date.
3. **First seen**: when the crawler first found a page or download.
4. **Updated**: when a page's text last changed.

If a document has a published date, that date is used. Otherwise it gets
the newest of the other three.

Everything found on a pool's first crawl is a *baseline*. Its first-seen date
is just the crawl date, so it isn't used. The same applies to pages cached
before first-seen tracking existed. From the next crawl onwards, any new page
or download is dated the day it first appears.

The crawler records these dates in two places:

- page frontmatter (`first_seen`, `updated`)
- `files/.file_index.json` for downloads, which also gives each document its
  source URL

Each crawl also appends its new and changed URLs to
`sources/web/<domain>/.crawl_log.jsonl`, keeping the last 365 runs.

Documents indexed before these dates existed need `mullet reindex --pool <p>
--rebuild` once to pick them up.

### Recency ranking (news pools)

```yaml
search:
  recency_boost: 3.0            # added to the rerank score of a just-dated chunk
  recency_half_life_days: 365   # the bonus halves every N days
  prioritize_data: false        # see below
```

The boost is applied after reranking and relevance filtering, so it reorders
relevant results rather than surfacing irrelevant new ones. Undated chunks get
no bonus and no penalty. The default rerank model (`ms-marco-MiniLM-L-6-v2`)
scores roughly -10 to +10, so a boost of 2–3 decides close calls without
overriding a clearly more relevant older article. Search results show
`**Published:**` and `**URL:**` lines when the metadata has them.

`prioritize_data` (on by default) is a reorder built for compliance pools. It
moves any chunk from a file named `*list*` or `*reference*`, or any chunk
that looks like a list, to the top regardless of score. Turn it off for news
pools, or it overrides both relevance and recency.

### What's new: `trawl_recent`

For "what's new" or "recent news" questions, use the `trawl_recent` MCP tool
rather than search. It lists a pool's documents newest first by `recency_ts`,
labels where each date came from, and accepts `since` (an ISO date) and
`limit`.

### Nightly crawl and the per-pool lock

Mullet's own scheduler stores tasks but never runs them, so nightly crawls use
a systemd user timer. For example, `example-news` uses
`~/.config/systemd/user/mullet-reindex-example-news.{service,timer}`, which runs
daily at 03:00. It is `Persistent=true`, so a run missed while the machine is
off happens at next boot.

```bash
systemctl --user list-timers mullet-reindex-example-news.timer
journalctl --user -u mullet-reindex-example-news.service   # last run's output
systemctl --user start mullet-reindex-example-news.service # run it now
```

Each reindex holds a per-pool lock (`<pool>/.reindex.lock`), and `--sync-sources`
crawling happens inside it. If the timer fires while a manual `--all` is
working on the same pool, the second run skips that pool and says so, rather
than writing to the same index at the same time.

## `--graph` (knowledge graph extraction)

Graph extraction is **hardcoded to local LLM via LM Studio only**.
The docstring in `mullet/knowledge/reindex.py` explains why: *"Graph extraction never
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
--sync-sources` per pool, logs to `overnight-reindex-<timestamp>.log` in the `logs` folder of the state directory (`mullet paths state`),
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
