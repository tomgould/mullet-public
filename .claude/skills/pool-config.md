# /pool-config — Analyse and Optimise a Mullet Pool Config

Use this skill to audit, fix, or create a `config.yaml` for one or all Mullet pools.
Triggered when: search quality is poor, graph produces 0 entities, reindex hangs, or user asks to review a pool.

Pools live at: `<pools-root>/<pool-name>/config.yaml`
Index data at: `<pools-root>/<pool-name>/index/`

`<pools-root>` is wherever this installation's Trawl pools directory is configured
(commonly a sibling directory next to the Mullet repo checkout) — ask the user if
it isn't already established in this conversation.

---

## Step 1 — Identify scope

If no pool name given, ask whether to audit one pool or all. For "all", iterate every directory under the pools root.

---

## Step 2 — Gather facts about the pool

Run these in parallel:

```bash
# Content type distribution
python3 -c "
import chromadb, collections
c = chromadb.PersistentClient(path='<pools-root>/<pool>/index')
colls = c.list_collections()
for col in colls:
    sample = col.get(limit=200, include=['metadatas'])
    exts = collections.Counter()
    for md in (sample['metadatas'] or []):
        src = (md or {}).get('source_path', md.get('source', ''))
        ext = src.rsplit('.',1)[-1].lower() if '.' in src else 'unknown'
        exts[ext] += 1
    print(f'Total chunks: {col.count()}')
    print(dict(exts.most_common(10)))
" 2>/dev/null

# Graph stats
sqlite3 <pools-root>/<pool>/index/knowledge_graph.db \
  "SELECT COUNT(*) as entities FROM entities; SELECT COUNT(*) as relationships FROM relationships;" 2>/dev/null || echo "No graph DB"

# Current config
cat <pools-root>/<pool>/config.yaml
```

---

## Step 3 — Diagnose issues

Check each area:

### Graph (`graph:` section)

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| 0 entities after graph build | Wrong `entity_types` for content | Update to match content type (see below) |
| Graph build takes > 2h | Pool too large, no cap | Add `max_graph_chunks: 3000` to graph section (note: currently CLI-only via `--max-graph-chunks`) |
| `graph.enabled: true` but no graph DB | Never been built separately | Run `mullet reindex --pool <name> --graph` |
| `graph.enabled: false` on content-rich pool | Was disabled as shortcut | Consider enabling with correct entity types |

**Entity types by content:**

- **Code repos** (Python, TypeScript, PHP, JS): `technology`, `library`, `framework`, `api`, `concept`, `pattern`
- **ML/research papers**: `person`, `organization`, `technology`, `concept`, `model`, `framework`
- **Docs / planning / notes**: `concept`, `technology`, `decision`, `requirement`, `framework`, `person`
- **Mixed code + docs**: `technology`, `library`, `framework`, `concept`, `person`, `organization`

Check source file extensions from Step 2 to classify. Majority `.py`/`.ts`/`.php` → code. Majority `.md`/`.txt` → docs.

### Search config (`search:` section)

| Setting | Good default | When to change |
|---------|-------------|----------------|
| `top_k` | 20 | Raise to 30 for research pools with long docs |
| `bm25_weight` / `vector_weight` | 0.5 / 0.5 | Raise bm25 for keyword-heavy code; raise vector for semantic docs |
| `hyde` | true | Disable for factual/reference pools (hallucination risk) |
| `multi_query` | true | Keep on; raises recall |
| `rerank_top_n` | 50 | Raise to 100 for large pools |
| `adaptive_rerank` | true | Keep on |

### Code indexing (`code:` section)

If `code.enabled: true`, verify:
- `exclude_patterns` covers `node_modules`, `dist`, `.git`, `__pycache__`, `*.min.js`
- No source maps (`.map`) or lockfiles being indexed
- `paths` points at the right subdirectory (usually `documents`)

### Embedding (`embedding:` section)

All pools should use `BAAI/bge-base-en-v1.5` (dimension 768) unless explicitly using a different model. Mismatched dimension causes silent failures on reindex.

---

## Step 4 — Apply changes

Edit `config.yaml` with targeted changes only. Do not reformat or reorder unrelated sections.

For `entity_types`, add under the `graph:` block:
```yaml
graph:
  enabled: true
  entity_types:
  - technology
  - library
  - framework
  - api
  - concept
  - pattern
  embed_batch_size: 64
  extract_workers: 1
  inter_batch_sleep: 0.0
  search_mode: mix
```

After editing, tell the user which reindex command to run:
- Vector only (fast): `mullet reindex --pool <name>`
- Graph only: `mullet reindex --pool <name> --graph`
- Both: `mullet reindex --pool <name>` then `mullet reindex --pool <name> --graph`
- Full rebuild: `mullet reindex --pool <name> --rebuild`

---

## Step 5 — Multi-pool report format

When auditing all pools, output a table:

```
Pool                  Chunks   Entities   Graph     Issue
------------------------------------------------------------
mullet-repos          10,973     618       ✓ ok      —
mullet-hermes         18,062       0       ✗ fix     entity_types: code types needed
ai_research           41,871       0       ✗ fix     entity_types: research types needed
docs-portal           58,997       0       ⚠ large   entity_types + consider max cap
...
```

Then list recommended changes grouped by fix type, not by pool — so the user can batch similar fixes.
