---
name: pool-improve
description: Audit a Mullet pool config against its actual content and apply improvements — entity types, search settings, code indexing, graph config
triggers:
  - improve pool
  - fix pool config
  - pool config
  - audit pool
  - pool quality
  - improve my pool
  - pool entity types
aliases:
  - pool-audit
  - improve-pool
  - pool-config
tags:
  - knowledge
  - pool
  - configuration
---

<Purpose>
Analyse a Mullet knowledge pool's config.yaml against the actual indexed content,
diagnose issues, and apply targeted improvements. Covers: graph entity types,
search tuning, code indexing exclusions, and embedding config.
</Purpose>

<PoolLocation>
Pool configs and indexes live at the path returned by:
  python3 -c "from mullet.core.env import get_pools_dir; print(get_pools_dir())"

Each pool directory contains:
  config.yaml       — the pool configuration
  documents/        — source documents (may be a symlink)
  index/            — ChromaDB vector index + BM25 cache + graph DB
</PoolLocation>

<Steps>

1. **Identify the pool**

   If no pool name given, list available pools:
   ```
   mullet reindex --list
   ```
   Ask the user which pool to audit, or whether to audit all.

2. **Gather facts** (run in parallel)

   ```bash
   POOL=<pool-name>
   POOLS_DIR=$(python3 -c "from mullet.core.env import get_pools_dir; print(get_pools_dir())")
   POOL_DIR="$POOLS_DIR/$POOL"

   # Current config
   cat "$POOL_DIR/config.yaml"

   # Source file extension distribution (sample 200 chunks)
   python3 -c "
   import chromadb, collections
   c = chromadb.PersistentClient(path='$POOL_DIR/index')
   for col in c.list_collections():
       sample = col.get(limit=200, include=['metadatas'])
       exts = collections.Counter()
       for md in (sample['metadatas'] or []):
           src = (md or {}).get('source_path', (md or {}).get('source', ''))
           ext = src.rsplit('.',1)[-1].lower() if '.' in src else 'none'
           exts[ext] += 1
       print(f'Total chunks: {col.count()}')
       print(dict(exts.most_common(15)))
   " 2>/dev/null

   # Graph stats
   sqlite3 "$POOL_DIR/index/knowledge_graph.db" \
     "SELECT COUNT(*) as entities FROM entities; SELECT COUNT(*) as rels FROM relationships;" \
     2>/dev/null || echo "No graph DB yet"
   ```

3. **Diagnose issues**

   **Graph entity types** — the most common problem:

   | Content majority | Correct entity_types |
   |-----------------|---------------------|
   | `.py`, `.ts`, `.php`, `.js` (code) | `technology`, `library`, `framework`, `api`, `concept`, `pattern` |
   | `.md`, `.txt`, `.pdf` (docs/research) | `person`, `organization`, `technology`, `concept`, `model`, `framework` |
   | Planning/decisions/notes | `concept`, `technology`, `decision`, `requirement`, `framework`, `person` |
   | Mixed code + docs | `technology`, `library`, `framework`, `concept`, `person`, `organization` |

   Symptom: graph build completes but shows `0 ent  0 rel` → entity types don't match the content.
   Symptom: `graph.enabled: true` but graph build never runs → run `mullet reindex --pool <name> --graph`

   **Search settings** to review:

   | Setting | Default | When to change |
   |---------|---------|----------------|
   | `top_k` | 20 | Raise to 30 for large research pools |
   | `bm25_weight` | 0.5 | Raise for keyword-heavy code pools |
   | `vector_weight` | 0.5 | Raise for semantic/conceptual docs |
   | `hyde` | true | Disable for factual/reference pools |
   | `rerank_top_n` | 50 | Raise to 100 for pools > 20k chunks |

   **Code indexing** (`code:` section) — check `exclude_patterns` covers:
   `**/node_modules/**`, `**/dist/**`, `**/.git/**`, `**/__pycache__/**`,
   `**/*.min.js`, `**/*.map`, `**/build/**`, `**/.next/**`

   **Embedding** — all pools should use `BAAI/bge-base-en-v1.5` (dimension: 768)
   unless deliberately using a different model. Mismatch causes silent search failures.

4. **Propose changes**

   Show only the fields that would change. Example:
   ```
   graph.entity_types: (not set) → [technology, library, framework, api, concept, pattern]
   search.rerank_top_n: 50 → 100
   code.exclude_patterns: missing → add **/node_modules/**, **/__pycache__/**
   ```
   Ask: "Apply these changes? (yes / no / show full config)"

5. **Apply changes**

   Edit only the affected fields in `config.yaml`. Do not reformat or reorder the file.

   Adding entity_types under an existing `graph:` block:
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

6. **Tell the user what to run next**

   After saving config.yaml, state clearly what reindex command is needed:

   - Entity types changed → `mullet reindex --pool <name> --graph-rebuild`
   - Search/code settings changed → `mullet reindex --pool <name>`
   - Both → `mullet reindex --pool <name>` then `mullet reindex --pool <name> --graph-rebuild`
   - Large pool (> 20k chunks) — warn: graph rebuild will be slow. Suggest running overnight
     or using `--max-graph-chunks 3000` to cap per-run.

</Steps>

<MultiPoolReport>
When auditing all pools, output a compact table then group recommendations:

```
Pool                  Chunks    Entities   Graph     Verdict
-------------------------------------------------------------
mullet-repos          10,973      618       ✓ ok      —
mullet-hermes         18,062        0       ✗ wrong   entity_types: code types needed
ai_research           41,871        0       ✗ wrong   entity_types: research types needed
docs-portal           58,997        0       ⚠ large   entity_types + graph rebuild will be slow
```

Group fixes: "The following pools need entity_types updated to code types: X, Y, Z"
rather than repeating the same fix for each pool individually.
</MultiPoolReport>

<Notes>
- Never modify the documents/ directory — only config.yaml
- Pool machine_name must match the directory name exactly
- Graph rebuild on pools > 20k chunks takes hours — always warn before starting
- The `--max-graph-chunks N` CLI flag caps chunks per run (not settable in config.yaml yet)
- After graph rebuild, run a test search and check entity count via sqlite3 to confirm it worked
</Notes>
