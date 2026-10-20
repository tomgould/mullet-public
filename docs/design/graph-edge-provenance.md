# Graph edge provenance — trust-tagging relationships by how they were derived

> Status: **proposal** (not scheduled, no implementation started)  
> Source: `mullet/knowledge/graph/`  
> Prompted by: audit of third-party tool Graphify (github.com/Graphify-Labs/graphify)

---

## 1. Overview

Mullet's knowledge graph (`mullet/knowledge/graph/`) extracts entities and
relationships from document chunks using a single LLM pass
(`extractor.py`). Every edge carries a `weight: float` confidence score,
but that score is **entirely self-reported by the LLM** — nothing
corroborates it against ground truth. There is no field distinguishing
"the LLM inferred this from prose" from "this is a structurally verifiable
fact."

Graphify (a local knowledge-graph tool audited via the `graphify` Trawl
pool) does something conceptually adjacent but tags every edge with a
provenance label:

- `EXTRACTED` — found via deterministic tree-sitter AST parsing
- `INFERRED` — resolved by its own LLM/heuristic pass
- `AMBIGUOUS` — uncertain

This lets Graphify's users filter or weight graph results by how much to
trust an edge. Mullet's graph engine has no equivalent — it's LLM-only,
so every edge is implicitly "inferred," but nothing says so.

This doc proposes adding provenance tagging to Mullet's graph, and goes
one step further than copying Graphify: Mullet already has a *separate*,
fully deterministic AST-based code index (`code.enabled` +
`trawl_find_symbol`) that Graphify has no equivalent of. Cross-referencing
the two gives provenance tags that are actually corroborated, not just
self-declared.

---

## 2. Current state

### 2.1 No provenance field today

`mullet/knowledge/graph/models.py`:

```python
@dataclass
class Relationship:
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str = ""
    weight: float = 1.0  # Confidence 0-1
    source_chunk_ids: list[str] = field(default_factory=list)
```

`Entity` has the same gap — no provenance field either. `weight` is the
only signal, and it conflates two different questions: "how confident is
the LLM" and "how was this edge derived." Those should be separate axes.

### 2.2 Everything comes from one LLM pass

`mullet/knowledge/graph/extractor.py` — `GraphExtractor` sends chunk text
through a single prompt (`EXTRACTION_PROMPT`, line 70) asking for JSON:

```
{"entities": [{"name": "...", "type": "...", "description": "..."}],
 "relationships": [{"source": "...", "target": "...", "type": "...",
                     "description": "...", "weight": 0.9}]}
```

The `weight` field is parsed and clamped at line 509
(`weight = min(1.0, max(0.0, float(rd.get("weight", 1.0))))`) — clamped
for range safety, not validated against anything external. There is no
second pass, no corroboration step, no distinction between "the model
read two variable names next to each other and guessed a relationship"
and "the model paraphrased an explicit `import` statement in the text."
Both come out with the same shape and, plausibly, similar weights.

### 2.3 A deterministic index already exists — and is unused by the graph

Separately from `knowledge/graph/`, Mullet has a fully deterministic,
non-LLM code-symbol index:

- `mullet/knowledge/code/parser.py` — `CodeParser`, tree-sitter-based,
  language-specific extractors in `mullet/knowledge/code/extractors/`
  (Python, TypeScript, PHP, Rust, etc.)
- `mullet/core/knowledge/symbol_finder.py` — `find_symbol()` /
  `find_symbols_by_type()`, exposed via the `trawl_find_symbol` MCP tool
- Enabled per-pool via `code.enabled: true` in `config.yaml`
  (`mullet/knowledge/engine/client.py:325`, `ingest.py:1191`)

This index knows, as AST fact rather than LLM guess: function/class
definitions, call sites, imports, and (language-dependent) inheritance.
It is a completely separate pipeline from `knowledge/graph/` — different
config flag, different storage, different extraction method. **The two
subsystems never talk to each other today.** A pool can have both
`code.enabled` and `graph.enabled` on and still get zero benefit from the
AST index when building the LLM graph.

---

## 3. Proposal

### 3.1 Add a provenance field, distinct from `weight`

Add to `Relationship` (and optionally `Entity`, for a future pass — see
§6):

```python
source_type: Literal["extracted", "inferred", "ambiguous"] = "inferred"
```

- **`extracted`** — corroborated by a deterministic fact (AST-derived
  call/import/inheritance edge from the code index, or some other
  non-LLM ground truth in future). High trust.
- **`inferred`** — found only via the LLM's prose reasoning, no
  corroboration available or attempted. This is what 100% of today's
  edges honestly are.
- **`ambiguous`** — low-weight and/or contradicted (e.g. the code index
  has a conflicting fact, or two LLM passes over overlapping chunks
  disagree on direction/type). Low trust; surfaced but flaggable.

Kept separate from `weight`: `weight` stays "how confident is the
extractor in this specific edge," `source_type` becomes "how was it
derived." A LLM could in principle report `weight=0.95` for a guess it's
very sure about — `source_type` still correctly says `inferred` unless
something external corroborates it.

### 3.2 Cross-reference against the code index (the actual new idea)

When a pool has **both** `code.enabled` and `graph.enabled` set, the
graph build gets a second signal source: after the LLM extraction pass
produces candidate relationships, check each one against the AST-derived
facts already sitting in the code-symbol index (`symbol_finder.py`).

Rough matching logic (naming/signature TBD at implementation time):

1. For a relationship like `(source_entity, relationship_type,
   target_entity)`, resolve `source_entity`/`target_entity` names against
   known symbols (function/class names) in the code index for the same
   pool.
2. If both resolve and the code index has a matching structural edge
   (call, import, inheritance) between them — tag `extracted`.
3. If the LLM's relationship type contradicts what the code index shows
   (e.g. LLM says "A calls B" but the AST shows no call edge and B
   doesn't exist in A's file/module) — tag `ambiguous`.
4. Otherwise (no corroborating or contradicting AST fact — e.g. the
   relationship is between concepts, not code symbols, or the code index
   doesn't cover the relevant language/file) — tag `inferred`, unchanged
   from today's behavior.

This is strictly additive to the existing extraction pipeline — it does
not change what the LLM pass produces, only how the output gets labeled
afterward.

### 3.3 Surfacing to users

The three read-side MCP tools that already touch the graph
(`mullet/server/mcp_knowledge.py`) gain provenance awareness:

- **`trawl_search_entities`** (`handle_search_entities`, line 1941) —
  already filters by `entity_type` (`args.get("entity_type")`); add an
  analogous `source_type` filter for relationship-bearing queries,
  following the same pattern.
- **`trawl_graph_export`** (`handle_graph_export`, line 2389, backed by
  `export_for_d3` in `mullet/knowledge/graph/export.py`) — include
  `source_type` in each edge's exported JSON so D3 consumers can color or
  filter by trust level; optionally accept a `min_provenance` /
  `source_types` export filter the way `max_nodes` is already accepted.
- **`trawl_graph_stats`** (`handle_graph_stats`, line 1875, backed by
  `graph_stats()` in `export.py`) — add a provenance breakdown alongside
  the existing entity-type distribution, e.g. `extracted: N, inferred: M,
  ambiguous: K`, so a pool owner can see at a glance how much of their
  graph is corroborated versus LLM-only.

The concrete use case this unlocks: "show only extracted edges" as a
high-trust traversal mode — e.g. for an agent that needs to reason about
actual call/import structure and cannot afford to walk a hallucinated
edge, versus a discovery/exploration mode that's happy to include
`inferred` edges for recall.

### 3.4 Backward compatibility

Every graph currently on disk was built before this field existed.
Default strategy:

- `source_type` defaults to `"inferred"` in the dataclass and in
  `from_dict()` deserialization (`models.py`) — this is the honest
  characterization of the current LLM-only pipeline; it is not a "no
  data" placeholder, it's an accurate label.
- No forced rebuild required for existing pools to keep working —
  `graph.load()` on an old graph file just sees every edge as `inferred`,
  which matches reality.
- Pools that want `extracted`/`ambiguous` tagging need a rebuild
  (`trawl_build_graph`) *after* upgrading, and only benefit if
  `code.enabled` is also on for that pool — see open questions below.

---

## 4. Scope and phasing

### Milestone 1 — schema + LLM-only default tagging

- Add `source_type` to `Relationship` (`models.py`), default `"inferred"`,
  round-tripped through `to_dict()`/`from_dict()`.
- No behavior change to extraction — `extractor.py` just writes
  `source_type="inferred"` on every edge it produces (or leaves it at the
  dataclass default).
- Wire `source_type` through `trawl_search_entities` filtering,
  `trawl_graph_export` output, and `trawl_graph_stats` breakdown (§3.3).
- Purely additive: no cross-referencing yet, existing graphs keep
  working unchanged, new graphs are labeled but not yet corroborated.

### Milestone 2 — cross-reference with the code index

- Build the matching step described in §3.2, gated on both
  `code.enabled` and `graph.enabled` being set for the pool.
- Promote `inferred` → `extracted`/`ambiguous` where the code index has a
  corroborating or contradicting fact.
- Likely lands as a post-processing pass over the extractor's output
  (or a step in the graph-build pipeline) rather than a change to the
  extraction prompt itself — the LLM doesn't need to know the code index
  exists.

### Open questions (unresolved, need answers before Milestone 2 is scheduled)

- **Rebuild requirement**: does cross-referencing require re-running
  `trawl_build_graph` on every existing pool with both flags on, or can
  it run as an incremental pass over an already-built graph? Leaning
  toward "requires rebuild" initially — simpler, matches how
  `code.enabled` itself already requires a (re-)ingest to take effect.
- **Performance cost of the join**: for a large pool, matching every
  LLM-extracted relationship against the symbol index is an O(edges ×
  lookup) pass. `symbol_finder.find_symbol()` needs to be reasonably
  indexed (not a linear scan) for this to be cheap enough to run on every
  graph build. Needs a benchmark on a real large pool before committing
  to "always cross-reference when both flags are on" versus an opt-in
  flag.
- **Name resolution ambiguity**: entity names in the graph are free text
  from the LLM; code symbols have exact qualified names. Matching
  `"UserService"` (LLM) to `UserService` (AST) in the right file/module
  when there might be multiple classes with the same name across files
  needs a defined resolution strategy (probably: match within the same
  `source_chunk_ids` file scope first, fall back to global name match
  with lower confidence).
- **Entity provenance**: this proposal only covers `Relationship`. Adding
  `source_type` to `Entity` too (e.g. "this entity name is a literal
  function/class name found by the AST parser" vs "the LLM invented a
  concept entity like 'Authentication Flow'") is a natural follow-up but
  out of scope for Milestone 1/2 — revisit once relationship-level
  tagging is shipped and useful.

---

## 5. What this is NOT

- **Not a rewrite of the extraction pipeline.** The LLM pass in
  `extractor.py` is unchanged; provenance is a label applied to its
  output, not a new extraction method.
- **Not a replacement for `weight`.** Confidence and provenance are
  orthogonal axes; both stay.
- **Not scheduled.** This is a design proposal only. No branch, no code
  changes, no milestone date.
