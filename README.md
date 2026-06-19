# mcp-super-memory

[![npm version](https://img.shields.io/npm/v/mcp-super-memory)](https://www.npmjs.com/package/mcp-super-memory)
[![Node.js](https://img.shields.io/badge/Node.js-20%2B-green)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**N:M associative memory graph for LLM agents — delivered as an MCP server.**

> Search **"Newton"** → reach **"strawberry"** through shared keys.
> Embedding similarity alone can't do this.

`mcp-super-memory` is an associative memory system for LLM agents built on a **Key/Value graph** — not a vector store. Memories live in a **Value Space**, accessed through a separate **Key Space** — one memory reachable via many keys, one key leading to many memories. This enables human-like associative leaps (multi-hop graph traversal) that pure embedding search fundamentally cannot replicate.

**Works with:** Claude Desktop · Claude Code · any MCP-compatible LLM agent

---

## Why Not Just Embeddings?

Every existing memory system (Mem0, A-MEM, MemGPT) stores memories as nodes and retrieves them by embedding similarity. This works until it doesn't:

```
Query: "Newton"
Embedding search finds: "Newton discovered gravity" ✅
Embedding search misses: "user likes strawberries"   ❌
```

Super Memory finds both — because "Newton" → apple memory → fruit key → strawberry memory. The **path exists in the key graph**, not in embedding space.

---

## How It Works

```
Key Space (concepts)         Value Space (memories)
─────────────────────        ──────────────────────────────
[Newton]  ──────────────────→ "Newton discovered gravity"
[apple]   ────────┬─────────→      ↑ same memory
[gravity] ────────┘
                  │
[apple]   ────────┼─────────→ "apples are red fruit"
[fruit]   ──────┬─┘
[red]     ──────┤
                │
[fruit]   ──────┼─────────→ "user likes strawberries"
[strawberry]────┘
```

`recall("Newton")` returns matching key clusters such as `[Newton]` and `[apple]`, not memory content. The agent then navigates explicitly: `read_key(apple)` → select the Newton memory → `read_memory(...)` → discover its `[fruit]` key → `read_key(fruit)` → select the strawberry memory.

The default MCP flow is therefore **Key → Memory → Key**. Full memory content enters the model context only when the agent deliberately calls `read_memory()`.

---

## Key Features

| Feature | Super Memory | A-MEM | Mem0 | MemGPT |
|---------|-------------|-------|------|--------|
| Key/Value separation | ✅ N:M | ❌ | ❌ | ❌ |
| Agent-driven Key → Memory navigation | ✅ built-in | ❌ | ❌ | ❌ |
| Associative multi-hop | ✅ built-in | ❌ | ❌ | ❌ |
| Depth system | ✅ | ❌ | ❌ | partial |
| Memory versioning | ✅ supersede | overwrites | overwrites | ❌ |
| Time decay | ✅ depth-weighted | ❌ | ❌ | ❌ |
| Key types | ✅ concept/name/proper_noun | ❌ | ❌ | ❌ |
| Key merge (IDF) | ✅ | ❌ | ❌ | ❌ |
| Hybrid retrieval (BM25 + dense + RRF) | ✅ | ❌ | partial | ❌ |
| Dual-path dense recall | ✅ key + content | ❌ | ❌ | ❌ |
| Hebbian link learning | ✅ | ❌ | ❌ | ❌ |
| Cross-encoder rerank | ✅ opt-in | ❌ | partial | ❌ |
| Depth-floored recall (`min_depth`) | ✅ | ❌ | ❌ | ❌ |
| Auto-download models (bge-m3 / reranker) | ✅ | n/a | n/a | n/a |

### Depth System

Every memory has a depth score `0.0 → 1.0`:

| Stage | Depth | Behavior |
|-------|-------|----------|
| Shallow | `< 0.3` | Recent, unverified. Easy to update or forget. |
| Medium | `0.3–0.7` | Confirmed multiple times. Stable. |
| Deep | `> 0.7` | Well-established fact. Resists correction. |

Depth increases `+0.05` each recall. Deep memories decay slower over time. If you try to correct a deep memory, it resists — its depth stays higher even after supersede.

### Key Types

Not all keys should behave the same. Names shouldn't match semantically — "동건" shouldn't match "뉴턴" just because they're both short Korean words.

| Type | Matching | Use Case |
|------|----------|----------|
| `concept` (default) | Embedding similarity ≥ threshold (0.28 OpenAI / 0.60 local) | Topics, categories, attributes |
| `name` | Exact match only | Person names |
| `proper_noun` | Exact match only | Brands, places |

Name/proper_noun keys also get IDF penalty (`×0.5`) when they become hub keys connected to many memories, preventing them from polluting unrelated searches.

### Versioning (not overwriting)

```
"user lives in Seoul"   (depth: 0.4 → weakened to 0.12, preserved)
        ↑ superseded by
"user moved to Busan"   (depth: 0.0, new)
```

Unlike A-MEM which overwrites memory on evolution, Super Memory keeps the full history. Every correction is traceable — when did the belief change, and from what session?

### Key Merging

```
Add key "파이썬"  → finds existing "Python" (similarity 0.87 > threshold 0.85)
                 → reuses existing key instead of creating duplicate
```

Prevents key space fragmentation. Same concept across languages or phrasing stays unified.

### Agent-driven Retrieval (default)

The default MCP API keeps Key Space and Value Space separate:

1. `recall(query)` searches canonical keys and aliases. It returns key IDs, concept labels, match scores, linked-memory counts, hub status, and specificity — never memory content.
2. `read_key(key_id)` returns ranked memory IDs and metadata, never content. Hub keys are paginated with `limit`/`offset` so broad concepts cannot flood context.
3. `read_memory(memory_id, via_key_id)` returns the full memory plus every connected key cluster. Only this full read increases memory depth/access count; only the traversed `via_key_id` edge receives Hebbian reinforcement.
4. The agent follows any returned key with another `read_key()` call, producing an explicit **Key → Memory → Key** graph walk.

Semantically merged keys are preserved as aliases on one canonical key cluster (for example `Python` + `파이썬`). A key linked to at least three active memories is surfaced as a hub with `is_hub`, `memory_count`, and `specificity` metadata rather than being hidden by IDF. Override the threshold with `SUPER_MEMORY_KEY_HUB_MIN_LINKS`.

### Direct Hybrid Retrieval (optional compatibility mode)

Set `SUPER_MEMORY_DIRECT_RECALL=true` to expose `recall_memories()`, the previous one-call memory retrieval path. Three signals run in parallel and are fused with **Reciprocal Rank Fusion** (`RRF_K = 60`):

- **BM25 (sparse):** lexical full-text search over memory content (MiniSearch, fuzzy + prefix). Catches exact terms, names, and rare tokens that embeddings blur.
- **Dense Path A (key matching):** query embedding → match keys → follow links → memories. Score = `keySim × IDF × linkWeight`, summed across all matching keys.
- **Dense Path B (content matching):** query embedding → directly compare against memory content embeddings. Finds memories even when they weren't tagged with the right keys.

Sparse and dense rank lists are merged by RRF, then modulated by depth and time before configurable multi-hop expansion (`hops=1–5`, default `2`). This compatibility tool is hidden by default so agents use explicit key navigation instead of collapsing the graph into one search call.

### Hebbian Link Learning

Reading a full memory is a **write**, not just a read. In the default flow, `recall()` and `read_key()` are read-only; `read_memory(memory_id, via_key_id)` reshapes the selected path:

- The traversed `via_key_id → memory_id` link is **reinforced** (`+0.1`, capped at `3.0`).
- Memory depth and access count increase only after the agent reads the full memory.

Reinforcement is scoped to the key the agent actually traversed — not every key attached to the memory. This is the literal Hebbian rule ("fire together, wire together") and prevents unrelated associations from growing when the memory is reached through a different concept. Weights are clamped to `[0.1, 3.0]`.

Link weights feed back into `read_key()` ranking, so repeatedly selected paths become easier to reach. Optional `recall_memories()` retains the previous matched-link reinforcement and explored-link decay behavior.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Key Space                          │
│   [name] [동건] [programming] [python] [fruit] [red]   │
│      ↓      ↓         ↓           ↓       ↓      ↓     │
│   [vec]  [exact]    [vec]       [vec]   [vec]  [vec]   │
└────────────────────────┬────────────────────────────────┘
                         │ N:M links
                         ↓
┌─────────────────────────────────────────────────────────┐
│                     Value Space                         │
│   "user's name is Donggeon"     depth: 0.85  (deep)    │
│   "user likes Python"           depth: 0.30  (medium)  │
│   "user likes strawberries"     depth: 0.05  (shallow) │
└─────────────────────────────────────────────────────────┘
```

**Default MCP navigation:**

1. Embed the query and match canonical key concepts plus exact aliases.
2. Return key clusters with `memory_count`, `is_hub`, and `specificity`; do not return memory content.
3. Rank a selected key's memory handles by link weight, depth, and time decay in `read_key()`.
4. Return full content and adjacent key clusters from `read_memory()`; reinforce only the traversed edge.
5. Repeat `read_key(next_key_id)` to walk the graph deliberately.

**Optional `recall_memories()` algorithm (hybrid, configurable 1–5 hops; default 2):**

Three retrieval signals run in parallel, then get fused and expanded:

1. **BM25 (sparse):** lexical search over memory content (MiniSearch, fuzzy `0.2` + prefix). Keep top 50.
2. **Dense Path A (keys):** embed query → match keys (concept: cosine ≥ threshold; name/proper_noun: substring match → score `1.0`) → take top 10 keys → follow links. Score = `keySim × IDF × linkWeight`, summed across matching keys.
3. **Dense Path B (content):** compare query embedding directly against memory content embeddings (cosine ≥ threshold).
4. **RRF fusion:** merge the BM25 and dense rank lists via `score += 1 / (RRF_K + rank + 1)` (`RRF_K = 60`).
5. **Depth & time modulation:** `score × (0.9 + depth × 0.1) × timeFactor`, where `timeFactor` is a depth-weighted 30-day half-life decay (deep memories decay slower).
6. **Associative expansion (`hops`, default 2):** breadth-first from the directly-matched set — each round follows shared keys (`× HOP_DECAY(0.3) × IDF × linkWeight`) and explicit `related_to` links (bidirectional, `× HOP_DECAY`) to the next frontier. `hops=N` walks up to N steps, so a memory's `hop` is its shortest chain distance. Score decays by `HOP_DECAY` per hop.
7. **Hebbian update:** reinforce matched-key links of returned memories (`+0.1`), decay explored-but-unreturned links (`−0.005`).
8. Return ranked results with `hop` field (`1` = direct, `2+` = associative distance).

### Similarity thresholds (calibrated per embedding model)

Embedding backends have very different cosine distributions, so a single threshold set cannot serve all of them. The thresholds below are calibrated per model (`getThresholdProfile()` in `src/embedding.ts`):

| Threshold | OpenAI | Local BGE (en) | Local e5 (multilingual) |
|-----------|--------|----------------|--------------------------|
| Key recall (query↔key cosine) | 0.28 | 0.60 | 0.85 |
| Content recall (query↔content cosine) | 0.28 | 0.50 | 0.80 |
| Key auto-link | 0.50 | 0.60 | 0.93 |
| Key merge | 0.85 | 0.85 | 0.97 |
| Memory dedup | 0.90 | 0.90 | 0.985 |

**Why e5 differs so much:** multilingual-e5 packs embeddings into a narrow high-cosine band (~0.86–0.99). Same-word query↔key pairs (asymmetric `query:`/`passage:` prefixes) still separate cleanly (~0.89 vs ≤0.82), but key↔key and content↔content do **not** — distinct facts like *"A uses Postgres"* and *"B uses Mongo"* sit at ~0.96, dangerously close to true paraphrases (~0.99). Hence e5's merge/dedup/auto-link thresholds are pushed high to avoid silently collapsing distinct memories.

**Drift escape hatch:** if you switch models or your data's character drifts, override any threshold without code changes:

```
SUPER_MEMORY_KEY_RECALL=0.82
SUPER_MEMORY_MEMORY_DEDUP=0.99
# also: _KEY_MERGE, _KEY_AUTOLINK, _CONTENT_RECALL  (values in [0,1])
```

**Score gate, distribution gate, and contradiction band** can also be tuned per deployment:

| Env var | Default (profile) | Description |
| --- | --- | --- |
| `SUPER_MEMORY_MIN_SCORE` | per-model (e.g. `0.55` for bge-m3) | Absolute cosine floor for optional `recall_memories()`. Set to `0` to disable. |
| `SUPER_MEMORY_GATE_Z` | `0` by default | Opt-in distribution gate for `recall_memories()` (robust-z, median/MAD). Values around 2–5 are typical; `0` disables it. |
| `SUPER_MEMORY_CONTRADICTION` | per-model (e.g. `0.80` for bge-m3) | Contradiction-band lower bound. Memory pairs whose cosine similarity falls in `[contradiction, memoryDedup)` are flagged as contradictions. `read_memory()`, `related()`, and optional `recall_memories()` expose conflicting IDs. |

**Why e5 gates are opt-in:** multilingual-e5's narrow cosine band (~0.86–0.99) makes a static floor unreliable, while held-out tests showed distribution and key-proximity gates can also overfit. Both are disabled by default to avoid hiding real memories. Use bge-m3 for reliable not-found behavior, or calibrate e5 gates on your own corpus.

Distribution gate parameters:
- **`gateZ`** — set via `SUPER_MEMORY_GATE_Z` or the `min_z` parameter of optional `recall_memories()`.
- **`0` disables** the gate (default for bge-m3, bge, openai, minilm — where `min_score` already works).
- Both gates **compose (AND)**: a result must clear both `min_score` and `gateZ` to be returned.
- A **literal name/proper-noun key match** (e.g. querying a stored `name`-typed key exactly) is always a definite anchor and bypasses the distribution gate.
- **`GATE_MIN_POPULATION = 8`**: the gate is skipped when fewer than 8 memories exist (too few samples for a reliable distribution), so early-session recall is unaffected. The gate's background population is **namespace-filtered** and excludes superseded/expired memories, so recall scoped to a sparse namespace may fall below this threshold and skip the gate entirely.
- **Known e5 limitation:** the optional gate keys off `maxContentSim` (content cosine only). A relevant fuzzy-key hit with a flat content distribution may be rejected; literal key matches bypass the gate.

An uncalibrated `LOCAL_EMBEDDING_MODEL` falls back to the BGE profile **and logs a warning** so the miscalibration is never silent.

> **Multilingual note:** cross-lingual *content* matching has a same-language bias (a Korean query scores Korean memories higher regardless of meaning). The reliable cross-lingual path is the **key graph** — tag memories with keys in multiple languages (e.g. `["딸기", "strawberry"]`) so recall hits the key exactly instead of relying on biased content similarity.

---

## MCP Tools

The memory system exposes 12 tools by default:

| Tool | Description |
| --- | --- |
| `recall(query, top_k?, namespace?)` | Search Key Space only. Returns canonical keys, aliases, scores, and hub metadata; never memory content. |
| `read_key(key_id, namespace?, limit?, offset?)` | List ranked memory IDs and metadata connected to one key. Never returns content; supports pagination for hubs. |
| `read_memory(memory_id, via_key_id?, namespace?)` | Read full memory content and connected keys. Increases depth/access and reinforces the traversed edge. |
| `remember(content, keys, key_types?, namespace?, ttl_seconds?, related_to?)` | Save memory with key concepts and optional type annotations |
| `correct(memory_id, content, keys?, key_types?, related_to?)` | Versioned update — old memory preserved but weakened |
| `related(memory_id)` | Find memories sharing keys (associative exploration) |
| `forget(memory_id)` | Permanently delete |
| `get_conversation(session_id, turn?)` | Load original conversation turns |
| `list_memories(namespace?)` | List all stored memories with keys, depth, access count |
| `remember_batch(items)` | Save multiple memories in one call |
| `cleanup_expired()` | Delete memories whose TTL has expired |
| `memory_stats()` | Get current key/memory/link counts |

Set `SUPER_MEMORY_DIRECT_RECALL=true` to expose a thirteenth compatibility tool, `recall_memories(...)`, with the previous BM25+dense+RRF multi-hop behavior.

A system prompt template is also available via `memory_system_prompt` MCP prompt — include it to instruct the agent to recall silently, use diverse keys, and never mention the memory system to users.

---

## Quick Start (MCP Server)

### Claude Desktop

Add to `claude_desktop_config.json`:

**OpenAI embeddings:**
```json
{
  "mcpServers": {
    "mcp-super-memory": {
      "command": "npx",
      "args": ["-y", "mcp-super-memory"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

**Local embeddings (no API key required) — bge-m3 recommended:**
```json
{
  "mcpServers": {
    "mcp-super-memory": {
      "command": "npx",
      "args": ["-y", "mcp-super-memory"],
      "env": {
        "EMBEDDING_BACKEND": "local",
        "LOCAL_EMBEDDING_MODEL": "bge-m3"
      }
    }
  }
}
```

> `bge-m3` (multilingual, recommended) **auto-downloads ~570MB on first run**, then caches. Omit `LOCAL_EMBEDDING_MODEL` for the lighter default (`fast-multilingual-e5-large`). Add `"SUPER_MEMORY_RERANK": "true"` to enable cross-encoder reranking (downloads a second model on first use).

### Claude Code

```bash
# OpenAI embeddings
claude mcp add mcp-super-memory -e OPENAI_API_KEY=your-openai-api-key -- npx -y mcp-super-memory

# Local embeddings (no API key required) — bge-m3 recommended (auto-downloads ~570MB on first run)
claude mcp add mcp-super-memory -e EMBEDDING_BACKEND=local -e LOCAL_EMBEDDING_MODEL=bge-m3 -- npx -y mcp-super-memory
```

### Manual / Development

```bash
git clone https://github.com/donggyun112/super-memory
cd super-memory
pnpm install
```

Create `.env`:
```
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Or use local embeddings (no API key required):
```
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=fast-multilingual-e5-large  # default; best fit for Korean/multilingual keys
```

**BGE-M3 (recommended, multilingual) — auto-downloaded:** set `LOCAL_EMBEDDING_MODEL=bge-m3` (aliases: `bgem3`, `baai/bge-m3`, `fast-bge-m3`). On first use the model is **fetched automatically if missing** — quantized ONNX (~570MB, from `onnx-community/bge-m3-ONNX`) plus the tokenizer/config (from `BAAI/bge-m3`) — and cached under `~/.super-memory/models/bge-m3`. No manual download needed:

```
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=bge-m3
# optional — point at an existing model dir to skip the download (backward compatible):
# LOCAL_EMBEDDING_MODEL_PATH=/absolute/path/to/model-dir   # dir with model.onnx + tokenizer files
# LOCAL_EMBEDDING_MODEL_FILE=model.onnx                    # optional; default is model.onnx
```

> First run downloads ~570MB once, then reuses the cache. If `LOCAL_EMBEDDING_MODEL_PATH` already holds the model it is used **as-is with no download** (a partial dir is self-healed — only missing files are fetched). Online-API backends (OpenAI) and fastembed built-ins are unaffected.

**Cross-encoder reranking (optional direct mode):** set `SUPER_MEMORY_DIRECT_RECALL=true` and `SUPER_MEMORY_RERANK=true` to re-score `recall_memories()` candidates with `bge-reranker-v2-m3`. The default key-navigation flow does not load the reranker. The model (~570MB, quantized) auto-downloads on first use and caches under `~/.super-memory/models/reranker`.

```
SUPER_MEMORY_RERANK=true
SUPER_MEMORY_DIRECT_RECALL=true
# optional: SUPER_MEMORY_RERANK_MODEL_PATH=/dir   SUPER_MEMORY_RERANK_POOL=30  (candidates re-scored)
```

> Off by default. If the model cannot load, `recall_memories()` falls back to fused ranking. Query decomposition remains the caller's responsibility.

**Reranker not-found gate (`SUPER_MEMORY_RERANK_MIN_SCORE`):** in direct compatibility mode, reject the complete `recall_memories()` result when the top cross-encoder logit is below this floor.

```
SUPER_MEMORY_RERANK=true
SUPER_MEMORY_DIRECT_RECALL=true
SUPER_MEMORY_RERANK_MIN_SCORE=0   # reject when top rerank logit < 0 (bge-reranker-v2-m3 scale)
```

> ⚠️ **Caveats.** (1) Unset by default — no gate. (2) The logit scale is **model-dependent**; `0` (≈ sigmoid 0.5) suits `bge-reranker-v2-m3` (measured: same-language found ≈ +3.9, not-found ≈ −5 to −6) — recalibrate for other rerankers. (3) **Trusted for SAME-LANGUAGE only.** Cross-lingual relevance logits run low even when relevant (KR query ↔ EN memory ≈ −5.4), so the gate auto-**bypasses on a script mismatch** (KR↔Latin) to avoid false-rejecting cross-lingual hits — which means **cross-lingual content must be reachable via bilingual keys** (`["Jiwoo","지우"]`), and cross-lingual *not-found* precision is a known limitation. Leave this off if you can't tag bilingual keys.

> **Prefix behavior:** BGE-M3 does **not** use `passage:`/`query:` prefixes — embeddings are passed through as-is. All other local models (e5, BGE-en, MiniLM) continue to use prefixes unchanged.

> **Recommended for multilingual / cross-lingual use: `bge-m3`.** It separates unrelated queries more reliably and performs substantially better than e5 on the project's Korean↔English fixtures. In optional direct mode, bge-m3's absolute `min_score` gate reaches ≈96% on the gate fixture; e5 requires corpus-specific tuning.

If `OPENAI_API_KEY` is not set and `EMBEDDING_BACKEND` is unset, the server automatically uses the local `fastembed` backend.
For English-only use or lower local resource usage, set `LOCAL_EMBEDDING_MODEL=fast-bge-base-en-v1.5` or `fast-bge-small-en-v1.5`.

> **Switching backends is safe.** The graph records an embedding **fingerprint** (backend + model id) identifying the vector space its embeddings live in. On startup, if the current backend's fingerprint **or** dimension differs from what is stored, the graph **auto-migrates** — every key and memory is re-embedded with the new backend while content, links, depth, and access history are preserved (a `graph.json.bak.*` backup is written first). The fingerprint matters because two models can share a dimension yet produce incompatible vectors (e.g. `fast-multilingual-e5-large` and `bge-m3` are **both 1024-d**); a dimension check alone would miss that swap and silently corrupt every similarity. Disable with `SUPER_MEMORY_AUTO_MIGRATE=false`. Re-embedding via OpenAI incurs one-time API cost proportional to your memory count.
>
> **Migrating a pre-fingerprint (legacy) graph across same-dimension models.** A graph written before fingerprinting has no recorded vector space, so a same-dimension model swap off it cannot be detected automatically. Set `SUPER_MEMORY_FORCE_REEMBED=true` for **one** startup to re-embed unconditionally and stamp the fingerprint; remove it afterward (left on, it re-embeds on every start). This is exactly the one-shot needed when moving an existing e5 graph to bge-m3.

```bash
pnpm dev
# or:
pnpm build
pnpm start
```

**Requirements:**
- Node.js 20+
- pnpm for local development
- OpenAI API key for OpenAI embeddings, or `fastembed` for local embeddings

---

## Data Storage

All data is local. No external database required.

```
~/.super-memory/
├── graph.json          # canonical keys, aliases, memories, weighted links
└── conversations/
    └── {session_id}.jsonl   # original conversation turns
```

Set `SUPER_MEMORY_DATA_DIR` to use a different storage directory.

---

## Limitations

- **Linear scan** — suitable for personal use (~10k memories). FAISS/ChromaDB integration planned for larger scale.
- **Agentic round trips** — the default `recall → read_key → read_memory` flow is more controllable and context-efficient, but needs more tool calls than one-shot retrieval.
- **Hub breadth** — broad keys can connect many memories. `read_key()` paginates hubs; the agent must choose whether to continue paging or follow a more specific adjacent key.
- **Agent quality matters** — key selection on `remember` affects retrieval quality. System prompt tuning is important.
- **Cross-lingual content bias** — with multilingual e5, raw content similarity favors same-language memories regardless of meaning. Tag memories with multilingual keys so the key graph (not biased content cosine) carries cross-lingual recall.
- **Threshold calibration** — thresholds are tuned per embedding model. A new/uncalibrated model falls back to the BGE profile (with a warning); recalibrate via the `SUPER_MEMORY_*` env overrides.

---

## Testing

```bash
pnpm test                        # unit tests (fast, no model download)
tsx test/scenarios.ts            # 21 end-to-end behavioral checks (local e5)
tsx test/robustness.ts           # threshold overrides + Hebbian pollution bounds
tsx test/migration.ts            # backend/dimension switch auto-migration (no brick)
tsx test/nhop.ts                 # N-hop chained traversal (recall hops parameter)
tsx test/depth-noise.ts          # deep-hop noise bounds + relative score floor
tsx test/live-multilingual.ts    # interactive multilingual recall demo

# Manual retriever-quality check (NOT part of pnpm test):
EMBEDDING_BACKEND=local npx tsx test/retriever-quality.live.ts
# For bge-m3: also set LOCAL_EMBEDDING_MODEL=bge-m3 LOCAL_EMBEDDING_MODEL_PATH=/abs/dir
```

`scenarios.ts` and `robustness.ts` exercise the real local embedding backend (direct/associative/cross-lingual recall, versioning, depth growth, dedup, TTL, Hebbian learning, namespace isolation). They double as a recalibration harness when tuning thresholds for a new model.

---

## Comparison with A-MEM

A-MEM (NeurIPS 2025) focuses on *memory evolution* — when new memories arrive, existing memories' descriptions update. Super Memory focuses on *memory access* — how to reach the right memory through associative paths.

They solve different problems. A-MEM asks "how do we keep memories well-organized?" Super Memory asks "how do we find memories the way humans actually think?"

The versioning approach also differs: A-MEM overwrites on evolution (current state only), Super Memory preserves history (full timeline).

---

## Roadmap

- [ ] FAISS/ChromaDB for scale
- [ ] Coding agent profile (different key strategies for code context)
- [ ] Memory export/import
- [ ] Multi-user support

---

## Author

**donggyun112** — [github.com/donggyun112](https://github.com/donggyun112)

Repository: [donggyun112/super-memory](https://github.com/donggyun112/super-memory) · Issues & PRs welcome.

## License

MIT © [donggyun112](https://github.com/donggyun112)
