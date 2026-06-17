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

Search `"Newton"` → matches `[Newton]`, `[apple]` keys (1-hop) → follows shared `[fruit]` key → reaches strawberry memory (2-hop, score decayed by 0.3×).

**Results include `hop` field** — you always know if a result is direct or associative.

---

## Key Features

| Feature | Super Memory | A-MEM | Mem0 | MemGPT |
|---------|-------------|-------|------|--------|
| Key/Value separation | ✅ N:M | ❌ | ❌ | ❌ |
| Associative multi-hop | ✅ built-in | ❌ | ❌ | ❌ |
| Depth system | ✅ | ❌ | ❌ | partial |
| Memory versioning | ✅ supersede | overwrites | overwrites | ❌ |
| Time decay | ✅ depth-weighted | ❌ | ❌ | ❌ |
| Key types | ✅ concept/name/proper_noun | ❌ | ❌ | ❌ |
| Key merge (IDF) | ✅ | ❌ | ❌ | ❌ |
| Hybrid retrieval (BM25 + dense + RRF) | ✅ | ❌ | partial | ❌ |
| Dual-path dense recall | ✅ key + content | ❌ | ❌ | ❌ |
| Hebbian link learning | ✅ | ❌ | ❌ | ❌ |

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

### Hybrid Retrieval (BM25 + dense, RRF-fused)

Recall is not a single similarity scan. Three signals run in parallel and are fused with **Reciprocal Rank Fusion** (`RRF_K = 60`):

- **BM25 (sparse):** lexical full-text search over memory content (MiniSearch, fuzzy + prefix). Catches exact terms, names, and rare tokens that embeddings blur.
- **Dense Path A (key matching):** query embedding → match keys → follow links → memories. Score = `keySim × IDF × linkWeight`, summed across all matching keys.
- **Dense Path B (content matching):** query embedding → directly compare against memory content embeddings. Finds memories even when they weren't tagged with the right keys.

Sparse and dense rank lists are merged by RRF, then modulated by depth and time before 2-hop expansion. Combining lexical and semantic signals is more robust than either alone.

### Hebbian Link Learning

Recall is a **write**, not just a read. Every recall reshapes the graph:

- Links whose key **actually matched the query** and led to a **returned** memory are **reinforced** (`+0.1`, capped at `3.0`).
- Links explored (matched key) but whose memory was **not returned** are **decayed** (`−0.005`, floored at `0.1`).

Reinforcement is scoped to the keys that *fired* for this query — not to every key of a returned memory. This is the literal Hebbian rule ("fire together, wire together") and it matters: reinforcing a returned memory's unrelated keys would let a stray association grow every time that memory surfaced for a *different* key, slowly polluting the graph. Weights are clamped to `[0.1, 3.0]`, so a hot memory's pull is bounded and the graph can recover from a bad reinforcement via subsequent decay.

Link weights feed directly back into scoring (`keySim × IDF × linkWeight`), so connections that repeatedly co-fire grow stronger and stale ones fade — the graph learns which associations actually matter from access patterns.

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

**Recall algorithm (hybrid, 2-hop):**

Three retrieval signals run in parallel, then get fused and expanded:

1. **BM25 (sparse):** lexical search over memory content (MiniSearch, fuzzy `0.2` + prefix). Keep top 50.
2. **Dense Path A (keys):** embed query → match keys (concept: cosine ≥ threshold; name/proper_noun: substring match → score `1.0`) → take top 10 keys → follow links. Score = `keySim × IDF × linkWeight`, summed across matching keys.
3. **Dense Path B (content):** compare query embedding directly against memory content embeddings (cosine ≥ threshold).
4. **RRF fusion:** merge the BM25 and dense rank lists via `score += 1 / (RRF_K + rank + 1)` (`RRF_K = 60`).
5. **Depth & time modulation:** `score × (0.9 + depth × 0.1) × timeFactor`, where `timeFactor` is a depth-weighted 30-day half-life decay (deep memories decay slower).
6. **2-hop expansion:** for each fused memory, follow *its* keys to associated memories (`× HOP_DECAY(0.3) × IDF × linkWeight`) and traverse explicit `related_to` links (bidirectional, `× HOP_DECAY`).
7. **Hebbian update:** reinforce matched-key links of returned memories (`+0.1`), decay explored-but-unreturned links (`−0.005`).
8. Return ranked results with `hop` field (`1` = direct, `2` = associative).

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

An uncalibrated `LOCAL_EMBEDDING_MODEL` falls back to the BGE profile **and logs a warning** so the miscalibration is never silent.

> **Multilingual note:** cross-lingual *content* matching has a same-language bias (a Korean query scores Korean memories higher regardless of meaning). The reliable cross-lingual path is the **key graph** — tag memories with keys in multiple languages (e.g. `["딸기", "strawberry"]`) so recall hits the key exactly instead of relying on biased content similarity.

---

## MCP Tools

The memory system exposes 10 tools via MCP:

| Tool | Description |
| --- | --- |
| `recall(query, top_k, namespace?, expand?)` | Hybrid search (BM25 + dense key/content, RRF-fused) with 2-hop associative traversal |
| `remember(content, keys, key_types?, namespace?, ttl_seconds?, related_to?)` | Save memory with key concepts and optional type annotations |
| `correct(memory_id, content, keys?, key_types?, related_to?)` | Versioned update — old memory preserved but weakened |
| `related(memory_id)` | Find memories sharing keys (associative exploration) |
| `forget(memory_id)` | Permanently delete |
| `get_conversation(session_id, turn?)` | Load original conversation turns |
| `list_memories(namespace?)` | List all stored memories with keys, depth, access count |
| `remember_batch(items)` | Save multiple memories in one call |
| `cleanup_expired()` | Delete memories whose TTL has expired |
| `memory_stats()` | Get current key/memory/link counts |

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

**Local embeddings (no API key required):**
```json
{
  "mcpServers": {
    "mcp-super-memory": {
      "command": "npx",
      "args": ["-y", "mcp-super-memory"],
      "env": {
        "EMBEDDING_BACKEND": "local"
      }
    }
  }
}
```

### Claude Code

```bash
# OpenAI embeddings
claude mcp add mcp-super-memory -e OPENAI_API_KEY=your-openai-api-key -- npx -y mcp-super-memory

# Local embeddings (no API key required)
claude mcp add mcp-super-memory -e EMBEDDING_BACKEND=local -- npx -y mcp-super-memory
```

### Manual / Development

```bash
git clone https://github.com/donggyun112/mcp-super-memory
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

If `OPENAI_API_KEY` is not set and `EMBEDDING_BACKEND` is unset, the server automatically uses the local `fastembed` backend.
For English-only use or lower local resource usage, set `LOCAL_EMBEDDING_MODEL=fast-bge-base-en-v1.5` or `fast-bge-small-en-v1.5`.

> **Switching backends is safe.** If you change the embedding backend/model, on next startup the graph **auto-migrates** — every key and memory is re-embedded with the new backend while content, links, depth, and access history are preserved (a `graph.json.bak.<dim>d` backup is written first). Disable with `SUPER_MEMORY_AUTO_MIGRATE=false`. Re-embedding via OpenAI incurs one-time API cost proportional to your memory count.

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
├── graph.json          # keys, memories, links
└── conversations/
    └── {session_id}.jsonl   # original conversation turns
```

Set `SUPER_MEMORY_DATA_DIR` to use a different storage directory.

---

## Limitations

- **Linear scan** — suitable for personal use (~10k memories). FAISS/ChromaDB integration planned for larger scale.
- **2-hop max** — deeper associative chains require `related()` tool calls by the agent.
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
tsx test/live-multilingual.ts    # interactive multilingual recall demo
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

## License

MIT
