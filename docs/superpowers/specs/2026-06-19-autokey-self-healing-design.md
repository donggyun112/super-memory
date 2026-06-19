# Auto-Key Self-Healing (heat-driven alias promotion)

**Date:** 2026-06-19
**Status:** Design approved, pending implementation plan
**Branch:** `feat/autokey-self-healing`

## Problem

Memory recall quality currently depends on how well the calling agent coins keys at
`remember()` time. The `remember` tool requires an agent-supplied `keys` array, and
`server.ts` ships a long prompt teaching the agent which keys are "good" vs "bad". If the
agent coins narrow or off-target keys (e.g. `["거주지", "성수동"]`), a later natural-language
query (`"어디 살아?"`) may only match weakly or miss entirely. The quality burden sits on the
agent's judgment.

MemoryOS sidesteps the analogous problem with heat-based promotion that the system runs
automatically, so quality does not gate on the agent. We want the same property for
super-memory, expressed through its existing graph + Hebbian + embedding-clustering design
rather than MemoryOS's tiered architecture.

## Goal

The system learns missing search vocabulary from **real usage** and folds it into the key
space automatically, with **no change to the agent-facing API** and no added agent burden.

Non-goals:
- No write-time LLM key extraction (keeps writes cheap/deterministic).
- No new tiered memory model; reuse `depth`/`access_count` as the existing heat analog.
- Not turning on the dormant write-time `shortKeyMerge` (that path is off on all profiles
  because e5 packs cosines too tightly; this design uses a behavioral signal instead).

## Core idea

The only channel through which *new* search vocabulary enters the system is the **recall
query text** — graph statistics alone can never invent a key the agent never coined. So the
learning signal is: a recall query that matched a key only **weakly** (semantic, not
exact/alias), which the agent then **confirmed** by calling `read_memory` via that key. That
behavioral confirmation is a stronger signal than cosine alone, which is what lets this work
on e5 (whose cosine bands do not separate cleanly).

Accumulate that signal as heat per `(key, query)`. Once it crosses a threshold, fold the
query into the key space — as an **alias** of the matched key when they are clear synonyms,
or as a **new key linked to the confirmed memory** when the query is a distinct facet.

## Architecture

### Components

1. **RecallBuffer** — runtime-only, NOT persisted.
   - Lives on the graph instance; bounded ring buffer (default ~32 recent recalls) with a
     TTL (default ~5 min).
   - Entry: `{ queryText, queryEmbedding, weakKeyIds: Set<keyId>, ts }`.
   - "Weak match" = a key matched during recall that is NOT `exact` / `alias` / `name`-exact,
     i.e. a `semantic` match whose score falls in the band `[minScore floor, strong cap)`.
     Keys already hit by exact/alias matching are good and are never learned from.

2. **aliasCandidates** — persisted on each key.
   - `key.aliasCandidates[normalizedQuery] = { count, lastSeen }`.
   - Incremented on each confirmed weak read; survives restart with the key.

3. **Promotion logic** — fires when `count >= PROMOTE_N`.
   - Gate: `isShortConcept(query)` (existing helper). Long natural-language queries are not
     promoted — the content path already handles them and they would pollute the alias set.
   - Branch on `cosine(query, key.concept)`:
     - `>= keyMerge` (clear synonym) → `_recordKeyAlias(key, query)` (fold into the cluster).
     - mid-band + short + distinct → create a new key from the query and link it to the
       confirmed memory(ies) via the existing `_autoLinkKeys` path.
   - Clear the candidate entry after promotion.

### Hook points (existing code)

- `recall()`: after match computation, classify weak key matches and push a RecallBuffer
  entry.
- `read_memory(memory_id, via_key_id)`: inside the **existing locked phase-3 block** where
  `depth` / `access_count` are already mutated (~`memoryGraph.ts:1444-1452`). Look up the
  RecallBuffer for a recent entry where `via_key_id ∈ weakKeyIds`; attribute to the most
  recent such entry within TTL; increment `(key, query)` heat; promote if threshold reached.
  Reuses the existing concurrency lock — no new lock introduced.

### Data flow (example)

```
recall("어디 살아")
  → key "거주지" matches semantic 0.6 (weak)
  → buffer push { query:"어디 살아", weakKeys:{거주지}, emb }
agent → read_key(거주지) → read_memory(성수동_mem, via_key_id=거주지)
  → buffer entry with 거주지 ∈ weakKeys found
  → 거주지.aliasCandidates["어디 살아"].count++  (→ 1)
... (accumulates to PROMOTE_N across sessions)
  → promote: cosine("어디 살아","거주지") high → add as alias of 거주지
next recall("어디 살아") → 거주지 exact/alias match, score 1.0  ✅ healed
```

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Misattribution (interleaved recalls) | Only TTL-fresh entries where `via_key_id` was actually weak; attribute to the most recent; `PROMOTE_N` makes a one-off misattribution harmless |
| Alias pollution | `isShortConcept` gate + cosine gate + per-key learned-alias cap (default 8) + prune learned aliases that get 0 exact/alias hits over M recalls |
| Garbage queries | Buffer entry requires `>= minScore` (already semantically related); must repeat `PROMOTE_N` times |
| Duplicate alias | `_recordKeyAlias` already dedups case-insensitively |
| New-key explosion | Prefer alias; new key only for mid-band + short + distinct; cap applies |
| Provenance | Learned aliases carry a `learned: true` flag, surfaced in `read_key` output for debug/rollback |

## Configuration (existing `SUPER_MEMORY_*` convention)

- `SUPER_MEMORY_AUTOKEY` — feature on/off. **Default ON**, set `=false` to disable
  (mirrors `SUPER_MEMORY_AUTO_MIGRATE`).
- `SUPER_MEMORY_AUTOKEY_PROMOTE_N` — heat threshold (default 3).
- `SUPER_MEMORY_AUTOKEY_MAX_ALIASES` — per-key learned-alias cap (default 8).
- Weak-band bounds derive from the active threshold profile (`keyRecall` / `minScore`) — no
  new magic numbers where an existing calibrated value fits.

## Testing

Use `__setTestEmbedder` for deterministic vectors.

- **Unit**: buffer TTL/eviction; weak-match classification; increment on confirmed read;
  promotion at N; short-concept gate; alias-vs-new-key branch; alias cap; pruning.
- **Integration**: full `recall → read_memory` loop run N times → promotion → subsequent
  `recall` hits via exact/alias. Assert `depth`/`access_count` are unaffected by the buffer
  logic.
- **Concurrency**: parallel recalls/reads do not double-promote or corrupt state (reuse the
  existing concurrency suite patterns).
- **Regression**: `npm run bench` — promotion must not reduce search quality; paraphrase
  recall should improve.

## Observability

`read_key` already returns `aliases`; learned aliases are tagged `learned: true` so authored
vs learned vocabulary is distinguishable for debugging and rollback.
