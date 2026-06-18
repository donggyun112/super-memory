# Retriever Quality Improvements — Design

**Date:** 2026-06-18
**Status:** Approved (design)
**Area:** `src/embedding.ts`, `src/memoryGraph.ts`, `src/types.ts`, `src/server.ts`, `test/`

## Problem

The associative-memory retriever has several quality gaps:

1. **No BGE-M3 support.** Multilingual recall quality (notably Korean) is weak on the
   current local backends. BGE-M3 (1024-dim, multilingual) would help but is not wired in.
2. **Cannot distinguish "not found."** `recall()` only has a *relative* score floor
   (`minRelScore`, a fraction of the top hit). When every candidate is noise, the top hit
   is also noise, so `topK` results are still returned — there is no absolute relevance gate.
3. **Key-merge distortion (A/B confusion).** Short, semantically-distinct concept keys
   (e.g. `"Agent A"` vs `"Agent B"`) embed at ~0.97+ and get merged by `findOrCreateKey()`,
   conflating distinct entities.
4. **Dedup too strict for well-separated models.** `memoryDedup` is tuned at 0.985 for e5,
   which is correct there (distinct facts ≈0.96 sit dangerously close to paraphrases ≈0.99),
   but is too strict for a better-separated model like BGE-M3 — real duplicates go uncaught
   and the graph fragments.
5. **No contradiction detection.** Conflicting facts about the same subject
   (e.g. "A uses Postgres" vs "A uses Mongo") embed with high similarity but are not
   duplicates. Today they either silently supersede each other or coexist with no relation.

## Critical Constraint: prefix back-compat

`embedLocal()` currently routes **every** local model through the library's
`passageEmbed()` / `queryEmbed()`, which unconditionally prepend the E5-specific
`"passage: "` / `"query: "` prefixes. This means existing **BGE / minilm stored embeddings
and their calibrated thresholds already have these prefixes baked in.** Changing that path
would invalidate stored data and calibration.

**Decision:** Only **bge-m3** gets the correct no-prefix path. Existing BGE / minilm / e5
behavior is left untouched.

## Design

### 1. BGE-M3 compatibility (`embedding.ts`)

fastembed@2.1.0 has **no built-in BGE-M3 enum** — only `CUSTOM`. "Compatibility only":
let a user who supplies BAAI/bge-m3 ONNX files locally point the backend at them.

- **Aliases** → resolve to fastembed `CUSTOM`:
  `bge-m3`, `bgem3`, `baai/bge-m3`, `fast-bge-m3`.
- **`getLocalModel()`**: when the resolved model is `CUSTOM`, pass to `FlagEmbedding.init`:
  - `modelAbsoluteDirPath` from env `LOCAL_EMBEDDING_MODEL_PATH`
  - `modelName` from env `LOCAL_EMBEDDING_MODEL_FILE` (default `model.onnx`)
  - Fail with a clear error if `CUSTOM` is selected but `LOCAL_EMBEDDING_MODEL_PATH` is unset.
- **`localModelFamily()`**: recognize the bge-m3 aliases and return a new `"bgem3"` family.
- **`embedLocal()`**: if family is `bgem3`, call `model.embed([text])` directly for **both**
  query and passage inputs (no `passage:`/`query:` prefix). All other families unchanged.
- **New threshold profile `bgem3`** (1024-dim; distribution closer to BGE than e5).
  Draft values — **calibration-pending, all env-overridable**:

  | field          | value |
  |----------------|-------|
  | keyMerge       | 0.86  |
  | memoryDedup    | 0.94  |
  | keyAutoLink    | 0.62  |
  | keyRecall      | 0.62  |
  | contentRecall  | 0.55  |
  | minScore       | 0.55  | (new — see §2) |
  | contradiction  | 0.88  | (new — see §5) |

  The dimension change (e.g. 768 → 1024) is handled by the existing auto-migration path
  (`_ensureEmbeddingDim` / `_migrateEmbeddings`).

### 2. Absolute score gate (`recall()`, `embedding.ts` profile)

Add an **absolute cosine floor** so "not found" returns empty instead of `topK` noise.

- Extend `ThresholdProfile` with `minScore`. Per-profile default ≈ that profile's
  `contentRecall`. Env override `SUPER_MEMORY_MIN_SCORE`.
- New `recall()` parameter `minScore` (defaults to the profile value).
- During recall, track each candidate memory's **raw max similarity**:
  `rawSim = max(contentSim, max(matched keySim))`, where a `name`/`proper_noun` **exact
  literal** key match counts as `1.0` (literal match ⇒ definitely relevant). `contentSim`
  is already computed for all memories in Dense Path B, so every candidate has a value.
- Just before returning, drop any candidate with `rawSim < minScore`. If nothing clears the
  gate, return `[]`.
- This is computed pre-fusion on raw cosine, so the gate is comparable across queries
  (unlike RRF fused scores). Independent of and composable with the existing relative
  `minRelScore`.
- Default behavior: profiles ship a sensible non-zero `minScore`; callers/env can set `0`
  to disable.

### 3. Key-merge A/B defense (`findOrCreateKey()`)

For `concept`-type keys only (name/proper_noun already match exactly):

- If the concept is **short** — `tokenCount ≤ 2` **or** `length ≤ 15` chars — skip semantic
  merge entirely and reuse an existing key only on **case-insensitive exact string match**;
  otherwise create a new key.
- Long concepts keep the existing semantic merge (`KEY_MERGE_THRESHOLD`).
- Thresholds (`2` tokens / `15` chars) are named constants for easy tuning.

**Tradeoff (accepted):** short near-synonyms (`auth` vs `authentication`) will no longer
auto-merge. This is the deliberate cost of preventing A/B-style over-merge.

### 4. Dedup lowered for bge-m3 only (`_findDuplicate()`)

- `bgem3` profile ships `memoryDedup ≈ 0.94` (vs e5's 0.985). e5 / openai / bge unchanged —
  no data-loss risk for existing graphs.
- Already env-tunable via `SUPER_MEMORY_MEMORY_DEDUP`.
- `_findDuplicate()` logic itself is unchanged; only the threshold value differs by profile.

### 5. Contradiction detection (heuristic) (`add()`, `supersede()`, `recall()`, `getRelated()`, `types.ts`)

Lightweight, no LLM. Surfaces a **signal**, does not enforce.

- **Type:** add `contradicts: string[]` to `Memory`. `load()` defaults it to `[]`;
  persisted in `graph.json` on the memory object (like `links`).
- **Contradiction band** per profile: `[contradiction_floor, memoryDedup)`.
  Draft floors: e5 `0.95`, bgem3 `0.88`, bge `0.85`, minilm `0.85`, openai `0.85`.
  Add `contradiction` to `ThresholdProfile`; env override `SUPER_MEMORY_CONTRADICTION`.
- **Detection (at `add()` / `supersede()`):** when the new memory's best similarity to an
  existing active memory falls in the band **and** the two share ≥1 key, record a
  **bidirectional** `contradicts` link between them. This is the band *just below* dedup:
  similar enough to be about the same subject, distinct enough not to be a paraphrase.
- **No auto-supersede:** both memories are kept; only the conflict is flagged.
- **Surfacing:** `recall()` results and `getRelated()` include `contradicts` so the agent
  can see and resolve conflicts (e.g. via `correct`/`supersede`).
- **Known limitation:** this is similarity-based, not semantic NLI — it will flag some
  non-contradictions (e.g. "A uses Postgres for cache" vs "A uses Postgres for main DB")
  and miss low-similarity contradictions. Accepted: it is a surfaced hint, not a gate.

## Implementation order

1. **BGE-M3 compat + `bgem3` profile** — establishes the threshold foundation.
2. **Absolute score gate** — extend profile + `recall()`.
3. **Key-merge short-key guard** — `findOrCreateKey()`.
4. **Dedup value** — `bgem3` profile only (mostly subsumed by step 1).
5. **Contradiction detection** — type + add/supersede + recall/related surfacing.

## Testing

Follow existing `test/` patterns (tsx node:test):

- **Score gate:** a query with no relevant memory returns `[]`; a relevant query still
  returns hits. `minScore` param and env override respected.
- **Short-key merge:** `"Agent A"` and `"Agent B"` produce two distinct keys; an exact
  repeat of `"Agent A"` reuses the same key; a long concept still semantic-merges.
- **BGE-M3:** family detection, no-prefix embed path, 1024-dim accepted; CUSTOM init errors
  clearly when `LOCAL_EMBEDDING_MODEL_PATH` is missing. (Embedding itself may be mocked /
  skipped if ONNX files are unavailable in CI.)
- **Contradiction:** two memories in the band sharing a key get mutual `contradicts` links
  and both survive (no supersede); a true duplicate (≥dedup) still supersedes; an unrelated
  pair (below band) gets no link.

## Non-goals

- LLM/NLI-based contradiction judging.
- Changing prefix behavior for existing BGE / minilm / e5 backends.
- Auto-resolving contradictions.
- Final empirical threshold values (drafts here; calibrated later, env-overridable meanwhile).
