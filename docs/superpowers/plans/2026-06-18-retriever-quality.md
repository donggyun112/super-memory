# Retriever Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve retriever quality — add BGE-M3 compatibility, an absolute "not-found" score gate, A/B-safe key merging, model-specific dedup, and heuristic contradiction detection.

**Architecture:** Threshold profiles in `src/embedding.ts` drive behavior in `src/memoryGraph.ts`. New per-model fields (`minScore`, `contradiction`) and a `bgem3` profile are added to the profile table; the graph's `recall()`, `findOrCreateKey()`, and `add()`/`supersede()` consume them. Decision logic is extracted into pure exported helpers so it can be unit-tested deterministically; a test-only embedder seam lets recall/add run end-to-end with crafted similarity vectors and no ONNX model.

**Tech Stack:** TypeScript (ESM, NodeNext), `fastembed` (local ONNX), `minisearch` (BM25), `node:test` + `tsx` for tests.

## Global Constraints

- ESM with `.js` import specifiers in TypeScript source (NodeNext). Match existing style.
- Embedding values must stay in `[0,1]` for threshold env overrides (`envThreshold` enforces this).
- **Do NOT change prefix behavior for existing `e5` / `bge` / `minilm` families.** Their stored embeddings and calibrated thresholds assume the current `passage:`/`query:` prefixing. Only `bgem3` gets the no-prefix path.
- `npm test` runs only `test/*.test.ts` (`tsx --test test/*.test.ts`). Fast deterministic tests go in a `*.test.ts` file. Model-dependent quality checks are manual scripts (e.g. `test/nhop.ts`), not part of `npm test`.
- Default behavior must not regress: new `minScore` is opt-in per profile; absent contradiction/contradicts data defaults to empty.
- Thresholds are calibration-pending drafts; every new threshold has an env override.
- Commit after every task. Run `npm run build` (tsc) before committing to catch type errors.

---

### Task 0: Test-only embedder seam

Lets later tasks drive `recall()`/`add()`/`findOrCreateKey()` with deterministic crafted embeddings (no ONNX download).

**Files:**
- Modify: `src/embedding.ts` (add seam to `embedTextAsync`, ~line 199)
- Test: `test/retriever-quality.test.ts` (create)

**Interfaces:**
- Produces: `__setTestEmbedder(fn: (text: string, inputType: "passage" | "query") => number[]): void`, `__clearTestEmbedder(): void`. When a test embedder is set, `embedTextAsync` returns its (synchronous) result wrapped in a resolved promise, bypassing all backends.

- [ ] **Step 1: Write the failing test**

Create `test/retriever-quality.test.ts`:

```typescript
import assert from "node:assert/strict";
import test from "node:test";

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

test("test embedder seam overrides embedTextAsync", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text) => (text === "hello" ? [1, 0] : [0, 1]));
  try {
    assert.deepEqual(await emb.embedTextAsync("hello"), [1, 0]);
    assert.deepEqual(await emb.embedTextAsync("world", "query"), [0, 1]);
  } finally {
    emb.__clearTestEmbedder();
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: FAIL — `emb.__setTestEmbedder is not a function`.

- [ ] **Step 3: Add the seam**

In `src/embedding.ts`, add module-level state below the `_localModel` declaration (near line 157):

```typescript
let _testEmbedder:
  | ((text: string, inputType: EmbeddingInputType) => number[])
  | null = null;

// Test-only seam. Lets tests drive embedding-dependent code paths with crafted
// vectors so cosine similarities are deterministic and no ONNX model is loaded.
// Never set this in production code.
export function __setTestEmbedder(
  fn: (text: string, inputType: EmbeddingInputType) => number[]
): void {
  _testEmbedder = fn;
}

export function __clearTestEmbedder(): void {
  _testEmbedder = null;
}
```

Then make `embedTextAsync` short-circuit. Change its opening (line ~199):

```typescript
export async function embedTextAsync(
  text: string,
  inputType: EmbeddingInputType = "passage"
): Promise<number[]> {
  if (_testEmbedder) return _testEmbedder(text, inputType);
  if (EMBEDDING_BACKEND === "local") {
    return embedLocal(text, inputType);
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/embedding.ts test/retriever-quality.test.ts
git commit -m "test: add test-only embedder seam for deterministic retriever tests"
```

---

### Task 1: BGE-M3 compatibility + profile fields

Adds the `bgem3` family, its threshold profile, the new `minScore`/`contradiction` profile fields (data only; logic lands in Tasks 2 & 5), correct no-prefix embedding for bge-m3, and CUSTOM-model plumbing.

**Files:**
- Modify: `src/embedding.ts` (aliases ~line 21, `ThresholdProfile` ~line 86, `THRESHOLD_PROFILES` ~line 94, family detection ~line 109, `getThresholdProfile` ~line 142, `getLocalModel` ~line 159, `embedLocal` ~line 180)
- Test: `test/retriever-quality.test.ts`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `familyForModel(modelName: string): "e5" | "bge" | "minilm" | "bgem3" | "unknown"` (pure)
  - `usesE5Prefix(family: string): boolean` (pure) — true only for `"e5"`
  - `customModelConfig(): { dir: string; file: string }` (pure; throws if `LOCAL_EMBEDDING_MODEL_PATH` unset)
  - `THRESHOLD_PROFILES` exported (record keyed by family)
  - `ThresholdProfile` gains `minScore: number` and `contradiction: number`

- [ ] **Step 1: Write the failing tests**

Append to `test/retriever-quality.test.ts`:

```typescript
test("familyForModel maps bge-m3 aliases to bgem3", async () => {
  const { familyForModel } = await import("../src/embedding.ts");
  for (const name of ["bge-m3", "bgem3", "BAAI/bge-m3", "fast-bge-m3", "BGE_M3"]) {
    assert.equal(familyForModel(name), "bgem3", name);
  }
  assert.equal(familyForModel("multilingual-e5-large"), "e5");
  assert.equal(familyForModel("bge-small-en-v1.5"), "bge");
  assert.equal(familyForModel("all-minilm-l6-v2"), "minilm");
  assert.equal(familyForModel("nonexistent-model"), "unknown");
});

test("only e5 uses the passage/query prefix", async () => {
  const { usesE5Prefix } = await import("../src/embedding.ts");
  assert.equal(usesE5Prefix("e5"), true);
  assert.equal(usesE5Prefix("bgem3"), false);
  assert.equal(usesE5Prefix("bge"), false);
});

test("bgem3 threshold profile exists with expected fields", async () => {
  const { THRESHOLD_PROFILES } = await import("../src/embedding.ts");
  const p = THRESHOLD_PROFILES.bgem3;
  assert.equal(p.memoryDedup, 0.94);
  assert.equal(p.minScore, 0.55);
  assert.equal(p.contradiction, 0.88);
  // every profile must define the new fields
  for (const fam of ["openai", "e5", "bge", "minilm", "bgem3"]) {
    assert.equal(typeof THRESHOLD_PROFILES[fam].minScore, "number", fam);
    assert.equal(typeof THRESHOLD_PROFILES[fam].contradiction, "number", fam);
  }
});

test("customModelConfig throws a clear error when path is unset", async () => {
  const { customModelConfig } = await import("../src/embedding.ts");
  const saved = process.env.LOCAL_EMBEDDING_MODEL_PATH;
  delete process.env.LOCAL_EMBEDDING_MODEL_PATH;
  try {
    assert.throws(() => customModelConfig(), /LOCAL_EMBEDDING_MODEL_PATH/);
  } finally {
    if (saved !== undefined) process.env.LOCAL_EMBEDDING_MODEL_PATH = saved;
  }
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: FAIL — `familyForModel`/`usesE5Prefix`/`THRESHOLD_PROFILES`/`customModelConfig` undefined, and `bgem3` profile missing.

- [ ] **Step 3: Add aliases**

In `src/embedding.ts`, add to `LOCAL_MODEL_ALIASES` (after the minilm entries, ~line 33):

```typescript
  "bge-m3": "CUSTOM",
  bgem3: "CUSTOM",
  "baai/bge-m3": "CUSTOM",
  "fast-bge-m3": "CUSTOM",
```

- [ ] **Step 4: Extend the ThresholdProfile interface and table**

Change the `ThresholdProfile` interface (~line 86) to add two fields:

```typescript
export interface ThresholdProfile {
  keyMerge: number;
  memoryDedup: number;
  keyAutoLink: number;
  keyRecall: number;
  contentRecall: number;
  // Absolute cosine floor: a recalled memory must have raw similarity (best of
  // content-sim / matched key-sim) >= minScore, else it is dropped. Lets recall
  // return [] for truly-unrelated queries instead of topK noise.
  minScore: number;
  // Lower bound of the contradiction band [contradiction, memoryDedup). New
  // memories whose best similarity to an existing one falls in this band AND
  // share a key are flagged (not deduped) as potential contradictions.
  contradiction: number;
}
```

Export the table (change `const THRESHOLD_PROFILES` to `export const THRESHOLD_PROFILES`) and add the new fields to every row plus the new `bgem3` row (~line 94):

```typescript
export const THRESHOLD_PROFILES: Record<string, ThresholdProfile> = {
  openai: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.5, keyRecall: 0.28, contentRecall: 0.28, minScore: 0.28, contradiction: 0.85 },
  bge: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.6, keyRecall: 0.6, contentRecall: 0.5, minScore: 0.5, contradiction: 0.85 },
  e5: { keyMerge: 0.97, memoryDedup: 0.985, keyAutoLink: 0.93, keyRecall: 0.85, contentRecall: 0.8, minScore: 0.8, contradiction: 0.95 },
  minilm: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.6, keyRecall: 0.5, contentRecall: 0.45, minScore: 0.45, contradiction: 0.85 },
  // bge-m3: multilingual, 1024-dim, well-separated (closer to bge than e5).
  // dedup lowered to 0.94 so real duplicates are caught without fragmenting.
  // Draft values — calibration-pending, env-overridable.
  bgem3: { keyMerge: 0.86, memoryDedup: 0.94, keyAutoLink: 0.62, keyRecall: 0.62, contentRecall: 0.55, minScore: 0.55, contradiction: 0.88 },
};
```

- [ ] **Step 5: Add pure family + prefix + custom-config helpers**

Replace `localModelFamily()` (~line 109) and add helpers. The pure `familyForModel` takes the name as an argument so it is testable without relying on module-level env consts:

```typescript
export function familyForModel(
  modelName: string
): "e5" | "bge" | "minilm" | "bgem3" | "unknown" {
  const normalized = normalizeModelName(modelName);
  if (["bge-m3", "bgem3", "baai/bge-m3", "fast-bge-m3", "bge_m3"].includes(normalized)) {
    return "bgem3";
  }
  const alias = LOCAL_MODEL_ALIASES[normalized];
  if (alias === "MLE5Large") return "e5";
  if (alias === "AllMiniLML6V2") return "minilm";
  if (alias === "BGEBaseENV15" || alias === "BGESmallENV15") return "bge";
  return "unknown";
}

export function usesE5Prefix(family: string): boolean {
  return family === "e5";
}

export function customModelConfig(): { dir: string; file: string } {
  const dir = process.env.LOCAL_EMBEDDING_MODEL_PATH ?? "";
  if (!dir.trim()) {
    throw new Error(
      `LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}" resolves to a CUSTOM model, ` +
        `so LOCAL_EMBEDDING_MODEL_PATH (absolute dir containing model.onnx + tokenizer ` +
        `files) is required. Optionally set LOCAL_EMBEDDING_MODEL_FILE (default model.onnx).`
    );
  }
  return { dir, file: process.env.LOCAL_EMBEDDING_MODEL_FILE ?? "model.onnx" };
}

function localModelFamily(): "e5" | "bge" | "minilm" | "bgem3" {
  const fam = familyForModel(LOCAL_EMBEDDING_MODEL);
  if (fam !== "unknown") return fam;
  // Unknown model: thresholds are NOT calibrated for it. Make the miscalibration
  // loud and point at the env override escape hatch (see original warning).
  if (!_warnedUncalibrated) {
    _warnedUncalibrated = true;
    console.error(
      `[super-memory] WARNING: no calibrated threshold profile for ` +
        `LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}". Falling back to BGE ` +
        `thresholds — the graph may mis-cluster. Override per-threshold with ` +
        `SUPER_MEMORY_KEY_MERGE / _MEMORY_DEDUP / _KEY_AUTOLINK / _KEY_RECALL / _CONTENT_RECALL.`
    );
  }
  return "bge";
}
```

- [ ] **Step 6: Wire env overrides for the new fields**

In `getThresholdProfile()` (~line 142), add the two new fields to the returned object:

```typescript
  return {
    keyMerge: envThreshold("SUPER_MEMORY_KEY_MERGE") ?? base.keyMerge,
    memoryDedup: envThreshold("SUPER_MEMORY_MEMORY_DEDUP") ?? base.memoryDedup,
    keyAutoLink: envThreshold("SUPER_MEMORY_KEY_AUTOLINK") ?? base.keyAutoLink,
    keyRecall: envThreshold("SUPER_MEMORY_KEY_RECALL") ?? base.keyRecall,
    contentRecall: envThreshold("SUPER_MEMORY_CONTENT_RECALL") ?? base.contentRecall,
    minScore: envThreshold("SUPER_MEMORY_MIN_SCORE") ?? base.minScore,
    contradiction: envThreshold("SUPER_MEMORY_CONTRADICTION") ?? base.contradiction,
  };
```

- [ ] **Step 7: No-prefix embedding + CUSTOM init plumbing**

Update `getLocalModel()` (~line 159) so a CUSTOM model is initialized from env:

```typescript
async function getLocalModel() {
  if (!_localModel) {
    try {
      const { FlagEmbedding, EmbeddingModel } = await import("fastembed");
      const model = resolveLocalModel(LOCAL_EMBEDDING_MODEL, EmbeddingModel);
      const initOpts: Record<string, unknown> = {
        model: model as never,
        cacheDir: LOCAL_EMBEDDING_CACHE_DIR,
      };
      if (model === EmbeddingModel.CUSTOM) {
        const { dir, file } = customModelConfig();
        initOpts.modelAbsoluteDirPath = dir;
        initOpts.modelName = file;
      }
      _localModel = await FlagEmbedding.init(initOpts as never);
    } catch (err) {
      throw new Error(
        "Failed to initialize local fastembed model.\n" +
          "Install with: npm install fastembed\n" +
          "Or set OPENAI_API_KEY to use OpenAI embeddings.\n" +
          `Cause: ${errorMessage(err)}`
      );
    }
  }
  return _localModel;
}
```

Update `embedLocal()` (~line 180) so only e5 uses the prefix methods; every other local family (including `bgem3`) embeds raw:

```typescript
async function embedLocal(
  text: string,
  inputType: EmbeddingInputType
): Promise<number[]> {
  const model = await getLocalModel();
  if (!usesE5Prefix(localModelFamily())) {
    // No passage:/query: prefix (bge-m3, custom, etc). Embed the raw text.
    for await (const batch of model.embed([text])) {
      return Array.from(batch[0]) as number[];
    }
    throw new Error("fastembed returned no embeddings");
  }
  if (inputType === "query" && typeof model.queryEmbed === "function") {
    return Array.from(await model.queryEmbed(text)) as number[];
  }
  const gen =
    typeof model.passageEmbed === "function"
      ? model.passageEmbed([text], 256)
      : model.embed([text]);
  for await (const batch of gen) {
    return Array.from(batch[0]) as number[];
  }
  throw new Error("fastembed returned no embeddings");
}
```

> NOTE: this changes `bge`/`minilm` from prefixed to raw embedding. Per Global Constraints we must NOT change their behavior. Therefore the `!usesE5Prefix(...)` branch must reproduce the OLD behavior for `bge`/`minilm` (which previously went through `passageEmbed`/`queryEmbed`). **Correction:** keep the prefix path for everything EXCEPT `bgem3`. Use the helper below instead.

Replace the guard with a bgem3-specific check so existing families are untouched:

```typescript
async function embedLocal(
  text: string,
  inputType: EmbeddingInputType
): Promise<number[]> {
  const model = await getLocalModel();
  const noPrefix = localModelFamily() === "bgem3";
  if (noPrefix) {
    for await (const batch of model.embed([text])) {
      return Array.from(batch[0]) as number[];
    }
    throw new Error("fastembed returned no embeddings");
  }
  if (inputType === "query" && typeof model.queryEmbed === "function") {
    return Array.from(await model.queryEmbed(text)) as number[];
  }
  const gen =
    typeof model.passageEmbed === "function"
      ? model.passageEmbed([text], 256)
      : model.embed([text]);
  for await (const batch of gen) {
    return Array.from(batch[0]) as number[];
  }
  throw new Error("fastembed returned no embeddings");
}
```

(`usesE5Prefix` remains exported for its unit test and documents intent; `embedLocal` uses the `=== "bgem3"` check to guarantee zero behavior change for `bge`/`minilm`/`e5`.)

- [ ] **Step 8: Run tests + build**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: PASS (all Task 0 + Task 1 tests).
Run: `npm run build`
Expected: no type errors.

- [ ] **Step 9: Commit**

```bash
git add src/embedding.ts test/retriever-quality.test.ts
git commit -m "feat: bge-m3 compatibility (no-prefix embed, CUSTOM plumbing, threshold profile)"
```

---

### Task 2: Absolute score gate in recall

Drops recalled memories whose raw similarity is below `minScore`; returns `[]` when nothing clears the gate. Realizes spec §2.

**Files:**
- Modify: `src/memoryGraph.ts` (thresholds block ~line 16-22; `recall()` signature ~line 650, raw-sim tracking in Dense Paths A/B ~line 690-752, filter before return ~line 856-865)
- Modify: `src/server.ts` (recall tool schema ~line 130, recall call ~line 283-290)
- Test: `test/retriever-quality.test.ts`

**Interfaces:**
- Consumes: `THRESHOLD_PROFILES`, test embedder seam (Task 0); `getThresholdProfile().minScore` (Task 1).
- Produces:
  - `passesAbsoluteGate(rawSim: number, minScore: number): boolean` (pure, exported from `memoryGraph.ts`)
  - `recall(query, topK?, namespace?, expand?, maxHops?, minRelScore?, minScore?)` — new 7th param `minScore` (default = profile `minScore`).

- [ ] **Step 1: Write the failing tests**

Append to `test/retriever-quality.test.ts`. The seam returns 2-D unit vectors; cosine equals the dot product, so we control similarity exactly.

```typescript
test("passesAbsoluteGate compares raw similarity to floor", async () => {
  const { passesAbsoluteGate } = await import("../src/memoryGraph.ts");
  assert.equal(passesAbsoluteGate(0.6, 0.55), true);
  assert.equal(passesAbsoluteGate(0.55, 0.55), true);
  assert.equal(passesAbsoluteGate(0.4, 0.55), false);
});

test("recall returns [] when nothing clears the absolute gate", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-gate-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;

  const emb = await import("../src/embedding.ts");
  // memory content embeds to [1,0]; the noise query embeds orthogonally to [0,1].
  emb.__setTestEmbedder((text) =>
    text === "노이즈쿼리" ? [0, 1] : [1, 0]
  );
  t.after(() => emb.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?gate=1`);
  const g = new MemoryGraph();
  await g.load();
  await g.add("사용자는 커피를 좋아한다", ["커피"]);

  // Relevant query (embeds to [1,0], cos=1 with content) clears the gate.
  const hit = (await g.recall("커피", 5)) as any[];
  assert.ok(hit.length >= 1, "relevant query should return results");

  // Orthogonal noise query (cos=0) is below any positive minScore -> [].
  const miss = (await g.recall("노이즈쿼리", 5, null, false, 2, 0, 0.5)) as any[];
  assert.equal(miss.length, 0, "noise query should return nothing");
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: FAIL — `passesAbsoluteGate` undefined; noise query currently returns BM25/loose results instead of `[]`.

- [ ] **Step 3: Add the pure gate helper + threshold constant**

In `src/memoryGraph.ts`, after the existing threshold constants (~line 22) add:

```typescript
const MIN_SCORE_THRESHOLD = _THRESHOLDS.minScore;
```

Near the other top-level pure helpers (e.g. after `batchCosineSim`, ~line 55) add:

```typescript
// A recalled memory must have raw similarity (best of content-sim and matched
// key-sim; exact name/proper-noun matches count as 1.0) at least minScore. This
// is computed on raw cosine BEFORE RRF fusion, so it is comparable across queries
// — unlike fused scores. minScore = 0 disables the gate.
export function passesAbsoluteGate(rawSim: number, minScore: number): boolean {
  return minScore <= 0 || rawSim >= minScore;
}
```

- [ ] **Step 4: Track raw similarity per memory during recall**

In `recall()` (~line 650), extend the signature and clamp:

```typescript
  async recall(
    query: string,
    topK = 5,
    namespace?: string | null,
    expand = false,
    maxHops = 2,
    minRelScore = 0,
    minScore = MIN_SCORE_THRESHOLD
  ): Promise<object[]> {
```

Add `minScore` clamping next to the existing `minRelScore` clamp (~line 663):

```typescript
    // Absolute cosine floor in [0,1]. 0 disables the gate.
    minScore = Math.max(0, Math.min(1, minScore));
```

Inside `_lock.runExclusive`, declare a raw-sim map alongside `memMatchedKeys`/`memHop` (~line 671):

```typescript
      const memRawSim: Record<string, number> = {};
      const bumpRaw = (mid: string, sim: number) => {
        if (sim > (memRawSim[mid] ?? -Infinity)) memRawSim[mid] = sim;
      };
```

In Dense Path A, inside the `for (const memId of ...)` loop where `denseScores` is updated (~line 718-726), record raw key similarity. For exact name/proper-noun matches `keySim` is already `1.0` (set at line 708), so this also covers literal matches:

```typescript
          denseScores[memId] = (denseScores[memId] ?? 0) + score;
          bumpRaw(memId, keySim);
```

In Dense Path B, where `cSim >= CONTENT_RECALL_THRESHOLD` (~line 740), record content similarity:

```typescript
          if (cSim >= CONTENT_RECALL_THRESHOLD) {
            bumpRaw(mid, cSim);
            const contentScore = cSim * 0.8;
```

- [ ] **Step 5: Apply the gate before slicing to topK**

Replace the floor/ranked block (~line 862-865) so the absolute gate runs alongside the relative floor:

```typescript
      const floor = sorted.length ? sorted[0][1] * minRelScore : 0;
      const ranked = sorted
        .filter(([, score]) => score >= floor)
        .filter(([mid]) => passesAbsoluteGate(memRawSim[mid] ?? 0, minScore))
        .slice(0, actualTopK);
```

> Memories that entered `memScores` only via BM25 or hop-traversal (no dense raw sim) have `memRawSim[mid] === undefined → 0`, so a positive `minScore` correctly drops them unless they also cleared a dense path. This is intended: the gate asks "did dense retrieval actually find this relevant?".

- [ ] **Step 6: Surface the param in the MCP tool**

In `src/server.ts`, add to the recall `inputSchema.properties` (~line 130):

```typescript
          min_rel_score: { type: "number" },
          min_score: { type: "number" },
```

And pass it in the `case "recall"` call (~line 289). Use `undefined` when omitted so the profile default applies:

```typescript
        const results = await graph.recall(
          a.query as string,
          typeof a.top_k === "number" ? a.top_k : 5,
          typeof a.namespace === "string" ? a.namespace : null,
          typeof a.expand === "boolean" ? a.expand : false,
          typeof a.hops === "number" ? a.hops : 2,
          typeof a.min_rel_score === "number" ? a.min_rel_score : 0,
          typeof a.min_score === "number" ? a.min_score : undefined
        );
```

> `recall(..., undefined)` uses the parameter default (`MIN_SCORE_THRESHOLD`), preserving the per-profile gate.

- [ ] **Step 7: Run tests + build**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: PASS.
Run: `npm run build`
Expected: no type errors.

- [ ] **Step 8: Commit**

```bash
git add src/memoryGraph.ts src/server.ts test/retriever-quality.test.ts
git commit -m "feat: absolute score gate so recall returns [] for unrelated queries"
```

---

### Task 3: A/B-safe short-key merging

Short concept keys merge only on exact string match, preventing "Agent A" / "Agent B" conflation. Realizes spec §3.

**Files:**
- Modify: `src/embedding.ts` (add `isShortConcept`)
- Modify: `src/memoryGraph.ts` (`findOrCreateKey()` ~line 440-479; import `isShortConcept`)
- Test: `test/retriever-quality.test.ts`

**Interfaces:**
- Consumes: test embedder seam (Task 0).
- Produces: `isShortConcept(concept: string): boolean` (pure, exported from `embedding.ts`). A concept is "short" when it has ≤ 2 whitespace tokens OR ≤ 15 characters (after trim).

- [ ] **Step 1: Write the failing tests**

Append to `test/retriever-quality.test.ts`:

```typescript
test("isShortConcept flags short or few-token concepts", async () => {
  const { isShortConcept } = await import("../src/embedding.ts");
  assert.equal(isShortConcept("Agent A"), true);   // 2 tokens
  assert.equal(isShortConcept("auth"), true);       // short
  assert.equal(isShortConcept("Agent B"), true);
  assert.equal(isShortConcept("distributed consensus protocol design"), false); // long + many tokens
});

test("short concept keys merge only on exact match", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-key-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;

  const emb = await import("../src/embedding.ts");
  // "Agent A" and "Agent B" embed nearly identically (cos ~0.9998) — high enough
  // that semantic merge WOULD merge them. The short-key guard must keep them apart.
  emb.__setTestEmbedder((text) =>
    text === "Agent A" ? [1, 0.02] : text === "Agent B" ? [1, 0.0] : [0.3, 0.95]
  );
  t.after(() => emb.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?key=1`);
  const g = new MemoryGraph();
  await g.load();

  const a1 = await g.findOrCreateKey("Agent A");
  const a2 = await g.findOrCreateKey("Agent A"); // exact repeat -> same key
  const b = await g.findOrCreateKey("Agent B");  // distinct short key -> new key

  assert.equal(a1, a2, "exact short repeat reuses the key");
  assert.notEqual(a1, b, "Agent A and Agent B must NOT merge");
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: FAIL — `isShortConcept` undefined; without the guard `Agent B` merges into `Agent A` (`a1 === b`).

- [ ] **Step 3: Add the pure helper**

In `src/embedding.ts`, near the other exported helpers, add:

```typescript
// Short concept keys (e.g. "Agent A" / "Agent B") embed almost identically and
// would over-merge under semantic matching, conflating distinct entities. Treat
// them like proper nouns: merge only on exact string match. Tunable.
export const SHORT_CONCEPT_MAX_TOKENS = 2;
export const SHORT_CONCEPT_MAX_CHARS = 15;

export function isShortConcept(concept: string): boolean {
  const trimmed = concept.trim();
  const tokens = trimmed.split(/\s+/).filter(Boolean);
  return tokens.length <= SHORT_CONCEPT_MAX_TOKENS || trimmed.length <= SHORT_CONCEPT_MAX_CHARS;
}
```

- [ ] **Step 4: Guard short concepts in findOrCreateKey**

In `src/memoryGraph.ts`, add `isShortConcept` to the embedding import (~line 7):

```typescript
import { embedTextAsync, EMBEDDING_BACKEND, getThresholdProfile, isShortConcept } from "./embedding.js";
```

In `findOrCreateKey()` for the concept branch (~line 457, before `const emb = await embedTextAsync(concept)`), add an exact-match-only fast path for short concepts:

```typescript
    // Short concept keys merge only on exact (case-insensitive) string match, so
    // near-identical-but-distinct short keys ("Agent A" vs "Agent B") stay separate.
    if (isShortConcept(concept)) {
      const lc = concept.toLowerCase();
      for (const [kid, k] of Object.entries(this.keys)) {
        if (k.key_type === "concept" && k.concept.toLowerCase() === lc) return kid;
      }
      const kid = uid();
      this.keys[kid] = {
        id: kid,
        concept,
        embedding: await embedTextAsync(concept),
        key_type: "concept",
      };
      return kid;
    }

    const emb = await embedTextAsync(concept);
```

(Long concepts fall through to the unchanged semantic-merge block below.)

- [ ] **Step 5: Run tests + build**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: PASS.
Run: `npm run build`
Expected: no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/embedding.ts src/memoryGraph.ts test/retriever-quality.test.ts
git commit -m "feat: short concept keys merge only on exact match (A/B defense)"
```

---

### Task 4: Contradiction detection (heuristic)

Flags conflicting-but-distinct memories with bidirectional `contradicts` links and surfaces them in `recall()`/`getRelated()`. Also verifies the bgem3 dedup boundary (spec §4) since the contradiction band sits just below `memoryDedup`. Realizes spec §5 (+ §4 verification).

**Files:**
- Modify: `src/types.ts` (`Memory.contradicts`)
- Modify: `src/embedding.ts` (add `inContradictionBand`)
- Modify: `src/memoryGraph.ts` (thresholds ~line 22; `load()` defaults ~line 362-372 and persistence is automatic via the memory object; `add()` detection ~line 513-551; `supersede()` detection ~line 597-642; `recall()` output ~line 872-887; `getRelated()` output ~line 925-1001)
- Test: `test/retriever-quality.test.ts`

**Interfaces:**
- Consumes: `THRESHOLD_PROFILES` / `getThresholdProfile()` (`contradiction`, `memoryDedup`); test embedder seam.
- Produces:
  - `inContradictionBand(sim: number, floor: number, dedup: number): boolean` (pure, exported from `embedding.ts`) — true when `floor <= sim < dedup`.
  - `Memory.contradicts: string[]`.
  - `recall()` result objects and `getRelated()` entries include a `contradicts: string[]` field.

- [ ] **Step 1: Write the failing tests**

Append to `test/retriever-quality.test.ts`:

```typescript
test("inContradictionBand is [floor, dedup)", async () => {
  const { inContradictionBand } = await import("../src/embedding.ts");
  assert.equal(inContradictionBand(0.9, 0.88, 0.94), true);
  assert.equal(inContradictionBand(0.88, 0.88, 0.94), true);  // inclusive floor
  assert.equal(inContradictionBand(0.94, 0.88, 0.94), false); // exclusive dedup (=> duplicate)
  assert.equal(inContradictionBand(0.5, 0.88, 0.94), false);  // below band
});

test("conflicting memories sharing a key get mutual contradicts links", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-contra-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;

  const emb = await import("../src/embedding.ts");
  // Two facts about "프로젝트A": Postgres vs Mongo. Craft cos ~0.92 — inside the
  // bgem3 band [0.88, 0.94), so NOT a duplicate but flagged as a contradiction.
  // (cos([1,0],[0.92, 0.392]) = 0.92)
  emb.__setTestEmbedder((text) =>
    text.includes("Postgres") ? [1, 0] : text.includes("Mongo") ? [0.92, 0.392] : [0, 1]
  );
  t.after(() => emb.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?contra=1`);
  const g = new MemoryGraph();
  await g.load();

  const [first] = await g.add("프로젝트A는 Postgres를 쓴다", ["프로젝트A"]);
  const [second, wasDup] = await g.add("프로젝트A는 Mongo를 쓴다", ["프로젝트A"]);

  assert.equal(wasDup, false, "should NOT be treated as a duplicate");
  assert.ok(second in g.memories && first in g.memories, "both memories survive");
  assert.ok(g.memories[second].contradicts.includes(first), "new -> old contradicts link");
  assert.ok(g.memories[first].contradicts.includes(second), "old -> new contradicts link");

  const results = (await g.recall("프로젝트A Postgres", 5)) as any[];
  const r = results.find((x) => x.id === first);
  assert.ok(r && Array.isArray(r.contradicts) && r.contradicts.includes(second),
    "recall surfaces contradicts");
});

test("load defaults contradicts to []", async (t) => {
  const { mkdtemp, rm, writeFile } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-load-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;
  // Pre-existing memory WITHOUT a contradicts field (older schema).
  await writeFile(
    join(dataDir, "graph.json"),
    JSON.stringify({
      keys: {},
      memories: { m1: { id: "m1", content: "x", embedding: [1, 0], created_at: 0 } },
      links: [],
    }),
    "utf-8"
  );
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?load=1`);
  const g = new MemoryGraph();
  await g.load();
  assert.deepEqual(g.memories.m1.contradicts, []);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: FAIL — `inContradictionBand` undefined; `contradicts` missing on memories; recall output lacks `contradicts`.

- [ ] **Step 3: Add `contradicts` to the type**

In `src/types.ts`, add to `Memory` (after `links`):

```typescript
  links: string[];
  contradicts: string[];
```

- [ ] **Step 4: Add the pure band helper**

In `src/embedding.ts`, near `isShortConcept`, add:

```typescript
// Contradiction band: similar enough to be about the same subject, but below the
// dedup threshold so it is a distinct (possibly conflicting) fact, not a paraphrase.
export function inContradictionBand(sim: number, floor: number, dedup: number): boolean {
  return sim >= floor && sim < dedup;
}
```

- [ ] **Step 5: Threshold constant + load default**

In `src/memoryGraph.ts`, add the constant after the others (~line 22):

```typescript
const CONTRADICTION_THRESHOLD = _THRESHOLDS.contradiction;
```

Add `inContradictionBand` to the embedding import (~line 7):

```typescript
import { embedTextAsync, EMBEDDING_BACKEND, getThresholdProfile, isShortConcept, inContradictionBand } from "./embedding.js";
```

In `load()`, add `contradicts` to the `defaults` object (~line 362) and normalize it like `links` (~line 373):

```typescript
      const defaults = {
        depth: 0.0,
        access_count: 0,
        last_accessed: 0,
        namespace: "default",
        ttl: null,
        links: [] as string[],
        contradicts: [] as string[],
        source: null,
        supersedes: null,
      };
      const mem: Memory = { ...defaults, ...m };
      mem.links = Array.isArray(mem.links)
        ? mem.links.filter((linkedId): linkedId is string => typeof linkedId === "string")
        : [];
      mem.contradicts = Array.isArray(mem.contradicts)
        ? mem.contradicts.filter((id): id is string => typeof id === "string" && id in (raw.memories ?? {}))
        : [];
```

> `contradicts` persists automatically: it's a field on the `Memory` object, and `save()` serializes the whole `this.memories` map.

- [ ] **Step 6: Add a contradiction-detection helper and call it from add()**

In `src/memoryGraph.ts`, add a private method to `MemoryGraph` (e.g. after `_findDuplicate`, ~line 314):

```typescript
  // Find an existing active memory that CONTRADICTS the new one: best similarity
  // sits in the contradiction band [CONTRADICTION_THRESHOLD, MEMORY_DEDUP_THRESHOLD)
  // AND the two share at least one key (same subject). Heuristic — surfaces a
  // signal, does not block or supersede. Returns the conflicting memory id or null.
  private _findContradiction(embedding: number[], keyIds: Iterable<string>): string | null {
    const newKeys = new Set(keyIds);
    if (newKeys.size === 0) return null;
    let bestId: string | null = null;
    let bestSim = -Infinity;
    for (const [mid, mem] of Object.entries(this.memories)) {
      if (mid in this._supersededBy) continue;
      const sim = cosineSim(embedding, mem.embedding);
      if (!inContradictionBand(sim, CONTRADICTION_THRESHOLD, MEMORY_DEDUP_THRESHOLD)) continue;
      const shares = [...(this._memToKeys[mid]?.keys() ?? [])].some((kid) => newKeys.has(kid));
      if (shares && sim > bestSim) {
        bestSim = sim;
        bestId = mid;
      }
    }
    return bestId;
  }
```

In `add()`, after the memory and its keys are linked (right after the `_autoLinkKeys(mid, embedding)` call, ~line 548), detect and record contradictions. The new memory's key ids are exactly those linked in the preceding loop, so collect them:

```typescript
      const linkedKeyIds = [...(this._memToKeys[mid]?.keys() ?? [])];
      this._autoLinkKeys(mid, embedding);
      const conflictId = this._findContradiction(embedding, linkedKeyIds);
      if (conflictId && conflictId !== mid) {
        if (!this.memories[mid].contradicts.includes(conflictId)) {
          this.memories[mid].contradicts.push(conflictId);
        }
        if (!this.memories[conflictId].contradicts.includes(mid)) {
          this.memories[conflictId].contradicts.push(mid);
        }
      }
      this._bm25.add({ id: mid, content });
      await this.save();
```

> Place the `linkedKeyIds` capture BEFORE `_autoLinkKeys` so contradiction "shared key" is judged on the explicit keys the caller supplied, not on fuzzy auto-links (which would inflate false positives). Detection runs against `embedding` (already computed) using the explicit keys.

Also ensure the new memory object in `add()` initializes `contradicts: []` (in the `this.memories[mid] = { ... }` literal, ~line 522, after `links: validLinks`):

```typescript
        links: validLinks,
        contradicts: [],
```

- [ ] **Step 7: Initialize contradicts in supersede() and detect there too**

In `supersede()`, add `contradicts: []` to the new memory literal (~line 597, after `links: validLinks`):

```typescript
        links: validLinks,
        contradicts: [],
```

After `this._autoLinkKeys(mid, newEmbedding)` (~line 641), add the same detection (using the keys just linked to `mid`):

```typescript
      const supKeyIds = [...(this._memToKeys[mid]?.keys() ?? [])];
      const supConflict = this._findContradiction(newEmbedding, supKeyIds);
      if (supConflict && supConflict !== mid && supConflict !== oldId) {
        if (!this.memories[mid].contradicts.includes(supConflict)) {
          this.memories[mid].contradicts.push(supConflict);
        }
        if (!this.memories[supConflict].contradicts.includes(mid)) {
          this.memories[supConflict].contradicts.push(mid);
        }
      }
```

> `_autoLinkKeys` runs before this in `supersede()`, so `supKeyIds` may include auto-linked keys. That is acceptable for superseded content (it inherits keys); the band + shared-key requirement still gates false positives.

- [ ] **Step 8: Surface contradicts in recall() and getRelated()**

In `recall()`, add `contradicts` to the pushed result object (~line 872, after `links: mem.links`):

```typescript
          links: mem.links,
          contradicts: mem.contradicts ?? [],
```

In `getRelated()`, when building each `related[...]` entry, contradictions are about a specific memory; expose the source memory's contradictions on the returned related entries by adding a `contradicts` field. Add to the entry shape and each literal. Update the type annotation (~line 928) to include `contradicts: string[]` and set it in every `related[mid] = { ... }` literal:

```typescript
            link_type: "key",
            depth: Math.round(mem.depth * 1000) / 1000,
            contradicts: mem.contradicts ?? [],
```

Apply the same `contradicts: mem.contradicts ?? []` line to the explicit-link and reverse-link literals in `getRelated()` (the two other `related[...] = { ... }` blocks).

- [ ] **Step 9: Clean up contradicts on delete**

In `_removeMemoryReferences()` (~line 189), also strip deleted ids from `contradicts` so dangling references don't accumulate:

```typescript
  private _removeMemoryReferences(memoryIds: Iterable<string>): void {
    const deleted = new Set(memoryIds);
    for (const [mid, mem] of Object.entries(this.memories)) {
      mem.links = this._validMemoryLinks(mem.links, mid).filter(
        (linkedId) => !deleted.has(linkedId)
      );
      if (Array.isArray(mem.contradicts)) {
        mem.contradicts = mem.contradicts.filter((id) => id in this.memories && !deleted.has(id));
      }
    }
  }
```

- [ ] **Step 10: Run tests + build**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: PASS (all tests across Tasks 0–4).
Run: `npm run build`
Expected: no type errors.
Run: `npm test`
Expected: existing `memoryGraph.test.ts` suite still PASS (the `memory()` helper there omits `contradicts`; verify `load()`/`delete()` tolerate its absence — they operate on injected objects, and the production code uses `mem.contradicts ?? []` / `Array.isArray` guards, so this is safe. If any existing test constructs a `Memory` and TypeScript flags the missing `contradicts` field, add `contradicts: []` to that test's `memory()` helper.)

- [ ] **Step 11: Commit**

```bash
git add src/types.ts src/embedding.ts src/memoryGraph.ts test/retriever-quality.test.ts
git commit -m "feat: heuristic contradiction detection with bidirectional contradicts links"
```

---

### Task 5: Docs + manual multilingual integration check

Document the new env vars and the bge-m3 setup; add a manual integration script for real-model confidence (mirrors `test/nhop.ts`).

**Files:**
- Modify: `README.md` (env var table / configuration section)
- Create: `test/retriever-quality.live.ts` (manual, not in `npm test`)

**Interfaces:**
- Consumes: real local backend (run manually with bge-m3 ONNX files present, or e5 default).

- [ ] **Step 1: Document new env vars in README**

In `README.md`, find the environment/configuration section and add entries for:
- `LOCAL_EMBEDDING_MODEL=bge-m3` (and aliases) — selects BGE-M3 via fastembed CUSTOM.
- `LOCAL_EMBEDDING_MODEL_PATH` — absolute dir containing `model.onnx` + tokenizer files (required for bge-m3/CUSTOM).
- `LOCAL_EMBEDDING_MODEL_FILE` — ONNX filename (default `model.onnx`).
- `SUPER_MEMORY_MIN_SCORE` — absolute cosine floor for recall (0 disables).
- `SUPER_MEMORY_CONTRADICTION` — contradiction-band lower bound.
- Note: bge-m3 uses no `passage:`/`query:` prefix; existing BGE/minilm/e5 behavior is unchanged.
- Note: recall now returns `[]` when nothing clears the gate, and result/related objects include a `contradicts` array.

- [ ] **Step 2: Create the manual integration script**

Create `test/retriever-quality.live.ts` (run with `EMBEDDING_BACKEND=local npx tsx test/retriever-quality.live.ts`):

```typescript
// Manual integration check (NOT part of `npm test`). Uses the real local model.
// Run: EMBEDDING_BACKEND=local npx tsx test/retriever-quality.live.ts
// For bge-m3: also set LOCAL_EMBEDDING_MODEL=bge-m3 LOCAL_EMBEDDING_MODEL_PATH=/abs/dir
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
const dataDir = await mkdtemp(join(tmpdir(), "sm-rq-live-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");
const g = new MemoryGraph();
await g.load();

let pass = 0, fail = 0; const fails: string[] = [];
const check = (name: string, cond: boolean, detail = "") => {
  if (cond) { pass++; console.log(`  ✅ ${name}`); }
  else { fail++; fails.push(name); console.log(`  ❌ ${name}  ${detail}`); }
};

console.log(`model=${LOCAL_EMBEDDING_MODEL}`);
await g.add("사용자는 커피를 좋아한다", ["커피", "음료"]);
await g.add("프로젝트A는 Postgres를 쓴다", ["프로젝트A", "데이터베이스"]);

const rel = (await g.recall("커피", 5)) as any[];
check("relevant query returns hits", rel.length >= 1, `${rel.length}`);

const noise = (await g.recall("양자역학 우주론 블랙홀", 5, null, false, 2, 0)) as any[];
check("unrelated query returns nothing (gate)", noise.length === 0, `${noise.length}`);

console.log(`\nRESULT: ${pass} passed, ${fail} failed`);
if (fail) console.log(`FAILED: ${fails.join(", ")}`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
```

- [ ] **Step 3: Run the manual script (with default e5 model)**

Run: `EMBEDDING_BACKEND=local npx tsx test/retriever-quality.live.ts`
Expected: both checks PASS. (The gate uses the e5 profile's `minScore=0.8` here; if the "unrelated" check is flaky on the live model, note the observed scores — thresholds are calibration-pending and env-tunable.)

- [ ] **Step 4: Commit**

```bash
git add README.md test/retriever-quality.live.ts
git commit -m "docs: document bge-m3 setup, score gate, and contradiction env vars"
```

---

## Self-Review

**Spec coverage:**
- §1 BGE-M3 compat → Task 1 (aliases, CUSTOM plumbing, no-prefix embed, `bgem3` profile). ✓
- §2 Absolute score gate → Task 2 (`minScore` profile field in Task 1; `passesAbsoluteGate` + recall wiring + MCP param in Task 2). ✓
- §3 Key-merge A/B defense → Task 3 (`isShortConcept` + exact-only short-key path). ✓
- §4 Dedup lowered for bge-m3 only → Task 1 (`bgem3.memoryDedup = 0.94`); boundary exercised by Task 4's contradiction band tests (`< 0.94` flagged, `≥ 0.94` deduped). ✓
- §5 Contradiction detection → Task 4 (type field, band helper, add/supersede detection, recall/related surfacing, delete cleanup). ✓
- Critical prefix back-compat constraint → Task 1 Step 7 uses `=== "bgem3"` so e5/bge/minilm are untouched. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. Task 1 Step 7 contains an explicit self-correction (the first guard would have changed bge/minilm behavior; the corrected `=== "bgem3"` version is the one to implement). ✓

**Type consistency:** `familyForModel`/`usesE5Prefix`/`customModelConfig`/`isShortConcept`/`inContradictionBand`/`passesAbsoluteGate` names are used identically where consumed. `Memory.contradicts: string[]` is initialized in `add()` and `supersede()` literals, defaulted in `load()`, surfaced in `recall()`/`getRelated()`, and cleaned in `_removeMemoryReferences()`. `recall()`'s new 7th param `minScore` matches the server call site. ✓
