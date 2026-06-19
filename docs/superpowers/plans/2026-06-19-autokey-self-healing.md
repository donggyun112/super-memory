# Auto-Key Self-Healing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the system learn missing search vocabulary from real usage — when a recall query matches a key only weakly (semantic, not literal) and the agent then confirms by reading a memory via that key, accumulate heat and fold the query into the key space — with zero agent-facing API change.

**Architecture:** A runtime-only `RecallBuffer` records recent weak (semantic) key matches per recall. On `readMemory(memoryId, viaKeyId)`, if `viaKeyId` was a recent weak match, increment a persisted per-`(key, query)` heat counter; once it crosses a threshold, promote the query to an **alias** of the matched key (high cosine) or a **new key linked to the confirmed memory** (mid cosine). All new state mutation happens inside the existing `_lock`. Decision logic and the buffer live in a new isolated `src/autokey.ts`; `memoryGraph.ts` only wires the two hook points.

**Tech Stack:** TypeScript (ESM, `"type": "module"`), Node ≥20, `node:test` + `node:assert/strict` run via `tsx --test`, existing `__setTestEmbedder` seam in `src/embedding.ts`.

## Global Constraints

- Node ≥20; ESM modules; all source imports use the `.js` extension for local files (e.g. `import { isShortConcept } from "./embedding.js"`) even though the file is `.ts`.
- Tests run with `npm test` (= `tsx --test test/*.test.ts`). Set `process.env.EMBEDDING_BACKEND = "local"` and `process.env.LOCAL_EMBEDDING_MODEL = "bge-m3"` at the top of any test file before importing modules, matching `test/retriever-quality.test.ts`.
- Integration tests that construct a real `MemoryGraph` MUST set `process.env.SUPER_MEMORY_DATA_DIR` to a unique temp dir before importing `memoryGraph.ts`, so they never touch `~/.super-memory`.
- No new runtime dependencies. Reuse existing helpers: `cosineSim` (memoryGraph.ts), `isShortConcept` (embedding.ts), `_recordKeyAlias`, `findOrCreateKey`, `_link`.
- Do NOT use `Date.now()` inside pure decision functions; inject a clock where time matters (the `RecallBuffer`). `memoryGraph.ts` may use `Date.now() / 1000` as it already does throughout.
- No Claude/AI attribution in commit messages.
- Per-key learned-alias growth is bounded by `AUTOKEY_MAX_ALIASES` (default 8). The buffer is bounded by capacity (32) + TTL (300s). Never introduce an unbounded collection.

---

## File Structure

- **Create** `src/autokey.ts` — config constants (env-driven), the `RecallBuffer` class, and the pure `decidePromotion` decision function. One responsibility: the auto-key learning policy + its ephemeral state, isolated from the graph so it is unit-testable without embeddings or disk.
- **Modify** `src/types.ts` — extend the `Key` interface with optional `aliasCandidates` (persisted heat ledger) and `learnedAliases` (provenance + prune metadata).
- **Modify** `src/memoryGraph.ts` — instantiate the buffer; push weak matches in `searchKeys`; confirm + promote in `readMemory` via a new private `_maybeLearnAlias`; restore the new Key fields in `load`; surface learned aliases in `_keyView`; prune stale learned aliases in `cleanupExpired`.
- **Create** `test/autokey.test.ts` — unit tests for `RecallBuffer` and `decidePromotion` (no graph, no embeddings).
- **Create** `test/autokey-integration.test.ts` — full `add → searchKeys → readMemory` loop with `__setTestEmbedder`, asserting promotion after N confirmations and that a subsequent recall hits via alias/new key.

---

### Task 1: Extend the `Key` type and persist the new fields

**Files:**
- Modify: `src/types.ts:1-7`
- Modify: `src/memoryGraph.ts:538-548` (the `load()` key-restore loop)
- Test: `test/autokey-integration.test.ts` (created here, one persistence test)

**Interfaces:**
- Produces: `Key.aliasCandidates?: Record<string, AliasCandidate>` where `AliasCandidate = { count: number; lastSeen: number; queryText: string }`; `Key.learnedAlias?` ledger `LearnedAlias = { alias: string; addedAt: number; hits: number }` as `Key.learnedAliases?: LearnedAlias[]`.

- [ ] **Step 1: Extend the `Key` interface**

In `src/types.ts`, replace the `Key` interface (lines 1-7) with:

```ts
export interface AliasCandidate {
  count: number;
  lastSeen: number;
  queryText: string;
}

export interface LearnedAlias {
  alias: string;
  addedAt: number;
  hits: number;
}

export interface Key {
  id: string;
  concept: string;
  aliases: string[];
  embedding: number[];
  key_type: "concept" | "name" | "proper_noun";
  // Heat ledger for auto-key self-healing: normalized recall query -> confirmation count.
  // Persisted with the key; cleared when a candidate is promoted. Optional/absent on
  // legacy graphs and on keys that have never received a weak-confirmed read.
  aliasCandidates?: Record<string, AliasCandidate>;
  // Aliases added by auto-key promotion (not authored at remember() time). Provenance for
  // read_key output and the basis for stale-alias pruning.
  learnedAliases?: LearnedAlias[];
}
```

- [ ] **Step 2: Restore the new fields defensively in `load()`**

In `src/memoryGraph.ts`, the load loop currently does `this.keys[kid] = { ...k, aliases };` (line 547). Replace that single line with a version that sanitizes the two new optional fields so a hand-edited or legacy `graph.json` cannot inject bad shapes:

```ts
      const aliasCandidates =
        k.aliasCandidates && typeof k.aliasCandidates === "object" && !Array.isArray(k.aliasCandidates)
          ? k.aliasCandidates
          : undefined;
      const learnedAliases = Array.isArray(k.learnedAliases)
        ? k.learnedAliases.filter(
            (l): l is { alias: string; addedAt: number; hits: number } =>
              !!l && typeof l.alias === "string" && typeof l.addedAt === "number" && typeof l.hits === "number"
          )
        : undefined;
      this.keys[kid] = { ...k, aliases, aliasCandidates, learnedAliases };
```

(No `save()` change is needed: `save()` serializes `this.keys` wholesale via `JSON.stringify`, so the new fields persist automatically.)

- [ ] **Step 3: Write the persistence round-trip test**

Create `test/autokey-integration.test.ts` with this content (more tests are appended in later tasks):

```ts
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
process.env.SUPER_MEMORY_DATA_DIR = mkdtempSync(join(tmpdir(), "autokey-"));

test("Key.aliasCandidates and learnedAliases survive save/load", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(() => [1, 0]);
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g1 = new MemoryGraph();
    const [mid] = await g1.add("동균은 성수동에 산다", ["거주지"]);
    const kid = g1.getKeysForMemory(mid)[0];
    g1.keys[kid].aliasCandidates = { "어디 살아": { count: 2, lastSeen: 100, queryText: "어디 살아" } };
    g1.keys[kid].learnedAliases = [{ alias: "사는곳", addedAt: 100, hits: 1 }];
    await g1.save();

    const g2 = new MemoryGraph();
    await g2.load();
    assert.equal(g2.keys[kid].aliasCandidates?.["어디 살아"].count, 2);
    assert.equal(g2.keys[kid].learnedAliases?.[0].alias, "사는곳");
  } finally {
    emb.__clearTestEmbedder();
  }
});
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: PASS (1 test). If `getKeysForMemory` returns an empty array, the embedder stub is fine — `add` always links the supplied `keyConcepts`, so `kid` is defined.

- [ ] **Step 5: Commit**

```bash
git add src/types.ts src/memoryGraph.ts test/autokey-integration.test.ts
git commit -m "feat(autokey): persist alias-candidate heat ledger and learned-alias provenance on keys"
```

---

### Task 2: `RecallBuffer` (ephemeral weak-match store)

**Files:**
- Create: `src/autokey.ts`
- Test: `test/autokey.test.ts`

**Interfaces:**
- Produces:
  - `class RecallBuffer` with constructor `new RecallBuffer(opts?: { capacity?: number; ttlSeconds?: number; now?: () => number })`.
  - `push(entry: { queryText: string; queryEmbedding: number[]; weakKeyScores: Map<string, number> }): void`
  - `consumeWeakMatch(keyId: string): { queryText: string; queryEmbedding: number[]; weakKeyScores: Map<string, number>; ts: number } | null` — returns the most-recent fresh entry whose `weakKeyScores` has `keyId`, and removes `keyId` from that entry's map so the same recall cannot re-fire for the same key.
  - `size(): number`

- [ ] **Step 1: Write failing tests**

Create `test/autokey.test.ts`:

```ts
import assert from "node:assert/strict";
import test from "node:test";
import { RecallBuffer } from "../src/autokey.ts";

test("consumeWeakMatch returns the most recent fresh entry for a key", () => {
  let clock = 1000;
  const buf = new RecallBuffer({ capacity: 4, ttlSeconds: 300, now: () => clock });
  buf.push({ queryText: "어디 살아", queryEmbedding: [1, 0], weakKeyScores: new Map([["k1", 0.9]]) });
  clock = 1010;
  buf.push({ queryText: "사는곳", queryEmbedding: [0, 1], weakKeyScores: new Map([["k1", 0.95]]) });

  const hit = buf.consumeWeakMatch("k1");
  assert.equal(hit?.queryText, "사는곳"); // most recent
  assert.equal(hit?.weakKeyScores.get("k1"), 0.95);
});

test("consumeWeakMatch removes the key so it cannot fire twice", () => {
  const buf = new RecallBuffer({ now: () => 0 });
  buf.push({ queryText: "q", queryEmbedding: [1], weakKeyScores: new Map([["k1", 0.9]]) });
  assert.ok(buf.consumeWeakMatch("k1"));
  assert.equal(buf.consumeWeakMatch("k1"), null);
});

test("entries past TTL are ignored", () => {
  let clock = 0;
  const buf = new RecallBuffer({ ttlSeconds: 300, now: () => clock });
  buf.push({ queryText: "q", queryEmbedding: [1], weakKeyScores: new Map([["k1", 0.9]]) });
  clock = 301;
  assert.equal(buf.consumeWeakMatch("k1"), null);
});

test("capacity evicts oldest entries", () => {
  const buf = new RecallBuffer({ capacity: 2, now: () => 0 });
  buf.push({ queryText: "a", queryEmbedding: [1], weakKeyScores: new Map([["k1", 0.9]]) });
  buf.push({ queryText: "b", queryEmbedding: [1], weakKeyScores: new Map([["k2", 0.9]]) });
  buf.push({ queryText: "c", queryEmbedding: [1], weakKeyScores: new Map([["k3", 0.9]]) });
  assert.equal(buf.size(), 2);
  assert.equal(buf.consumeWeakMatch("k1"), null); // evicted
  assert.ok(buf.consumeWeakMatch("k3"));
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx tsx --test test/autokey.test.ts`
Expected: FAIL — `Cannot find module '../src/autokey.ts'`.

- [ ] **Step 3: Implement `RecallBuffer`**

Create `src/autokey.ts`:

```ts
export interface RecallBufferEntry {
  queryText: string;
  queryEmbedding: number[];
  weakKeyScores: Map<string, number>;
  ts: number;
}

// Runtime-only ring buffer of recent recalls that matched one or more keys only
// *weakly* (semantic, not literal). Never persisted. Bounded by capacity + TTL so
// it can never grow without bound or attribute a confirmation to a stale query.
export class RecallBuffer {
  private _entries: RecallBufferEntry[] = [];
  private readonly _capacity: number;
  private readonly _ttl: number;
  private readonly _now: () => number;

  constructor(opts: { capacity?: number; ttlSeconds?: number; now?: () => number } = {}) {
    this._capacity = Math.max(1, Math.floor(opts.capacity ?? 32));
    this._ttl = Math.max(1, opts.ttlSeconds ?? 300);
    this._now = opts.now ?? (() => Date.now() / 1000);
  }

  push(entry: { queryText: string; queryEmbedding: number[]; weakKeyScores: Map<string, number> }): void {
    this._entries.push({ ...entry, ts: this._now() });
    if (this._entries.length > this._capacity) {
      this._entries.splice(0, this._entries.length - this._capacity);
    }
  }

  // Most-recent fresh entry that weakly matched keyId; removes keyId from that
  // entry so a single recall confirms a given key at most once.
  consumeWeakMatch(keyId: string): RecallBufferEntry | null {
    const cutoff = this._now() - this._ttl;
    for (let i = this._entries.length - 1; i >= 0; i--) {
      const e = this._entries[i];
      if (e.ts < cutoff) continue;
      if (e.weakKeyScores.has(keyId)) {
        const result: RecallBufferEntry = {
          queryText: e.queryText,
          queryEmbedding: e.queryEmbedding,
          weakKeyScores: new Map(e.weakKeyScores),
          ts: e.ts,
        };
        e.weakKeyScores.delete(keyId);
        return result;
      }
    }
    return null;
  }

  size(): number {
    return this._entries.length;
  }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx tsx --test test/autokey.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/autokey.ts test/autokey.test.ts
git commit -m "feat(autokey): add ephemeral RecallBuffer for weak-match tracking"
```

---

### Task 3: Config constants + `decidePromotion` policy

**Files:**
- Modify: `src/autokey.ts`
- Test: `test/autokey.test.ts`

**Interfaces:**
- Produces:
  - `AUTOKEY_ENABLED: boolean`, `AUTOKEY_PROMOTE_N: number`, `AUTOKEY_MAX_ALIASES: number`, `AUTOKEY_BUFFER_CAPACITY: number`, `AUTOKEY_BUFFER_TTL_SECONDS: number`, `AUTOKEY_PRUNE_AGE_SECONDS: number`.
  - `function decidePromotion(args: { count: number; query: string; cosine: number; learnedAliasCount: number; aliasThreshold: number; newKeyThreshold: number; promoteN: number; maxAliases: number }): "alias" | "newKey" | "none"`

- [ ] **Step 1: Write failing tests**

Append to `test/autokey.test.ts`:

```ts
import { decidePromotion } from "../src/autokey.ts";

const base = {
  query: "사는곳", learnedAliasCount: 0,
  aliasThreshold: 0.86, newKeyThreshold: 0.62, promoteN: 3, maxAliases: 8,
};

test("decidePromotion: below promoteN does nothing", () => {
  assert.equal(decidePromotion({ ...base, count: 2, cosine: 0.99 }), "none");
});

test("decidePromotion: high cosine at threshold -> alias", () => {
  assert.equal(decidePromotion({ ...base, count: 3, cosine: 0.9 }), "alias");
});

test("decidePromotion: mid cosine -> newKey", () => {
  assert.equal(decidePromotion({ ...base, count: 3, cosine: 0.7 }), "newKey");
});

test("decidePromotion: cosine below newKey floor -> none", () => {
  assert.equal(decidePromotion({ ...base, count: 3, cosine: 0.5 }), "none");
});

test("decidePromotion: long query is never promoted", () => {
  const longQ = "동균이 요즘 즐겨 마시는 음료가 무엇인지";
  assert.equal(decidePromotion({ ...base, query: longQ, count: 5, cosine: 0.99 }), "none");
});

test("decidePromotion: alias cap forces none even at high cosine", () => {
  assert.equal(decidePromotion({ ...base, count: 3, cosine: 0.99, learnedAliasCount: 8 }), "none");
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx tsx --test test/autokey.test.ts`
Expected: FAIL — `decidePromotion` is not exported.

- [ ] **Step 3: Implement config + policy**

Append to `src/autokey.ts`:

```ts
import { isShortConcept } from "./embedding.js";

function envInt(name: string, fallback: number, min: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= min ? Math.floor(n) : fallback;
}

// Feature flag. Default ON; set SUPER_MEMORY_AUTOKEY=false to disable (mirrors
// SUPER_MEMORY_AUTO_MIGRATE). Read once at import.
export const AUTOKEY_ENABLED = process.env.SUPER_MEMORY_AUTOKEY !== "false";
export const AUTOKEY_PROMOTE_N = envInt("SUPER_MEMORY_AUTOKEY_PROMOTE_N", 3, 1);
export const AUTOKEY_MAX_ALIASES = envInt("SUPER_MEMORY_AUTOKEY_MAX_ALIASES", 8, 0);
export const AUTOKEY_BUFFER_CAPACITY = 32;
export const AUTOKEY_BUFFER_TTL_SECONDS = 300;
export const AUTOKEY_PRUNE_AGE_SECONDS = envInt(
  "SUPER_MEMORY_AUTOKEY_PRUNE_AGE", 30 * 24 * 3600, 0
);

// Pure policy: given accumulated heat and the recall-time query↔key cosine, decide
// whether to fold the query into the key space and how. Short-concept gate keeps
// natural-language queries out of the alias set; the content path already serves those.
export function decidePromotion(args: {
  count: number;
  query: string;
  cosine: number;
  learnedAliasCount: number;
  aliasThreshold: number;
  newKeyThreshold: number;
  promoteN: number;
  maxAliases: number;
}): "alias" | "newKey" | "none" {
  if (args.count < args.promoteN) return "none";
  if (!isShortConcept(args.query)) return "none";
  if (args.cosine >= args.aliasThreshold) {
    return args.learnedAliasCount < args.maxAliases ? "alias" : "none";
  }
  if (args.cosine >= args.newKeyThreshold) return "newKey";
  return "none";
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx tsx --test test/autokey.test.ts`
Expected: PASS (10 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/autokey.ts test/autokey.test.ts
git commit -m "feat(autokey): add promotion policy and env-driven config"
```

---

### Task 4: Push weak matches into the buffer from `searchKeys`

**Files:**
- Modify: `src/memoryGraph.ts:11-31` (imports + field), `src/memoryGraph.ts:1011-1015` (end of `searchKeys`)
- Test: `test/autokey-integration.test.ts`

**Interfaces:**
- Consumes: `RecallBuffer`, `AUTOKEY_ENABLED`, `AUTOKEY_BUFFER_CAPACITY`, `AUTOKEY_BUFFER_TTL_SECONDS` from `./autokey.js`.
- Produces: `MemoryGraph` private field `_recallBuffer: RecallBuffer`; weak (semantic) returned keys are pushed per `searchKeys` call.

- [ ] **Step 1: Add the import and the buffer field**

At the top of `src/memoryGraph.ts`, after the existing imports, add:

```ts
import { RecallBuffer, AUTOKEY_ENABLED, AUTOKEY_BUFFER_CAPACITY, AUTOKEY_BUFFER_TTL_SECONDS } from "./autokey.js";
```

Inside `class MemoryGraph`, alongside the other private fields (near line 209, after `_bm25`), add:

```ts
  private _recallBuffer = new RecallBuffer({
    capacity: AUTOKEY_BUFFER_CAPACITY,
    ttlSeconds: AUTOKEY_BUFFER_TTL_SECONDS,
  });
```

- [ ] **Step 2: Write the failing test**

Append to `test/autokey-integration.test.ts`:

```ts
test("searchKeys records weak (semantic) matches in the recall buffer", async () => {
  const emb = await import("../src/embedding.ts");
  // "거주지" key embeds [1,0]; a paraphrase query embeds close but not literal.
  emb.__setTestEmbedder((text) => (text.includes("성수") || text === "거주지" ? [1, 0] : [0.95, 0.31]));
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("동균은 성수동에 산다", ["거주지"]);
    // getKeysForMemory returns concept strings; resolve the concept to its key ID.
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "거주지")!;

    await g.searchKeys("어디 살아"); // semantic match on 거주지, not literal
    const hit = (g as unknown as { _recallBuffer: { consumeWeakMatch(k: string): unknown } })._recallBuffer
      .consumeWeakMatch(kid);
    assert.ok(hit, "expected 거주지 to be recorded as a weak match");
  } finally {
    emb.__clearTestEmbedder();
  }
});
```

- [ ] **Step 3: Run to verify failure**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: FAIL — `consumeWeakMatch` returns null (nothing pushed yet).

- [ ] **Step 4: Push weak matches at the end of `searchKeys`**

In `src/memoryGraph.ts`, `searchKeys` currently ends its locked block with:

```ts
      return candidates
        .sort((a, b) => Number(b._literal) - Number(a._literal) || b.score - a.score || b.specificity - a.specificity)
        .slice(0, topK)
        .map(({ _literal, ...candidate }) => candidate);
```

Replace that `return` with:

```ts
      const result = candidates
        .sort((a, b) => Number(b._literal) - Number(a._literal) || b.score - a.score || b.specificity - a.specificity)
        .slice(0, topK)
        .map(({ _literal, ...candidate }) => candidate);

      if (AUTOKEY_ENABLED) {
        const weak = result.filter((c) => c.match_type === "semantic");
        if (weak.length > 0) {
          this._recallBuffer.push({
            queryText: cleanQuery,
            queryEmbedding: qEmb,
            weakKeyScores: new Map(weak.map((c) => [c.key_id, c.score])),
          });
        }
      }
      return result;
```

- [ ] **Step 5: Run to verify pass**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/memoryGraph.ts test/autokey-integration.test.ts
git commit -m "feat(autokey): record weak semantic key matches in recall buffer"
```

---

### Task 5: Confirm + promote in `readMemory` (`_maybeLearnAlias`)

**Files:**
- Modify: `src/memoryGraph.ts` (import line from Task 4; new private method; one call in `readMemory` after line 1080)
- Test: `test/autokey-integration.test.ts`

**Interfaces:**
- Consumes: `decidePromotion`, `AUTOKEY_ENABLED`, `AUTOKEY_PROMOTE_N`, `AUTOKEY_MAX_ALIASES` from `./autokey.js`; existing `KEY_MERGE_THRESHOLD`, `KEY_AUTO_LINK_THRESHOLD`, `_recordKeyAlias`, `findOrCreateKey`, `_link`.
- Produces: private `async _maybeLearnAlias(keyId: string, memoryId: string): Promise<void>`.

- [ ] **Step 1: Extend the autokey import**

Update the Task 4 import line in `src/memoryGraph.ts` to also pull the policy + thresholds:

```ts
import {
  RecallBuffer, decidePromotion,
  AUTOKEY_ENABLED, AUTOKEY_BUFFER_CAPACITY, AUTOKEY_BUFFER_TTL_SECONDS,
  AUTOKEY_PROMOTE_N, AUTOKEY_MAX_ALIASES,
} from "./autokey.js";
```

- [ ] **Step 2: Write the failing test (full heal loop)**

Append to `test/autokey-integration.test.ts`:

```ts
test("repeated weak-confirmed reads promote the query (new key) and heal recall", async () => {
  const emb = await import("../src/embedding.ts");
  // Query embeds in the mid band vs the key (cos ~0.95 with bge-m3 keyMerge 0.86 would
  // alias; to exercise the newKey branch we make the query orthogonal-ish but still a
  // surfaced semantic match by also making it literal-free). Use a value below keyMerge
  // (0.86) but above keyAutoLink (0.62): [0.8,0.6] · [1,0] = 0.8.
  // Key "거주지" and the memory content embed to [1,0]. The recall query "살곳" embeds
  // to [0.8,0.6] → cosine 0.8 vs the key: above keyAutoLink (0.62) so it surfaces as a
  // SEMANTIC match, below keyMerge (0.86) so promotion takes the newKey branch. "살곳"
  // shares no substring with "거주지", so it can never be a literal/concept match.
  emb.__setTestEmbedder((text) => (text === "거주지" || text.includes("성수") ? [1, 0] : [0.8, 0.6]));
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("동균은 성수동에 산다", ["거주지"]);
    // getKeysForMemory returns concept strings; resolve the concept to its key ID.
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "거주지")!;

    const QUERY = "살곳"; // short concept, semantic match on 거주지, no substring overlap
    for (let i = 0; i < 3; i++) {
      const keys = (await g.searchKeys(QUERY)) as Array<{ key_id: string; match_type: string }>;
      assert.ok(keys.some((k) => k.key_id === kid && k.match_type === "semantic"));
      await g.readMemory(mid, kid);
    }

    // After 3 confirmations a NEW key for the query exists and links to the memory.
    const healedKid = Object.keys(g.keys).find((k) => g.keys[k].concept === QUERY);
    assert.ok(healedKid, "expected a new key coined from the query");
    assert.ok(g.getKeysForMemory(mid).includes(healedKid!), "new key must link the memory");
  } finally {
    emb.__clearTestEmbedder();
  }
});

test("depth/access_count still increment exactly once per readMemory", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(() => [1, 0]);
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("x", ["kx"]);
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "kx")!;
    const before = (await g.readMemory(mid, kid)) as { memory: { access_count: number } };
    const after = (await g.readMemory(mid, kid)) as { memory: { access_count: number } };
    assert.equal(after.memory.access_count, before.memory.access_count + 1);
  } finally {
    emb.__clearTestEmbedder();
  }
});
```

- [ ] **Step 3: Run to verify failure**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: FAIL — no new key is coined (the heal-loop test fails); the access_count test passes.

- [ ] **Step 4: Implement `_maybeLearnAlias` and wire it into `readMemory`**

In `src/memoryGraph.ts`, add this private method to `class MemoryGraph` (place it directly above `readMemory`):

```ts
  // Auto-key self-healing: a memory was just confirmed (read) via viaKeyId. If that key
  // was a recent WEAK (semantic) recall match, the originating query is candidate
  // vocabulary the key is missing. Accumulate heat; promote at threshold. Runs inside
  // readMemory's lock; readMemory's unconditional save() persists any mutation.
  private async _maybeLearnAlias(keyId: string, memoryId: string): Promise<void> {
    const entry = this._recallBuffer.consumeWeakMatch(keyId);
    if (!entry) return;
    const key = this.keys[keyId];
    if (!key) return;
    const q = entry.queryText.trim();
    if (q.length < 2) return;
    const norm = q.toLowerCase();
    if (key.concept.toLowerCase() === norm) return;
    if ((key.aliases ?? []).some((a) => a.toLowerCase() === norm)) return;

    key.aliasCandidates ??= {};
    const prev = key.aliasCandidates[norm];
    const candidate = { count: (prev?.count ?? 0) + 1, lastSeen: Date.now() / 1000, queryText: q };
    key.aliasCandidates[norm] = candidate;

    const decision = decidePromotion({
      count: candidate.count,
      query: q,
      cosine: entry.weakKeyScores.get(keyId) ?? 0,
      learnedAliasCount: key.learnedAliases?.length ?? 0,
      aliasThreshold: KEY_MERGE_THRESHOLD,
      newKeyThreshold: KEY_AUTO_LINK_THRESHOLD,
      promoteN: AUTOKEY_PROMOTE_N,
      maxAliases: AUTOKEY_MAX_ALIASES,
    });

    if (decision === "alias") {
      this._recordKeyAlias(keyId, q);
      key.learnedAliases ??= [];
      key.learnedAliases.push({ alias: q, addedAt: Date.now() / 1000, hits: 0 });
      delete key.aliasCandidates[norm];
    } else if (decision === "newKey") {
      const newKid = await this.findOrCreateKey(q, "concept");
      this._link(newKid, memoryId);
      delete key.aliasCandidates[norm];
    }
  }
```

Then, inside `readMemory`, immediately after the `if (viaKeyId) { this._setLinkWeight(...) }` block (after line 1080, before `const connectedKeys = ...`), add:

```ts
      if (AUTOKEY_ENABLED && viaKeyId) {
        await this._maybeLearnAlias(viaKeyId, memoryId);
      }
```

(Placing it before `connectedKeys` is assembled means a freshly linked new key appears in the returned `keys` array — the agent sees the heal immediately. `findOrCreateKey` does not acquire `_lock`, so calling it here is safe and non-reentrant.)

- [ ] **Step 5: Run to verify pass**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the full suite to check for regressions**

Run: `npm test`
Expected: all pre-existing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/memoryGraph.ts test/autokey-integration.test.ts
git commit -m "feat(autokey): promote weak-confirmed queries to aliases/new keys on read"
```

---

### Task 6: Surface learned aliases in `read_key` output

**Files:**
- Modify: `src/memoryGraph.ts:441-453` (`_keyView`)
- Test: `test/autokey-integration.test.ts`

**Interfaces:**
- Produces: `_keyView` return object gains `learned_aliases: string[]`.

- [ ] **Step 1: Write the failing test**

Append to `test/autokey-integration.test.ts`:

```ts
test("read_key surfaces learned aliases as provenance", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(() => [1, 0]);
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("동균은 성수동에 산다", ["거주지"]);
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "거주지")!;
    g.keys[kid].learnedAliases = [{ alias: "사는곳", addedAt: 1, hits: 0 }];

    const view = g.readKey(kid) as { key: { learned_aliases: string[] } };
    assert.deepEqual(view.key.learned_aliases, ["사는곳"]);
  } finally {
    emb.__clearTestEmbedder();
  }
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: FAIL — `learned_aliases` is undefined.

- [ ] **Step 3: Add `learned_aliases` to `_keyView`**

In `src/memoryGraph.ts`, `_keyView` returns an object literal (lines 444-452). Add one field after `aliases`:

```ts
      aliases: key.aliases ?? [],
      learned_aliases: (key.learnedAliases ?? []).map((l) => l.alias),
```

- [ ] **Step 4: Run to verify pass**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/memoryGraph.ts test/autokey-integration.test.ts
git commit -m "feat(autokey): expose learned aliases in read_key output"
```

---

### Task 7: Track alias hits + prune stale learned aliases

**Files:**
- Modify: `src/memoryGraph.ts` `searchKeys` (alias-match hit bump, near line 983) and `cleanupExpired` (prune pass, near line 1648)
- Test: `test/autokey-integration.test.ts`

**Interfaces:**
- Consumes: `AUTOKEY_PRUNE_AGE_SECONDS` from `./autokey.js`.
- Produces: learned aliases gain `hits` on literal recall; `cleanupExpired` removes learned aliases with `hits === 0` older than `AUTOKEY_PRUNE_AGE_SECONDS` (from both `learnedAliases` and `aliases`).

- [ ] **Step 1: Extend the autokey import**

Add `AUTOKEY_PRUNE_AGE_SECONDS` to the existing `./autokey.js` import in `src/memoryGraph.ts`.

- [ ] **Step 2: Write the failing tests**

Append to `test/autokey-integration.test.ts`:

```ts
test("a literal hit on a learned alias bumps its hit count", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(() => [1, 0]);
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("동균은 성수동에 산다", ["거주지"]);
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "거주지")!;
    g.keys[kid].aliases.push("사는곳");
    g.keys[kid].learnedAliases = [{ alias: "사는곳", addedAt: 1, hits: 0 }];

    await g.searchKeys("사는곳"); // literal alias match
    assert.equal(g.keys[kid].learnedAliases?.[0].hits, 1);
  } finally {
    emb.__clearTestEmbedder();
  }
});

test("cleanupExpired prunes stale, never-hit learned aliases", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(() => [1, 0]);
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("동균은 성수동에 산다", ["거주지"]);
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "거주지")!;
    g.keys[kid].aliases.push("쓸모없는별칭");
    g.keys[kid].learnedAliases = [{ alias: "쓸모없는별칭", addedAt: 0, hits: 0 }]; // addedAt epoch 0 = very old

    await g.cleanupExpired();
    assert.equal(g.keys[kid].learnedAliases?.length ?? 0, 0);
    assert.ok(!g.keys[kid].aliases.includes("쓸모없는별칭"));
  } finally {
    emb.__clearTestEmbedder();
  }
});
```

- [ ] **Step 3: Run to verify failure**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: FAIL — hits stays 0; stale alias not pruned.

- [ ] **Step 4: Bump hits on literal alias match in `searchKeys`**

In `src/memoryGraph.ts` `searchKeys`, just after `matchedAlias` is computed (after line 983, the `const matchedAlias = ...` line), add:

```ts
        if (matchedAlias && key.learnedAliases) {
          const la = key.learnedAliases.find((l) => l.alias.toLowerCase() === matchedAlias.toLowerCase());
          if (la) la.hits += 1;
        }
```

(Hit counts are non-critical metadata; they are persisted opportunistically by the next `save()` from any write path. `searchKeys` is not given its own disk write to keep the recall hot path free of I/O.)

- [ ] **Step 5: Add the prune pass to `cleanupExpired`**

In `src/memoryGraph.ts` `cleanupExpired`, inside the `runExclusive` callback after the expired-memory loop and before the final `await this.save()`, add:

```ts
      const now = Date.now() / 1000;
      for (const key of Object.values(this.keys)) {
        if (!key.learnedAliases?.length) continue;
        const keep = key.learnedAliases.filter(
          (l) => l.hits > 0 || now - l.addedAt < AUTOKEY_PRUNE_AGE_SECONDS
        );
        if (keep.length === key.learnedAliases.length) continue;
        const dropped = new Set(
          key.learnedAliases.filter((l) => !keep.includes(l)).map((l) => l.alias.toLowerCase())
        );
        key.learnedAliases = keep;
        key.aliases = (key.aliases ?? []).filter((a) => !dropped.has(a.toLowerCase()));
      }
```

- [ ] **Step 6: Run to verify pass**

Run: `npx tsx --test test/autokey-integration.test.ts`
Expected: PASS (7 tests).

- [ ] **Step 7: Full suite + commit**

Run: `npm test`
Expected: all PASS.

```bash
git add src/memoryGraph.ts test/autokey-integration.test.ts
git commit -m "feat(autokey): track learned-alias hits and prune stale ones"
```

---

### Task 8: Documentation + benchmark regression check

**Files:**
- Modify: `README.md` (env-var/config section)
- Run: `npm run bench`

**Interfaces:** none (docs + verification).

- [ ] **Step 1: Document the new env vars in `README.md`**

Find the section listing `SUPER_MEMORY_*` environment variables (search `README.md` for `SUPER_MEMORY_`). Add:

```markdown
| `SUPER_MEMORY_AUTOKEY` | `true` | Auto-key self-healing: learn missing search terms from real usage. Set `false` to disable. |
| `SUPER_MEMORY_AUTOKEY_PROMOTE_N` | `3` | Weak-confirmed reads of a `(key, query)` pair before the query is folded into the key space. |
| `SUPER_MEMORY_AUTOKEY_MAX_ALIASES` | `8` | Max learned aliases promoted per key. |
| `SUPER_MEMORY_AUTOKEY_PRUNE_AGE` | `2592000` | Seconds before a never-hit learned alias is pruned by `cleanup_expired`. |
```

If no such table exists, add a short "Auto-key self-healing" subsection describing the behavior and these four variables.

- [ ] **Step 2: Run the benchmark to confirm no quality regression**

Run: `npm run bench`
Expected: completes without error; recall-quality scores are no worse than the pre-change baseline (the feature only adds vocabulary, never removes a working match). Record the numbers in the commit message if the bench prints a summary.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(autokey): document self-healing env vars"
```

---

## Self-Review

**1. Spec coverage:**
- Weak-match learning signal → Tasks 4 (push) + 5 (confirm). ✓
- Ephemeral RecallBuffer (bounded + TTL, not persisted) → Task 2. ✓
- Persisted `(key, query)` heat ledger → Task 1 (`aliasCandidates`) + Task 5 (increment). ✓
- Heat threshold `PROMOTE_N`, short-concept gate, alias-vs-new-key by cosine → Task 3 (`decidePromotion`). ✓
- Reuse `_recordKeyAlias` / `_autoLinkKeys`-style linking / `isShortConcept` / embeddings → Tasks 3, 5. ✓ (new key uses `findOrCreateKey` + `_link`, the same primitive `_autoLinkKeys` calls.)
- Hook into existing locked phase of `readMemory` (no new lock) → Task 5. ✓
- Config (`SUPER_MEMORY_AUTOKEY*`, default ON) → Task 3 + Task 8 docs. ✓
- Risk mitigations: misattribution (TTL + consume-once + N-threshold) Task 2/5; alias pollution (short-concept + cosine + cap) Task 3; pruning Task 7; dedup (`_recordKeyAlias` case-insensitive) Task 5. ✓
- Provenance flag in `read_key` → Task 6. ✓
- Tests: unit (buffer, policy) Tasks 2/3; integration (heal loop, depth/access invariance) Task 5; pruning Task 7. ✓
- Regression: `npm run bench` → Task 8. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**3. Type consistency:** `aliasCandidates: Record<string, AliasCandidate>` and `learnedAliases: LearnedAlias[]` defined in Task 1 and used unchanged in Tasks 5/6/7. `decidePromotion` signature defined in Task 3 matches its call in Task 5 (`count, query, cosine, learnedAliasCount, aliasThreshold, newKeyThreshold, promoteN, maxAliases`). `RecallBuffer.consumeWeakMatch`/`push`/`size` defined in Task 2 match usage in Tasks 4/5. `weakKeyScores` is a `Map<string, number>` throughout. ✓

**Note on concurrency:** `searchKeys` and `readMemory` both run inside `this._lock.runExclusive`, so all buffer and key mutations are serialized — no new lock is introduced and no read/write can interleave mid-mutation.
