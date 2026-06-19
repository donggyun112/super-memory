# Dedup/Contradiction Calibration Corpus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a labeled pair corpus and a threshold-sweep harness so the dedup/contradiction thresholds can be calibrated against data with a train/held-out split, and the 1순위 bug (dup vs contradiction not separable by cosine alone) is quantified.

**Architecture:** Pure scoring/sweep logic lives in `bench/calibrate-lib.ts` (no I/O, fully unit-tested). A thin runner `bench/calibrate.ts` reads `bench/pairs.json`, embeds each pair with the real `embedTextAsync`, and calls the orchestrator, then prints a scorecard. Mirrors the live `MemoryGraph` decision path (≥dedup → duplicate; band+shared-key → contradiction; else independent).

**Tech Stack:** TypeScript (ESM, NodeNext), `tsx`, `node:test`, existing `src/embedding.ts` (`embedTextAsync`, `__setTestEmbedder`).

## Global Constraints

- Node ≥ 20; ESM with `.ts` import specifiers (e.g. `import "./calibrate-lib.ts"`), matching existing `bench/*.ts`.
- Do NOT modify `bench/fixture.json` or `bench/run.ts` (independent retrieval benchmark).
- No new runtime dependencies.
- No Claude/AI attribution in commit messages.
- Classification mirrors `src/memoryGraph.ts` exactly: `simAB >= dedupCut → "duplicate"`; `floor <= simAB < dedupCut AND sharedKey → "contradiction"`; else `"independent"`.
- Diagnostic scope only — do NOT change the supersede decision or add a recovery path.

---

### Task 1: Corpus seed + labeling guideline

**Files:**
- Create: `bench/pairs.json`
- Create: `bench/LABELING.md`
- Test: `test/calibrate-corpus.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `bench/pairs.json` with shape `{ pairs: Pair[] }` where `Pair = { id: string; a: string; b: string; keys_a: string[]; keys_b: string[]; relation: "duplicate"|"contradiction"|"independent"; confidence: "high"|"low"; split: "train"|"held-out" }`. ~30 seed pairs covering all 3 relations in KO and EN.

- [ ] **Step 1: Write the failing test**

```typescript
// test/calibrate-corpus.test.ts
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

const RELATIONS = new Set(["duplicate", "contradiction", "independent"]);

test("pairs.json is well-formed and covers all three relations", async () => {
  const raw = JSON.parse(await readFile(resolve("bench/pairs.json"), "utf-8"));
  assert.ok(Array.isArray(raw.pairs), "pairs must be an array");
  assert.ok(raw.pairs.length >= 30, `expected >=30 seed pairs, got ${raw.pairs.length}`);

  const seen = new Set<string>();
  const counts: Record<string, number> = { duplicate: 0, contradiction: 0, independent: 0 };
  for (const p of raw.pairs) {
    for (const f of ["id", "a", "b"]) assert.equal(typeof p[f], "string", `${f} must be a string`);
    assert.ok(Array.isArray(p.keys_a) && Array.isArray(p.keys_b), "keys must be arrays");
    assert.ok(RELATIONS.has(p.relation), `bad relation: ${p.relation}`);
    assert.ok(["high", "low"].includes(p.confidence), `bad confidence: ${p.confidence}`);
    assert.ok(["train", "held-out"].includes(p.split), `bad split: ${p.split}`);
    assert.ok(!seen.has(p.id), `duplicate id: ${p.id}`);
    seen.add(p.id);
    counts[p.relation]++;
  }
  for (const r of RELATIONS) assert.ok(counts[r] >= 5, `need >=5 ${r} pairs, got ${counts[r]}`);

  // contradiction pairs must share a key (same subject) — else they can't be detected.
  for (const p of raw.pairs.filter((x: any) => x.relation === "contradiction")) {
    const a = new Set(p.keys_a.map((k: string) => k.toLowerCase()));
    assert.ok(p.keys_b.some((k: string) => a.has(k.toLowerCase())), `contradiction ${p.id} must share a key`);
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/calibrate-corpus.test.ts`
Expected: FAIL — `ENOENT` (bench/pairs.json does not exist yet).

- [ ] **Step 3: Create `bench/pairs.json` with ~30 seed pairs**

Author ~30 pairs by hand (KO+EN, Mina persona from `fixture.json`), ≥5 per relation, stratified across `split`. Contradiction pairs MUST share a key. Mark genuinely ambiguous boundary pairs `confidence: "low"`. Example shape (extend to ≥30):

```json
{
  "_doc": "Calibration corpus for dedup/contradiction thresholds. Each pair labels the relation between two facts. Drives threshold sweeps in bench/calibrate.ts. See LABELING.md.",
  "pairs": [
    { "id": "dup-01", "a": "미나는 아이스 라떼를 마신다", "b": "미나가 즐겨 마시는 건 아이스 라떼야", "keys_a": ["음료","라떼"], "keys_b": ["음료","라떼","취향"], "relation": "duplicate", "confidence": "high", "split": "train" },
    { "id": "dup-02", "a": "Mina works as a backend engineer", "b": "Mina is a backend developer", "keys_a": ["job","engineer"], "keys_b": ["job","developer"], "relation": "duplicate", "confidence": "high", "split": "held-out" },
    { "id": "con-01", "a": "회의는 월요일이다", "b": "회의는 금요일이다", "keys_a": ["회의","일정"], "keys_b": ["회의","일정"], "relation": "contradiction", "confidence": "high", "split": "held-out" },
    { "id": "con-02", "a": "Mina codes primarily in Go", "b": "Mina codes primarily in Rust", "keys_a": ["언어","Mina"], "keys_b": ["언어","Mina"], "relation": "contradiction", "confidence": "high", "split": "train" },
    { "id": "ind-01", "a": "미나는 강아지를 키운다", "b": "미나는 재즈를 듣는다", "keys_a": ["반려동물"], "keys_b": ["음악"], "relation": "independent", "confidence": "high", "split": "train" },
    { "id": "ind-02", "a": "미나는 서울 마포구에 산다", "b": "미나는 새벽에 집중이 잘 된다", "keys_a": ["거주지"], "keys_b": ["작업스타일"], "relation": "independent", "confidence": "high", "split": "held-out" }
  ]
}
```

- [ ] **Step 4: Create `bench/LABELING.md`**

```markdown
# Calibration Pair Labeling Guide

Each pair (a, b) gets one `relation`:

- **duplicate** — same fact restated (paraphrase, reorder, synonym). System should
  supersede (dedup). e.g. "라떼를 마신다" / "마시는 건 라떼야".
- **contradiction** — same subject, conflicting value. System should flag, keep both.
  MUST share a key. e.g. "회의는 월요일" / "회의는 금요일".
- **independent** — unrelated or coexisting facts. Keep both, no flag.

## Boundary rules
- Refinement (general vs specific: "서울 산다" / "마포구 산다") → `independent`, `confidence: "low"`.
- If two reasonable people would disagree on the label → set `confidence: "low"`.
- Different subjects can never be `contradiction` (no shared key) → `independent`.

## Generation rules (avoid bias)
- Use natural, conversational phrasing. Vary sentence structure between a and b.
- Cap single-token minimal-pairs (e.g. only day-of-week swapped) at ~1/3 of
  contradictions; their artificially high cosine biases the threshold.
- Mix KO and EN; reuse the Mina persona for realism.
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npx tsx --test test/calibrate-corpus.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bench/pairs.json bench/LABELING.md test/calibrate-corpus.test.ts
git commit -m "test(bench): add calibration pair corpus seed + labeling guide"
```

---

### Task 2: Core classification (`sharedKey`, `classifyPair`)

**Files:**
- Create: `bench/calibrate-lib.ts`
- Test: `test/calibrate-lib.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `type Relation = "duplicate" | "contradiction" | "independent"`
  - `interface Pair { id: string; a: string; b: string; keys_a: string[]; keys_b: string[]; relation: Relation; confidence: "high"|"low"; split: "train"|"held-out" }`
  - `interface ScoredPair { pair: Pair; simAB: number; sharedKey: boolean }`
  - `sharedKey(keysA: string[], keysB: string[]): boolean`
  - `classifyPair(simAB: number, shared: boolean, floor: number, dedupCut: number): Relation`

- [ ] **Step 1: Write the failing test**

```typescript
// test/calibrate-lib.test.ts
import assert from "node:assert/strict";
import test from "node:test";
import { sharedKey, classifyPair } from "../bench/calibrate-lib.ts";

test("sharedKey is case-insensitive overlap", () => {
  assert.equal(sharedKey(["회의", "일정"], ["일정"]), true);
  assert.equal(sharedKey(["Mina"], ["mina"]), true);
  assert.equal(sharedKey(["음료"], ["음악"]), false);
  assert.equal(sharedKey([], ["x"]), false);
});

test("classifyPair mirrors the MemoryGraph decision path", () => {
  // >= dedupCut -> duplicate (regardless of shared key)
  assert.equal(classifyPair(0.96, true, 0.80, 0.94), "duplicate");
  assert.equal(classifyPair(0.96, false, 0.80, 0.94), "duplicate");
  // in band AND shared -> contradiction
  assert.equal(classifyPair(0.88, true, 0.80, 0.94), "contradiction");
  // in band but NOT shared -> independent
  assert.equal(classifyPair(0.88, false, 0.80, 0.94), "independent");
  // below floor -> independent
  assert.equal(classifyPair(0.50, true, 0.80, 0.94), "independent");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/calibrate-lib.test.ts`
Expected: FAIL — cannot resolve `../bench/calibrate-lib.ts`.

- [ ] **Step 3: Write minimal implementation**

```typescript
// bench/calibrate-lib.ts
export type Relation = "duplicate" | "contradiction" | "independent";

export interface Pair {
  id: string;
  a: string;
  b: string;
  keys_a: string[];
  keys_b: string[];
  relation: Relation;
  confidence: "high" | "low";
  split: "train" | "held-out";
}

export interface ScoredPair {
  pair: Pair;
  simAB: number;
  sharedKey: boolean;
}

export function sharedKey(keysA: string[], keysB: string[]): boolean {
  const a = new Set(keysA.map((k) => k.toLowerCase()));
  return keysB.some((k) => a.has(k.toLowerCase()));
}

export function classifyPair(
  simAB: number,
  shared: boolean,
  floor: number,
  dedupCut: number
): Relation {
  if (simAB >= dedupCut) return "duplicate";
  if (simAB >= floor && shared) return "contradiction";
  return "independent";
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsx --test test/calibrate-lib.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/calibrate-lib.ts test/calibrate-lib.test.ts
git commit -m "feat(bench): add pair classification mirroring the graph decision path"
```

---

### Task 3: Metrics (`prf`, `range`)

**Files:**
- Modify: `bench/calibrate-lib.ts`
- Test: `test/calibrate-lib.test.ts` (append)

**Interfaces:**
- Consumes: `ScoredPair`, `classifyPair`, `Relation`.
- Produces:
  - `interface PRF { p: number; r: number; f1: number }`
  - `interface Scorecard { perClass: Record<Relation, PRF>; macroF1: number }`
  - `prf(scored: ScoredPair[], floor: number, dedupCut: number): Scorecard`
  - `range(lo: number, hi: number, step: number): number[]` (inclusive, rounded to 2 dp)

- [ ] **Step 1: Write the failing test**

```typescript
// append to test/calibrate-lib.test.ts
import { prf, range } from "../bench/calibrate-lib.ts";

function sp(relation: string, simAB: number, shared: boolean) {
  return { pair: { id: "x", a: "", b: "", keys_a: [], keys_b: [], relation, confidence: "high", split: "train" }, simAB, sharedKey: shared } as any;
}

test("prf computes per-class precision/recall/f1 and macro", () => {
  // floor=0.80, cut=0.94
  const scored = [
    sp("duplicate", 0.96, true),      // pred duplicate  -> TP duplicate
    sp("duplicate", 0.88, true),      // pred contradiction -> FN duplicate, FP contradiction
    sp("contradiction", 0.88, true),  // pred contradiction -> TP contradiction
    sp("independent", 0.50, false),   // pred independent -> TP independent
  ];
  const sc = prf(scored, 0.80, 0.94);
  // duplicate: TP1 FP0 FN1 -> p=1, r=0.5, f1=0.6667
  assert.equal(sc.perClass.duplicate.p, 1);
  assert.equal(sc.perClass.duplicate.r, 0.5);
  assert.ok(Math.abs(sc.perClass.duplicate.f1 - 2 / 3) < 1e-9);
  // contradiction: TP1 FP1 FN0 -> p=0.5, r=1, f1=0.6667
  assert.equal(sc.perClass.contradiction.p, 0.5);
  assert.equal(sc.perClass.contradiction.r, 1);
  // independent: TP1 FP0 FN0 -> f1=1
  assert.equal(sc.perClass.independent.f1, 1);
});

test("range is inclusive and 2-dp rounded", () => {
  assert.deepEqual(range(0.80, 0.82, 0.01), [0.8, 0.81, 0.82]);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/calibrate-lib.test.ts`
Expected: FAIL — `prf`/`range` not exported.

- [ ] **Step 3: Write minimal implementation**

```typescript
// append to bench/calibrate-lib.ts
const RELATIONS: Relation[] = ["duplicate", "contradiction", "independent"];

export interface PRF { p: number; r: number; f1: number; }
export interface Scorecard { perClass: Record<Relation, PRF>; macroF1: number; }

export function prf(scored: ScoredPair[], floor: number, dedupCut: number): Scorecard {
  const tp: Record<Relation, number> = { duplicate: 0, contradiction: 0, independent: 0 };
  const fp: Record<Relation, number> = { duplicate: 0, contradiction: 0, independent: 0 };
  const fn: Record<Relation, number> = { duplicate: 0, contradiction: 0, independent: 0 };
  for (const s of scored) {
    const pred = classifyPair(s.simAB, s.sharedKey, floor, dedupCut);
    const act = s.pair.relation;
    if (pred === act) tp[act]++;
    else { fp[pred]++; fn[act]++; }
  }
  const perClass = {} as Record<Relation, PRF>;
  let macro = 0;
  for (const c of RELATIONS) {
    const p = tp[c] + fp[c] === 0 ? 0 : tp[c] / (tp[c] + fp[c]);
    const r = tp[c] + fn[c] === 0 ? 0 : tp[c] / (tp[c] + fn[c]);
    const f1 = p + r === 0 ? 0 : (2 * p * r) / (p + r);
    perClass[c] = { p, r, f1 };
    macro += f1;
  }
  return { perClass, macroF1: macro / RELATIONS.length };
}

export function range(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsx --test test/calibrate-lib.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/calibrate-lib.ts test/calibrate-lib.test.ts
git commit -m "feat(bench): add per-class P/R/F1 scorecard and range helper"
```

---

### Task 4: Prior-aware FP + split (`priorWeightedFP`, `splitPairs`)

**Files:**
- Modify: `bench/calibrate-lib.ts`
- Test: `test/calibrate-lib.test.ts` (append)

**Interfaces:**
- Consumes: `ScoredPair`, `classifyPair`, `Pair`.
- Produces:
  - `priorWeightedFP(scored: ScoredPair[], floor: number, dedupCut: number, indepPrior: number): number` — `indepPrior * (falseFlags / independentCount)`, where a false flag is a truly-`independent` pair predicted non-independent. Returns 0 if no independent pairs.
  - `splitPairs(scored: ScoredPair[]): { train: ScoredPair[]; heldOut: ScoredPair[] }` — partition by `pair.split`.

- [ ] **Step 1: Write the failing test**

```typescript
// append to test/calibrate-lib.test.ts
import { priorWeightedFP, splitPairs } from "../bench/calibrate-lib.ts";

test("priorWeightedFP weights the independent false-flag rate by the prior", () => {
  // 2 independent pairs; one gets mis-flagged (in band + shared) -> rate 0.5
  const scored = [
    sp("independent", 0.88, true),   // pred contradiction -> false flag
    sp("independent", 0.50, false),  // pred independent -> correct
  ];
  assert.equal(priorWeightedFP(scored, 0.80, 0.94, 0.95), 0.95 * 0.5);
  assert.equal(priorWeightedFP([], 0.80, 0.94, 0.95), 0);
});

test("splitPairs partitions by split field", () => {
  const a = sp("duplicate", 0.96, true); a.pair.split = "train";
  const b = sp("contradiction", 0.88, true); b.pair.split = "held-out";
  const { train, heldOut } = splitPairs([a, b]);
  assert.equal(train.length, 1);
  assert.equal(heldOut.length, 1);
  assert.equal(heldOut[0].pair.relation, "contradiction");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/calibrate-lib.test.ts`
Expected: FAIL — `priorWeightedFP`/`splitPairs` not exported.

- [ ] **Step 3: Write minimal implementation**

```typescript
// append to bench/calibrate-lib.ts
export function priorWeightedFP(
  scored: ScoredPair[],
  floor: number,
  dedupCut: number,
  indepPrior: number
): number {
  const indeps = scored.filter((s) => s.pair.relation === "independent");
  if (indeps.length === 0) return 0;
  const falseFlags = indeps.filter(
    (s) => classifyPair(s.simAB, s.sharedKey, floor, dedupCut) !== "independent"
  ).length;
  return indepPrior * (falseFlags / indeps.length);
}

export function splitPairs(scored: ScoredPair[]): { train: ScoredPair[]; heldOut: ScoredPair[] } {
  return {
    train: scored.filter((s) => s.pair.split === "train"),
    heldOut: scored.filter((s) => s.pair.split === "held-out"),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsx --test test/calibrate-lib.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/calibrate-lib.ts test/calibrate-lib.test.ts
git commit -m "feat(bench): add prior-weighted false-flag rate and train/held-out split"
```

---

### Task 5: Sweep + separability + orchestrator

**Files:**
- Modify: `bench/calibrate-lib.ts`
- Test: `test/calibrate-lib.test.ts` (append)

**Interfaces:**
- Consumes: all of the above.
- Produces:
  - `interface SweepRow { floor: number; dedupCut: number; macroF1: number; scorecard: Scorecard }`
  - `sweep(scored: ScoredPair[], floors: number[], cuts: number[]): SweepRow[]` — skips rows where `floor >= dedupCut`.
  - `bestByMacroF1(rows: SweepRow[]): SweepRow`
  - `interface Separability { floor: number; dedupCut: number; dupF1: number; contraF1: number; minF1: number }`
  - `bestJointSeparability(rows: SweepRow[]): Separability` — the row maximizing `min(dupF1, contraF1)`. Low `minF1` is the 1순위 evidence that cosine alone cannot separate duplicate from contradiction.
  - `interface CalibrationResult { trainN: number; heldOutN: number; best: SweepRow; heldOut: Scorecard; overfitDelta: number; separability: Separability; priorFP: number }`
  - `calibrate(scored: ScoredPair[], opts: { floors: number[]; cuts: number[]; indepPrior: number }): CalibrationResult`

- [ ] **Step 1: Write the failing test**

```typescript
// append to test/calibrate-lib.test.ts
import { sweep, bestByMacroF1, bestJointSeparability, calibrate } from "../bench/calibrate-lib.ts";

test("sweep skips floor>=cut and bestByMacroF1 picks the max", () => {
  const scored = [sp("duplicate", 0.96, true), sp("contradiction", 0.88, true), sp("independent", 0.5, false)];
  const rows = sweep(scored, [0.80, 0.95], [0.90, 0.94]);
  assert.ok(rows.every((r) => r.floor < r.dedupCut), "no floor>=cut rows");
  const best = bestByMacroF1(rows);
  assert.ok(best.macroF1 >= Math.max(...rows.map((r) => r.macroF1)) - 1e-9);
});

test("bestJointSeparability reports the best simultaneous dup/contra F1", () => {
  // dup pair at 0.96, contra pair at 0.95 (above cut 0.94 -> misclassified as duplicate)
  // No cut cleanly separates: raising cut to catch the 0.95 contra would drop the 0.96 dup.
  const scored = [
    sp("duplicate", 0.96, true),
    sp("contradiction", 0.95, true),
  ];
  const rows = sweep(scored, [0.80], [0.94, 0.97]);
  const sep = bestJointSeparability(rows);
  assert.ok(sep.minF1 < 1, "classes overlap -> cannot get both F1=1 (1순위 evidence)");
});

test("calibrate ties it together with overfit delta", () => {
  const tr = sp("duplicate", 0.96, true); tr.pair.split = "train";
  const ho = sp("contradiction", 0.88, true); ho.pair.split = "held-out";
  const res = calibrate([tr, ho], { floors: [0.80], cuts: [0.94], indepPrior: 0.95 });
  assert.equal(res.trainN, 1);
  assert.equal(res.heldOutN, 1);
  assert.equal(typeof res.overfitDelta, "number");
  assert.ok(res.separability);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/calibrate-lib.test.ts`
Expected: FAIL — `sweep`/`bestByMacroF1`/`bestJointSeparability`/`calibrate` not exported.

- [ ] **Step 3: Write minimal implementation**

```typescript
// append to bench/calibrate-lib.ts
export interface SweepRow { floor: number; dedupCut: number; macroF1: number; scorecard: Scorecard; }

export function sweep(scored: ScoredPair[], floors: number[], cuts: number[]): SweepRow[] {
  const rows: SweepRow[] = [];
  for (const floor of floors) {
    for (const dedupCut of cuts) {
      if (floor >= dedupCut) continue;
      const scorecard = prf(scored, floor, dedupCut);
      rows.push({ floor, dedupCut, macroF1: scorecard.macroF1, scorecard });
    }
  }
  return rows;
}

export function bestByMacroF1(rows: SweepRow[]): SweepRow {
  return rows.reduce((best, r) => (r.macroF1 > best.macroF1 ? r : best));
}

export interface Separability {
  floor: number; dedupCut: number; dupF1: number; contraF1: number; minF1: number;
}

export function bestJointSeparability(rows: SweepRow[]): Separability {
  let best: Separability | null = null;
  for (const r of rows) {
    const dupF1 = r.scorecard.perClass.duplicate.f1;
    const contraF1 = r.scorecard.perClass.contradiction.f1;
    const minF1 = Math.min(dupF1, contraF1);
    if (!best || minF1 > best.minF1) {
      best = { floor: r.floor, dedupCut: r.dedupCut, dupF1, contraF1, minF1 };
    }
  }
  return best!;
}

export interface CalibrationResult {
  trainN: number;
  heldOutN: number;
  best: SweepRow;
  heldOut: Scorecard;
  overfitDelta: number;
  separability: Separability;
  priorFP: number;
}

export function calibrate(
  scored: ScoredPair[],
  opts: { floors: number[]; cuts: number[]; indepPrior: number }
): CalibrationResult {
  const { train, heldOut } = splitPairs(scored);
  const rows = sweep(train, opts.floors, opts.cuts);
  const best = bestByMacroF1(rows);
  const heldOutScore = prf(heldOut, best.floor, best.dedupCut);
  return {
    trainN: train.length,
    heldOutN: heldOut.length,
    best,
    heldOut: heldOutScore,
    overfitDelta: Math.abs(best.macroF1 - heldOutScore.macroF1),
    separability: bestJointSeparability(rows),
    priorFP: priorWeightedFP(train, best.floor, best.dedupCut, opts.indepPrior),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsx --test test/calibrate-lib.test.ts`
Expected: PASS.

- [ ] **Step 5: Run typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add bench/calibrate-lib.ts test/calibrate-lib.test.ts
git commit -m "feat(bench): add threshold sweep, separability signal, and calibrate orchestrator"
```

---

### Task 6: Runner + npm script

**Files:**
- Create: `bench/calibrate.ts`
- Modify: `package.json` (add `bench:calibrate` script)

**Interfaces:**
- Consumes: `calibrate`, `sharedKey`, `range`, `Pair`, `ScoredPair` from `./calibrate-lib.ts`; `embedTextAsync` from `../src/embedding.ts`.
- Produces: a CLI that prints a scorecard. No exported API.

- [ ] **Step 1: Add the npm script**

In `package.json` `scripts`, after `"bench:keynav": "tsx bench/keynav.ts"`, add:

```json
    "bench:calibrate": "tsx bench/calibrate.ts"
```

- [ ] **Step 2: Write the runner**

```typescript
// bench/calibrate.ts
// Dedup/contradiction threshold calibration. Builds simAB per labeled pair with the
// real embedder, sweeps thresholds on the train split, validates on held-out, and
// prints the 1순위 separability signal. Run: npm run bench:calibrate
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

import { sharedKey, calibrate, range, type Pair, type ScoredPair } from "./calibrate-lib.ts";
const { embedTextAsync } = await import("../src/embedding.ts");
const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na === 0 || nb === 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

const raw = JSON.parse(await readFile(resolve("bench/pairs.json"), "utf-8")) as { pairs: Pair[] };

const scored: ScoredPair[] = [];
for (const pair of raw.pairs) {
  const ea = await embedTextAsync(pair.a);
  const eb = await embedTextAsync(pair.b);
  scored.push({ pair, simAB: cosine(ea, eb), sharedKey: sharedKey(pair.keys_a, pair.keys_b) });
}

const res = calibrate(scored, {
  floors: range(0.75, 0.90, 0.05),
  cuts: range(0.88, 0.99, 0.01),
  indepPrior: 0.95,
});

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
console.log(`\nsuper-memory calibration — model=${LOCAL_EMBEDDING_MODEL} | train=${res.trainN} held-out=${res.heldOutN}`);
console.log("─".repeat(64));
console.log(`BEST (train macro-F1): floor=${res.best.floor} cut=${res.best.dedupCut} -> ${res.best.macroF1.toFixed(2)}`);
console.log(`HELD-OUT macro-F1: ${res.heldOut.macroF1.toFixed(2)}  Δ=${res.overfitDelta.toFixed(2)} ${res.overfitDelta > 0.10 ? "⚠ OVERFIT" : "OK"}`);
console.log(`prior-weighted FP (indep prior 0.95): ${pct(res.priorFP)}`);
console.log("─".repeat(64));
const s = res.separability;
console.log(`best joint dup/contra F1: dup=${s.dupF1.toFixed(2)} contra=${s.contraF1.toFixed(2)} min=${s.minF1.toFixed(2)} @ floor=${s.floor} cut=${s.dedupCut}`);
if (s.minF1 < 0.75) {
  console.log(`⚠ duplicate and contradiction are not separable by cosine alone (min joint F1 ${s.minF1.toFixed(2)}).`);
  console.log(`  This is the 1순위 evidence: a token-level discriminator (harness v2) is needed.`);
}
```

- [ ] **Step 3: Run the calibration (smoke test)**

Run: `npm run bench:calibrate`
Expected: prints the scorecard block with `BEST`, `HELD-OUT`, `prior-weighted FP`, and the `best joint dup/contra F1` line — no errors/stack traces. (First run downloads bge-m3 ~570MB if not cached.)

- [ ] **Step 4: Run the full non-live test suite (no regressions)**

Run: `npx tsx --test test/*.test.ts`
Expected: all pass (existing 77 + new corpus/lib tests).

- [ ] **Step 5: Commit**

```bash
git add bench/calibrate.ts package.json
git commit -m "feat(bench): add calibrate runner and bench:calibrate script"
```

---

## Self-Review

**1. Spec coverage:**
- ① schema (pairs.json + confidence) → Task 1 ✓
- ② harness pair-cosine + decision mirror → Task 2 (classify) + Task 6 (cosine/embed) ✓
- ③ metrics P/R/F1 + sweep + held-out delta + prior-FP + 1순위 signal → Tasks 3,4,5,6 ✓
- ④ file layout (pairs.json, calibrate.ts, LABELING.md, package script) → Tasks 1,6 ✓
- Refinement 1 (confidence + LABELING) → Task 1 ✓
- Refinement 2 (prior-FP) → Task 4/6 ✓
- Refinement 3 (generation diversity) → LABELING.md (Task 1) ✓
- Refinement 4 (stratified split + honesty) → Task 1 (split field) + Task 5 (delta is the signal) ✓
- Note: spec's "contra-F1 ceiling unaffected by cut" illustrative example is replaced by the more accurate `bestJointSeparability` (min joint F1) signal — documented in Task 5 interface.

**2. Placeholder scan:** No TBD/TODO; every code step has complete code. Task 1 Step 3 asks the author to extend seed pairs to ≥30 (a data-authoring task, not a code placeholder) — the schema, example rows, and the validating test are all concrete.

**3. Type consistency:** `Pair`, `ScoredPair`, `Relation`, `Scorecard`, `PRF`, `SweepRow`, `Separability`, `CalibrationResult` defined in Task 2/3/5 and consumed consistently in Tasks 4–6. `classifyPair(simAB, shared, floor, dedupCut)` signature identical across all call sites. `calibrate(scored, {floors, cuts, indepPrior})` matches runner usage in Task 6.
