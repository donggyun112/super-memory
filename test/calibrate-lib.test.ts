import assert from "node:assert/strict";
import test from "node:test";
import { sharedKey, classifyPair, prf, range, priorWeightedFP, splitPairs } from "../bench/calibrate-lib.ts";

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
