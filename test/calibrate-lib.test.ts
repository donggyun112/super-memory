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
