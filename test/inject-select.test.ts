import assert from "node:assert/strict";
import test from "node:test";
import { selectInject, type InjectCandidate } from "../src/inject.ts";

// candidates in relevance order (best first); depths vary
const C: InjectCandidate[] = [
  { id: "a", depth: 0.10 },
  { id: "b", depth: 0.80 },
  { id: "c", depth: 0.30 },
  { id: "d", depth: 0.05 },
];

test("no opts: keeps relevance order", () => {
  assert.deepEqual(selectInject(C, 2), ["a", "b"]);
});

test("preferDepth: orders the relevance-selected window deepest-first (never widens membership)", () => {
  // Membership stays the top-2 by relevance (a, b); preferDepth only reorders WITHIN them so the
  // confirmed (deep) one leads. It must NOT reach past the window to pull the deeper rank-2 'c' in.
  assert.deepEqual(selectInject(C, 2, { preferDepth: true }), ["b", "a"]); // depth 0.80, 0.10
});

test("preferDepth must NOT promote a deep but low-relevance candidate over relevant ones", () => {
  // ICJC-doc regression: a frequently-read (deep) but barely-relevant doc sits at the BOTTOM of
  // relevance order (BM25 0.013, read 8x → depth 0.40). It must never displace the top relevance
  // hits — even though they are shallow new memories.
  const cands: InjectCandidate[] = [
    { id: "rel1", depth: 0.05 }, // most relevant, shallow (new memory)
    { id: "rel2", depth: 0.08 }, // relevant, shallow
    { id: "junk", depth: 0.40 }, // barely relevant, but deep
  ];
  const r = selectInject(cands, 2, { preferDepth: true });
  assert.ok(!r.includes("junk"), `deep-but-irrelevant 'junk' must not enter top-2, got ${r.join(",")}`);
  assert.deepEqual([...r].sort(), ["rel1", "rel2"]);
});

test("exploreShallow: reserves a slot for the shallowest relevant candidate", () => {
  // preferDepth picks [b,c]; explore must surface the global shallowest (d, 0.05)
  const r = selectInject(C, 2, { preferDepth: true, exploreShallow: true });
  assert.ok(r.includes("d"), `shallowest 'd' should be surfaced, got ${r.join(",")}`);
  assert.ok(r.includes("b"), `deepest 'b' should remain, got ${r.join(",")}`);
  assert.equal(r.length, 2);
});

test("exploreShallow is a no-op when all candidates already fit", () => {
  assert.equal(selectInject(C, 4, { exploreShallow: true }).length, 4);
});
