import assert from "node:assert/strict";
import test from "node:test";
import { RecallBuffer, decidePromotion } from "../src/autokey.ts";

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

test("TTL boundary: entry at cutoff edge is still fresh, expires one second later", () => {
  let clock = 0;
  const buf = new RecallBuffer({ ttlSeconds: 300, now: () => clock });
  buf.push({ queryText: "q", queryEmbedding: [1], weakKeyScores: new Map([["k1", 0.9]]) });

  // At now=300: cutoff = 300 - 300 = 0; ts=0; 0 < 0 is false → still fresh
  clock = 300;
  assert.ok(buf.consumeWeakMatch("k1"), "entry at exact cutoff boundary should still be fresh");

  // Push again for the expiry check (previous was consumed)
  clock = 0;
  const buf2 = new RecallBuffer({ ttlSeconds: 300, now: () => clock });
  buf2.push({ queryText: "q", queryEmbedding: [1], weakKeyScores: new Map([["k1", 0.9]]) });
  clock = 301;
  assert.equal(buf2.consumeWeakMatch("k1"), null, "entry should expire at now=301");
});

test("capacity evicts oldest entries", () => {
  const buf = new RecallBuffer({ capacity: 2, now: () => 0 });
  buf.push({ queryText: "a", queryEmbedding: [1], weakKeyScores: new Map([["k1", 0.9]]) });
  buf.push({ queryText: "b", queryEmbedding: [1], weakKeyScores: new Map([["k2", 0.9]]) });
  buf.push({ queryText: "c", queryEmbedding: [1], weakKeyScores: new Map([["k3", 0.9]]) });
  assert.equal(buf.size(), 2);
  assert.equal(buf.consumeWeakMatch("k1"), null); // evicted
  assert.ok(buf.consumeWeakMatch("k2")); // second-oldest survived
  assert.ok(buf.consumeWeakMatch("k3"));
});

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

test("decidePromotion: cosine exactly at aliasThreshold -> alias", () => {
  assert.equal(decidePromotion({ ...base, count: 3, cosine: 0.86 }), "alias");
});

test("decidePromotion: cosine exactly at newKeyThreshold -> newKey", () => {
  assert.equal(decidePromotion({ ...base, count: 3, cosine: 0.62 }), "newKey");
});
