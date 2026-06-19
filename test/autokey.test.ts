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
