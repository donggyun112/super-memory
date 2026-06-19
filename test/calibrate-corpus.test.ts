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
