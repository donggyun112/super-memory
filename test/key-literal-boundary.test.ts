// Literal key matching must respect word boundaries. A short common-noun key like "name"
// must NOT spuriously match inside a longer word ("namespace"), which would let a low-value
// key spike to score 1 and bury the genuinely relevant key. Whole-word mentions still match.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
// 6-dim controlled space. Query aligns with the topical key, is orthogonal to "name".
const vecs: Record<string, number[]> = {
  "namespace management": [1, 0, 0, 0, 0, 0],
  "the name field": [0, 0, 1, 0, 0, 0],
  관리: [0.95, 0.3122, 0, 0, 0, 0], // cos(query) ~= 0.95
  name: [0, 0, 1, 0, 0, 0], // cos(query) = 0 -> only a literal hit could surface it
  topicMem: [0, 0, 0, 1, 0, 0],
  personMem: [0, 0, 0, 0, 1, 0],
};
const vec = (t: string): number[] => vecs[t] ?? [0, 0, 0, 0, 0, 1];

async function freshGraph(t: any) {
  const dir = await mkdtemp(join(tmpdir(), "sm-litbound-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?lit=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  return g;
}

test("short key does not substring-match inside a longer word", async (t) => {
  const g = await freshGraph(t);
  await g.add("topicMem", ["관리"], {});
  await g.add("personMem", ["name"], {});

  const keys = (await g.searchKeys("namespace management", 10)) as Array<{ concept: string; score: number }>;
  const concepts = keys.map((k) => k.concept);
  // "name" is a substring of "namespace" but NOT a whole word -> must be excluded.
  assert.ok(!concepts.includes("name"), `"name" must not match "namespace management", got ${concepts.join(",")}`);
  // The genuinely relevant key must survive.
  assert.ok(concepts.includes("관리"), `relevant key "관리" missing, got ${concepts.join(",")}`);
});

test("whole-word mention still matches literally", async (t) => {
  const g = await freshGraph(t);
  await g.add("personMem", ["name"], {});
  const keys = (await g.searchKeys("the name field", 10)) as Array<{ concept: string }>;
  assert.ok(keys.map((k) => k.concept).includes("name"), "whole-word 'name' must match");
});
