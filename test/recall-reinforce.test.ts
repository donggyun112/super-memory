// Hebbian reinforcement scope: recall() must strengthen ONLY the top-ranked returned memory's
// matched-key links, not the whole returned tail. Reinforcing every returned memory inflates
// weak, frequently-co-retrieved-but-unused links over time (graph pollution).
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// 3-axis embedder. Query == M1's content axis (top hit, cos 1.0). M2 sits at cos 0.7 — high
// enough to be returned, but ranked below M1.
function vec(tx: string): number[] {
  const t = tx.toLowerCase();
  if (t.includes("secondary")) return [0.7, 0.7141428, 0]; // cos 0.7 to [1,0,0] — checked first
  if (t.includes("topic") || t.includes("primary") || t.includes("query")) return [1, 0, 0];
  return [0, 0, 1];
}

test("recall reinforces only the top-ranked result, not the lower-ranked returned tail", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-reinf-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(vec);
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?reinf=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  const [m1] = await g.add("primary topic answer", ["topic"]); // top hit (cos 1.0)
  const [m2] = await g.add("secondary topic note", ["topic"]); // returned, but ranked below (cos 0.7)
  const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "topic")!;

  const gw = g as unknown as { _getLinkWeight(k: string, m: string): number };
  const w1Before = gw._getLinkWeight(kid, m1);
  const w2Before = gw._getLinkWeight(kid, m2);

  const res = (await g.recall("topic query", 5)) as Array<{ id: string }>;
  // Sanity: both memories are returned, M1 first.
  assert.equal(res[0].id, m1, "M1 should rank first");
  assert.ok(res.some((r) => r.id === m2), "M2 should also be returned");

  const w1After = gw._getLinkWeight(kid, m1);
  const w2After = gw._getLinkWeight(kid, m2);
  assert.ok(w1After > w1Before, "top-ranked result's link must be reinforced");
  assert.equal(w2After, w2Before, "lower-ranked returned tail must NOT be reinforced");
});
