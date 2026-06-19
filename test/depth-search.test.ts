// recall(min_depth) filters to well-established memories: only those whose depth (raised
// each recall) is >= the floor are returned. min_depth=0 (default) keeps the old behavior.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
function vec(t: string): number[] {
  const m: Record<string, number[]> = {
    QQ: [1, 0, 0, 0], DEEP: [1, 0, 0, 0], SHALLOW: [0.9, 0, 0.4359, 0],
    kd: [0, 1, 0, 0], ks: [0, 0, 0, 1],
  };
  return m[t] ?? [0, 1, 0, 0];
}

test("min_depth returns only memories at or above the depth floor", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-depth-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?depth=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  const [deep] = await g.add("DEEP", ["kd"], {});
  const [shallow] = await g.add("SHALLOW", ["ks"], {});
  g.memories[deep].depth = 0.9;     // well-established
  g.memories[shallow].depth = 0.2;  // recent/unverified

  // No floor → both returned.
  const all = (await g.recall("QQ", 10, null, false, 2, 0, 0, 0, 0, 0)) as any[];
  assert.deepEqual(new Set(all.map((m) => m.id)), new Set([deep, shallow]), "min_depth=0 returns both");

  // Floor 0.5 → only the deep memory.
  const deepOnly = (await g.recall("QQ", 10, null, false, 2, 0, 0, 0, 0, 0.5)) as any[];
  assert.deepEqual(deepOnly.map((m) => m.id), [deep], `min_depth=0.5 must drop the shallow memory, got ${deepOnly.map((m) => m.content).join(",")}`);
});
