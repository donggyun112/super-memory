// Concurrency guarantees for recall():
//  1. The cross-encoder rerank runs OUTSIDE the graph lock, so a write can land
//     while a rerank is in flight (no long mutex hold). Asserted as a deadlock
//     detector: if the lock were held across rerank, the interleaved add would
//     block forever and the test would time out.
//  2. flush() runs outside the lock but save() is serialized (per-write unique
//     temp name + _saveLock), so a storm of concurrent recalls/adds leaves the
//     graph file valid and uncorrupted.
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
const vecs: Record<string, number[]> = {
  QQ: [1, 0, 0, 0, 0],
  A: [0.9, 0.4359, 0, 0, 0], // cos(QQ)=0.90
  B: [0.85, 0, 0.5268, 0, 0], // cos(QQ)=0.85
  ka: [0, 0, 0, 1, 0],
  kb: [0, 0, 0, 0, 1],
};
const vec = (t: string): number[] => {
  const extra = t.match(/^extra (\d+)$/);
  if (extra) {
    const angle = (Number(extra[1]) * 2 * Math.PI) / 10;
    return [0, 0, 0, Math.cos(angle), Math.sin(angle)];
  }
  return vecs[t] ?? [0, 0, 0, 1, 0];
};

async function freshGraph(t: any, tag: string) {
  const dir = await mkdtemp(join(tmpdir(), `sm-${tag}-`));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const rer = await import("../src/reranker.ts");
  t.after(() => rer.__clearTestReranker());

  const mg = await import(`../src/memoryGraph.ts?cc=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  return { g, rer, dir };
}

test("a write proceeds while a rerank is in flight (rerank is not under the lock)", async (t) => {
  const { g, rer } = await freshGraph(t, "cc-nolock");
  await g.add("A", ["ka"], {});
  await g.add("B", ["kb"], {});

  // Barrier: rerank announces it started, then blocks until the interleaved add
  // has fully completed. If recall held the lock across rerank, that add could
  // never acquire the lock → deadlock → this test times out (the assertion of the fix).
  let signalStarted!: () => void;
  const rerankStarted = new Promise<void>((res) => (signalStarted = res));
  let addDone = false;
  let releaseRerank!: () => void;
  const rerankGate = new Promise<void>((res) => (releaseRerank = res));

  rer.__setTestReranker(async (_q, texts) => {
    signalStarted();
    await rerankGate; // held open until the add lands
    return texts.map((_t, i) => 10 - i);
  });

  const recallP = g.recall("QQ", 5, null, false, 2, 0, 0, 0, 0) as Promise<any[]>;
  await rerankStarted; // recall is now parked inside rerank, lock released

  // This add must be able to acquire the lock RIGHT NOW, mid-rerank.
  const addP = g.add("A second memory", ["kc"], {}).then(() => {
    addDone = true;
  });
  await addP;
  assert.equal(addDone, true, "add must complete while rerank is in flight");

  releaseRerank(); // let rerank (and thus recall) finish
  const res = await recallP;
  assert.ok(res.length > 0, "recall still returns its results after the unlocked rerank");
});

test("concurrent recalls + adds leave the graph file valid (no save race)", async (t) => {
  const { g, rer, dir } = await freshGraph(t, "cc-save");
  await g.add("A", ["ka"], {});
  await g.add("B", ["kb"], {});
  rer.__setTestReranker((_q, texts) => texts.map((_t, i) => 10 - i));

  // Fire a storm: 20 recalls (each markDirty + flush) interleaved with 10 adds.
  // Each flush() runs outside the lock; only serialized save() + unique temp names
  // keep them from clobbering the file.
  const ops: Promise<unknown>[] = [];
  for (let i = 0; i < 20; i++) ops.push(g.recall("QQ", 5, null, false, 2, 0, 0, 0, 0));
  for (let i = 0; i < 10; i++) ops.push(g.add(`extra ${i}`, [`k${i}`], {}));
  await Promise.all(ops);

  // The persisted graph must be parseable and complete (no half-written/renamed file).
  const raw = await readFile(join(dir, "graph.json"), "utf-8");
  const parsed = JSON.parse(raw); // throws if corrupt
  assert.ok(parsed.memories && typeof parsed.memories === "object");
  assert.ok(Object.keys(parsed.memories).length >= 12, "all adds persisted");

  // No leftover temp files dangling after the storm settles.
  const stragglers = (await readdir(dir)).filter((f) => f.endsWith(".tmp"));
  assert.equal(stragglers.length, 0, `temp files must be renamed away, found: ${stragglers}`);
});
