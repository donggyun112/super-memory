// read_memory currently calls save() on every call, rewriting the whole graph.json
// (O(graph) per read — measured 263ms @ 3k memories). Reinforcement signals
// (depth/access/link) are soft: they should accumulate in RAM and be flushed, not
// force a full-file write per read. Behavior test (no timing): a read must NOT rewrite
// graph.json; flush() must persist the accumulated depth.
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

test("read_memory defers persistence; flush() writes it", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-defer-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => { const v = new Array(8).fill(0); v[tx.length % 8] = 1; return v; });
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?defer=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  const [mid] = await g.add("a fact to reinforce by reading", ["reinforce-key"], {});
  const diskDepth = async () => JSON.parse(await readFile(join(dir, "graph.json"), "utf-8")).memories[mid].depth;

  assert.equal(await diskDepth(), 0, "new memory persists at depth 0");

  await g.readMemory(mid, null, null); // +0.05 in RAM
  await g.readMemory(mid, null, null); // +0.05 in RAM
  assert.equal(await diskDepth(), 0, "read_memory must NOT rewrite graph.json on every read");

  await g.flush();
  assert.ok((await diskDepth()) >= 0.1 - 1e-9, "flush() persists the accumulated depth");
});
