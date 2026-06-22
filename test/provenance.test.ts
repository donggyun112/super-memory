// Provenance storage contract: a `source` object passed to add() must round-trip
// intact through read_memory(), and a memory saved without one stays null. The MCP
// `remember`/`correct`/`remember_batch` handlers rely on this to stamp every saved
// memory with its session/tool/timestamp.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
// Distinct one-hot vectors keyed off text length so the two memories never dedup.
function vec(tx: string): number[] {
  const v = new Array(16).fill(0);
  v[tx.length % 16] = 1;
  return v;
}

test("source provenance round-trips through add() and read_memory()", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-prov-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?prov=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  const source = {
    session: "sess-abc",
    tool: "remember",
    saved_at: "2026-06-22T00:00:00.000Z",
    conversation: "conv-1",
  };
  const [mid] = await g.add("a fact that should carry provenance here", ["provkeyone"], { source });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const res: any = await g.readMemory(mid, null, null);
  assert.equal(res.memory.source?.session, "sess-abc");
  assert.equal(res.memory.source?.tool, "remember");
  assert.equal(res.memory.source?.conversation, "conv-1");

  // Baseline: a memory saved with no source stays null (the old behavior).
  const [mid2] = await g.add("short different fact", ["provkeytwo"], {});
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const res2: any = await g.readMemory(mid2, null, null);
  assert.equal(res2.memory.source, null);
});
