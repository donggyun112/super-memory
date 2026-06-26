// read_memory surfaces a ready-to-run get_conversation call (`trace`) when the
// memory carries a host transcript link, so the agent can drill to the original
// conversation without remapping host_session/host_agent/host_turn into the
// tool's session_id/agent/turn params. No host link → no trace.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
function vec(tx: string): number[] {
  const v = new Array(16).fill(0);
  v[tx.length % 16] = 1;
  return v;
}

async function freshGraph(t: any) {
  const dir = await mkdtemp(join(tmpdir(), "sm-trace-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?trace=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  return g;
}

test("read_memory returns a get_conversation trace when a host link is present", async (t) => {
  const g = await freshGraph(t);
  const source = {
    session: "keymem-srv",
    tool: "remember",
    host_agent: "claude",
    host_session: "1cf71208-353e-4f03-b8ac-2eee63aa8e8f",
    host_turn: 123,
  };
  const [mid] = await g.add("a fact worth tracing to its source", ["tracekeyone"], { source });
  const res: any = await g.readMemory(mid, null, null);

  assert.deepEqual(res.trace, {
    tool: "get_conversation",
    args: { session_id: "1cf71208-353e-4f03-b8ac-2eee63aa8e8f", agent: "claude", turn: 123 },
  });
});

test("read_memory has no trace when the memory carries no host link", async (t) => {
  const g = await freshGraph(t);
  const [mid] = await g.add("a fact with no host link", ["tracekeytwo"], {
    source: { session: "keymem-srv", tool: "remember" },
  });
  const res: any = await g.readMemory(mid, null, null);

  assert.equal(res.trace, null);
});
