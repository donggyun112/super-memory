// Concurrency verification. The graph serializes mutations with one Mutex (_lock) and
// disk I/O with another (_saveLock); recall reads/flushes outside _lock. These tests hammer
// the public API concurrently and assert the structural invariants that the locking must
// uphold: no lost updates, symmetric key<->memory links, no dangling refs, atomic disk
// writes, and recall never observing torn state. They also pin down the one intentional
// non-atomicity (best-effort dedup) as non-corrupting rather than silently regressing it.
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
const D = 256;
// Deterministic embedder: identical text -> identical vector (dedup fires); distinct text ->
// near-orthogonal one-hot (no accidental dedup or key<->content auto-linking).
function vec(t: string): number[] {
  let h = 0;
  for (let i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) >>> 0;
  const v = new Array(D).fill(0);
  v[h % D] = 1;
  return v;
}

async function freshGraph(t: { after: (fn: () => unknown) => void }) {
  const dir = await mkdtemp(join(tmpdir(), "sm-conc-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?conc=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  return { g, dir, mg };
}

// Asserts the graph's internal indexes are mutually consistent and reference only live nodes.
function assertInvariants(g: any) {
  const k2m = g._keyToMems as Record<string, Map<string, number>>;
  const m2k = g._memToKeys as Record<string, Map<string, number>>;
  const keys = g.keys as Record<string, unknown>;
  const mems = g.memories as Record<string, unknown>;
  for (const [kid, mm] of Object.entries(k2m)) {
    assert.ok(kid in keys, `key->mem index references missing key ${kid}`);
    assert.ok(mm.size > 0, `key ${kid} left with an empty link map (should be pruned)`);
    for (const [mid, w] of mm) {
      assert.ok(mid in mems, `key->mem link ${kid}->${mid} points at a dangling memory`);
      assert.equal(m2k[mid]?.get(kid), w, `asymmetric link ${kid}<->${mid}`);
    }
  }
  for (const [mid, kk] of Object.entries(m2k)) {
    assert.ok(mid in mems, `mem->key index references missing memory ${mid}`);
    for (const [kid, w] of kk) {
      assert.equal(k2m[kid]?.get(mid), w, `asymmetric link ${mid}<->${kid}`);
    }
  }
}

test("concurrent distinct adds: no lost updates, links stay symmetric", async (t) => {
  const { g } = await freshGraph(t);
  const N = 40;
  const added = await Promise.all(
    Array.from({ length: N }, (_, i) => g.add(`distinct memory number ${i}`, [`key${i}`], {}))
  );
  const ids = added.map(([id]: [string, boolean]) => id);
  assert.equal(new Set(ids).size, N, "every concurrent add must yield a unique id");
  assert.equal(added.filter(([, dup]: [string, boolean]) => dup).length, 0, "distinct content must not be deduped");
  for (const id of ids) assert.ok(id in g.memories, `add ${id} was lost`);
  assert.equal(Object.keys(g.memories).length, N, "memory count must equal the number of adds");
  assertInvariants(g);
});

test("concurrent delete/add/recall chaos: stays consistent, recall never throws", async (t) => {
  const { g } = await freshGraph(t);
  const seed = await Promise.all(
    Array.from({ length: 20 }, (_, i) => g.add(`seed memory ${i}`, [`s${i}`], {}))
  );
  const seedIds = seed.map(([id]: [string, boolean]) => id);

  let recallThrew: string | null = null;
  await Promise.all([
    ...seedIds.slice(0, 10).map((id: string) => g.delete(id)),
    ...Array.from({ length: 10 }, (_, i) => g.add(`fresh memory ${i}`, [`f${i}`], {})),
    ...Array.from({ length: 20 }, (_, i) =>
      g.recall(`seed memory ${i}`, 5).catch((e: Error) => { recallThrew = e.message; })
    ),
  ]);
  assert.equal(recallThrew, null, `recall threw during concurrent writes: ${recallThrew}`);
  assertInvariants(g);
  for (const id of seedIds.slice(0, 10)) assert.ok(!(id in g.memories), `deleted memory ${id} survived`);
});

test("concurrent mutations produce an atomic, round-trippable disk image", async (t) => {
  const { g, dir, mg } = await freshGraph(t);
  await Promise.all(
    Array.from({ length: 25 }, (_, i) => g.add(`persisted memory ${i}`, [`p${i}`, "shared"], {}))
  );
  await g.flush();

  const raw = await readFile(join(dir, "graph.json"), "utf-8");
  assert.doesNotThrow(() => JSON.parse(raw), "graph.json must be a complete (non-torn) JSON document");

  const g2 = new mg.MemoryGraph();
  await g2.load();
  assert.equal(
    Object.keys(g2.memories).length,
    Object.keys(g.memories).length,
    "reloaded memory count must match in-memory state"
  );
  assert.equal(g2.linkCount, g.linkCount, "reloaded link count must match in-memory state");
  assertInvariants(g2);
});

// Dedup must hold under concurrency. add() detects duplicates and inserts under a SINGLE lock
// acquisition (atomic check+insert), and supersede() follows the supersession chain to the
// current live head, so concurrent identical adds collapse into one live memory instead of
// forking. The superseded predecessors remain as tombstones (read paths skip them) but there
// must be exactly one LIVE memory for the content.
test("concurrent identical adds dedupe to a single live memory", async (t) => {
  const { g } = await freshGraph(t);
  const M = 8;
  await Promise.all(
    Array.from({ length: M }, () => g.add("one identical sentence stored many times", ["dupkey"], {}))
  );
  const superseded = (g as any)._supersededBy as Record<string, string>;
  const live = Object.keys(g.memories).filter((id) => !(id in superseded));
  assert.equal(live.length, 1, `concurrent identical adds must leave exactly one live memory, got ${live.length}`);
  assertInvariants(g);
});
