// related() must stay navigable at scale: a neighbor connected by a RARE (specific)
// shared key should rank above one connected only by a HUB key (shared by many), and
// the result list must be capped so a hub doesn't flood the chain. This is what lets an
// LLM "찾아 들어가기" (recall -> related -> related) drill down meaningfully instead of
// drowning in hub neighbors.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
// Distinct one-hot vectors (24-dim) so nothing dedups or auto-links: contents/keys on
// fixed dims, each hubmem on its own dim, fallback (probe) on a dedicated dim.
function vec(t: string): number[] {
  const v = new Array(24).fill(0);
  const map: Record<string, number> = { ca: 0, cb: 1, cc: 2, cd: 3, ce: 4, cf: 5, HUB: 6, RARE: 7 };
  if (t in map) { v[map[t]] = 1; return v; }
  const m = t.match(/^hubmem(\d+)$/);
  if (m) { v[8 + Number(m[1])] = 1; return v; }
  v[23] = 1; return v;
}

test("related ranks rare-key neighbors above hub-key neighbors and caps output", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-relhub-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.SUPER_MEMORY_RELATED_LIMIT = "10";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?relhub=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  // A shares HUB with many, and RARE only with C.
  const [a] = await g.add("ca", ["HUB", "RARE"], {});
  const [c] = await g.add("cc", ["RARE"], {}); // rare-key neighbor of A
  // 12 hub-only neighbors so HUB is a big hub and total neighbors exceed the cap.
  for (let i = 0; i < 12; i++) await g.add(`hubmem${i}`, ["HUB"], {});

  const rel = g.getRelated(a) as any[];
  assert.ok(rel.length > 0, "should return neighbors");
  assert.equal(rel[0].id, c, `rare-key neighbor must rank #1, got ${JSON.stringify(rel.map((r) => r.shared_keys))}`);
  assert.ok(rel.length <= 10, `output must be capped at SUPER_MEMORY_RELATED_LIMIT=10, got ${rel.length}`);
});
