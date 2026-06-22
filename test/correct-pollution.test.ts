// correct() with no keys inherits the old keys. That is right for a same-topic update
// (Seoul -> Busan keeps "residence") but pollutes recall for an off-topic correction
// (strawberries -> peanut allergy keeps "strawberry"). Fix: drop inherited *concept*
// keys the corrected content has drifted away from; keep relevant ones and exact-match
// anchors. Topic-based test embedder: same dim = same topic (cosine 1), else cosine 0.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
function vec(tx: string): number[] {
  const v = new Array(16).fill(0);
  const t = tx.toLowerCase();
  if (t.includes("strawberr") || t.includes("딸기")) v[0] = 1;
  else if (t.includes("peanut") || t.includes("allerg")) v[1] = 1;
  else if (t.includes("residence") || t.includes("거주") || t.includes("seoul") || t.includes("busan") || t.includes("lives") || t.includes("moved")) v[2] = 1;
  else v[15] = 1;
  return v;
}

async function freshGraph(t: { after: (fn: () => unknown) => void }) {
  const dir = await mkdtemp(join(tmpdir(), "sm-pollute-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?pollute=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  return g;
}

test("off-topic correction without keys drops the now-irrelevant topic tag", async (t) => {
  const g = await freshGraph(t);
  const [mid] = await g.add("the user loves strawberries", ["strawberry", "딸기"], {});
  const nid = await g.supersede(mid, "the user is allergic to peanuts", {});
  const keys: string[] = g.getKeysForMemory(nid);
  assert.ok(
    !keys.includes("strawberry") && !keys.includes("딸기"),
    `off-topic memory should drop the stale topic, got: [${keys.join(", ")}]`
  );
});

test("same-topic correction without keys keeps the still-relevant key", async (t) => {
  const g = await freshGraph(t);
  const [mid] = await g.add("the user lives in Seoul", ["residence"], {});
  const nid = await g.supersede(mid, "the user moved to Busan", {});
  const keys: string[] = g.getKeysForMemory(nid);
  assert.ok(keys.includes("residence"), `same-topic memory should keep its key, got: [${keys.join(", ")}]`);
});
