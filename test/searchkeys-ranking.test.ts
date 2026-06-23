// searchKeys ranking: a literal *token* match on a generic concept key must not bury a
// higher-relevance semantic answer. Literal-first ordering is kept only for entity keys
// (name/proper_noun), where an exact literal hit IS the answer — mirroring recall()'s policy.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// 3-axis topic embedder. Query "메모리 도구" sits at cos 0.3 to the 메모리 axis and cos 0.7
// to the 연상 axis, so the semantically-closer key (연상기억장치) should win even though the
// query literally contains the word "메모리".
function vec(tx: string): number[] {
  const t = tx.toLowerCase();
  if (t === "메모리") return [1, 0, 0];
  if (t === "연상기억장치") return [0, 1, 0];
  if (t === "메모리 도구") return [0.3, 0.7, 0.6480741];
  if (t.includes("연상")) return [0, 1, 0]; // memory B content
  if (t.includes("저장")) return [1, 0, 0]; // memory A content
  return [0, 0, 1];
}

async function freshGraph(t: import("node:test").TestContext) {
  const dir = await mkdtemp(join(tmpdir(), "sm-rank-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(vec);
  t.after(() => emb.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?rank=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  return g;
}

test("a partial literal token match does not outrank a higher-cosine semantic answer", async (t) => {
  const g = await freshGraph(t);
  await g.add("메모리 저장 기능", ["메모리"]); // literal-token match on the query
  await g.add("연상 작용", ["연상기억장치"]); // the semantically-closer answer (cos 0.7)

  const r = (await g.searchKeys("메모리 도구")) as Array<{ concept: string; score: number }>;
  assert.equal(
    r[0].concept,
    "연상기억장치",
    "the cos-0.7 semantic key must rank above the cos-0.3 literal-token key"
  );
});

test("name/proper_noun literal match still ranks first (entity precedence preserved)", async (t) => {
  const g = await freshGraph(t);
  // A concept key sitting at cos 0.7 to the query, plus a NAME key literally present in it.
  const [mid] = await g.add("연상 작용", ["연상기억장치"]); // concept, cos 0.7
  const kidName = await g.findOrCreateKey("메모리", "name"); // entity literally in the query
  (g as unknown as { _link(k: string, m: string): void })._link(kidName, mid);

  const r = (await g.searchKeys("메모리 도구")) as Array<{ key_id: string; key_type: string }>;
  assert.equal(r[0].key_type, "name", "an exact entity (name) literal hit outranks a semantic concept");
});
