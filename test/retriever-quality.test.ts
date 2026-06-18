import assert from "node:assert/strict";
import test from "node:test";

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

test("test embedder seam overrides embedTextAsync", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text) => (text === "hello" ? [1, 0] : [0, 1]));
  try {
    assert.deepEqual(await emb.embedTextAsync("hello"), [1, 0]);
    assert.deepEqual(await emb.embedTextAsync("world", "query"), [0, 1]);
  } finally {
    emb.__clearTestEmbedder();
  }
});

test("familyForModel maps bge-m3 aliases to bgem3", async () => {
  const { familyForModel } = await import("../src/embedding.ts");
  for (const name of ["bge-m3", "bgem3", "BAAI/bge-m3", "fast-bge-m3", "BGE_M3"]) {
    assert.equal(familyForModel(name), "bgem3", name);
  }
  assert.equal(familyForModel("multilingual-e5-large"), "e5");
  assert.equal(familyForModel("bge-small-en-v1.5"), "bge");
  assert.equal(familyForModel("all-minilm-l6-v2"), "minilm");
  assert.equal(familyForModel("nonexistent-model"), "unknown");
});

test("only e5 uses the passage/query prefix", async () => {
  const { usesE5Prefix } = await import("../src/embedding.ts");
  assert.equal(usesE5Prefix("e5"), true);
  assert.equal(usesE5Prefix("bgem3"), false);
  assert.equal(usesE5Prefix("bge"), false);
});

test("bgem3 threshold profile exists with expected fields", async () => {
  const { THRESHOLD_PROFILES } = await import("../src/embedding.ts");
  const p = THRESHOLD_PROFILES.bgem3;
  assert.equal(p.memoryDedup, 0.94);
  assert.equal(p.minScore, 0.55);
  assert.equal(p.contradiction, 0.88);
  // every profile must define the new fields
  for (const fam of ["openai", "e5", "bge", "minilm", "bgem3"]) {
    assert.equal(typeof THRESHOLD_PROFILES[fam].minScore, "number", fam);
    assert.equal(typeof THRESHOLD_PROFILES[fam].contradiction, "number", fam);
  }
});

test("customModelConfig throws a clear error when path is unset", async () => {
  const { customModelConfig } = await import("../src/embedding.ts");
  const saved = process.env.LOCAL_EMBEDDING_MODEL_PATH;
  delete process.env.LOCAL_EMBEDDING_MODEL_PATH;
  try {
    assert.throws(() => customModelConfig(), /LOCAL_EMBEDDING_MODEL_PATH/);
  } finally {
    if (saved !== undefined) process.env.LOCAL_EMBEDDING_MODEL_PATH = saved;
  }
});

test("passesAbsoluteGate compares raw similarity to floor", async () => {
  const { passesAbsoluteGate } = await import("../src/memoryGraph.ts");
  assert.equal(passesAbsoluteGate(0.6, 0.55), true);
  assert.equal(passesAbsoluteGate(0.55, 0.55), true);
  assert.equal(passesAbsoluteGate(0.4, 0.55), false);
});

test("recall returns [] when nothing clears the absolute gate", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-gate-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;

  const emb = await import("../src/embedding.ts");
  // memory content embeds to [1,0]; the noise query embeds orthogonally to [0,1].
  emb.__setTestEmbedder((text) =>
    text === "노이즈쿼리" ? [0, 1] : [1, 0]
  );
  t.after(() => emb.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?gate=1`);
  const g = new MemoryGraph();
  await g.load();
  await g.add("사용자는 커피를 좋아한다", ["커피"]);

  // Relevant query (embeds to [1,0], cos=1 with content) clears the gate.
  const hit = (await g.recall("커피", 5)) as any[];
  assert.ok(hit.length >= 1, "relevant query should return results");

  // Orthogonal noise query (cos=0) is below any positive minScore -> [].
  const miss = (await g.recall("노이즈쿼리", 5, null, false, 2, 0, 0.5)) as any[];
  assert.equal(miss.length, 0, "noise query should return nothing");
});
