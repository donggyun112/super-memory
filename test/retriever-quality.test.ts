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

test("isShortConcept flags short or few-token concepts", async () => {
  const { isShortConcept } = await import("../src/embedding.ts");
  assert.equal(isShortConcept("Agent A"), true);   // 2 tokens
  assert.equal(isShortConcept("auth"), true);       // short
  assert.equal(isShortConcept("Agent B"), true);
  assert.equal(isShortConcept("distributed consensus protocol design"), false); // long + many tokens
});

test("short concept keys merge only on exact match", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-key-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;

  const emb = await import("../src/embedding.ts");
  // "Agent A" and "Agent B" embed nearly identically (cos ~0.9998) — high enough
  // that semantic merge WOULD merge them. The short-key guard must keep them apart.
  emb.__setTestEmbedder((text) =>
    text === "Agent A" ? [1, 0.02] : text === "Agent B" ? [1, 0.0] : [0.3, 0.95]
  );
  t.after(() => emb.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?key=1`);
  const g = new MemoryGraph();
  await g.load();

  const a1 = await g.findOrCreateKey("Agent A");
  const a2 = await g.findOrCreateKey("Agent A"); // exact repeat -> same key
  const b = await g.findOrCreateKey("Agent B");  // distinct short key -> new key

  assert.equal(a1, a2, "exact short repeat reuses the key");
  assert.notEqual(a1, b, "Agent A and Agent B must NOT merge");
});

test("inContradictionBand is [floor, dedup)", async () => {
  const { inContradictionBand } = await import("../src/embedding.ts");
  assert.equal(inContradictionBand(0.9, 0.88, 0.94), true);
  assert.equal(inContradictionBand(0.88, 0.88, 0.94), true);  // inclusive floor
  assert.equal(inContradictionBand(0.94, 0.88, 0.94), false); // exclusive dedup (=> duplicate)
  assert.equal(inContradictionBand(0.5, 0.88, 0.94), false);  // below band
});

test("conflicting memories sharing a key get mutual contradicts links", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-contra-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;

  const emb = await import("../src/embedding.ts");
  // Two facts about "프로젝트A": Postgres vs Mongo. Craft cos ~0.92 — inside the
  // bgem3 band [0.88, 0.94), so NOT a duplicate but flagged as a contradiction.
  // (cos([1,0],[0.92, 0.392]) = 0.92)
  emb.__setTestEmbedder((text) =>
    text.includes("Postgres") ? [1, 0] : text.includes("Mongo") ? [0.92, 0.392] : [0, 1]
  );
  t.after(() => emb.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?contra=1`);
  const g = new MemoryGraph();
  await g.load();

  const [first] = await g.add("프로젝트A는 Postgres를 쓴다", ["프로젝트A"]);
  const [second, wasDup] = await g.add("프로젝트A는 Mongo를 쓴다", ["프로젝트A"]);

  assert.equal(wasDup, false, "should NOT be treated as a duplicate");
  assert.ok(second in g.memories && first in g.memories, "both memories survive");
  assert.ok(g.memories[second].contradicts.includes(first), "new -> old contradicts link");
  assert.ok(g.memories[first].contradicts.includes(second), "old -> new contradicts link");

  const results = (await g.recall("프로젝트A Postgres", 5)) as any[];
  const r = results.find((x) => x.id === first);
  assert.ok(r && Array.isArray(r.contradicts) && r.contradicts.includes(second),
    "recall surfaces contradicts");
});

test("load defaults contradicts to []", async (t) => {
  const { mkdtemp, rm, writeFile } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-load-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;
  // Pre-existing memory WITHOUT a contradicts field (older schema).
  await writeFile(
    join(dataDir, "graph.json"),
    JSON.stringify({
      keys: {},
      memories: { m1: { id: "m1", content: "x", embedding: [1, 0], created_at: 0 } },
      links: [],
    }),
    "utf-8"
  );
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?load=1`);
  const g = new MemoryGraph();
  await g.load();
  assert.deepEqual(g.memories.m1.contradicts, []);
});
