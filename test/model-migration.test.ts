// A backend switch between two models of the SAME embedding dimension (e.g.
// e5-large ↔ bge-m3, both 1024-d) must still re-embed: the vectors live in
// different spaces, so leaving the stored embeddings in place silently corrupts
// every cosine. The dimension-only guard cannot see this; an embedding-model
// fingerprint stored in the graph can.
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let importCounter = 0;

async function loadModules(dataDir: string) {
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;
  const emb = await import("../src/embedding.ts");
  const mg = await import(`../src/memoryGraph.ts?modelmig=${importCounter++}`);
  return { emb, mg };
}

test("re-embeds when the embedding model changes at the same dimension", async (t) => {
  const dataDir = await mkdtemp(join(tmpdir(), "sm-modelmig-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));

  // Graph written by a prior backend: 2-dim vectors, fingerprinted as e5.
  const stored = {
    meta: { embeddingFingerprint: "local:fast-multilingual-e5-large" },
    keys: {
      k1: { id: "k1", concept: "딸기", embedding: [1, 0], key_type: "concept" },
    },
    memories: {
      m1: {
        id: "m1", content: "사용자는 딸기를 좋아한다", embedding: [1, 0],
        created_at: 1000, source: null, supersedes: null, depth: 0.7,
        access_count: 5, last_accessed: 1000, namespace: "default",
        ttl: null, links: [], contradicts: [],
      },
    },
    links: [{ key_id: "k1", memory_id: "m1", weight: 1.0 }],
  };
  await writeFile(join(dataDir, "graph.json"), JSON.stringify(stored), "utf-8");

  // Current backend: a DIFFERENT model (bge-m3) that happens to produce the SAME
  // 2-dim vectors. Same dim, different space → must re-embed. The test embedder
  // returns a distinct vector so a re-embed is observable.
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const { emb, mg } = await loadModules(dataDir);
  emb.__setTestEmbedder(() => [0, 1]); // same dim (2), different value
  t.after(() => emb.__clearTestEmbedder());

  const g = new mg.MemoryGraph();
  await g.load();

  assert.deepEqual(
    g.memories.m1.embedding, [0, 1],
    "memory must be re-embedded with the new model, not left at the stored e5 vector"
  );
  assert.deepEqual(g.keys.k1.embedding, [0, 1], "keys must be re-embedded too");
  assert.equal(g.memories.m1.depth, 0.7, "depth preserved through migration");
  assert.equal(g.memories.m1.access_count, 5, "access_count preserved");

  const backups = (await readdir(dataDir)).filter((f) => f.includes(".bak."));
  assert.ok(backups.length >= 1, `pre-migration backup expected, found: ${backups.join(",")}`);

  const persisted = JSON.parse(await readFile(join(dataDir, "graph.json"), "utf-8"));
  assert.equal(
    persisted.meta?.embeddingFingerprint, "local:bge-m3",
    "persisted fingerprint must reflect the new model"
  );
  assert.deepEqual(persisted.memories.m1.embedding, [0, 1], "migration persisted to disk");
});

test("SUPER_MEMORY_FORCE_REEMBED re-embeds a legacy graph (no fingerprint, same dim)", async (t) => {
  // A graph written before fingerprinting (meta absent) gives no way to detect a
  // same-dimension model swap automatically. FORCE_REEMBED is the one-shot escape
  // hatch for that switch: re-embed unconditionally and stamp the fingerprint.
  const dataDir = await mkdtemp(join(tmpdir(), "sm-modelmig-force-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));

  const legacy = {
    keys: { k1: { id: "k1", concept: "딸기", embedding: [1, 0], key_type: "concept" } },
    memories: {
      m1: {
        id: "m1", content: "사용자는 딸기를 좋아한다", embedding: [1, 0],
        created_at: 1000, source: null, supersedes: null, depth: 0.7,
        access_count: 5, last_accessed: 1000, namespace: "default",
        ttl: null, links: [], contradicts: [],
      },
    },
    links: [{ key_id: "k1", memory_id: "m1", weight: 1.0 }],
  };
  await writeFile(join(dataDir, "graph.json"), JSON.stringify(legacy), "utf-8");

  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.SUPER_MEMORY_FORCE_REEMBED = "true";
  t.after(() => { delete process.env.SUPER_MEMORY_FORCE_REEMBED; });
  const { emb, mg } = await loadModules(dataDir);
  emb.__setTestEmbedder(() => [0, 1]);
  t.after(() => emb.__clearTestEmbedder());

  const g = new mg.MemoryGraph();
  await g.load();

  assert.deepEqual(g.memories.m1.embedding, [0, 1], "forced re-embed must replace the legacy vector");
  const persisted = JSON.parse(await readFile(join(dataDir, "graph.json"), "utf-8"));
  assert.equal(persisted.meta?.embeddingFingerprint, "local:bge-m3", "fingerprint stamped after forced re-embed");
});

test("does NOT re-embed when the model is unchanged (same fingerprint, same dim)", async (t) => {
  const dataDir = await mkdtemp(join(tmpdir(), "sm-modelmig-noop-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));

  const stored = {
    meta: { embeddingFingerprint: "local:bge-m3" },
    keys: { k1: { id: "k1", concept: "딸기", embedding: [1, 0], key_type: "concept" } },
    memories: {
      m1: {
        id: "m1", content: "사용자는 딸기를 좋아한다", embedding: [0.6, 0.8],
        created_at: 1000, source: null, supersedes: null, depth: 0.7,
        access_count: 5, last_accessed: 1000, namespace: "default",
        ttl: null, links: [], contradicts: [],
      },
    },
    links: [{ key_id: "k1", memory_id: "m1", weight: 1.0 }],
  };
  await writeFile(join(dataDir, "graph.json"), JSON.stringify(stored), "utf-8");

  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const { emb, mg } = await loadModules(dataDir);
  emb.__setTestEmbedder(() => [0, 1]); // would change the vector IF a re-embed ran
  t.after(() => emb.__clearTestEmbedder());

  const g = new mg.MemoryGraph();
  await g.load();

  assert.deepEqual(
    g.memories.m1.embedding, [0.6, 0.8],
    "unchanged model must NOT trigger a re-embed"
  );
  const backups = (await readdir(dataDir)).filter((f) => f.includes(".bak."));
  assert.equal(backups.length, 0, `no migration backup expected, found: ${backups.join(",")}`);
});
