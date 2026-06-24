// Opt-in inject recall: recallInject(query) should return, in ONE call, the connected-but-
// dissimilar memory reachable only via a shared key — without the agent manually walking
// read_key -> read_memory. Topic-based test embedder: "job" topic = dim 0, "founding" = dim 1,
// shared entity "Acme" = dim 2. The answer memory B ("Acme founded 1990") is dim-1, so it has
// ~0 content similarity to the "user job" query — it is reachable ONLY via the shared Acme key.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
function vec(tx: string): number[] {
  const v = new Array(16).fill(0);
  const t = tx.toLowerCase();
  if (t.includes("found") || t.includes("1990") || t.includes("history")) v[1] = 1;
  else if (t.includes("job") || t.includes("works")) v[0] = 1;
  else if (t.includes("acme")) v[2] = 1;
  else v[15] = 1;
  return v;
}

test("recallInject surfaces the connected-but-dissimilar memory in one call", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-inject-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?inject=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  await g.add("user works at Acme", ["job", "Acme"], {});          // A — matches "user job"
  const [bId] = await g.add("Acme was founded in 1990", ["Acme", "history"], {}); // B — connected via Acme only

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r: any = await g.recallInject("user job", 5, null);
  assert.ok(Array.isArray(r.keys) && r.keys.length > 0, "returns navigation keys");
  assert.ok(
    r.memories.some((m: { id: string }) => m.id === bId),
    "auto-injects the connected-but-dissimilar memory (B) without manual traversal"
  );
});

// Regression: a query with NO genuine anchor (the cross-lingual / topic-mismatch failure) must not
// scrape coincidental BM25 hits into the injected set. Here "shared" is a stray token both the query
// and a topically-unrelated memory contain, but their embeddings are orthogonal (fruit≠metal), so
// the memory has zero dense similarity to the query — it is pure BM25 noise. Inject must return [].
function fruitMetalVec(tx: string): number[] {
  const v = new Array(16).fill(0);
  const t = tx.toLowerCase();
  if (t.includes("fruit")) v[0] = 1; // query topic
  else if (t.includes("metal")) v[5] = 1; // unrelated topic, orthogonal to the query
  else v[15] = 1;
  return v;
}

test("recallInject returns nothing when the query has no real anchor (no BM25-only junk)", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-inject-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => fruitMetalVec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?inject=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  // Only a topically-unrelated memory exists; it shares the stray token "shared" with the query.
  await g.add("shared metal device control panel", ["metal"], {});

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r: any = await g.recallInject("shared fruit nutrition", 5, null);
  assert.equal(
    r.memories.length,
    0,
    `no genuine anchor → inject must return nothing, got ${r.memories.length} BM25-only hits`
  );
});

test("recallInject still injects when a genuine anchor exists (anchor gate does not over-filter)", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-inject-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => fruitMetalVec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?inject=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  await g.add("shared metal device control panel", ["metal"], {}); // BM25 noise
  const [goodId] = await g.add("fruit nutrition facts", ["fruit"], {}); // genuine anchor

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r: any = await g.recallInject("shared fruit nutrition", 5, null);
  assert.ok(
    r.memories.some((m: { id: string }) => m.id === goodId),
    "a genuinely-matching memory must still be injected"
  );
});

test("recallInject excludes BM25-only lexical noise riding alongside a real anchor", async (t) => {
  // The live failure: an English query genuinely anchors on some memories (so the gate passes), but
  // unrelated docs that merely share a stray token ride in via BM25 and fill the slots. Their fused
  // scores sit in the SAME noise band as the real cross-lingual hits, so a relative floor can't
  // separate them — only matched_via provenance can. Inject must keep dense/graph-supported hits and
  // drop the BM25-only ones.
  const dir = await mkdtemp(join(tmpdir(), "sm-inject-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => fruitMetalVec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?inject=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  // Two junk docs that SHARE a key ("device") among themselves but not with the query. Each is a
  // BM25-only hop-1 hit; in traversal they cross-tag each other "device(via)" — so naive
  // "has a non-bm25 tag" would let them through. The real discriminator: the (via) hop originates
  // from another junk doc, not a genuine anchor.
  const [junk1] = await g.add("shared metal device control panel", ["metal", "device"], {});
  const [junk2] = await g.add("shared metal device remote unit", ["metal", "device"], {});
  const [goodId] = await g.add("fruit nutrition facts", ["fruit"], {}); // genuine dense anchor

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r: any = await g.recallInject("shared fruit nutrition", 8, null);
  const ids = r.memories.map((m: { id: string }) => m.id);
  assert.ok(ids.includes(goodId), "the genuine anchor must be injected");
  assert.ok(!ids.includes(junk1), `BM25-only junk1 must NOT be injected, got ${ids.join(",")}`);
  assert.ok(!ids.includes(junk2), `BM25-only junk2 must NOT be injected, got ${ids.join(",")}`);
});

test("recallInject is passive — it does not reinforce or depth-bump the memories it surfaces", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-inject-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?inject=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  const [aId] = await g.add("user works at Acme", ["job", "Acme"], {});
  const [bId] = await g.add("Acme was founded in 1990", ["Acme", "history"], {});

  // Injection auto-surfaces memories the agent never asked for. Strengthening links / depth
  // for the whole surfaced (and the even wider internal) pool would inflate noise on every
  // call. Reinforcement must come only from a real read_memory, not from passive injection.
  await g.recallInject("user job", 1, null);

  assert.equal(g.memories[aId].access_count, 0, "inject must not bump access_count");
  assert.equal(g.memories[bId].access_count, 0, "inject must not bump access_count");
  assert.equal(g.memories[aId].depth, g.memories[bId].depth, "inject must not depth-bump unevenly");
});
