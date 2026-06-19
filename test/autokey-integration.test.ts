import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
process.env.SUPER_MEMORY_DATA_DIR = mkdtempSync(join(tmpdir(), "autokey-"));

test("Key.aliasCandidates and learnedAliases survive save/load", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(() => [1, 0]);
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g1 = new MemoryGraph();
    const [mid] = await g1.add("동균은 성수동에 산다", ["거주지"]);
    const concept = g1.getKeysForMemory(mid)[0];
    const kid = Object.keys(g1.keys).find((k) => g1.keys[k].concept === concept)!;
    assert.ok(kid, "expected a key for the concept");
    g1.keys[kid].aliasCandidates = { "어디 살아": { count: 2, lastSeen: 100, queryText: "어디 살아" } };
    g1.keys[kid].learnedAliases = [{ alias: "사는곳", addedAt: 100, hits: 1 }];
    await g1.save();

    const g2 = new MemoryGraph();
    await g2.load();
    assert.equal(g2.keys[kid].aliasCandidates?.["어디 살아"].count, 2);
    assert.equal(g2.keys[kid].learnedAliases?.[0].alias, "사는곳");
  } finally {
    emb.__clearTestEmbedder();
  }
});

test("searchKeys records weak (semantic) matches in the recall buffer", async () => {
  const emb = await import("../src/embedding.ts");
  // "거주지" key embeds [1,0]; a paraphrase query embeds close but not literal.
  emb.__setTestEmbedder((text) => (text.includes("성수") || text === "거주지" ? [1, 0] : [0.95, 0.31]));
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("동균은 성수동에 산다", ["거주지"]);
    // getKeysForMemory returns concept strings; resolve the concept to its key ID.
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "거주지")!;

    await g.searchKeys("어디 살아"); // semantic match on 거주지, not literal
    const hit = (g as unknown as { _recallBuffer: { consumeWeakMatch(k: string): unknown } })._recallBuffer
      .consumeWeakMatch(kid);
    assert.ok(hit, "expected 거주지 to be recorded as a weak match");
  } finally {
    emb.__clearTestEmbedder();
  }
});

test("repeated weak-confirmed reads promote the query (new key) and heal recall", async () => {
  const emb = await import("../src/embedding.ts");
  // Query embeds in the mid band vs the key (cos ~0.95 with bge-m3 keyMerge 0.86 would
  // alias; to exercise the newKey branch we make the query orthogonal-ish but still a
  // surfaced semantic match by also making it literal-free). Use a value below keyMerge
  // (0.86) but above keyAutoLink (0.62): [0.8,0.6] · [1,0] = 0.8.
  // Key "거주지" and the memory content embed to [1,0]. The recall query "살곳" embeds
  // to [0.8,0.6] → cosine 0.8 vs the key: above keyAutoLink (0.62) so it surfaces as a
  // SEMANTIC match, below keyMerge (0.86) so promotion takes the newKey branch. "살곳"
  // shares no substring with "거주지", so it can never be a literal/concept match.
  emb.__setTestEmbedder((text) => (text === "거주지" || text.includes("성수") ? [1, 0] : [0.8, 0.6]));
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("동균은 성수동에 산다", ["거주지"]);
    // getKeysForMemory returns concept strings; resolve the concept to its key ID.
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "거주지")!;

    const QUERY = "살곳"; // short concept, semantic match on 거주지, no substring overlap
    for (let i = 0; i < 3; i++) {
      const keys = (await g.searchKeys(QUERY)) as Array<{ key_id: string; match_type: string }>;
      assert.ok(keys.some((k) => k.key_id === kid && k.match_type === "semantic"));
      await g.readMemory(mid, kid);
    }

    // After 3 confirmations a NEW key for the query exists and links to the memory.
    const healedKid = Object.keys(g.keys).find((k) => g.keys[k].concept === QUERY);
    assert.ok(healedKid, "expected a new key coined from the query");
    assert.ok(g.getKeysForMemory(mid).includes(QUERY), "new key must link the memory");
  } finally {
    emb.__clearTestEmbedder();
  }
});

test("depth/access_count still increment exactly once per readMemory", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(() => [1, 0]);
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("x", ["kx"]);
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "kx")!;
    const before = (await g.readMemory(mid, kid)) as { memory: { access_count: number } };
    const after = (await g.readMemory(mid, kid)) as { memory: { access_count: number } };
    assert.equal(after.memory.access_count, before.memory.access_count + 1);
  } finally {
    emb.__clearTestEmbedder();
  }
});

test("read_key surfaces learned aliases as provenance", async () => {
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(() => [1, 0]);
  try {
    const { MemoryGraph } = await import("../src/memoryGraph.ts");
    const g = new MemoryGraph();
    const [mid] = await g.add("동균은 성수동에 산다", ["거주지"]);
    const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "거주지")!;
    g.keys[kid].learnedAliases = [{ alias: "사는곳", addedAt: 1, hits: 0 }];

    const view = (await g.readKey(kid)) as { key: { learned_aliases: string[] } };
    assert.deepEqual(view.key.learned_aliases, ["사는곳"]);
  } finally {
    emb.__clearTestEmbedder();
  }
});
