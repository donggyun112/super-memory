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
