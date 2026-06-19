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
