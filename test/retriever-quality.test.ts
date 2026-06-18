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
