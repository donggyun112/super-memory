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
