// recall() applies the cross-encoder reranker (when enabled) as a precision pass: it
// reorders the gated top candidates by the reranker's scores while keeping the same set.
// Driven here by a test reranker (no model load) so the wiring is verified deterministically.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
function vec(t: string): number[] {
  const m: Record<string, number[]> = {
    QQ: [1, 0, 0, 0, 0, 0],
    AA: [1, 0, 0, 0, 0, 0],         // cos(query)=1.00 → fused #1
    BB: [0.9, 0, 0.4359, 0, 0, 0],  // cos=0.90 → fused #2
    CC: [0.8, 0, 0, 0.6, 0, 0],     // cos=0.80 → fused #3
    ka: [0, 1, 0, 0, 0, 0], kb: [0, 0, 0, 0, 1, 0], kc: [0, 0, 0, 0, 0, 1],
  };
  return m[t] ?? [0, 1, 0, 0, 0, 0];
}

test("rerank reorders the gated results by reranker score (set unchanged)", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-rerank-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  // Shared singleton (no query-buster) so memoryGraph's `import from "./reranker.js"`
  // sees the same instance we set the seam on.
  const rer = await import("../src/reranker.ts");
  t.after(() => rer.__clearTestReranker());

  const mg = await import(`../src/memoryGraph.ts?rerank=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("AA", ["ka"], {});
  await g.add("BB", ["kb"], {});
  await g.add("CC", ["cc-key-distinct"], {}); // distinct key, won't matter

  // Baseline (no reranker): fused order AA > BB > CC.
  rer.__clearTestReranker();
  const off = (await g.recall("QQ", 10, null, false, 2, 0, 0, 0, 0)) as any[];
  assert.equal(off[0].content, "AA", `fused top should be AA, got ${off.map((m) => m.content).join(",")}`);

  // Reranker scores the LAST fused candidate highest → CC should jump to #1.
  rer.__setTestReranker((_q, texts) => texts.map((_t, i) => i));
  const on = (await g.recall("QQ", 10, null, false, 2, 0, 0, 0, 0)) as any[];
  assert.equal(on[0].content, "CC", `rerank should lift CC to #1, got ${on.map((m) => m.content).join(",")}`);
  assert.deepEqual(
    new Set(on.map((m) => m.content)), new Set(off.map((m) => m.content)),
    "rerank must keep the same result set, only reorder"
  );
});
