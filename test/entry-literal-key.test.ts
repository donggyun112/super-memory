// Entry-recall accuracy: a memory whose key the query LITERALLY contains must win, even
// when that key's embedding doesn't clear the keyRecall threshold (so it never entered the
// dense keyScores). Previously the lexical exact-key boost only iterated keyScores, so a
// literally-named key like "동물" was skipped and the right memory sank under hub/content
// noise. A literal key mention is a strong, model-independent signal — always boost it
// (IDF-weighted so hubs don't dominate).
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
function vec(t: string): number[] {
  const v = new Array(7).fill(0);
  const m: Record<string, number> = { "PET fact": 2, "동물": 1, "키우기": 3, "기타": 5 };
  if (t === "동물 키우기") { v[0] = 1; return v; }           // query
  if (t === "DISTRACT fact") { v[0] = 0.9; v[4] = 0.4358899; return v; } // cos(query)=0.9
  if (t in m) { v[m[t]] = 1; return v; }
  v[6] = 1; return v; // fallback (probe etc.)
}

test("a literally-named key wins entry recall even when its embedding is below keyRecall", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-entry-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?entry=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  // PET: keys literally in the query, but their embeddings are orthogonal to it (so they
  // never enter keyScores). PET content is also orthogonal → only the literal key boost
  // can surface it.
  const [pet] = await g.add("PET fact", ["동물", "키우기"], {});
  // DISTRACT: strong content match to the query, no literal-key overlap.
  await g.add("DISTRACT fact", ["기타"], {});

  const r = (await g.recall("동물 키우기", 5, null, false, 2, 0, 0, 0, 0)) as any[];
  assert.ok(r.length > 0, "should return results");
  assert.equal(
    r[0].id, pet,
    `the literally-keyed memory must rank #1, got: ${r.map((m) => m.content).join(" | ")}`
  );
});
