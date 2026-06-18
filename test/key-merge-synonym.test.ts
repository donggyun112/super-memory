// Write-time key reconciliation: the LLM stores memories without knowing the existing
// key vocabulary, so it picks synonymous-but-different keys ("코딩" vs "프로그래밍") and
// the n:M chain fragments. Short concept keys currently merge only on EXACT string match.
// Add a CONSERVATIVE semantic merge: an incoming short key folds into an existing key only
// at high cosine (clear synonym), so synonyms unify while distinct concepts (음식 vs 음료,
// Agent A vs Agent B) stay separate. Under-merge is safe (status quo); over-merge corrupts
// the graph, so the threshold is high.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
function vec(t: string): number[] {
  // keys live in dims 0-1 (so similarity is tunable); contents one-hot in dims 2-5.
  const m: Record<string, number[]> = {
    "프로그래밍": [1, 0, 0, 0, 0, 0],
    "코딩": [0.95, 0.3122499, 0, 0, 0, 0],   // cos(프로그래밍)=0.95 -> clear synonym, MERGE
    "음식": [0.78, 0.6257795, 0, 0, 0, 0],   // cos(프로그래밍)=0.78 -> distinct, KEEP
    m1: [0, 0, 1, 0, 0, 0], m2: [0, 0, 0, 1, 0, 0], m3: [0, 0, 0, 0, 1, 0],
  };
  return m[t] ?? [0, 0, 0, 0, 0, 1];
}

test("a clear synonym short-key merges into the existing key; a distinct one stays separate", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-keymerge-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.SUPER_MEMORY_SHORT_KEY_MERGE = "0.9"; // merge >=0.9 cosine
  t.after(() => { delete process.env.SUPER_MEMORY_SHORT_KEY_MERGE; });

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?keymerge=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  const [m1] = await g.add("m1", ["프로그래밍"], {});
  const [m2] = await g.add("m2", ["코딩"], {});   // synonym -> should reuse 프로그래밍's key
  const [m3] = await g.add("m3", ["음식"], {});   // distinct -> own key

  // m1 and m2 now share the (merged) concept key -> related connects them.
  const relM1 = (g.getRelated(m1) as any[]).map((r) => r.id);
  assert.ok(relM1.includes(m2), "synonym keys must merge so the two memories share a key");
  assert.ok(!relM1.includes(m3), "a distinct concept must NOT merge in");

  // Exactly 2 concept keys exist (프로그래밍[+코딩 merged], 음식), not 3.
  const conceptKeys = Object.values(g.keys).filter((k: any) => k.key_type === "concept");
  assert.equal(conceptKeys.length, 2, `expected 2 concept keys after merge, got ${conceptKeys.length}`);
});
