// Live functional test: local e5 multilingual backend, real embeddings.
// Run: EMBEDDING_BACKEND=local tsx test/live-multilingual.ts
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local"; // force local fastembed (e5 default)
const dataDir = await mkdtemp(join(tmpdir(), "sm-live-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { EMBEDDING_BACKEND, LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");
console.log(`backend=${EMBEDDING_BACKEND} model=${LOCAL_EMBEDDING_MODEL} dataDir=${dataDir}`);

const g = new MemoryGraph();
await g.load();

const t0 = Date.now();
await g.add("Newton discovered gravity", ["Newton", "apple", "gravity"]);
await g.add("apples are red fruit", ["apple", "fruit", "red"]);
await g.add("user likes strawberries", ["fruit", "strawberry"]);
await g.add("사용자의 이름은 동건이다", ["이름", "동건"], { keyTypes: { 동건: "name" } });
await g.add("user enjoys programming in Python", ["Python", "programming"]);
console.log(`\nseeded 5 memories in ${Date.now() - t0}ms`);

const dim = Object.values(g.memories)[0].embedding.length;
console.log(`embedding dim = ${dim}\n`);

function show(label: string, results: any[]) {
  console.log(`▶ ${label}`);
  if (results.length === 0) { console.log("  (no results)\n"); return; }
  for (const r of results) {
    console.log(
      `  [hop${r.hop} score=${r.score}] ${r.content}\n` +
      `        via: ${(r.matched_via || []).join(", ")}`
    );
  }
  console.log("");
}

// 1) Headline associative leap: Newton -> (apple) -> fruit -> strawberry
show("EN query 'Newton' (expect strawberry via 2-hop)", await g.recall("Newton", 5));

// 2) Cross-lingual: Korean query reaching English memory
show("KO query '뉴턴이 발견한 것' (cross-lingual -> Newton/gravity)", await g.recall("뉴턴이 발견한 것", 3));

// 3) Cross-lingual content match: Korean query for English 'strawberries'
show("KO query '딸기 좋아하는 사람' (cross-lingual -> strawberries)", await g.recall("딸기 좋아하는 사람", 3));

// 4) name-type key exact behavior
show("query '동건' (name key)", await g.recall("동건", 3));

// 5) cross-lingual key merge: does '파이썬' merge into existing 'Python' key?
const before = Object.values(g.keys).filter((k: any) => k.key_type === "concept").length;
await g.add("user is learning 파이썬 frameworks", ["파이썬"]);
const pyKeys = Object.values(g.keys).filter((k: any) =>
  ["python", "파이썬"].includes((k as any).concept.toLowerCase())
);
console.log(`▶ cross-lingual key merge check`);
console.log(`  concept keys before=${before}, Python/파이썬 keys now=${pyKeys.map((k:any)=>k.concept).join(" | ")}`);
console.log(`  ${pyKeys.length === 1 ? "MERGED ✅ (파이썬 reused Python)" : "SEPARATE (no merge at threshold)"}\n`);

await rm(dataDir, { recursive: true, force: true });
console.log("done.");
