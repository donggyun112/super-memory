// Manual integration check (NOT part of `npm test`). Uses the real local model.
// Run: EMBEDDING_BACKEND=local npx tsx test/retriever-quality.live.ts
// For bge-m3: also set LOCAL_EMBEDDING_MODEL=bge-m3 LOCAL_EMBEDDING_MODEL_PATH=/abs/dir
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
const dataDir = await mkdtemp(join(tmpdir(), "sm-rq-live-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");
const g = new MemoryGraph();
await g.load();

let pass = 0, fail = 0; const fails: string[] = [];
const check = (name: string, cond: boolean, detail = "") => {
  if (cond) { pass++; console.log(`  ✅ ${name}`); }
  else { fail++; fails.push(name); console.log(`  ❌ ${name}  ${detail}`); }
};

console.log(`model=${LOCAL_EMBEDDING_MODEL}`);
await g.add("사용자는 커피를 좋아한다", ["커피", "음료"]);
await g.add("프로젝트A는 Postgres를 쓴다", ["프로젝트A", "데이터베이스"]);

const rel = (await g.recall("커피", 5)) as any[];
check("relevant query returns hits", rel.length >= 1, `${rel.length}`);

const noise = (await g.recall("양자역학 우주론 블랙홀", 5, null, false, 2)) as any[];
check("unrelated query returns nothing (gate)", noise.length === 0, `${noise.length}`);

console.log(`\nRESULT: ${pass} passed, ${fail} failed`);
if (fail) console.log(`FAILED: ${fails.join(", ")}`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
