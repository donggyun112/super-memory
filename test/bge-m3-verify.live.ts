// Manual end-to-end verification on the REAL bge-m3 model (fastembed CUSTOM).
// Run:
//   EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 \
//   LOCAL_EMBEDDING_MODEL_PATH=/abs/dir/with/model.onnx \
//   npx tsx test/bge-m3-verify.live.ts
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";
const dataDir = await mkdtemp(join(tmpdir(), "sm-bgem3-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { embedTextAsync, LOCAL_EMBEDDING_MODEL, getThresholdProfile } = await import(
  "../src/embedding.ts"
);

let pass = 0,
  fail = 0;
const fails: string[] = [];
const check = (name: string, cond: boolean, detail = "") => {
  if (cond) {
    pass++;
    console.log(`  ✅ ${name}${detail ? "  " + detail : ""}`);
  } else {
    fail++;
    fails.push(name);
    console.log(`  ❌ ${name}  ${detail}`);
  }
};

function cos(a: number[], b: number[]): number {
  let d = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

const p = getThresholdProfile();
console.log(`model=${LOCAL_EMBEDDING_MODEL}`);
console.log(
  `profile: minScore=${p.minScore} contentRecall=${p.contentRecall} keyRecall=${p.keyRecall} contradiction=${p.contradiction} memoryDedup=${p.memoryDedup} keyMerge=${p.keyMerge}\n`
);

// ── 1. dimension + no-prefix embedding ──
const v = await embedTextAsync("사용자는 커피를 좋아한다", "passage");
check("bge-m3 embeds to 1024-dim", v.length === 1024, `dim=${v.length}`);

const g = new MemoryGraph();
await g.load();

// ── 2. absolute score gate (relevant vs unrelated) ──
await g.add("사용자는 커피를 좋아한다", ["커피", "음료"]);
await g.add("사용자의 이름은 동건이다", ["이름", "동건"], { keyTypes: { 동건: "name" } });
const rel = (await g.recall("좋아하는 음료", 5)) as any[];
check("gate: relevant query returns hits", rel.length >= 1, `n=${rel.length}`);
const noise = (await g.recall("양자역학 블랙홀 우주론", 5)) as any[];
check("gate: unrelated query returns []", noise.length === 0, `n=${noise.length}`);

// ── 3. short-key A/B defense ──
const a1 = await g.findOrCreateKey("Agent A");
const a2 = await g.findOrCreateKey("Agent A");
const b = await g.findOrCreateKey("Agent B");
const ka = await embedTextAsync("Agent A", "passage");
const kb = await embedTextAsync("Agent B", "passage");
check("short-key: 'Agent A' exact repeat reuses key", a1 === a2);
check(
  "short-key: 'Agent A' vs 'Agent B' stay distinct",
  a1 !== b,
  `cos(A,B)=${cos(ka, kb).toFixed(4)} (would merge at keyMerge=${p.keyMerge} without guard)`
);

// ── 4. dedup: near-paraphrase superseded ──
const ns = "dedup";
const [d1] = await g.add("프로젝트A는 PostgreSQL을 데이터베이스로 쓴다", ["프로젝트A", "DB"], {
  namespace: ns,
});
const [, wasDup] = await g.add("프로젝트A는 데이터베이스로 PostgreSQL을 사용한다", ["프로젝트A", "DB"], {
  namespace: ns,
});
const e1 = g.memories[d1]?.embedding;
const eDup = await embedTextAsync("프로젝트A는 데이터베이스로 PostgreSQL을 사용한다", "passage");
check(
  "dedup: near-paraphrase detected as duplicate",
  wasDup === true,
  `cos=${e1 ? cos(e1, eDup).toFixed(4) : "?"} vs memoryDedup=${p.memoryDedup}`
);

// ── 5. contradiction band: distinct conflicting fact ──
const ns2 = "contra";
const [c1] = await g.add("프로젝트B는 PostgreSQL을 쓴다", ["프로젝트B"], { namespace: ns2 });
const [c2, dup2] = await g.add("프로젝트B는 MongoDB를 쓴다", ["프로젝트B"], { namespace: ns2 });
const ec1 = g.memories[c1]?.embedding;
const ec2 = g.memories[c2]?.embedding;
const csim = ec1 && ec2 ? cos(ec1, ec2) : NaN;
console.log(
  `\n  [info] cos('쓴다 PostgreSQL','쓴다 MongoDB')=${csim.toFixed(4)} — contradiction band [${p.contradiction}, ${p.memoryDedup})`
);
check("contradiction: not silently deduped (both survive)", dup2 === false && c1 in g.memories);
const linked = !!g.memories[c2]?.contradicts?.includes(c1) && !!g.memories[c1]?.contradicts?.includes(c2);
check(
  "contradiction: bidirectional contradicts link recorded",
  linked,
  linked ? "" : `(cos=${csim.toFixed(4)} fell outside band — calibration signal, not a logic bug)`
);

console.log(`\n${"=".repeat(50)}`);
console.log(`RESULT: ${pass} passed, ${fail} failed`);
if (fail) console.log(`FAILED: ${fails.join(", ")}`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
