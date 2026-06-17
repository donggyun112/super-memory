// N-hop chained traversal, isolated from the content path so depth is the only
// variable. We raise CONTENT_RECALL very high (content matching effectively off)
// so memories are reachable ONLY through the key graph — then verify recall(hops=N)
// walks exactly N steps along a chain A→B→C→D whose ends share no key.
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
process.env.SUPER_MEMORY_CONTENT_RECALL = "0.97"; // disable content-path flooding
process.env.SUPER_MEMORY_KEY_RECALL = "0.85";
const dataDir = await mkdtemp(join(tmpdir(), "sm-nhop-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const g = new MemoryGraph();
await g.load();

let pass = 0, fail = 0; const fails: string[] = [];
function check(name: string, cond: boolean, detail = "") {
  if (cond) { pass++; console.log(`  ✅ ${name}`); }
  else { fail++; fails.push(name); console.log(`  ❌ ${name}  ${detail}`); }
}
const ns = "chain";
const has = (rs: any[], frag: string) => rs.some(r => r.content.includes(frag));
const recall = (h: number) => g.recall("알파프로젝트", 10, ns, false, h) as Promise<any[]>;

// Chain: A —(인증모듈)— B —(결제게이트웨이)— C —(배송추적)— D. Ends (A,D) share no key.
await g.add("A: 알파 프로젝트 인증", ["알파프로젝트", "인증모듈"], { namespace: ns });
await g.add("B: 결제 게이트웨이 연동", ["인증모듈", "결제게이트웨이"], { namespace: ns });
await g.add("C: 배송 추적 시스템", ["결제게이트웨이", "배송추적"], { namespace: ns });
await g.add("D: 리뷰와 별점 수집", ["배송추적", "리뷰시스템"], { namespace: ns });

const fmt = (rs: any[]) => rs.map(x => x.content[0] + "#" + x.hop).join(" ");
const h1 = await recall(1), h2 = await recall(2), h3 = await recall(3), h4 = await recall(4);
console.log(`  hops=1: ${fmt(h1)}\n  hops=2: ${fmt(h2)}\n  hops=3: ${fmt(h3)}\n  hops=4: ${fmt(h4)}\n`);

check("hops=1 → only A (direct, no traversal)", has(h1, "A:") && !has(h1, "B:") && !has(h1, "C:"), fmt(h1));
check("hops=2 → reaches B, not C", has(h2, "B:") && !has(h2, "C:"), fmt(h2));
check("hops=3 → reaches C, not D", has(h3, "C:") && !has(h3, "D:"), fmt(h3));
check("hops=4 → reaches D at the chain's far end", has(h4, "D:"), fmt(h4));
const d = h4.find(x => x.content.includes("D:"));
check("D tagged hop=4 (true chain distance)", d?.hop === 4, `hop=${d?.hop}`);

console.log(`\n${"=".repeat(50)}`);
console.log(`RESULT: ${pass} passed, ${fail} failed`);
if (fail) console.log(`FAILED: ${fails.join(", ")}`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
