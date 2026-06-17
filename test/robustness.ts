// Robustness suite: threshold escape hatches + Hebbian pollution bounds.
// Run: tsx test/robustness.ts
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
const dataDir = await mkdtemp(join(tmpdir(), "sm-rob-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { getThresholdProfile } = await import("../src/embedding.ts");

let pass = 0, fail = 0; const fails: string[] = [];
function check(name: string, cond: boolean, detail = "") {
  if (cond) { pass++; console.log(`  ✅ ${name}`); }
  else { fail++; fails.push(name); console.log(`  ❌ ${name}  ${detail}`); }
}

// ── R3: env override escape hatch (pure config) ──
console.log("R3 threshold env override");
{
  process.env.SUPER_MEMORY_KEY_RECALL = "0.42";
  check("env override applies", getThresholdProfile().keyRecall === 0.42, JSON.stringify(getThresholdProfile()));
  process.env.SUPER_MEMORY_KEY_RECALL = "banana";
  check("invalid override ignored (falls back to profile)", getThresholdProfile().keyRecall !== 0.42 && getThresholdProfile().keyRecall > 0, `${getThresholdProfile().keyRecall}`);
  delete process.env.SUPER_MEMORY_KEY_RECALL;
}

const g = new MemoryGraph();
await g.load();
const recall = (q: string, ns?: string) => g.recall(q, 5, ns ?? null, false) as Promise<any[]>;
const W = (kid: string, mid: string) => (g as any)._keyToMems[kid]?.get(mid) as number;
const keyId = (concept: string) => Object.entries(g.keys).find(([, k]: any) => k.concept === concept)?.[0] as string;

// ── R1 + R2: Hebbian bounds and anti-pollution ──
console.log("R1/R2 Hebbian pollution under repeated (mis)recall");
{
  const ns = "heb";
  const [correctId] = await g.add("사용자는 점심에 김밥을 먹었다", ["점심", "김밥"], { namespace: ns });
  const [hubId] = await g.add("사용자는 매일 아침 운동을 한다", ["운동", "점심"], { namespace: ns });
  const lunchKey = keyId("점심");

  const before = await recall("점심", ns);
  const rankCorrectBefore = before.findIndex(r => r.id === correctId) + 1;
  const rankHubBefore = before.findIndex(r => r.id === hubId) + 1;
  const wBaseline = W(lunchKey, hubId); // weight right before hammering an unrelated key

  // Hammer the hub memory via its OWN key 30x (simulates repeated recall on '운동')
  for (let i = 0; i < 30; i++) await recall("운동", ns);

  const wHub = W(lunchKey, hubId);
  check("R1: link weight stays bounded (<= 3.0)", wHub <= 3.0 + 1e-9, `wHub=${wHub}`);

  // The '점심' link of the hub was NOT a matched key in any '운동' recall, so 30
  // recalls on '운동' must leave it untouched (no cross-key pollution).
  check("R2a: unrelated '점심' link not inflated by '운동' recalls", Math.abs(wHub - wBaseline) < 1e-6, `${wBaseline} -> ${wHub}`);

  const after = await recall("점심", ns);
  const rankCorrectAfter = after.findIndex(r => r.id === correctId) + 1;
  const rankHubAfter = after.findIndex(r => r.id === hubId) + 1;
  check("R2b: correct memory still retrievable for its own query", rankCorrectAfter > 0, `rank=${rankCorrectAfter}`);
  check("R2c: hub did not crowd out the correct memory on '점심'", rankCorrectAfter <= rankHubAfter, `correct#${rankCorrectAfter} vs hub#${rankHubAfter}`);
  console.log(`     ('점심' ranks  before: correct#${rankCorrectBefore} hub#${rankHubBefore}  |  after: correct#${rankCorrectAfter} hub#${rankHubAfter})`);
}

console.log(`\n${"=".repeat(50)}`);
console.log(`RESULT: ${pass} passed, ${fail} failed`);
if (fail) console.log(`FAILED: ${fails.join(", ")}`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
