// Does deep traversal create noise? Measure how a hub key floods results as hops
// grow, and whether HOP_DECAY + IDF keep that noise scored well below real hits.
// Content path off so we isolate pure graph-traversal noise.
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
process.env.SUPER_MEMORY_CONTENT_RECALL = "0.97";
process.env.SUPER_MEMORY_KEY_RECALL = "0.85";
const dataDir = await mkdtemp(join(tmpdir(), "sm-noise-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const g = new MemoryGraph();
await g.load();
const ns = "noise";

// A —(bridge)— B, and B sits on a HUB key shared by 6 unrelated "noise" memories.
// Two fully-isolated memories must NEVER appear.
await g.add("SEED 정답 시드", ["seed_a", "bridge_ab"], { namespace: ns });
await g.add("CHAIN B 노드", ["bridge_ab", "hub"], { namespace: ns });
for (let i = 1; i <= 6; i++) await g.add(`HUBNOISE ${i}`, ["hub", `u${i}`], { namespace: ns });
await g.add("ISOLATED X", ["iso_x"], { namespace: ns });
await g.add("ISOLATED Y", ["iso_y"], { namespace: ns });

const recall = (h: number) => g.recall("seed_a", 20, ns, false, h) as Promise<any[]>;
const tag = (c: string) => c.split(" ")[0];

let pass = 0, fail = 0;
const ck = (n: string, c: boolean, d = "") => { if (c) { pass++; console.log(`  ✅ ${n}`); } else { fail++; console.log(`  ❌ ${n}  ${d}`); } };

for (const h of [1, 2, 3, 4, 5]) {
  const r = await recall(h);
  const byHop: Record<number, number> = {};
  for (const x of r) byHop[x.hop] = (byHop[x.hop] ?? 0) + 1;
  const top = r[0]?.score ?? 0;
  const noise = r.filter(x => tag(x.content) === "HUBNOISE");
  const noiseMax = noise.length ? Math.max(...noise.map(x => x.score)) : 0;
  console.log(`hops=${h}: n=${r.length}  byHop=${JSON.stringify(byHop)}  top=${top}  hubNoise=${noise.length} (maxScore=${noiseMax}, ${top ? (noiseMax / top * 100).toFixed(1) : 0}% of top)`);
}

const h5 = await recall(5);
ck("isolated nodes never leak in (even at hops=5)", !h5.some(x => tag(x.content) === "ISOLATED"));
const seed = h5.find(x => tag(x.content) === "SEED");
const noiseMax = Math.max(0, ...h5.filter(x => tag(x.content) === "HUBNOISE").map(x => x.score));
ck("deep hub-noise scores far below the direct hit (<10%)", noiseMax < (seed?.score ?? 1) * 0.1, `noiseMax=${noiseMax} seed=${seed?.score}`);

// Relative score floor: minRelScore=0.05 should drop the ~2% hub-noise but keep real hits.
const floored = await g.recall("seed_a", 20, ns, false, 5, 0.05) as any[];
console.log(`\nhops=5 + minRelScore=0.05: ${floored.map(x => tag(x.content) + "#" + x.hop).join(" ")}`);
ck("floor drops the hub-noise flood", !floored.some(x => tag(x.content) === "HUBNOISE"), floored.map(x => tag(x.content)).join(","));
ck("floor keeps the direct hit (SEED)", floored.some(x => tag(x.content) === "SEED"));
ck("floor=0 (default) is unchanged from no-floor recall", (await recall(5)).length === h5.length);

console.log(`\nRESULT: ${pass} passed, ${fail} failed`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
