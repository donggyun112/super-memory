// Latency/perf bench for keymem's OWN data-structure cost (not embedding inference).
// A synthetic 1024-dim embedder isolates: linear key scan (searchKeys), per-read
// graph.json rewrite (read_memory calls save()), and add() cost growth during build.
// Run: tsx bench/perf.ts            (defaults below)
//      tsx bench/perf.ts 500 2000 5000
import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function vecFor(s: string): number[] {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  const rnd = mulberry32(h >>> 0);
  const v = new Array(1024);
  for (let i = 0; i < 1024; i++) v[i] = rnd() * 2 - 1;
  return v;
}
const pct = (a: number[], p: number) => {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor((s.length - 1) * p)];
};
const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
const emb = await import("../src/embedding.ts");
emb.__setTestEmbedder((tx: string) => vecFor(tx));

async function bench(N: number, tag: number) {
  const dir = await mkdtemp(join(tmpdir(), `kmperf-${N}-`));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  const mg = await import(`../src/memoryGraph.ts?perf=${tag}`);
  const g = new mg.MemoryGraph();
  await g.load();

  const ids: string[] = [];
  const addLat: number[] = [];
  for (let i = 0; i < N; i++) {
    const s = performance.now();
    const [id] = await g.add(`memory number ${i} about topic ${i % 50} with some filler text`, [`topic${i % 50}`, `item${i}`], {});
    addLat.push(performance.now() - s);
    ids.push(id);
  }
  const addFirst = avg(addLat.slice(0, 50));
  const addLast = avg(addLat.slice(-50));

  const sk: number[] = [];
  for (let q = 0; q < 50; q++) { const s = performance.now(); await g.searchKeys(`topic${q % 50}`, 8, null); sk.push(performance.now() - s); }

  const rmLat: number[] = [];
  for (let q = 0; q < 50; q++) { const s = performance.now(); await g.readMemory(ids[(q * 7) % N], null, null); rmLat.push(performance.now() - s); }

  const sizeMB = (await stat(join(dir, "graph.json"))).size / 1e6;
  await rm(dir, { recursive: true, force: true });
  return { N, addFirst, addLast, skP50: pct(sk, 0.5), skP95: pct(sk, 0.95), rmP50: pct(rmLat, 0.5), rmP95: pct(rmLat, 0.95), sizeMB };
}

const Ns = (process.argv.slice(2).map(Number).filter(Boolean));
const sizes = Ns.length ? Ns : [500, 1500, 3000];
console.log(`# keymem perf (synthetic 1024-dim embedder; measures graph ops, not embedding inference)`);
console.log(`N\tadd_first50(ms)\tadd_last50(ms)\tsearchKeys p50/p95(ms)\tread_memory p50/p95(ms)\tgraph.json(MB)`);
let tag = 0;
for (const N of sizes) {
  const r = await bench(N, tag++);
  console.log(`${r.N}\t${r.addFirst.toFixed(2)}\t\t${r.addLast.toFixed(2)}\t\t${r.skP50.toFixed(2)}/${r.skP95.toFixed(2)}\t\t${r.rmP50.toFixed(2)}/${r.rmP95.toFixed(2)}\t\t${r.sizeMB.toFixed(1)}`);
}
console.log("# done");
