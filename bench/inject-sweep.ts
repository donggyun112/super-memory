// Inject N-sweep: is auto-injecting associated memories worth it, and at what noise cost?
// On the blind-agent-keyed bridge set (the honest data), sweep the injected top-N and report,
// per N: both@N (got BOTH gold supports), support-recall@N, and the avg # of NON-support slots
// injected (the "noise" the consuming agent must tolerate). DIRECT (expand=false) vs GRAPH
// (expand=true) at the same N answers the key control: does just taking more flat results catch
// up, or does graph traversal still win — i.e. does inject add value beyond a bigger k?
//
//   tsx bench/inject-sweep.ts
import { mkdtemp, rm, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

type Row = { question: string; type: string; support: string[]; titles: string[]; paras: string[] };
const all = JSON.parse(await readFile(resolve("bench/hotpot-slice.json"), "utf-8")) as Row[];
const bridge = all.filter((r) => r.type === "bridge").slice(0, 40);
const agentKeys = JSON.parse(await readFile(resolve("bench/hotpot-agentkeys.json"), "utf-8")) as Record<string, string[]>;

const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");

const NS = [2, 3, 5, 8, 10];
type Agg = { n: number; both: number; recall: number; noise: number };
const mk = (): Agg => ({ n: 0, both: 0, recall: 0, noise: 0 });
// out[cond][N]
const out: Record<string, Record<number, Agg>> = { DIRECT: {}, GRAPH: {} };
for (const c of ["DIRECT", "GRAPH"]) for (const N of NS) out[c][N] = mk();

const dir = await mkdtemp(join(tmpdir(), "km-sweep-"));
let qi = 0;
for (const r of bridge) {
  process.env.KEYMEM_DATA_DIR = await mkdtemp(join(dir, "q-"));
  const mg = await import(`../src/memoryGraph.ts?sw=${qi}`);
  const g = new mg.MemoryGraph();
  await g.load();
  const gidToTitle: Record<string, string> = {};
  for (let pi = 0; pi < r.titles.length; pi++) {
    const keys = agentKeys[`q${qi}p${pi}`] ?? [r.titles[pi]];
    const [gid] = await g.add(r.paras[pi] ?? "", keys, {});
    gidToTitle[gid] = r.titles[pi];
  }
  for (const cond of ["DIRECT", "GRAPH"]) {
    // request the largest N once, then score prefixes
    const res = (await g.recall(r.question, 10, null, cond === "GRAPH", cond === "GRAPH" ? 2 : 1, 0, 0)) as Array<{ id: string }>;
    const titles = res.map((x) => gidToTitle[x.id]).filter(Boolean);
    for (const N of NS) {
      const top = titles.slice(0, N);
      const found = r.support.filter((s) => top.includes(s)).length;
      const a = out[cond][N];
      a.n++; a.both += found === r.support.length ? 1 : 0; a.recall += found / r.support.length;
      a.noise += top.length - found; // non-support slots injected
    }
  }
  qi++;
}
await rm(dir, { recursive: true, force: true });

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
console.log(`\nkeymem inject N-sweep — HotpotQA bridge, blind agent keys — model=${LOCAL_EMBEDDING_MODEL}, N=${qi}`);
console.log("─".repeat(72));
console.log(`topN   both@N (DIRECT→GRAPH)   support-recall (D→G)   avg noise slots (G)`);
for (const N of NS) {
  const d = out.DIRECT[N], g = out.GRAPH[N];
  console.log(
    `${String(N).padEnd(4)}   ${pct(d.both / d.n).padStart(4)} → ${pct(g.both / g.n).padStart(4)}` +
    `            ${pct(d.recall / d.n).padStart(4)} → ${pct(g.recall / g.n).padStart(4)}` +
    `          ${(g.noise / g.n).toFixed(1)} / ${N}`
  );
}
console.log("─".repeat(72));
console.log(`(both@N: got BOTH gold supports within top-N; noise slots: injected non-support memories)`);
await writeFile(resolve("bench/inject-sweep-results.json"), JSON.stringify({ model: LOCAL_EMBEDDING_MODEL, N: qi, NS, aggregates: out }, null, 2));
console.log("results → bench/inject-sweep-results.json");
