// Operational robustness: a backend/dimension switch must NOT brick the graph.
// We write a graph.json whose embeddings are the "wrong" dimension (simulating
// data created by a different backend), then load() and assert auto-migration
// re-embeds everything, preserves content/links/depth, and restores recall.
import { mkdtemp, writeFile, readFile, rm, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
const dataDir = await mkdtemp(join(tmpdir(), "sm-mig-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

let pass = 0, fail = 0; const fails: string[] = [];
function check(name: string, cond: boolean, detail = "") {
  if (cond) { pass++; console.log(`  ✅ ${name}`); }
  else { fail++; fails.push(name); console.log(`  ❌ ${name}  ${detail}`); }
}

// Craft a graph with 3-dim (wrong) embeddings but real content/keys/links/depth.
const wrongDim = {
  keys: {
    k1: { id: "k1", concept: "딸기", embedding: [0.1, 0.2, 0.3], key_type: "concept" },
    k2: { id: "k2", concept: "fruit", embedding: [0.2, 0.1, 0.4], key_type: "concept" },
  },
  memories: {
    m1: { id: "m1", content: "사용자는 딸기를 매우 좋아한다", embedding: [0.1, 0.2, 0.3],
      created_at: 1000, source: null, supersedes: null, depth: 0.73, access_count: 9,
      last_accessed: 1000, namespace: "default", ttl: null, links: ["m2"] },
    m2: { id: "m2", content: "딸기는 봄에 제철이다", embedding: [0.3, 0.2, 0.1],
      created_at: 1000, source: null, supersedes: null, depth: 0.4, access_count: 3,
      last_accessed: 1000, namespace: "default", ttl: null, links: [] },
  },
  links: [
    { key_id: "k1", memory_id: "m1", weight: 1.5 },
    { key_id: "k2", memory_id: "m1", weight: 1.0 },
    { key_id: "k1", memory_id: "m2", weight: 1.0 },
  ],
};
await writeFile(join(dataDir, "graph.json"), JSON.stringify(wrongDim), "utf-8");

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const g = new MemoryGraph();
await g.load(); // should auto-migrate 3-dim -> real dim

const realDim = g.memories.m1.embedding.length;
check("memories re-embedded to current backend dim (not 3)", realDim > 3, `dim=${realDim}`);
check("all memories share the new dim", g.memories.m2.embedding.length === realDim, "");
check("depth preserved through migration", Math.abs(g.memories.m1.depth - 0.73) < 1e-9, `${g.memories.m1.depth}`);
check("access_count preserved", g.memories.m1.access_count === 9, "");
check("explicit link preserved", g.memories.m1.links.includes("m2"), "");
check("key link weight preserved", (g as any)._keyToMems["k1"]?.get("m1") === 1.5, `${(g as any)._keyToMems["k1"]?.get("m1")}`);

const backups = (await readdir(dataDir)).filter(f => f.includes(".bak."));
check("pre-migration backup written", backups.length >= 1, `files=${backups.join(",")}`);

// The whole point: recall works again after the backend switch.
const r = await g.recall("딸기", 5, "default", false) as any[];
check("recall works post-migration (no brick)", r.length > 0, `n=${r.length}`);
check("recall returns the strawberry memory", r.some(x => x.content.includes("딸기를 매우 좋아")), `top=${r[0]?.content}`);

// Persisted file is now the new dim — a second cold load needs no migration.
const persisted = JSON.parse(await readFile(join(dataDir, "graph.json"), "utf-8"));
check("persisted embeddings are migrated on disk", persisted.memories.m1.embedding.length === realDim, "");

console.log(`\n${"=".repeat(50)}`);
console.log(`RESULT: ${pass} passed, ${fail} failed`);
if (fail) console.log(`FAILED: ${fails.join(", ")}`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
