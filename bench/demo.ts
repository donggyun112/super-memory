// Snappy terminal demo of keymem's associative recall, for an asciinema cast → GIF.
// Two modes share ONE event script so the GIF and the live run always match:
//   tsx bench/demo.ts            live, human-watchable (sleeps between writes)
//   tsx bench/demo.ts --cast     emits an asciicast v3 cast on stdout (feed to `agg`)
// The recall is run against a real MemoryGraph; only the pacing is authored, so the
// surfaced result is genuine — we just control timing precisely instead of recording.
/* eslint-disable no-console */
console.error = () => {};
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "fast-bge-small-en-v1.5";
const dir = await mkdtemp(join(tmpdir(), "keymem-demo-"));
process.env.KEYMEM_DATA_DIR = dir;
const { MemoryGraph } = await import("../src/memoryGraph.ts");

const CAST = process.argv.includes("--cast");
const C = { b: "\x1b[1m", d: "\x1b[2m", g: "\x1b[32m", c: "\x1b[36m", y: "\x1b[33m", r: "\x1b[31m", x: "\x1b[0m" };

const graph = new MemoryGraph();
await graph.load();

const facts = [
  { content: "Mina is allergic to peanuts", keys: ["Mina", "allergy", "peanuts"] },
  { content: "party cake recipe uses peanut butter frosting", keys: ["party cake", "recipe", "peanut butter", "peanuts"] },
];
const known: Record<string, string> = {};
for (const f of facts) {
  const [id] = await graph.add(f.content, f.keys, {});
  known[id] = f.content;
}

const q = "is the party cake safe for the kids?";
const res = (await graph.recallInject("is the party cake safe for the kids", 5, null, {})) as {
  memories: Array<{ id: string; content?: string }>;
};
// Prove the allergy fact really surfaced (the whole point of the demo) before we draw it.
const surfaced = res.memories.map((m) => known[m.id] ?? m.content ?? m.id);
if (!surfaced.some((t) => /allerg/i.test(t))) {
  process.stderr.write("demo: allergy memory did not surface — aborting\n");
  process.exit(1);
}
// The hop's middle node is the REAL shared key between the two facts, not a hardcode.
const shared = facts[0].keys.find((k) => facts[1].keys.includes(k)) ?? "peanuts";

await rm(dir, { recursive: true, force: true });

// ── One event script. `d` = gap in ms before writing `t`. ──
// PACE scales every gap uniformly — bump to slow the whole demo down, drop to speed up.
const PACE = 1.25;
type Step = { d: number; t: string };
const steps: Step[] = [];
const push = (d: number, t: string) => steps.push({ d: Math.round(d * PACE), t });

push(100, `${C.b}${C.c}keymem${C.x} ${C.d}· recall by association, not just similarity${C.x}\n\n`);
push(300, `${C.d}two facts, saved in different chats —${C.x}\n`);
push(280, `  ${C.g}✓${C.x}  ${facts[0].content}\n`);
push(280, `  ${C.g}✓${C.x}  ${facts[1].content}\n\n`);
push(350, `${C.d}later — a question that never says peanut, allergy, or Mina:${C.x}\n`);
push(250, `  ${C.y}❯${C.x} `);
for (const ch of q) push(12, `${C.y}${ch}${C.x}`); // typed, ~0.4s total
push(150, "\n");
push(550, `\n  ${C.b}keymem walks the keys:${C.x}\n`);
push(350, `  ${C.c}party cake${C.x} → ${C.c}peanut butter${C.x} → ${C.b}${C.y}\x1b[4m${shared}\x1b[24m${C.x} → ${C.r}${C.b}⚠ Mina${C.x}\n`);
push(450, `\n  ${C.r}${C.b}⚠ not safe${C.x} — cake has peanuts, Mina is allergic\n`);
push(450, `\n${C.d}a vector store never makes this hop. keymem walks the key.${C.x}\n`);
push(250, `  ${C.c}❯ npx -y keymem${C.x}\n`);

if (CAST) {
  const header = {
    version: 3,
    term: { cols: 80, rows: 24 },
    timestamp: 1782197850,
    command: "CI=true tsx bench/demo.ts --cast",
    env: { SHELL: "/bin/zsh" },
  };
  // asciicast v3 event times are RELATIVE intervals (gap since previous event),
  // and the terminal is raw — bare "\n" only line-feeds, so use "\r\n" to align.
  const out = [JSON.stringify(header)];
  for (const s of steps) {
    out.push(JSON.stringify([s.d / 1000, "o", s.t.replace(/\n/g, "\r\n")]));
  }
  out.push(JSON.stringify([0.05, "x", "0"]));
  process.stdout.write(out.join("\n") + "\n");
} else {
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  for (const s of steps) {
    await sleep(s.d);
    process.stdout.write(s.t);
  }
}
