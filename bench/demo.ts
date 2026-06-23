// Snappy terminal demo of keymem's associative recall, for an asciinema cast → GIF.
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

const C = { b: "\x1b[1m", d: "\x1b[2m", g: "\x1b[32m", c: "\x1b[36m", y: "\x1b[33m", r: "\x1b[31m", x: "\x1b[0m" };
const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));
const line = (s = "") => process.stdout.write(s + "\n");
const w = (s: string) => process.stdout.write(s);

const graph = new MemoryGraph();
await graph.load();

line(`${C.b}${C.c}keymem${C.x} ${C.d}· recall by association, not just similarity${C.x}`);
line();
line(`${C.d}two facts, saved across different chats —${C.x}`);
const facts = [
  { content: "Mina is allergic to peanuts", keys: ["Mina", "allergy", "peanuts"] },
  { content: "party cake recipe uses peanut butter frosting", keys: ["party cake", "recipe", "peanut butter", "peanuts"] },
];
const known: Record<string, string> = {};
for (const f of facts) {
  const [id] = await graph.add(f.content, f.keys, {});
  known[id] = f.content;
  line(`  ${C.g}✓${C.x}  ${f.content}`);
}
line();
line(`${C.d}later — a question that never says peanut, allergy, or Mina:${C.x}`);
w(`  ${C.y}❯${C.x} `);
const q = "is the party cake safe for the kids?";
for (const ch of q) { w(`${C.y}${ch}${C.x}`); await sleep(8); } // typed
line();
const res = (await graph.recallInject("is the party cake safe for the kids", 5, null, {})) as {
  memories: Array<{ id: string; content?: string }>;
};
line();
line(`  ${C.b}keymem surfaces, in one call:${C.x}`);
for (const m of res.memories) {
  const txt = known[m.id] ?? m.content ?? m.id;
  if (/allerg/i.test(txt)) line(`     ${C.r}${C.b}⚠ ${txt}${C.x}   ${C.r}← via shared key 'peanuts'${C.x}`);
  else line(`       ${txt}`);
}
line();
line(`${C.d}a vector store never connects these. keymem walks the key.${C.x}`);
line(`  ${C.c}❯ npx -y keymem${C.x}`);

await rm(dir, { recursive: true, force: true });
