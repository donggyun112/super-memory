import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { Memory } from "../src/types.js";

let importCounter = 0;

async function tempDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "super-memory-test-"));
}

async function loadMemoryModule(dataDir: string) {
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;
  return import(`../src/memoryGraph.ts?test=${importCounter++}`);
}

function memory(id: string, links: string[] = [], ttl: number | null = null): Memory {
  return {
    id,
    content: `memory ${id}`,
    embedding: [1, 0],
    created_at: 0,
    source: null,
    supersedes: null,
    depth: 0,
    access_count: 0,
    last_accessed: 0,
    namespace: "default",
    ttl,
    links,
  };
}

test("load fails on malformed graph files", async (t) => {
  const dataDir = await tempDir();
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  await writeFile(join(dataDir, "graph.json"), "{not json", "utf-8");

  const { MemoryGraph } = await loadMemoryModule(dataDir);
  await assert.rejects(
    () => new MemoryGraph().load(),
    /Failed to load memory graph/
  );
});

test("conversation session ids cannot escape the data directory", async (t) => {
  const dataDir = await tempDir();
  t.after(() => rm(dataDir, { recursive: true, force: true }));

  const { loadConversation, saveTurn } = await loadMemoryModule(dataDir);

  await assert.rejects(
    () => saveTurn("../escape", "user", "outside"),
    /Invalid session_id/
  );
  await assert.rejects(
    () => loadConversation("../escape"),
    /Invalid session_id/
  );

  await assert.rejects(
    () => readFile(join(dataDir, "escape.jsonl"), "utf-8"),
    (err: unknown) => (err as NodeJS.ErrnoException).code === "ENOENT"
  );
});

test("conversation logs report malformed JSONL lines", async (t) => {
  const dataDir = await tempDir();
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  await mkdir(join(dataDir, "conversations"), { recursive: true });
  await writeFile(
    join(dataDir, "conversations", "session-1.jsonl"),
    "{\"turn\":0}\nnot json\n",
    "utf-8"
  );

  const { loadConversation } = await loadMemoryModule(dataDir);
  await assert.rejects(
    () => loadConversation("session-1"),
    /Invalid conversation log session-1 at line 2/
  );
});

test("delete removes explicit links to the deleted memory", async (t) => {
  const dataDir = await tempDir();
  t.after(() => rm(dataDir, { recursive: true, force: true }));

  const { MemoryGraph } = await loadMemoryModule(dataDir);
  const graph = new MemoryGraph();
  graph.memories = {
    a: memory("a"),
    b: memory("b", ["a", "c", "a", "missing", "b"]),
    c: memory("c"),
  };

  assert.equal(await graph.delete("a"), true);
  assert.deepEqual(graph.memories.b.links, ["c"]);
});

test("cleanupExpired removes explicit links to expired memories", async (t) => {
  const dataDir = await tempDir();
  t.after(() => rm(dataDir, { recursive: true, force: true }));

  const { MemoryGraph } = await loadMemoryModule(dataDir);
  const graph = new MemoryGraph();
  graph.memories = {
    expired: memory("expired", [], Date.now() / 1000 - 1),
    active: memory("active", ["expired"]),
  };

  assert.equal(await graph.cleanupExpired(), 1);
  assert.deepEqual(graph.memories.active.links, []);
});
