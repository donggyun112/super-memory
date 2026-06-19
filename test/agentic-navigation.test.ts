import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(text: string): number[] {
  const vectors: Record<string, number[]> = {
    "프로그래밍": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "코딩": [0.95, 0.3122499, 0, 0, 0, 0, 0, 0, 0, 0],
    HUB: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    RARE: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "first memory": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "second memory": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "third memory": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  };
  return vectors[text] ?? [0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
}

test("agent navigates Key → Memory → Key while aliases and hubs stay visible", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-agentic-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.SUPER_MEMORY_SHORT_KEY_MERGE = "0.9";
  t.after(() => delete process.env.SUPER_MEMORY_SHORT_KEY_MERGE);

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const module = await import(`../src/memoryGraph.ts?agentic=${n++}`);
  const graph = new module.MemoryGraph();
  await graph.load();

  const [firstId] = await graph.add("first memory", ["프로그래밍", "HUB", "RARE"]);
  await graph.add("second memory", ["코딩", "HUB"]);
  await graph.add("third memory", ["프로그래밍", "HUB"]);

  const programming = Object.values(graph.keys).find((key) => key.concept === "프로그래밍");
  assert.ok(programming, "canonical programming key should exist");
  assert.deepEqual(programming.aliases, ["코딩"]);

  const candidates = (await graph.searchKeys("코딩", 5)) as any[];
  const keyCandidate = candidates.find((candidate) => candidate.key_id === programming.id);
  assert.ok(keyCandidate, "querying an alias should return its canonical key cluster");
  assert.equal(keyCandidate.match_type, "alias");
  assert.equal(keyCandidate.memory_count, 3);
  assert.equal(keyCandidate.is_hub, true);
  assert.equal(keyCandidate.cluster_size, 2);
  assert.ok(!("content" in keyCandidate), "recall candidates must not expose memory content");

  const keyRead = graph.readKey(programming.id, { limit: 2 }) as any;
  assert.equal(keyRead.total, 3);
  assert.equal(keyRead.memories.length, 2);
  assert.equal(keyRead.next_offset, 2);
  assert.ok(
    keyRead.memories.every((memory: any) => !("content" in memory) && !("preview" in memory)),
    "read_key must expose handles, not memory content"
  );
  assert.equal(graph.memories[firstId].access_count, 0, "read_key must not count as a full memory read");

  const memoryRead = (await graph.readMemory(firstId, programming.id)) as any;
  assert.equal(memoryRead.memory.content, "first memory");
  assert.equal(memoryRead.memory.access_count, 1);
  assert.equal(memoryRead.memory.depth, 0.05);

  const traversedKey = memoryRead.keys.find((key: any) => key.key_id === programming.id);
  assert.equal(traversedKey.traversed_from, true);
  assert.equal(traversedKey.link_weight, 1.1, "only the traversed edge should be reinforced");
  assert.deepEqual(traversedKey.aliases, ["코딩"]);

  const hubKey = memoryRead.keys.find((key: any) => key.concept === "HUB");
  assert.equal(hubKey.is_hub, true);
  assert.equal(hubKey.memory_count, 3);

  await assert.rejects(
    () => graph.readMemory(firstId, "not-linked"),
    /is not linked/,
    "via_key_id must identify a real traversed edge"
  );

  const reloaded = new module.MemoryGraph();
  await reloaded.load();
  const persisted = Object.values(reloaded.keys).find((key) => key.concept === "프로그래밍");
  assert.deepEqual(persisted?.aliases, ["코딩"], "aliases must survive persistence");
});
