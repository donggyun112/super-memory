// Auto-download for the local CUSTOM embedding model (bge-m3), with strict backward
// compatibility: a complete model dir downloads NOTHING; a partial/empty one fetches only
// the missing files (self-healing). The fetcher is injected so these run offline. Online-API
// backends and fastembed built-ins never call ensureCustomEmbeddingModel.
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
const ALL = ["model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"];

test("backward compat: a COMPLETE model dir downloads nothing", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-mdl-full-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  for (const f of ALL) await writeFile(join(dir, f), "stub");
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.LOCAL_EMBEDDING_MODEL_PATH = dir;
  delete process.env.LOCAL_EMBEDDING_MODEL_FILE;

  const emb = await import(`../src/embedding.ts?dl=${n++}`);
  let calls = 0;
  const r = await emb.ensureCustomEmbeddingModel(async () => { calls++; });
  assert.equal(r.dir, dir);
  assert.equal(calls, 0, "a complete dir must not download anything");
});

test("auto-download: an EMPTY dir fetches model + tokenizer + config", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-mdl-empty-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.LOCAL_EMBEDDING_MODEL_PATH = dir;
  delete process.env.LOCAL_EMBEDDING_MODEL_FILE;

  const emb = await import(`../src/embedding.ts?dl=${n++}`);
  const fetched: string[] = [];
  await emb.ensureCustomEmbeddingModel(async (_u: string, dest: string) => { fetched.push(dest); writeFileSync(dest, "x"); });
  assert.ok(fetched.some((d) => d.endsWith("model.onnx")), "model.onnx fetched");
  assert.ok(fetched.some((d) => d.endsWith("tokenizer.json")), "tokenizer fetched");
  assert.ok(fetched.length >= 3, `expected >=3 files, got ${fetched.length}`);
});

test("self-heal: a PARTIAL dir (model only) fetches just the missing tokenizer/config", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-mdl-partial-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(join(dir, "model.onnx"), "stub"); // model present, tokenizer/config missing
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.LOCAL_EMBEDDING_MODEL_PATH = dir;
  delete process.env.LOCAL_EMBEDDING_MODEL_FILE;

  const emb = await import(`../src/embedding.ts?dl=${n++}`);
  const fetched: string[] = [];
  await emb.ensureCustomEmbeddingModel(async (_u: string, dest: string) => { fetched.push(dest); writeFileSync(dest, "x"); });
  assert.ok(!fetched.some((d) => d.endsWith("model.onnx")), "model.onnx already present → not re-downloaded");
  assert.ok(fetched.some((d) => d.endsWith("tokenizer.json")), "missing tokenizer must be fetched");
});
