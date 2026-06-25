// The local embedding model must download to ONE deterministic, absolute location regardless of the
// model family (e5 built-in vs bge-m3 custom) or the process's working directory. The old default
// was a RELATIVE "local_cache", which scattered the ~500MB model wherever the server happened to be
// launched. It should default to <keymem home>/models (i.e. ~/.keymem/models), the same parent that
// bge-m3 already uses, so everything lands together.
import assert from "node:assert/strict";
import { isAbsolute, join } from "node:path";
import test from "node:test";

let n = 0;

test("LOCAL_EMBEDDING_CACHE_DIR defaults to an absolute <home>/models, not a relative dir", async () => {
  delete process.env.LOCAL_EMBEDDING_CACHE_DIR;
  const env = await import("../src/env.ts");
  const emb = await import(`../src/embedding.ts?cachedir=${n++}`);

  const expected = join(env.homeBaseDir(), "models");
  assert.equal(emb.LOCAL_EMBEDDING_CACHE_DIR, expected, "default cache dir should be <keymem home>/models");
  assert.ok(isAbsolute(emb.LOCAL_EMBEDDING_CACHE_DIR), "cache dir must be absolute, not relative");
  assert.ok(emb.LOCAL_EMBEDDING_CACHE_DIR.endsWith(join(".keymem", "models")) ||
            emb.LOCAL_EMBEDDING_CACHE_DIR.endsWith(join(".super-memory", "models")),
            `should live under the keymem home, got ${emb.LOCAL_EMBEDDING_CACHE_DIR}`);
});

test("an explicit LOCAL_EMBEDDING_CACHE_DIR still wins (e.g. Docker pins it to /data/models)", async () => {
  process.env.LOCAL_EMBEDDING_CACHE_DIR = "/data/models";
  const emb = await import(`../src/embedding.ts?cachedir=${n++}`);
  assert.equal(emb.LOCAL_EMBEDDING_CACHE_DIR, "/data/models");
  delete process.env.LOCAL_EMBEDDING_CACHE_DIR;
});
