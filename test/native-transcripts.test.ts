import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile, utimes } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let importCounter = 0;

async function tempDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "native-transcripts-test-"));
}

// Loads the module fresh with the given native agent home overrides. Always
// clears CLAUDE_CODE_SESSION_ID unless explicitly set, so the test runner's own
// host session (this process is itself a Claude Code child) never leaks in.
async function loadModule(env: {
  claudeConfigDir?: string;
  codexHome?: string;
  claudeSessionId?: string;
  codexThreadId?: string;
}) {
  if (env.claudeConfigDir) process.env.CLAUDE_CONFIG_DIR = env.claudeConfigDir;
  else delete process.env.CLAUDE_CONFIG_DIR;
  if (env.codexHome) process.env.CODEX_HOME = env.codexHome;
  else delete process.env.CODEX_HOME;
  if (env.claudeSessionId) process.env.CLAUDE_CODE_SESSION_ID = env.claudeSessionId;
  else delete process.env.CLAUDE_CODE_SESSION_ID;
  if (env.codexThreadId) process.env.CODEX_THREAD_ID = env.codexThreadId;
  else delete process.env.CODEX_THREAD_ID;
  return import(`../src/nativeTranscripts.ts?test=${importCounter++}`);
}

const UUID = "e7f5b1d2-1602-4180-ac66-9f9acbd1f673";

async function writeClaudeFixture(configDir: string, uuid: string) {
  const projDir = join(configDir, "projects", "-Users-me-proj");
  await mkdir(projDir, { recursive: true });
  const lines = [
    { type: "summary", summary: "ignore me", leafUuid: "x" },
    {
      type: "user",
      sessionId: uuid,
      cwd: "/Users/me/proj",
      timestamp: "2026-01-01T00:00:00.000Z",
      message: { role: "user", content: "hello there" },
    },
    { type: "system", content: "tool noise" },
    {
      type: "assistant",
      sessionId: uuid,
      timestamp: "2026-01-01T00:00:01.000Z",
      message: {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "private reasoning" },
          { type: "text", text: "hi back" },
          { type: "tool_use", name: "Bash", input: {} },
        ],
      },
    },
  ];
  await writeFile(
    join(projDir, `${uuid}.jsonl`),
    lines.map((l) => JSON.stringify(l)).join("\n") + "\n",
    "utf-8"
  );
}

async function writeCodexFixture(codexHome: string, uuid: string): Promise<string> {
  const dayDir = join(codexHome, "sessions", "2026", "01", "01");
  await mkdir(dayDir, { recursive: true });
  const lines = [
    {
      type: "session_meta",
      timestamp: "2026-01-01T00:00:00.000Z",
      payload: { id: uuid, cwd: "/Users/me/proj", timestamp: "2026-01-01T00:00:00.000Z" },
    },
    {
      type: "response_item",
      timestamp: "2026-01-01T00:00:01.000Z",
      payload: { type: "message", role: "user", content: [{ type: "input_text", text: "hello codex" }] },
    },
    { type: "event_msg", payload: { type: "token_count" } },
    {
      type: "response_item",
      timestamp: "2026-01-01T00:00:02.000Z",
      payload: { type: "message", role: "assistant", content: [{ type: "output_text", text: "hi from codex" }] },
    },
  ];
  const path = join(dayDir, `rollout-2026-01-01T00-00-00-${uuid}.jsonl`);
  await writeFile(path, lines.map((l) => JSON.stringify(l)).join("\n") + "\n", "utf-8");
  return path;
}

async function writeClaudeFixturePath(configDir: string, uuid: string): Promise<string> {
  await writeClaudeFixture(configDir, uuid);
  return join(configDir, "projects", "-Users-me-proj", `${uuid}.jsonl`);
}

test("parses a Claude transcript into normalized turns (text only)", async (t) => {
  const dir = await tempDir();
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeClaudeFixture(dir, UUID);

  const { loadNativeConversation } = await loadModule({ claudeConfigDir: dir });
  const turns = await loadNativeConversation("claude", UUID);

  assert.deepEqual(turns, [
    { turn: 0, role: "user", content: "hello there", ts: "2026-01-01T00:00:00.000Z" },
    { turn: 1, role: "assistant", content: "hi back", ts: "2026-01-01T00:00:01.000Z" },
  ]);
});

test("parses a Codex rollout into normalized turns", async (t) => {
  const dir = await tempDir();
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeCodexFixture(dir, UUID);

  const { loadNativeConversation } = await loadModule({ codexHome: dir });
  const turns = await loadNativeConversation("codex", UUID);

  assert.deepEqual(turns, [
    { turn: 0, role: "user", content: "hello codex", ts: "2026-01-01T00:00:01.000Z" },
    { turn: 1, role: "assistant", content: "hi from codex", ts: "2026-01-01T00:00:02.000Z" },
  ]);
});

test("turn windowing returns ±2 around the requested turn", async (t) => {
  const dir = await tempDir();
  t.after(() => rm(dir, { recursive: true, force: true }));
  const projDir = join(dir, "projects", "-Users-me-proj");
  await mkdir(projDir, { recursive: true });
  const lines = Array.from({ length: 10 }, (_, i) => ({
    type: i % 2 === 0 ? "user" : "assistant",
    sessionId: UUID,
    timestamp: `2026-01-01T00:00:0${i}.000Z`,
    message: { role: i % 2 === 0 ? "user" : "assistant", content: `turn ${i}` },
  }));
  await writeFile(
    join(projDir, `${UUID}.jsonl`),
    lines.map((l) => JSON.stringify(l)).join("\n") + "\n",
    "utf-8"
  );

  const { loadNativeConversation } = await loadModule({ claudeConfigDir: dir });
  const turns = await loadNativeConversation("claude", UUID, 5);

  assert.deepEqual(
    turns.map((tn: any) => tn.turn),
    [3, 4, 5, 6, 7]
  );
});

test("rejects non-UUID session ids (path traversal guard)", async (t) => {
  const dir = await tempDir();
  t.after(() => rm(dir, { recursive: true, force: true }));
  const { loadNativeConversation } = await loadModule({ claudeConfigDir: dir });

  await assert.rejects(
    () => loadNativeConversation("claude", "../../etc/passwd"),
    /Invalid session_id/
  );
  await assert.rejects(
    () => loadNativeConversation("codex", "not-a-uuid"),
    /Invalid session_id/
  );
});

test("returns empty array when the session is not found", async (t) => {
  const dir = await tempDir();
  t.after(() => rm(dir, { recursive: true, force: true }));
  await mkdir(join(dir, "projects"), { recursive: true });
  const { loadNativeConversation } = await loadModule({ claudeConfigDir: dir });

  const turns = await loadNativeConversation("claude", "00000000-0000-4000-8000-000000000000");
  assert.deepEqual(turns, []);
});

test("auto-detects the agent from the session id across roots", async (t) => {
  const claudeDir = await tempDir();
  const codexDir = await tempDir();
  t.after(() => rm(claudeDir, { recursive: true, force: true }));
  t.after(() => rm(codexDir, { recursive: true, force: true }));
  await writeClaudeFixture(claudeDir, UUID);
  const codexUuid = "11111111-2222-4333-8444-555555555555";
  await writeCodexFixture(codexDir, codexUuid);

  const { loadNativeAuto } = await loadModule({
    claudeConfigDir: claudeDir,
    codexHome: codexDir,
  });

  const claudeTurns = await loadNativeAuto(UUID);
  assert.equal(claudeTurns[0].content, "hello there");
  const codexTurns = await loadNativeAuto(codexUuid);
  assert.equal(codexTurns[0].content, "hello codex");

  // Non-UUID and unknown ids resolve to empty without throwing, so callers
  // can safely fall back to other lookups.
  assert.deepEqual(await loadNativeAuto("not-a-uuid"), []);
  assert.deepEqual(await loadNativeAuto("99999999-0000-4000-8000-000000000000"), []);
});

test("detectActiveSession picks the most-recently-modified transcript", async (t) => {
  const claudeDir = await tempDir();
  const codexDir = await tempDir();
  t.after(() => rm(claudeDir, { recursive: true, force: true }));
  t.after(() => rm(codexDir, { recursive: true, force: true }));
  const claudePath = await writeClaudeFixturePath(claudeDir, UUID);
  const codexUuid = "11111111-2222-4333-8444-555555555555";
  const codexPath = await writeCodexFixture(codexDir, codexUuid);

  // Claude older, Codex newer → the active session is Codex.
  await utimes(claudePath, new Date(1_000_000), new Date(1_000_000));
  await utimes(codexPath, new Date(2_000_000), new Date(2_000_000));

  const { detectActiveSession } = await loadModule({
    claudeConfigDir: claudeDir,
    codexHome: codexDir,
  });
  const active = await detectActiveSession({ now: 2_000_000, maxAgeMs: 10_000_000 });

  assert.equal(active.agent, "codex");
  assert.equal(active.session_id, codexUuid);
  assert.equal(active.turn, 1); // 2 turns (user, assistant) → latest index
});

test("detectActiveSession trusts CLAUDE_CODE_SESSION_ID over mtime", async (t) => {
  const claudeDir = await tempDir();
  const codexDir = await tempDir();
  t.after(() => rm(claudeDir, { recursive: true, force: true }));
  t.after(() => rm(codexDir, { recursive: true, force: true }));
  // The env session's transcript is OLDER than a competing Codex rollout, yet
  // the env var must win — it is authoritative, not a guess.
  const claudePath = await writeClaudeFixturePath(claudeDir, UUID);
  const codexUuid = "11111111-2222-4333-8444-555555555555";
  const codexPath = await writeCodexFixture(codexDir, codexUuid);
  await utimes(claudePath, new Date(1_000_000), new Date(1_000_000)); // older
  await utimes(codexPath, new Date(9_000_000), new Date(9_000_000)); // newer

  const { detectActiveSession } = await loadModule({
    claudeConfigDir: claudeDir,
    codexHome: codexDir,
    claudeSessionId: UUID,
  });
  const active = await detectActiveSession({ now: 9_000_000, maxAgeMs: 100_000_000 });

  assert.equal(active.agent, "claude");
  assert.equal(active.session_id, UUID); // env wins despite Codex being fresher
  assert.equal(active.turn, 1);
});

test("detectActiveSession trusts CODEX_THREAD_ID over mtime", async (t) => {
  const claudeDir = await tempDir();
  const codexDir = await tempDir();
  t.after(() => rm(claudeDir, { recursive: true, force: true }));
  t.after(() => rm(codexDir, { recursive: true, force: true }));
  // Codex thread ids are UUIDv7-style (e.g. 019f0226-…); the value matches the
  // rollout file's session id. A fresher Claude transcript must not win.
  const codexUuid = "019f0226-8525-7ab0-8369-7dc41701eea1";
  const codexPath = await writeCodexFixture(codexDir, codexUuid);
  const claudePath = await writeClaudeFixturePath(claudeDir, UUID);
  await utimes(codexPath, new Date(1_000_000), new Date(1_000_000)); // older
  await utimes(claudePath, new Date(9_000_000), new Date(9_000_000)); // newer

  const { detectActiveSession } = await loadModule({
    claudeConfigDir: claudeDir,
    codexHome: codexDir,
    codexThreadId: codexUuid,
  });
  const active = await detectActiveSession({ now: 9_000_000, maxAgeMs: 100_000_000 });

  assert.equal(active.agent, "codex");
  assert.equal(active.session_id, codexUuid); // env wins despite Claude being fresher
  assert.equal(active.turn, 1);
});

test("detectActiveSession falls back to mtime when no env session is set", async (t) => {
  const claudeDir = await tempDir();
  const codexDir = await tempDir();
  t.after(() => rm(claudeDir, { recursive: true, force: true }));
  t.after(() => rm(codexDir, { recursive: true, force: true }));
  const codexUuid = "11111111-2222-4333-8444-555555555555";
  const codexPath = await writeCodexFixture(codexDir, codexUuid);
  await utimes(codexPath, new Date(2_000_000), new Date(2_000_000));

  const { detectActiveSession } = await loadModule({
    claudeConfigDir: claudeDir, // empty
    codexHome: codexDir,
  });
  const active = await detectActiveSession({ now: 2_000_000, maxAgeMs: 10_000_000 });

  assert.equal(active.agent, "codex");
  assert.equal(active.session_id, codexUuid);
});

test("detectActiveSession returns null when the freshest transcript is stale", async (t) => {
  const claudeDir = await tempDir();
  const codexDir = await tempDir(); // empty — isolate from the real ~/.codex
  t.after(() => rm(claudeDir, { recursive: true, force: true }));
  t.after(() => rm(codexDir, { recursive: true, force: true }));
  const claudePath = await writeClaudeFixturePath(claudeDir, UUID);
  await utimes(claudePath, new Date(2_000_000), new Date(2_000_000));

  const { detectActiveSession } = await loadModule({
    claudeConfigDir: claudeDir,
    codexHome: codexDir,
  });
  const active = await detectActiveSession({ now: 5_000_000, maxAgeMs: 1000 });

  assert.equal(active, null);
});

test("lists recent sessions across agents with cwd and preview", async (t) => {
  const claudeDir = await tempDir();
  const codexDir = await tempDir();
  t.after(() => rm(claudeDir, { recursive: true, force: true }));
  t.after(() => rm(codexDir, { recursive: true, force: true }));
  await writeClaudeFixture(claudeDir, UUID);
  const otherUuid = "11111111-2222-4333-8444-555555555555";
  await writeCodexFixture(codexDir, otherUuid);

  const { listNativeSessions } = await loadModule({
    claudeConfigDir: claudeDir,
    codexHome: codexDir,
  });
  const sessions = await listNativeSessions({ limit: 10 });

  const byId = Object.fromEntries(sessions.map((s: any) => [s.session_id, s]));
  assert.ok(byId[UUID], "claude session listed");
  assert.equal(byId[UUID].agent, "claude");
  assert.equal(byId[UUID].cwd, "/Users/me/proj");
  assert.equal(byId[UUID].preview, "hello there");
  assert.ok(byId[otherUuid], "codex session listed");
  assert.equal(byId[otherUuid].agent, "codex");
  assert.equal(byId[otherUuid].preview, "hello codex");
});
