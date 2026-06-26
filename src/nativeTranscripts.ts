// Read and normalize conversation transcripts that local coding agents
// (Claude Code, Codex) already persist on disk. keymem never writes these —
// it only reads the host agent's own logs from well-known, per-OS paths.

import { readFile, readdir, stat, realpath } from "fs/promises";
import { homedir } from "os";
import { join, basename, resolve, sep } from "path";

export type Agent = "claude" | "codex";

export interface NormalizedTurn {
  turn: number;
  role: string;
  content: string;
  ts: string | null;
}

export interface SessionInfo {
  agent: Agent;
  session_id: string;
  path: string;
  cwd: string | null;
  modified: number;
  preview: string | null;
}

// Both agents name their session files by a v4-style UUID. Restricting
// session_id to this shape is the primary path-traversal guard: the value is
// only ever used as a filename filter, never concatenated into a path segment
// that could escape the agent's root.
const UUID_PATTERN =
  /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

function assertUuid(sessionId: string): void {
  if (!UUID_PATTERN.test(sessionId)) {
    throw new Error(
      `Invalid session_id "${sessionId}". Expected a UUID as written by the agent's transcript file.`
    );
  }
}

// ── Roots (honour the agents' own env overrides) ──

export function claudeRoot(): string {
  const home = process.env.CLAUDE_CONFIG_DIR || join(homedir(), ".claude");
  return join(home, "projects");
}

export function codexRoot(): string {
  const home = process.env.CODEX_HOME || join(homedir(), ".codex");
  return join(home, "sessions");
}

function rootFor(agent: Agent): string {
  return agent === "claude" ? claudeRoot() : codexRoot();
}

// The host agent stamps its current session id into the MCP server's env.
function envSession(): { agent: Agent; session_id: string } | null {
  const claude = process.env.CLAUDE_CODE_SESSION_ID;
  if (claude && UUID_PATTERN.test(claude)) return { agent: "claude", session_id: claude };
  const codex = process.env.CODEX_THREAD_ID;
  if (codex && UUID_PATTERN.test(codex)) return { agent: "codex", session_id: codex };
  return null;
}

// ── Parsing ──

function parseLines(text: string): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line) as Record<string, unknown>);
    } catch {
      // External logs we don't own — skip malformed lines instead of throwing.
    }
  }
  return out;
}

// Flatten a content field (string or array of typed blocks) to plain text,
// keeping only human/assistant prose (text/input_text/output_text), dropping
// thinking, tool calls, and other non-conversational blocks.
function blocksToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const b = block as Record<string, unknown>;
    const type = b.type;
    if (type === "text" || type === "input_text" || type === "output_text") {
      if (typeof b.text === "string") parts.push(b.text);
    }
  }
  return parts.join("\n");
}

function parseClaude(text: string): NormalizedTurn[] {
  const turns: NormalizedTurn[] = [];
  let idx = 0;
  for (const line of parseLines(text)) {
    if (line.type !== "user" && line.type !== "assistant") continue;
    const message = (line.message as Record<string, unknown>) ?? {};
    const role = typeof message.role === "string" ? message.role : (line.type as string);
    const content = blocksToText(message.content);
    if (!content.trim()) continue;
    turns.push({
      turn: idx++,
      role,
      content,
      ts: typeof line.timestamp === "string" ? line.timestamp : null,
    });
  }
  return turns;
}

function parseCodex(text: string): NormalizedTurn[] {
  const turns: NormalizedTurn[] = [];
  let idx = 0;
  for (const line of parseLines(text)) {
    if (line.type !== "response_item") continue;
    const payload = (line.payload as Record<string, unknown>) ?? {};
    if (payload.type !== "message") continue;
    const role = payload.role;
    if (role !== "user" && role !== "assistant") continue;
    const content = blocksToText(payload.content);
    if (!content.trim()) continue;
    turns.push({
      turn: idx++,
      role: role as string,
      content,
      ts: typeof line.timestamp === "string" ? line.timestamp : null,
    });
  }
  return turns;
}

function parseFor(agent: Agent, text: string): NormalizedTurn[] {
  return agent === "claude" ? parseClaude(text) : parseCodex(text);
}

// ── File resolution ──

async function walkJsonl(dir: string): Promise<string[]> {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  const files: string[] = [];
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walkJsonl(full)));
    } else if (entry.isFile() && entry.name.endsWith(".jsonl")) {
      files.push(full);
    }
  }
  return files;
}

function fileMatchesSession(agent: Agent, file: string, sessionId: string): boolean {
  const name = basename(file);
  return agent === "claude"
    ? name === `${sessionId}.jsonl`
    : name.endsWith(`-${sessionId}.jsonl`);
}

// Confirm the resolved file truly lives under the agent root, following any
// symlinks — defence in depth on top of the UUID filter.
async function withinRoot(root: string, file: string): Promise<boolean> {
  try {
    const realRoot = await realpath(root);
    const realFile = await realpath(file);
    return realFile === realRoot || realFile.startsWith(realRoot + sep);
  } catch {
    return false;
  }
}

async function findSessionFile(agent: Agent, sessionId: string): Promise<string | null> {
  const root = rootFor(agent);
  for (const file of await walkJsonl(root)) {
    if (fileMatchesSession(agent, file, sessionId)) {
      if (await withinRoot(root, resolve(file))) return file;
    }
  }
  return null;
}

// ── Public API ──

export async function loadNativeConversation(
  agent: Agent,
  sessionId: string,
  turn?: number | null
): Promise<NormalizedTurn[]> {
  assertUuid(sessionId);
  const file = await findSessionFile(agent, sessionId);
  if (!file) return [];
  const turns = parseFor(agent, await readFile(file, "utf-8"));
  if (turn != null) {
    const start = Math.max(0, turn - 2);
    const end = Math.min(turns.length, turn + 3);
    return turns.slice(start, end);
  }
  return turns;
}

// Best-effort lookup when the caller doesn't know which agent produced the
// session: try each root in turn. Non-UUID or unknown ids resolve to an empty
// array (never throw) so callers can fall back to other stores.
export async function loadNativeAuto(
  sessionId: string,
  turn?: number | null
): Promise<NormalizedTurn[]> {
  if (!UUID_PATTERN.test(sessionId)) return [];
  for (const agent of ["claude", "codex"] as Agent[]) {
    const turns = await loadNativeConversation(agent, sessionId, turn);
    if (turns.length) return turns;
  }
  return [];
}

// Identify the session being written *right now* — the active conversation that
// triggered a remember() call is, by construction, the most-recently-touched
// transcript on disk. A staleness guard avoids mislabelling memories saved
// outside any live agent session (e.g. a script). Returns the host link to
// stamp on the memory's provenance.
export async function detectActiveSession(
  opts: { maxAgeMs?: number; now?: number } = {}
): Promise<{ agent: Agent; session_id: string; turn: number } | null> {
  const maxAgeMs = opts.maxAgeMs ?? 5 * 60 * 1000;
  const now = opts.now ?? Date.now();

  // Tier 1 — authoritative: the host injects its session id into every MCP
  // server it spawns (Claude Code → CLAUDE_CODE_SESSION_ID, Codex →
  // CODEX_THREAD_ID). When present it identifies the session exactly, with no
  // mtime guessing or cross-session ambiguity.
  const fromEnv = envSession();
  if (fromEnv) {
    const file = await findSessionFile(fromEnv.agent, fromEnv.session_id);
    const turns = file ? parseFor(fromEnv.agent, await readFile(file, "utf-8")) : [];
    return { ...fromEnv, turn: Math.max(0, turns.length - 1) };
  }

  // Tier 2 — heuristic fallback (Codex, Claude Desktop, older clients): the
  // transcript being appended right now is the most-recently-modified file.
  let best: { agent: Agent; file: string; mtime: number } | null = null;
  for (const agent of ["claude", "codex"] as Agent[]) {
    for (const file of await walkJsonl(rootFor(agent))) {
      let mtime: number;
      try {
        mtime = (await stat(file)).mtimeMs;
      } catch {
        continue;
      }
      if (!best || mtime > best.mtime) best = { agent, file, mtime };
    }
  }
  if (!best || now - best.mtime > maxAgeMs) return null;

  let text: string;
  try {
    text = await readFile(best.file, "utf-8");
  } catch {
    return null;
  }
  const lines = parseLines(text);
  const sessionId =
    best.agent === "claude" ? claudeSessionId(best.file) : codexSessionMeta(lines)?.id ?? null;
  if (!sessionId) return null;
  const turns = parseFor(best.agent, text);
  return { agent: best.agent, session_id: sessionId, turn: Math.max(0, turns.length - 1) };
}

function claudeSessionId(file: string): string | null {
  const name = basename(file, ".jsonl");
  return UUID_PATTERN.test(name) ? name : null;
}

function codexSessionMeta(lines: Record<string, unknown>[]): { id: string; cwd: string | null } | null {
  for (const line of lines) {
    if (line.type !== "session_meta") continue;
    const payload = (line.payload as Record<string, unknown>) ?? {};
    if (typeof payload.id === "string" && UUID_PATTERN.test(payload.id)) {
      return { id: payload.id, cwd: typeof payload.cwd === "string" ? payload.cwd : null };
    }
  }
  return null;
}

function claudeCwd(lines: Record<string, unknown>[]): string | null {
  for (const line of lines) {
    if (typeof line.cwd === "string") return line.cwd;
  }
  return null;
}

async function sessionInfo(agent: Agent, file: string): Promise<SessionInfo | null> {
  let modified = 0;
  try {
    modified = (await stat(file)).mtimeMs;
  } catch {
    return null;
  }
  let text: string;
  try {
    text = await readFile(file, "utf-8");
  } catch {
    return null;
  }
  const lines = parseLines(text);
  const turns = parseFor(agent, text);
  const preview = turns.find((tn) => tn.role === "user")?.content ?? null;

  if (agent === "claude") {
    const id = claudeSessionId(file);
    if (!id) return null;
    return { agent, session_id: id, path: file, cwd: claudeCwd(lines), modified, preview };
  }
  const meta = codexSessionMeta(lines);
  if (!meta) return null;
  return { agent, session_id: meta.id, path: file, cwd: meta.cwd, modified, preview };
}

export async function listNativeSessions(
  opts: { agent?: Agent; limit?: number } = {}
): Promise<SessionInfo[]> {
  const agents: Agent[] = opts.agent ? [opts.agent] : ["claude", "codex"];
  const sessions: SessionInfo[] = [];
  for (const agent of agents) {
    for (const file of await walkJsonl(rootFor(agent))) {
      const info = await sessionInfo(agent, file);
      if (info) sessions.push(info);
    }
  }
  sessions.sort((a, b) => b.modified - a.modified);
  return typeof opts.limit === "number" ? sessions.slice(0, opts.limit) : sessions;
}
