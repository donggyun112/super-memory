import { readFile, writeFile, mkdir, appendFile, rename, copyFile } from "fs/promises";
import { randomBytes } from "crypto";
import { join } from "path";
import { Mutex } from "async-mutex";
import { cfgRaw, dataDir } from "./env.js";
import { selectInject } from "./inject.js";
import MiniSearch from "minisearch";
import { embedTextAsync, EMBEDDING_BACKEND, embeddingFingerprint, getThresholdProfile, isShortConcept, inContradictionBand } from "./embedding.js";
import { rerankEnabled, rerankScores } from "./reranker.js";
import type { Key, Memory, GraphData } from "./types.js";
import {
  RecallBuffer, decidePromotion,
  AUTOKEY_ENABLED, AUTOKEY_BUFFER_CAPACITY, AUTOKEY_BUFFER_TTL_SECONDS,
  AUTOKEY_PROMOTE_N, AUTOKEY_MAX_ALIASES, AUTOKEY_PRUNE_AGE_SECONDS,
  AUTOKEY_CONFIRM_FLOOR,
} from "./autokey.js";

const DATA_DIR = dataDir();
const GRAPH_FILE = join(DATA_DIR, "graph.json");
const CONVERSATIONS_DIR = join(DATA_DIR, "conversations");
const SESSION_ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/;

// Thresholds are calibrated per embedding backend/model (see embedding.ts).
const _THRESHOLDS = getThresholdProfile();
const KEY_MERGE_THRESHOLD = _THRESHOLDS.keyMerge;
const MEMORY_DEDUP_THRESHOLD = _THRESHOLDS.memoryDedup;
const KEY_AUTO_LINK_THRESHOLD = _THRESHOLDS.keyAutoLink;
const KEY_RECALL_THRESHOLD = _THRESHOLDS.keyRecall;
const CONTENT_RECALL_THRESHOLD = _THRESHOLDS.contentRecall;
const MIN_SCORE_THRESHOLD = _THRESHOLDS.minScore;
const GATE_Z_THRESHOLD = _THRESHOLDS.gateZ;
const GATE_MIN_POPULATION = 8;
const KEY_GATE_THRESHOLD = _THRESHOLDS.keyGate;
const SHORT_KEY_MERGE_THRESHOLD = _THRESHOLDS.shortKeyMerge;
const CONTRADICTION_THRESHOLD = _THRESHOLDS.contradiction;
const DEPTH_INCREMENT = 0.05;
const DEPTH_MAX = 1.0;
const DEPTH_DEEP_THRESHOLD = 0.7;

const RRF_K = 60;
const BM25_RESULT_DEPTH = 50;
const DENSE_RESULT_DEPTH = 50;

// related() ranks neighbors by shared-key specificity (IDF) and caps the list, so a hub
// key (shared by many) can't flood the chain. Keeps recall→related→related navigable.
const RELATED_LIMIT = Number(cfgRaw("RELATED_LIMIT") ?? 20);
const RELATED_EXPLICIT_BONUS = 1.0; // an explicit link is the strongest connection signal
const _hubMinLinks = Number(cfgRaw("KEY_HUB_MIN_LINKS") ?? 3);
const KEY_HUB_MIN_LINKS = Number.isFinite(_hubMinLinks)
  ? Math.max(2, Math.floor(_hubMinLinks))
  : 3;

// When the cross-encoder reranker is on (KEYMEM_RERANK), re-score this many of the
// top fused candidates by joint (query, memory) relevance, then keep the requested top_k.
// A wider pool than top_k lets the reranker rescue a right answer the fused score buried.
const RERANK_POOL = Number(cfgRaw("RERANK_POOL") ?? 30);

// Rerank-based not-found gate (opt-in). The cross-encoder's absolute relevance logit is a
// stronger "does this memory actually answer the query" signal than bi-encoder cosine, so a
// low top logit means the query is unanswerable → return []. Unset = disabled. A definite
// key anchor (literal name/proper-noun match) bypasses it. NOTE: reliable for SAME-LANGUAGE
// queries only — cross-lingual relevance logits run low even when relevant, so cross-lingual
// not-found must lean on key anchors, not this floor.
const RERANK_MIN_SCORE =
  cfgRaw("RERANK_MIN_SCORE") !== undefined ? Number(cfgRaw("RERANK_MIN_SCORE")) : null;

// KR↔Latin script check. The rerank not-found gate only trusts its logit when the query and
// the top candidate share script — cross-lingual (e.g. Korean query ↔ English memory) logits
// run low even when relevant, so a script mismatch means "don't trust the low logit, keep it".
const hasHangul = (s: string): boolean => /[㄰-㆏가-힣]/.test(s);

// Literal key match must land on a word boundary (unicode-aware) so a short common-noun key
// like "name" does not spuriously match inside a longer word ("namespace") and spike to the
// top. Terms shorter than 2 chars never match literally.
function literalKeyMatch(queryLower: string, term: string): boolean {
  if (!term || term.length < 2) return false;
  const esc = term.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^\\p{L}\\p{N}])${esc}($|[^\\p{L}\\p{N}])`, "u").test(queryLower);
}

const LINK_WEIGHT_DEFAULT = 1.0;
const LINK_WEIGHT_MIN = 0.1;
const LINK_WEIGHT_MAX = 3.0;
const LINK_REINFORCE_AMOUNT = 0.1;
const LINK_DECAY_RATE = 0.005;

// ── Vector math ──

function cosineSim(a: number[], b: number[]): number {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const norm = Math.sqrt(normA) * Math.sqrt(normB);
  return norm === 0 ? 0 : dot / norm;
}

function batchCosineSim(query: number[], matrix: number[][]): number[] {
  if (matrix.length === 0) return [];
  return matrix.map((row) => cosineSim(query, row));
}

// A recalled memory must have raw similarity (best of content-sim and matched
// key-sim; exact name/proper-noun matches count as 1.0) at least minScore. This
// is computed on raw cosine BEFORE RRF fusion, so it is comparable across queries
// — unlike fused scores. minScore = 0 disables the gate.
export function passesAbsoluteGate(rawSim: number, minScore: number): boolean {
  return minScore <= 0 || rawSim >= minScore;
}

// Median of a numeric array (sorted copy; average of middle two for even length).
function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// Robust z-score of `top` against the distribution of `values`, using median and
// MAD (median absolute deviation) scaled by 1.4826 so the result reads in
// sigma-like units. Robust to the skewed, packed cosine bands that e5 produces,
// where a true match is a right-tail outlier even though every value is "high".
// Returns Infinity when the distribution is degenerate (empty, or MAD == 0) —
// no meaningful z can be computed, so callers treat it as "do not block".
export function robustZScore(top: number, values: number[]): number {
  if (values.length === 0) return Infinity;
  const med = median(values);
  const mad = median(values.map((v) => Math.abs(v - med)));
  if (mad === 0) return Infinity;
  return (top - med) / (1.4826 * mad);
}

// Distribution "not-found" gate. Passes (true) when disabled (gateZ <= 0), when
// the population is too small to be reliable (< minCount), or when the z is
// non-finite (degenerate distribution). Otherwise the top hit must be at least a
// gateZ-sigma outlier of the similarity distribution to count as "found".
export function passesDistributionGate(
  top: number,
  values: number[],
  gateZ: number,
  minCount: number
): boolean {
  if (gateZ <= 0) return true;
  if (values.length < minCount) return true;
  const z = robustZScore(top, values);
  return Number.isFinite(z) ? z >= gateZ : true;
}

// ── Utils ──

function uid(): string {
  return randomBytes(6).toString("hex");
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function isNodeError(err: unknown): err is NodeJS.ErrnoException {
  return err instanceof Error && "code" in err;
}

function conversationPath(sessionId: string): string {
  if (!SESSION_ID_PATTERN.test(sessionId)) {
    throw new Error(
      "Invalid session_id. Use 1-128 characters: letters, numbers, dot, underscore, or hyphen."
    );
  }
  return join(CONVERSATIONS_DIR, `${sessionId}.jsonl`);
}

export function sanitizeKeys(keys: unknown): string[] {
  let arr: unknown[];
  if (typeof keys === "string") {
    try {
      arr = JSON.parse(keys);
    } catch {
      arr = [keys];
    }
  } else if (Array.isArray(keys)) {
    arr = keys;
  } else {
    return [];
  }
  return arr
    .filter((k): k is string => typeof k === "string" && k.trim().length >= 2)
    .map((k) => k.trim());
}

// ── MemoryGraph ──

export class MemoryGraph {
  keys: Record<string, Key> = {};
  memories: Record<string, Memory> = {};

  private _keyToMems: Record<string, Map<string, number>> = {};
  private _memToKeys: Record<string, Map<string, number>> = {};
  private _supersededBy: Record<string, string> = {};
  private _storedDim: number | null = null;
  // Embedding-space fingerprint read from graph.json (null = legacy graph with no
  // provenance). Drives re-embed on a same-dimension model swap; see embeddingFingerprint.
  private _storedFingerprint: string | null = null;
  private _lock = new Mutex();
  // Serializes disk writes independently of _lock so a flush() done OUTSIDE _lock
  // (recall's tail) can never race another save on the temp file or interleave
  // renames. Lock order is always _lock → _saveLock (writes) or _saveLock alone
  // (recall flush); nothing acquires _saveLock then _lock, so no deadlock.
  private _saveLock = new Mutex();
  private _saveSeq = 0;
  private _dirty = false;
  private _bm25: MiniSearch;
  private _recallBuffer = new RecallBuffer({
    capacity: AUTOKEY_BUFFER_CAPACITY,
    ttlSeconds: AUTOKEY_BUFFER_TTL_SECONDS,
  });

  constructor() {
    this._bm25 = new MiniSearch({
      fields: ["content"],
      storeFields: [],
      idField: "id",
      tokenize: (text: string) =>
        text
          .toLowerCase()
          .split(/[\s\p{P}]+/u)
          .filter((t) => t.length >= 1),
      processTerm: (term: string) => (term.length < 1 ? false : term.toLowerCase()),
    });
  }

  static readonly HOP_DECAY = 0.3;
  static readonly TIME_HALF_LIFE = 30 * 24 * 3600;

  get linkCount(): number {
    return Object.values(this._keyToMems).reduce(
      (sum, mids) => sum + mids.size,
      0
    );
  }

  private _link(keyId: string, memId: string, weight = LINK_WEIGHT_DEFAULT): void {
    if (!this._keyToMems[keyId]) this._keyToMems[keyId] = new Map();
    if (!this._keyToMems[keyId].has(memId)) {
      this._keyToMems[keyId].set(memId, weight);
    }
    if (!this._memToKeys[memId]) this._memToKeys[memId] = new Map();
    if (!this._memToKeys[memId].has(keyId)) {
      this._memToKeys[memId].set(keyId, weight);
    }
  }

  private _hasLink(keyId: string, memId: string): boolean {
    return this._keyToMems[keyId]?.has(memId) ?? false;
  }

  private _getLinkWeight(keyId: string, memId: string): number {
    return this._keyToMems[keyId]?.get(memId) ?? LINK_WEIGHT_DEFAULT;
  }

  private _setLinkWeight(keyId: string, memId: string, weight: number): void {
    const clamped = Math.max(LINK_WEIGHT_MIN, Math.min(LINK_WEIGHT_MAX, weight));
    this._keyToMems[keyId]?.set(memId, clamped);
    this._memToKeys[memId]?.set(keyId, clamped);
  }

  private _unlinkMemory(memId: string): void {
    const kids = this._memToKeys[memId];
    if (kids) {
      for (const kid of kids.keys()) {
        const mems = this._keyToMems[kid];
        if (mems) {
          mems.delete(memId);
          if (mems.size === 0) delete this._keyToMems[kid];
        }
      }
      delete this._memToKeys[memId];
    }
  }

  private _validMemoryLinks(links: unknown, selfId: string): string[] {
    if (!Array.isArray(links)) return [];
    const seen = new Set<string>();
    const valid: string[] = [];
    for (const linkedId of links) {
      if (typeof linkedId !== "string") continue;
      if (linkedId === selfId || seen.has(linkedId)) continue;
      if (!(linkedId in this.memories)) continue;
      seen.add(linkedId);
      valid.push(linkedId);
    }
    return valid;
  }

  private _removeMemoryReferences(memoryIds: Iterable<string>): void {
    const deleted = new Set(memoryIds);
    for (const [mid, mem] of Object.entries(this.memories)) {
      mem.links = this._validMemoryLinks(mem.links, mid).filter(
        (linkedId) => !deleted.has(linkedId)
      );
      if (Array.isArray(mem.contradicts)) {
        mem.contradicts = mem.contradicts.filter((id) => id in this.memories && !deleted.has(id));
      }
    }
  }

  private _pruneDanglingExplicitLinks(): void {
    for (const [mid, mem] of Object.entries(this.memories)) {
      mem.links = this._validMemoryLinks(mem.links, mid);
    }
  }

  private _pruneOrphanKeys(): void {
    for (const kid of Object.keys(this.keys)) {
      const mems = this._keyToMems[kid];
      if (!mems || mems.size === 0) delete this.keys[kid];
    }
  }

  private _checkDim(embedding: number[]): void {
    const dim = embedding.length;
    if (this._storedDim === null) {
      this._storedDim = dim;
      return;
    }
    if (dim !== this._storedDim) {
      throw new Error(
        `Embedding dimension mismatch: existing data uses ${this._storedDim}-dim, ` +
          `current backend (${EMBEDDING_BACKEND}) produces ${dim}-dim.\n` +
          `Restart the server to auto-migrate (re-embeds all data with the current ` +
          `backend, preserving content), or set KEYMEM_AUTO_MIGRATE=false to opt out.`
      );
    }
  }

  // Recover from an embedding-backend/dimension change instead of bricking.
  // Switching backends (e.g. OpenAI 1536-dim → local 768/1024-dim) used to make
  // every recall/remember throw forever. Here we detect the mismatch on load and
  // re-embed all keys and memories with the current backend — content, links,
  // depth, and access history are preserved. Disable with KEYMEM_AUTO_MIGRATE=false.
  private async _ensureEmbeddingDim(): Promise<void> {
    if (this._storedDim === null) return;
    let probeDim: number;
    try {
      probeDim = (await embedTextAsync("dimension probe")).length;
    } catch (err) {
      console.error(`[graph] could not probe embedding dimension: ${errorMessage(err)}`);
      return;
    }
    const currentFp = embeddingFingerprint();
    const dimChanged = probeDim !== this._storedDim;
    // A same-dimension model swap (e5-large ↔ bge-m3, both 1024-d) leaves the
    // dimension unchanged but moves the data into an incompatible vector space.
    // Only a known stored fingerprint that differs can prove this — a legacy graph
    // (null fingerprint) is left alone to avoid a spurious re-embed and just gets
    // stamped on the next save.
    const modelChanged = this._storedFingerprint !== null && this._storedFingerprint !== currentFp;
    // Legacy graphs (no fingerprint) can't be diffed automatically, so a
    // same-dim swap off them is undetectable. FORCE_REEMBED is the explicit
    // one-shot for that switch: re-embed everything with the current backend.
    const forced = cfgRaw("FORCE_REEMBED") === "true";
    if (!dimChanged && !modelChanged && !forced) return;

    if (!forced && cfgRaw("AUTO_MIGRATE") === "false") {
      const reason = dimChanged
        ? `stored embeddings are ${this._storedDim}-dim but the current backend produces ${probeDim}-dim`
        : `stored embeddings were built by "${this._storedFingerprint}" but the current backend is "${currentFp}"`;
      console.error(
        `[graph] WARNING: ${reason} (backend ${EMBEDDING_BACKEND}). Auto-migration is ` +
          `disabled — recall/remember will be unreliable until the original backend is restored.`
      );
      return;
    }
    await this._migrateEmbeddings(probeDim, currentFp);
  }

  private async _migrateEmbeddings(newDim: number, newFingerprint: string): Promise<void> {
    const nKeys = Object.keys(this.keys).length;
    const nMems = Object.keys(this.memories).length;
    const change =
      newDim !== this._storedDim
        ? `dimension changed ${this._storedDim} -> ${newDim}`
        : `model changed "${this._storedFingerprint}" -> "${newFingerprint}" (same ${newDim}-dim)`;
    console.error(
      `[graph] embedding ${change}. Re-embedding ${nKeys} keys + ${nMems} memories with the ` +
        `current backend (${EMBEDDING_BACKEND}); content and links are preserved. One-time migration.`
    );
    const tag = newDim !== this._storedDim ? `${this._storedDim}d` : (this._storedFingerprint ?? `${this._storedDim}d-legacy`).replace(/[^A-Za-z0-9._-]/g, "_");
    try {
      await copyFile(GRAPH_FILE, `${GRAPH_FILE}.bak.${tag}`);
    } catch (err) {
      console.error(`[graph] pre-migration backup failed (continuing): ${errorMessage(err)}`);
    }
    for (const mem of Object.values(this.memories)) {
      mem.embedding = await embedTextAsync(mem.content, "passage");
    }
    for (const key of Object.values(this.keys)) {
      key.embedding = await embedTextAsync(key.concept, "passage");
    }
    this._storedDim = newDim;
    this._storedFingerprint = newFingerprint;
    await this.save();
    console.error(`[graph] migration complete: now ${newDim}-dim / "${newFingerprint}" (backup saved).`);
  }

  private _isExpired(mem: Memory): boolean {
    return mem.ttl != null && Date.now() / 1000 > mem.ttl;
  }

  private _timeFactor(mem: Memory): number {
    const age = Date.now() / 1000 - mem.created_at;
    const decayRate = 1.0 - mem.depth * 0.7;
    const decay = Math.exp((-age * decayRate) / MemoryGraph.TIME_HALF_LIFE);
    return 0.5 + 0.5 * decay;
  }

  private _keyIdf(keyId: string): number {
    const freq = this._keyToMems[keyId]?.size ?? 0;
    if (freq <= 1) return 1.0;
    let idf = 1.0 / freq;
    const kt = this.keys[keyId]?.key_type;
    if (kt === "name" || kt === "proper_noun") idf *= 0.5;
    return idf;
  }

  private _recordKeyAlias(keyId: string, alias: string): void {
    const key = this.keys[keyId];
    if (!key) return;
    const clean = alias.trim();
    if (clean.length < 2 || key.concept.toLowerCase() === clean.toLowerCase()) return;
    key.aliases ??= [];
    if (!key.aliases.some((existing) => existing.toLowerCase() === clean.toLowerCase())) {
      key.aliases.push(clean);
    }
  }

  // Move every memory link from `fromId` onto `intoId` (keeping the stronger weight on conflict),
  // fold the merged key's concept + aliases into `intoId` as aliases, then delete `fromId`. The
  // relink primitive behind fragmentation healing.
  private _mergeKeyInto(fromId: string, intoId: string): void {
    if (fromId === intoId || !this.keys[fromId] || !this.keys[intoId]) return;
    for (const [mid, w] of this._keyToMems[fromId] ?? new Map<string, number>()) {
      if (this._hasLink(intoId, mid)) {
        this._setLinkWeight(intoId, mid, Math.max(this._getLinkWeight(intoId, mid), w));
      } else {
        this._link(intoId, mid, w);
      }
      this._memToKeys[mid]?.delete(fromId);
    }
    this._recordKeyAlias(intoId, this.keys[fromId].concept);
    for (const a of this.keys[fromId].aliases ?? []) this._recordKeyAlias(intoId, a);
    delete this._keyToMems[fromId];
    delete this.keys[fromId];
  }

  // One-time heal: collapse keys that share a surface string but were split across key_types (a name
  // hub + a concept twin from before cross-type reconciliation existed) into a single cluster.
  // Canonical = the name/proper_noun key (literal-match); ties broken by most-linked. Idempotent —
  // once healed and saved, later loads find no duplicates. Returns the number of keys merged away.
  private _healFragmentedKeys(): number {
    const byString = new Map<string, string[]>();
    for (const [kid, key] of Object.entries(this.keys)) {
      const norm = key.concept.toLowerCase();
      (byString.get(norm) ?? byString.set(norm, []).get(norm)!).push(kid);
    }
    const rank = (kid: string): number => {
      const t = this.keys[kid].key_type;
      const strong = t === "name" || t === "proper_noun" ? 1 : 0;
      return strong * 1e7 + (this._keyToMems[kid]?.size ?? 0);
    };
    let merged = 0;
    for (const kids of byString.values()) {
      if (kids.length < 2) continue;
      const canonical = kids.reduce((a, b) => (rank(b) > rank(a) ? b : a));
      for (const kid of kids) {
        if (kid === canonical) continue;
        this._mergeKeyInto(kid, canonical);
        merged++;
      }
    }
    if (merged > 0) this.markDirty();
    return merged;
  }

  private _activeMemoryIdsForKey(keyId: string, namespace?: string | null): string[] {
    const active: string[] = [];
    for (const mid of this._keyToMems[keyId]?.keys() ?? []) {
      const mem = this.memories[mid];
      if (!mem || this._isExpired(mem) || mid in this._supersededBy) continue;
      if (namespace && mem.namespace !== namespace) continue;
      active.push(mid);
    }
    return active;
  }

  private _keyView(keyId: string, namespace?: string | null): object {
    const key = this.keys[keyId];
    const memoryCount = this._activeMemoryIdsForKey(keyId, namespace).length;
    return {
      key_id: keyId,
      concept: key.concept,
      aliases: key.aliases ?? [],
      learned_aliases: (key.learnedAliases ?? []).map((l) => l.alias),
      key_type: key.key_type,
      memory_count: memoryCount,
      is_hub: memoryCount >= KEY_HUB_MIN_LINKS,
      specificity: memoryCount > 0 ? Math.round((1 / memoryCount) * 1000) / 1000 : 0,
    };
  }

  private _findDuplicate(embedding: number[]): string | null {
    const activeMems = Object.entries(this.memories).filter(
      ([mid]) => !(mid in this._supersededBy)
    );
    if (activeMems.length === 0) return null;
    const matrix = activeMems.map(([, mem]) => mem.embedding);
    const sims = batchCosineSim(embedding, matrix);
    let bestIdx = 0,
      bestSim = -Infinity;
    for (let i = 0; i < sims.length; i++) {
      if (sims[i] > bestSim) {
        bestSim = sims[i];
        bestIdx = i;
      }
    }
    return bestSim >= MEMORY_DEDUP_THRESHOLD ? activeMems[bestIdx][0] : null;
  }

  // Find an existing active memory that CONTRADICTS the new one: best similarity
  // sits in the contradiction band [CONTRADICTION_THRESHOLD, MEMORY_DEDUP_THRESHOLD)
  // AND the two share at least one key (same subject). Heuristic — surfaces a
  // signal, does not block or supersede. Returns the conflicting memory id or null.
  private _findContradiction(embedding: number[], keyIds: Iterable<string>): string | null {
    const newKeys = new Set(keyIds);
    if (newKeys.size === 0) return null;
    let bestId: string | null = null;
    let bestSim = -Infinity;
    for (const [mid, mem] of Object.entries(this.memories)) {
      if (mid in this._supersededBy) continue;
      const sim = cosineSim(embedding, mem.embedding);
      if (!inContradictionBand(sim, CONTRADICTION_THRESHOLD, MEMORY_DEDUP_THRESHOLD)) continue;
      const shares = [...(this._memToKeys[mid]?.keys() ?? [])].some((kid) => newKeys.has(kid));
      if (shares && sim > bestSim) {
        bestSim = sim;
        bestId = mid;
      }
    }
    return bestId;
  }

  private _autoLinkKeys(memId: string, embedding: number[]): void {
    const keyIds = Object.keys(this.keys);
    if (keyIds.length === 0) return;
    const matrix = keyIds.map((kid) => this.keys[kid].embedding);
    const sims = batchCosineSim(embedding, matrix);
    for (let i = 0; i < keyIds.length; i++) {
      if (sims[i] >= KEY_AUTO_LINK_THRESHOLD && !this._hasLink(keyIds[i], memId)) {
        this._link(keyIds[i], memId, sims[i]);
      }
    }
  }

  private _rebuildBm25Index(): void {
    this._bm25.removeAll();
    const docs = Object.entries(this.memories)
      .filter(([mid]) => !(mid in this._supersededBy))
      .filter(([, mem]) => !this._isExpired(mem))
      .map(([mid, mem]) => ({ id: mid, content: mem.content }));
    if (docs.length > 0) this._bm25.addAll(docs);
  }

  getKeysForMemory(memId: string): string[] {
    const kids = this._memToKeys[memId];
    if (!kids) return [];
    return [...kids.keys()]
      .filter((kid) => kid in this.keys)
      .map((kid) => this.keys[kid].concept);
  }

  // Like getKeysForMemory but carries each key's id alongside its concept. Used by inject so a
  // consumer can jump straight to read_key(key_id) — no concept→id resolution round trip.
  getKeyRefsForMemory(memId: string): Array<{ concept: string; key_id: string }> {
    const kids = this._memToKeys[memId];
    if (!kids) return [];
    return [...kids.keys()]
      .filter((kid) => kid in this.keys)
      .map((kid) => ({ concept: this.keys[kid].concept, key_id: kid }));
  }

  // ── I/O ──

  async load(): Promise<void> {
    let raw: GraphData;
    try {
      const text = await readFile(GRAPH_FILE, "utf-8");
      raw = JSON.parse(text) as GraphData;
    } catch (err) {
      if (isNodeError(err) && err.code === "ENOENT") return;
      throw new Error(`Failed to load memory graph at ${GRAPH_FILE}: ${errorMessage(err)}`);
    }

    this._storedFingerprint = raw.meta?.embeddingFingerprint ?? null;

    for (const [kid, k] of Object.entries(raw.keys ?? {})) {
      const seen = new Set<string>();
      const aliases = (Array.isArray(k.aliases) ? k.aliases : []).filter((alias) => {
        if (typeof alias !== "string" || alias.trim().length < 2) return false;
        const normalized = alias.trim().toLowerCase();
        if (normalized === k.concept.toLowerCase() || seen.has(normalized)) return false;
        seen.add(normalized);
        return true;
      });
      const aliasCandidates =
        k.aliasCandidates && typeof k.aliasCandidates === "object" && !Array.isArray(k.aliasCandidates)
          ? Object.fromEntries(
              Object.entries(k.aliasCandidates).filter(
                (entry): entry is [string, { count: number; lastSeen: number; queryText: string }] => {
                  const v = entry[1];
                  return (
                    !!v &&
                    typeof v === "object" &&
                    typeof (v as { count?: unknown }).count === "number" &&
                    typeof (v as { lastSeen?: unknown }).lastSeen === "number" &&
                    typeof (v as { queryText?: unknown }).queryText === "string"
                  );
                }
              )
            )
          : undefined;
      const learnedAliases = Array.isArray(k.learnedAliases)
        ? k.learnedAliases.filter(
            (l): l is { alias: string; addedAt: number; hits: number } =>
              !!l && typeof l.alias === "string" && typeof l.addedAt === "number" && typeof l.hits === "number"
          )
        : undefined;
      this.keys[kid] = { ...k, aliases, aliasCandidates, learnedAliases };
    }

    for (const [mid, m] of Object.entries(raw.memories ?? {})) {
      const defaults = {
        depth: 0.0,
        access_count: 0,
        last_accessed: 0,
        namespace: "default",
        ttl: null,
        links: [] as string[],
        contradicts: [] as string[],
        source: null,
        supersedes: null,
      };
      const mem: Memory = { ...defaults, ...m };
      mem.links = Array.isArray(mem.links)
        ? mem.links.filter((linkedId): linkedId is string => typeof linkedId === "string")
        : [];
      mem.contradicts = Array.isArray(mem.contradicts)
        ? mem.contradicts.filter((id): id is string => typeof id === "string" && id in (raw.memories ?? {}))
        : [];
      if (!mem.embedding || mem.embedding.length === 0) {
        mem.embedding = await embedTextAsync(mem.content);
      }
      this.memories[mid] = mem;
    }

    if (Object.keys(this.memories).length > 0) {
      const firstMem = Object.values(this.memories)[0];
      this._storedDim = firstMem.embedding.length;
    }

    for (const lnk of raw.links ?? []) {
      if (lnk.key_id in this.keys && lnk.memory_id in this.memories) {
        this._link(lnk.key_id, lnk.memory_id, lnk.weight ?? LINK_WEIGHT_DEFAULT);
      }
    }

    this._pruneDanglingExplicitLinks();

    // Heal keys fragmented across key_types in stores written before cross-type reconciliation.
    const healed = this._healFragmentedKeys();
    if (healed > 0) console.error(`[graph] healed ${healed} fragmented key(s)`);

    for (const [mid, mem] of Object.entries(this.memories)) {
      if (mem.supersedes) {
        this._supersededBy[mem.supersedes] = mid;
      }
    }

    await this._ensureEmbeddingDim();

    this._rebuildBm25Index();

    console.error(
      `[graph] loaded ${Object.keys(this.keys).length} keys, ` +
        `${Object.keys(this.memories).length} memories, ${this.linkCount} links`
    );
  }

  async save(): Promise<void> {
    const links: Array<{ key_id: string; memory_id: string; weight: number }> = [];
    for (const [kid, mids] of Object.entries(this._keyToMems)) {
      for (const [mid, weight] of mids) {
        links.push({ key_id: kid, memory_id: mid, weight });
      }
    }
    // Stamp the embedding-space fingerprint so a later same-dimension backend swap
    // is detected on load. Keep _storedFingerprint in sync for in-process reloads.
    const fingerprint = embeddingFingerprint();
    this._storedFingerprint = fingerprint;
    const data: GraphData = {
      keys: this.keys,
      memories: this.memories,
      links,
      meta: { embeddingFingerprint: fingerprint },
    };
    // Snapshot is built synchronously above (callers mutate under _lock without
    // awaiting mid-mutation, so this read is consistent). Serialize the actual I/O
    // so concurrent saves can't collide: a per-write unique temp name + single-flight
    // _saveLock together guarantee one clean writeFile→rename at a time.
    const json = JSON.stringify(data, null, 2);
    await this._saveLock.runExclusive(async () => {
      await mkdir(DATA_DIR, { recursive: true });
      const tmp = `${GRAPH_FILE}.${process.pid}.${++this._saveSeq}.tmp`;
      await writeFile(tmp, json, "utf-8");
      await rename(tmp, GRAPH_FILE);
    });
    this._dirty = false;
  }

  markDirty(): void {
    this._dirty = true;
  }

  async flush(): Promise<void> {
    if (this._dirty) await this.save();
  }

  // ── Key management ──

  async findOrCreateKey(
    concept: string,
    keyType: "concept" | "name" | "proper_noun" = "concept"
  ): Promise<string> {
    const normalizedConcept = concept.toLowerCase();

    // Cross-type reconciliation FIRST: the same surface string must resolve to ONE key cluster
    // regardless of the requested type. Without this, a `name` hub and a `concept` key for the same
    // string ("동균") fragment the same entity — pivoting onto the 1-memory twin is a dead end.
    // name/proper_noun is canonical (it carries literal-match + entity precedence), so a concept
    // request joins an existing name key, and a name request PROMOTES an existing concept twin in
    // place: its id, memories and links are kept, it just gains literal-match semantics.
    for (const [kid, key] of Object.entries(this.keys)) {
      const terms = [key.concept, ...(key.aliases ?? [])];
      if (!terms.some((term) => term.toLowerCase() === normalizedConcept)) continue;
      if ((keyType === "name" || keyType === "proper_noun") && key.key_type === "concept") {
        key.key_type = keyType; // promote the twin in place
      }
      this._recordKeyAlias(kid, concept);
      return kid;
    }

    // No same-string key exists. name/proper_noun never semantic-merges (distinct entities can be
    // near-synonyms), so create one outright.
    if (keyType === "name" || keyType === "proper_noun") {
      const kid = uid();
      this.keys[kid] = {
        id: kid,
        concept,
        aliases: [],
        embedding: await embedTextAsync(concept),
        key_type: keyType,
      };
      return kid;
    }

    // Short concept keys merge only on exact (case-insensitive) string match, so
    // near-identical-but-distinct short keys ("Agent A" vs "Agent B") stay separate.
    if (isShortConcept(concept)) {
      const emb = await embedTextAsync(concept);
      // Conservative semantic merge: fold an incoming short key into an existing concept
      // key only at high cosine (clear synonym). Reconciles state-blind LLM key choices
      // without conflating distinct concepts. Disabled by default (threshold 0).
      if (SHORT_KEY_MERGE_THRESHOLD > 0) {
        const conceptKeys = Object.entries(this.keys).filter(([, k]) => k.key_type === "concept");
        if (conceptKeys.length > 0) {
          const sims = batchCosineSim(emb, conceptKeys.map(([, k]) => k.embedding));
          let bestIdx = 0, bestSim = -Infinity;
          for (let i = 0; i < sims.length; i++) if (sims[i] > bestSim) { bestSim = sims[i]; bestIdx = i; }
          if (bestSim >= SHORT_KEY_MERGE_THRESHOLD) {
            const existingId = conceptKeys[bestIdx][0];
            this._recordKeyAlias(existingId, concept);
            return existingId;
          }
        }
      }
      const kid = uid();
      this.keys[kid] = { id: kid, concept, aliases: [], embedding: emb, key_type: "concept" };
      return kid;
    }

    const emb = await embedTextAsync(concept);
    const conceptKeys = Object.entries(this.keys).filter(
      ([, k]) => k.key_type === "concept"
    );
    if (conceptKeys.length > 0) {
      const matrix = conceptKeys.map(([, k]) => k.embedding);
      const sims = batchCosineSim(emb, matrix);
      let bestIdx = 0,
        bestSim = -Infinity;
      for (let i = 0; i < sims.length; i++) {
        if (sims[i] > bestSim) {
          bestSim = sims[i];
          bestIdx = i;
        }
      }
      if (bestSim >= KEY_MERGE_THRESHOLD) {
        const existingId = conceptKeys[bestIdx][0];
        this._recordKeyAlias(existingId, concept);
        return existingId;
      }
    }

    const kid = uid();
    this.keys[kid] = { id: kid, concept, aliases: [], embedding: emb, key_type: "concept" };
    return kid;
  }

  // ── Add ──

  async add(
    content: string,
    keyConcepts: string[],
    options: {
      keyTypes?: Record<string, string> | null;
      source?: Record<string, unknown> | null;
      namespace?: string;
      ttlSeconds?: number | null;
      relatedTo?: string[] | null;
    } = {}
  ): Promise<[string, boolean, string | null, boolean]> {
    const embedding = await embedTextAsync(content); // outside lock

    // Duplicate detection and insertion run under a SINGLE lock acquisition so they are
    // atomic: two concurrent identical adds serialize, and the second observes the first's
    // memory as a duplicate instead of both clearing the check and inserting twice. The dup
    // path defers to supersede() only AFTER releasing the lock (the mutex is non-reentrant).
    let dupId: string | null = null;
    let resultMid = "";
    await this._lock.runExclusive(async () => {
      this._checkDim(embedding);
      dupId = this._findDuplicate(embedding);
      if (dupId !== null) return; // defer to supersede() once the lock is released

      const mid = uid();
      resultMid = mid;
      const now = Date.now() / 1000;
      const expiresAt =
        options.ttlSeconds != null ? now + options.ttlSeconds : null;
      const validLinks = this._validMemoryLinks(options.relatedTo ?? [], mid);

      this.memories[mid] = {
        id: mid,
        content,
        embedding,
        created_at: now,
        source: options.source ?? null,
        supersedes: null,
        depth: 0.0,
        access_count: 0,
        last_accessed: now,
        namespace: options.namespace ?? "default",
        ttl: expiresAt,
        links: validLinks,
        contradicts: [],
      };

      const sanitized = sanitizeKeys(keyConcepts);
      const keyTypes = options.keyTypes ?? {};
      for (const concept of sanitized) {
        const kt = ((keyTypes[concept] ?? "concept") as
          | "concept"
          | "name"
          | "proper_noun");
        const kid = await this.findOrCreateKey(concept, kt);
        if (!this._hasLink(kid, mid)) this._link(kid, mid);
      }

      const linkedKeyIds = [...(this._memToKeys[mid]?.keys() ?? [])];
      this._autoLinkKeys(mid, embedding);
      const conflictId = this._findContradiction(embedding, linkedKeyIds);
      if (conflictId && conflictId !== mid) {
        if (!this.memories[mid].contradicts.includes(conflictId)) {
          this.memories[mid].contradicts.push(conflictId);
        }
        if (!this.memories[conflictId].contradicts.includes(mid)) {
          this.memories[conflictId].contradicts.push(mid);
        }
      }
      this._bm25.add({ id: mid, content });
      await this.save();
    });

    if (dupId !== null) {
      // Threshold-free conflict signal: does the soon-to-be-superseded memory share a
      // key with the incoming one? A shared key + high similarity is the same shape the
      // contradiction detector keys off, so surface it rather than silently overwriting.
      // The supersede DECISION is unchanged — telling a paraphrase from a high-similarity
      // conflict needs the calibration corpus; this only makes the overwrite observable.
      const sanitized = sanitizeKeys(keyConcepts);
      const dupConcepts = new Set<string>();
      for (const kid of this._memToKeys[dupId]?.keys() ?? []) {
        const k = this.keys[kid];
        if (!k) continue;
        dupConcepts.add(k.concept.toLowerCase());
        for (const alias of k.aliases ?? []) dupConcepts.add(alias.toLowerCase());
      }
      const conflict = sanitized.some((c) => dupConcepts.has(c.toLowerCase()));

      const newId = await this.supersede(dupId, content, {
        keyConcepts,
        keyTypes: options.keyTypes ?? undefined,
        source: options.source,
        namespace: options.namespace,
        relatedTo: options.relatedTo,
      });
      return [newId, true, dupId, conflict];
    }

    return [resultMid, false, null, false];
  }

  // ── Supersede ──

  async supersede(
    oldId: string,
    newContent: string,
    options: {
      keyConcepts?: string[] | null;
      keyTypes?: Record<string, string> | null;
      source?: Record<string, unknown> | null;
      namespace?: string | null;
      relatedTo?: string[] | null;
    } = {}
  ): Promise<string> {
    const newEmbedding = await embedTextAsync(newContent); // outside lock

    let resultMid = "";
    await this._lock.runExclusive(async () => {
      // Follow the supersession chain to the current live head. Normally oldId is already
      // live (callers pass an id from _findDuplicate, which skips superseded memories) so
      // this is a no-op. Under concurrency it serializes multiple supersedes of the same
      // target into one linear chain instead of forking parallel successors.
      while (oldId in this._supersededBy) oldId = this._supersededBy[oldId];
      if (!(oldId in this.memories)) {
        // The head was superseded and pruned by a concurrent supersede (grandparent cleanup
        // deletes it). Re-resolve against the current live state so concurrent supersedes of
        // the same content collapse into one chain instead of erroring or forking successors.
        const reResolved = this._findDuplicate(newEmbedding);
        if (reResolved === null) {
          throw new Error(`Memory ${oldId} not found`);
        }
        oldId = reResolved;
      }

      const old = this.memories[oldId];

      // Chain cleanup: keep depth max 1 (new -> old; grandparent deleted)
      const grandparentId = old.supersedes;
      if (grandparentId && grandparentId in this.memories) {
        delete this.memories[grandparentId];
        this._unlinkMemory(grandparentId);
        this._removeMemoryReferences([grandparentId]);
        delete this._supersededBy[grandparentId];
        this._pruneOrphanKeys();
        try { this._bm25.discard(grandparentId); } catch { /* already removed */ }
      }

      const mid = uid();
      resultMid = mid;
      const now = Date.now() / 1000;
      const ns = options.namespace ?? old.namespace;
      const nextLinks = options.relatedTo === undefined ? old.links : options.relatedTo;
      const validLinks = this._validMemoryLinks(nextLinks ?? [], mid);

      this.memories[mid] = {
        id: mid,
        content: newContent,
        embedding: newEmbedding,
        created_at: now,
        source: options.source ?? null,
        supersedes: oldId,
        depth: 0.0,
        access_count: 0,
        last_accessed: now,
        namespace: ns,
        ttl: old.ttl,
        links: validLinks,
        contradicts: [],
      };

      this._bm25.add({ id: mid, content: newContent });

      // Weaken old memory depth
      old.depth =
        old.depth >= DEPTH_DEEP_THRESHOLD
          ? old.depth * 0.8
          : old.depth * 0.3;
      this._supersededBy[oldId] = mid;
      // Remove stale contradiction back-references to the now-superseded oldId.
      // read-at-time already skips superseded memories, so this is cleanup only.
      for (const mem of Object.values(this.memories)) {
        if (Array.isArray(mem.contradicts)) {
          mem.contradicts = mem.contradicts.filter((id) => id !== oldId);
        }
      }
      try { this._bm25.discard(oldId); } catch { /* already removed */ }

      const keyConcepts = options.keyConcepts;
      if (keyConcepts && keyConcepts.length > 0) {
        const sanitized = sanitizeKeys(keyConcepts);
        const keyTypes = options.keyTypes ?? {};
        for (const concept of sanitized) {
          const kt = ((keyTypes[concept] ?? "concept") as
            | "concept"
            | "name"
            | "proper_noun");
          const kid = await this.findOrCreateKey(concept, kt);
          this._link(kid, mid);
        }
      } else {
        // Inherit the old keys, but drop CONCEPT keys the corrected content has drifted
        // away from. A same-topic update (Seoul -> Busan) stays close to "residence" and
        // keeps it; an off-topic correction (strawberries -> peanut allergy) must not
        // remain tagged "strawberry" and pollute that key's recall. Exact-match anchors
        // (name/proper_noun) are always kept — they only match literal queries, so they
        // cannot pollute semantic recall. (snapshot to avoid mutation during iteration)
        for (const [kid, w] of [...(this._memToKeys[oldId] ?? new Map())]) {
          const key = this.keys[kid];
          if (
            key &&
            key.key_type === "concept" &&
            cosineSim(newEmbedding, key.embedding) < KEY_RECALL_THRESHOLD
          ) {
            continue; // stale topic key — drop it so the off-topic memory is not mis-tagged
          }
          this._link(kid, mid, w);
        }
      }

      this._autoLinkKeys(mid, newEmbedding);
      // Deliberately captured AFTER _autoLinkKeys (unlike add(), which captures
      // explicit keys BEFORE auto-linking). Superseded content inherits all keys
      // including auto-linked ones; the band + shared-key requirement still gates
      // false positives. Do NOT reorder to match add().
      const supKeyIds = [...(this._memToKeys[mid]?.keys() ?? [])];
      const supConflict = this._findContradiction(newEmbedding, supKeyIds);
      if (supConflict && supConflict !== mid && supConflict !== oldId) {
        if (!this.memories[mid].contradicts.includes(supConflict)) {
          this.memories[mid].contradicts.push(supConflict);
        }
        if (!this.memories[supConflict].contradicts.includes(mid)) {
          this.memories[supConflict].contradicts.push(mid);
        }
      }
      await this.save();
    });

    return resultMid;
  }

  // ── Agent-driven key navigation ──

  async searchKeys(
    query: string,
    topK = 8,
    namespace?: string | null
  ): Promise<object[]> {
    const cleanQuery = query.trim();
    if (!cleanQuery || Object.keys(this.keys).length === 0) return [];

    const qEmb = await embedTextAsync(cleanQuery, "query");
    this._checkDim(qEmb);
    topK = Math.max(1, Math.min(20, Math.floor(topK)));

    // Content signal: max cosine of a key's member memories to the query. Lets a key whose
    // CONTENT matches surface even when its coined concept does not lexically/semantically hit
    // the query — the cure for key-coining dependence. Computed OUTSIDE the lock (read-only
    // cosine over a synchronous snapshot) to keep the lock hold short, matching the rerank/flush
    // off-lock design. A memory added after this snapshot simply scores 0 for this query.
    const memIds = Object.keys(this.memories);
    const memSimArr = batchCosineSim(qEmb, memIds.map((mid) => this.memories[mid].embedding));
    const memSim = new Map<string, number>();
    for (let j = 0; j < memIds.length; j++) memSim.set(memIds[j], memSimArr[j]);

    return this._lock.runExclusive(async () => {
      const queryLower = cleanQuery.toLowerCase();
      const isShortQuery = isShortConcept(cleanQuery);
      const nearMiss = new Map<string, number>(); // gate-dropped keys in the confirmation band
      const keyIds = Object.keys(this.keys);
      const sims = batchCosineSim(qEmb, keyIds.map((kid) => this.keys[kid].embedding));
      const candidates: Array<{
        key_id: string;
        concept: string;
        aliases: string[];
        key_type: Key["key_type"];
        score: number;
        match_type: "concept" | "alias" | "semantic" | "content";
        memory_count: number;
        is_hub: boolean;
        specificity: number;
        cluster_size: number;
        evidence: "index_only";
        suggested_tool: "read_key";
        _literal: boolean;
        _contentMid: string;
      }> = [];

      for (let i = 0; i < keyIds.length; i++) {
        const kid = keyIds[i];
        const key = this.keys[kid];
        const activeIds = this._activeMemoryIdsForKey(kid, namespace);
        if (activeIds.length === 0) continue;

        const aliases = key.aliases ?? [];
        const conceptLiteral = literalKeyMatch(queryLower, key.concept);
        const matchedAlias = aliases.find((alias) => literalKeyMatch(queryLower, alias));
        if (matchedAlias && key.learnedAliases) {
          const la = key.learnedAliases.find((l) => l.alias.toLowerCase() === matchedAlias.toLowerCase());
          if (la) la.hits += 1;
        }
        const literal = conceptLiteral || matchedAlias !== undefined;
        let contentSim = 0;
        let contentMid = "";
        for (const mid of activeIds) {
          const s = memSim.get(mid) ?? 0;
          if (s > contentSim) { contentSim = s; contentMid = mid; }
        }
        const keySim = sims[i];
        if (
          (key.key_type === "name" || key.key_type === "proper_noun")
            ? !literal
            : !literal && keySim < KEY_RECALL_THRESHOLD && contentSim < CONTENT_RECALL_THRESHOLD
        ) {
          // Gate-dropped, but a concept key whose embedding sits just below the recall gate
          // (in the confirmation band) is a learning signal: if the agent later confirms the
          // right memory via this key, autokey can fold the (short-concept) query in as an
          // alias. Record only for short-concept queries — long sentences never promote.
          if (
            AUTOKEY_ENABLED && isShortQuery && !literal &&
            key.key_type !== "name" && key.key_type !== "proper_noun" &&
            keySim >= AUTOKEY_CONFIRM_FLOOR && keySim < KEY_RECALL_THRESHOLD
          ) {
            nearMiss.set(kid, Math.round(keySim * 1000) / 1000);
          }
          continue;
        }
        // Entity keys (name/proper_noun) enter ONLY via literal match, and an exact entity
        // hit IS the answer — so they keep relevance 1. A concept key matched by a literal
        // *token* (e.g. query "메모리 도구" hitting a key named "메모리"), however, must compete
        // on actual semantic relevance, or a generic word match buries the real answer.
        const isEntityKey = key.key_type === "name" || key.key_type === "proper_noun";
        const relevance = literal && isEntityKey ? 1 : Math.max(keySim, contentSim);

        const memoryCount = activeIds.length;
        candidates.push({
          key_id: kid,
          concept: key.concept,
          aliases,
          key_type: key.key_type,
          score: Math.round(relevance * 1000) / 1000,
          match_type: matchedAlias ? "alias" : conceptLiteral ? "concept" : contentSim > keySim ? "content" : "semantic",
          memory_count: memoryCount,
          is_hub: memoryCount >= KEY_HUB_MIN_LINKS,
          specificity: Math.round((1 / memoryCount) * 1000) / 1000,
          cluster_size: 1 + aliases.length,
          evidence: "index_only",
          suggested_tool: "read_key",
          _literal: literal,
          _contentMid: contentMid,
        });
      }

      const claimedContentMids = new Set<string>();
      const result = candidates
        // Rank by relevance first; literal is only a tiebreak (entity literals already carry
        // relevance 1, so they still surface at the top — without burying a stronger semantic hit).
        .sort((a, b) => b.score - a.score || Number(b._literal) - Number(a._literal) || b.specificity - a.specificity)
        .filter((c) => {
          // Collapse synonym keys that surface only because they share the same content-matched
          // memory: keep the highest-ranked one so one memory's aliases can't flood the results.
          if (c.match_type !== "content") return true;
          if (claimedContentMids.has(c._contentMid)) return false;
          claimedContentMids.add(c._contentMid);
          return true;
        })
        .slice(0, topK)
        .map(({ _literal, _contentMid, ...candidate }) => candidate);

      if (AUTOKEY_ENABLED) {
        // Surfaced semantic matches (passed the gate) and gate-dropped near-misses (in the
        // confirmation band) are both learning signals: a later confirmed read via any of
        // these keys lets autokey fold the query in.
        const weakKeyScores = new Map<string, number>(nearMiss);
        for (const c of result) if (c.match_type === "semantic") weakKeyScores.set(c.key_id, c.score);
        if (weakKeyScores.size > 0) {
          this._recallBuffer.push({ queryText: cleanQuery, weakKeyScores });
        }
      }
      return result;
    });
  }

  async readKey(
    keyId: string,
    options: { namespace?: string | null; limit?: number; offset?: number; query?: string | null } = {}
  ): Promise<object> {
    if (!(keyId in this.keys)) throw new Error(`Key ${keyId} not found`);
    const namespace = options.namespace ?? null;
    const limit = Math.max(1, Math.min(50, Math.floor(options.limit ?? 10)));
    const offset = Math.max(0, Math.floor(options.offset ?? 0));
    // Query-aware ranking: when a query is supplied, order this key's memories by content
    // relevance to it (not only by link weight). This is what makes a generic hub key usable —
    // the target rises to the top instead of being buried among the hub's other members.
    // Omitted query reproduces the prior link-weight ordering exactly (rel = 1).
    const cleanQuery = options.query?.trim();
    const qEmb = cleanQuery ? await embedTextAsync(cleanQuery, "query") : null;
    if (qEmb) this._checkDim(qEmb);

    const ranked = this._activeMemoryIdsForKey(keyId, namespace)
      .map((mid) => {
        const mem = this.memories[mid];
        const linkWeight = this._getLinkWeight(keyId, mid);
        const rel = qEmb ? cosineSim(qEmb, mem.embedding) : 1;
        const score = rel * linkWeight * (0.9 + mem.depth * 0.1) * this._timeFactor(mem);
        return { mid, mem, linkWeight, score };
      })
      .sort((a, b) => b.score - a.score || b.mem.created_at - a.mem.created_at);

    const page = ranked.slice(offset, offset + limit).map(({ mid, mem, linkWeight, score }) => ({
      memory_id: mid,
      evidence: "unread" as const,
      suggested_tool: "read_memory" as const,
      depth: Math.round(mem.depth * 1000) / 1000,
      created_at: mem.created_at,
      namespace: mem.namespace,
      link_weight: Math.round(linkWeight * 1000) / 1000,
      score: Math.round(score * 1000) / 1000,
    }));

    return {
      key: this._keyView(keyId, namespace),
      memories: page,
      total: ranked.length,
      next_offset: offset + limit < ranked.length ? offset + limit : null,
    };
  }

  // Auto-key self-healing: a memory was just confirmed (read) via viaKeyId. If that key
  // was a recent WEAK (semantic) recall match, the originating query is candidate
  // vocabulary the key is missing. Accumulate heat; promote at threshold. Runs inside
  // readMemory's lock; readMemory's unconditional save() persists any mutation.
  private async _maybeLearnAlias(keyId: string, memoryId: string): Promise<void> {
    const entry = this._recallBuffer.consumeWeakMatch(keyId);
    if (!entry) return;
    const key = this.keys[keyId];
    if (!key) return;
    const q = entry.queryText.trim();
    if (q.length < 2) return;
    const norm = q.toLowerCase();
    if (key.concept.toLowerCase() === norm) return;
    if ((key.aliases ?? []).some((a) => a.toLowerCase() === norm)) return;

    key.aliasCandidates ??= {};
    const prev = key.aliasCandidates[norm];
    const candidate = { count: (prev?.count ?? 0) + 1, lastSeen: Date.now() / 1000, queryText: q };
    key.aliasCandidates[norm] = candidate;

    const decision = decidePromotion({
      count: candidate.count,
      query: q,
      cosine: entry.weakKeyScores.get(keyId) ?? 0,
      learnedAliasCount: key.learnedAliases?.length ?? 0,
      aliasThreshold: KEY_MERGE_THRESHOLD,
      newKeyThreshold: KEY_AUTO_LINK_THRESHOLD,
      promoteN: AUTOKEY_PROMOTE_N,
      maxAliases: AUTOKEY_MAX_ALIASES,
      confirmFloor: AUTOKEY_CONFIRM_FLOOR,
    });

    if (decision === "alias") {
      this._recordKeyAlias(keyId, q);
      key.learnedAliases ??= [];
      key.learnedAliases.push({ alias: q, addedAt: Date.now() / 1000, hits: 0 });
      delete key.aliasCandidates[norm];
    } else if (decision === "newKey") {
      const newKid = await this.findOrCreateKey(q, "concept");
      this._link(newKid, memoryId);
      delete key.aliasCandidates[norm];
    }
  }

  async readMemory(
    memoryId: string,
    viaKeyId?: string | null,
    namespace?: string | null
  ): Promise<object> {
    return this._lock.runExclusive(async () => {
      const mem = this.memories[memoryId];
      if (!mem || this._isExpired(mem)) throw new Error(`Memory ${memoryId} not found`);
      if (namespace && mem.namespace !== namespace) throw new Error(`Memory ${memoryId} not found`);
      if (memoryId in this._supersededBy) {
        throw new Error(`Memory ${memoryId} was superseded by ${this._supersededBy[memoryId]}`);
      }
      if (viaKeyId && !this._hasLink(viaKeyId, memoryId)) {
        throw new Error(`Key ${viaKeyId} is not linked to memory ${memoryId}`);
      }

      mem.depth = Math.min(mem.depth + DEPTH_INCREMENT, DEPTH_MAX);
      mem.access_count += 1;
      mem.last_accessed = Date.now() / 1000;
      if (viaKeyId) {
        this._setLinkWeight(
          viaKeyId,
          memoryId,
          this._getLinkWeight(viaKeyId, memoryId) + LINK_REINFORCE_AMOUNT
        );
      }

      if (AUTOKEY_ENABLED && viaKeyId) {
        await this._maybeLearnAlias(viaKeyId, memoryId);
      }

      const connectedKeys = [...(this._memToKeys[memoryId] ?? new Map())]
        .filter(([kid]) => kid in this.keys)
        .map(([kid, weight]) => ({
          ...this._keyView(kid, mem.namespace),
          link_weight: Math.round(weight * 1000) / 1000,
          traversed_from: kid === viaKeyId,
        }))
        .sort((a, b) => b.link_weight - a.link_weight);

      // Defer persistence: reinforcement is soft and a full save() here is O(graph)
      // (measured ~263ms @ 3k memories). markDirty() holds it in RAM; flush()/the next
      // content write makes it durable. Mirrors recall()'s existing deferred-flush path.
      this.markDirty();
      // When the memory was saved with a host transcript link, hand the agent a
      // ready-to-run get_conversation call so it can drill to the verbatim
      // exchange without remapping source fields to tool params. A passive hint,
      // not a directive — use it only when the recalled fact is too compressed.
      const src = mem.source;
      const trace =
        src && typeof src.host_session === "string"
          ? {
              tool: "get_conversation" as const,
              args: {
                session_id: src.host_session,
                agent: src.host_agent,
                turn: src.host_turn,
              },
            }
          : null;
      return {
        evidence: "read",
        grounded: true,
        suggested_tool: null,
        trace,
        memory: {
          id: memoryId,
          content: mem.content,
          depth: Math.round(mem.depth * 1000) / 1000,
          access_count: mem.access_count,
          last_accessed: mem.last_accessed,
          created_at: mem.created_at,
          source: mem.source,
          namespace: mem.namespace,
          expires_at: mem.ttl,
          supersedes: mem.supersedes,
          superseded_by: this._supersededBy[memoryId] ?? null,
          related_to: mem.links,
          contradicts: mem.contradicts ?? [],
        },
        keys: connectedKeys,
        via_key_id: viaKeyId ?? null,
      };
    });
  }

  // Opt-in one-shot "inject" recall: navigation keys PLUS the top-N associatively-expanded
  // memories, so an agent gets connected-but-dissimilar memories in one call without manual
  // read_key/read_memory traversal. Trades the deliberate-navigation default's context-efficiency
  // and precision (the injected set carries lower-precision associative neighbours) for fewer
  // round trips — hence opt-in, never the default. (TDD stub — replaced below.)
  async recallInject(
    query: string,
    topK = 5,
    namespace: string | null = null,
    opts: { preferDepth?: boolean; exploreShallow?: boolean } = {}
  ): Promise<{ keys: object[]; memories: object[] }> {
    // Navigation keys (so the agent can still steer) + selected top-N expanded memories. Keep the
    // default absolute anchor gate (minScore = MIN_SCORE_THRESHOLD). The gate is ANCHOR-based, not
    // per-hit: it does NOT drop HOP_DECAY'd associative neighbours — when any genuine dense anchor
    // exists the full fused/traversed set is kept. It only requires that the query actually matches
    // SOMETHING before we inject. Passing minScore=0 disabled it, so a query with no real match
    // (e.g. a cross-lingual miss) filled every slot with coincidental BM25 hits — pure junk. Pull a
    // wider candidate pool, then let selectInject pick by relevance / depth / exploration.
    const keys = await this.searchKeys(query, 8, namespace);
    const pool = (await this.recall(
      query, Math.max(topK * 3, 15), namespace, true, 2, 0, MIN_SCORE_THRESHOLD,
      GATE_Z_THRESHOLD, KEY_GATE_THRESHOLD, 0, false // reinforce=false: injection is passive
    )) as Array<{ id: string; matched_via?: string[] }>;
    // Inject is associative/semantic expansion, so only memories with DENSE or GRAPH support belong
    // in it: a content/key cosine match, or a (via)/(linked) hop from a genuine anchor. Once any
    // anchor clears the gate, recall keeps the WHOLE fused set — which includes memories pulled in
    // by BM25 alone (a stray shared token, e.g. an English query grazing "user"/"control" in
    // unrelated wiki docs). Those ride in beside the real hits and fill slots with junk; their fused
    // scores sit in the same noise band as real cross-lingual hits, so a relative floor can't tell
    // them apart — the matched_via provenance can. Drop candidates whose ONLY signal is "(bm25)".
    // Plain recall (deliberate navigation) still returns BM25 hits; this exclusion is inject-scoped.
    const supported = pool.filter((m) => (m.matched_via ?? []).some((v) => v !== "(bm25)"));
    const cands = supported.map((m) => ({ id: m.id, depth: this.memories[m.id]?.depth ?? 0 }));
    const byId = new Map(supported.map((m) => [m.id, m]));
    const memories = selectInject(cands, topK, opts)
      .map((id) => byId.get(id))
      .filter((m): m is { id: string } => Boolean(m))
      // Enrich keys with key_id so the agent can read_key() directly — inject's whole point is
      // skipping round trips, and a bare concept would force a resolution step right back in.
      .map((m) => ({ ...m, keys: this.getKeyRefsForMemory(m.id) }));
    return { keys, memories };
  }

  // ── Direct memory recall (internal / compatibility mode) ──

  async recall(
    query: string,
    topK = 5,
    namespace?: string | null,
    expand = false,
    maxHops = 2,
    minRelScore = 0,
    minScore = MIN_SCORE_THRESHOLD,
    minZ = GATE_Z_THRESHOLD,
    minKeyGate = KEY_GATE_THRESHOLD,
    minDepth = 0,
    // When false, recall is a pure read: no depth/access bump, no Hebbian reinforce/decay.
    // recallInject uses this so passively-surfaced (and the wider internal candidate) memories
    // aren't reinforced — only a real read_memory should strengthen the graph.
    reinforce = true
  ): Promise<object[]> {
    if (Object.keys(this.memories).length === 0) return [];

    // Depth floor in [0,1]: keep only well-established (frequently-recalled) memories.
    // 0 = no filter (default). Lets a caller ask for "only deep/important facts".
    minDepth = Math.max(0, Math.min(1, minDepth));

    // Clamp traversal depth: 1 = direct only, up to 5 for deep associative drill-down.
    maxHops = Math.max(1, Math.min(5, Math.floor(maxHops)));
    // Relative score floor in [0, 0.9): fraction of the top score below which results are dropped.
    minRelScore = Math.max(0, Math.min(0.9, minRelScore));
    // Absolute cosine floor in [0,1]. 0 disables the gate.
    minScore = Math.max(0, Math.min(1, minScore));
    minZ = Math.max(0, minZ);
    // Key-proximity gate floor in [0,1]. 0 disables it.
    minKeyGate = Math.max(0, Math.min(1, minKeyGate));
    const qEmb = await embedTextAsync(query, "query"); // outside lock
    this._checkDim(qEmb);

    const results: object[] = [];
    const queryLower = query.toLowerCase().trim();
    const memMatchedKeys: Record<string, string[]> = {};
    const memHop: Record<string, number> = {};
    let keyScores: [number, string][] = [];

    // Hoisted to method scope so Phase 3 (a separate locked section) can reuse it.
    const skip = (mid: string): boolean => {
      if (!(mid in this.memories)) return true;
      const mem = this.memories[mid];
      if (this._isExpired(mem)) return true;
      if (namespace && mem.namespace !== namespace) return true;
      if (mid in this._supersededBy) return true;
      return false;
    };

    // Phase-1 outputs, consumed by the unlocked rerank (Phase 2) + commit (Phase 3).
    let gated: [string, number][] = [];
    let definiteAnchor = false;
    const actualTopK = expand ? topK * 2 : topK;

    // ── Phase 1 (locked, fully synchronous) ── retrieve + fuse + gate. No await runs
    // inside this section, so the lock is held only for fast in-memory work, never
    // across model inference or disk I/O.
    await this._lock.runExclusive(async () => {
      const memRawSim: Record<string, number> = {};
      const allContentSims: number[] = [];
      const bumpRaw = (mid: string, sim: number) => {
        if (sim > (memRawSim[mid] ?? -Infinity)) memRawSim[mid] = sim;
      };

      // ── BM25 sparse search ──
      const bm25Ranked: Array<{ id: string; score: number }> = [];
      const bm25Results = this._bm25.search(query, { fuzzy: 0.2, prefix: true });
      for (const r of bm25Results.slice(0, BM25_RESULT_DEPTH)) {
        if (!skip(r.id)) bm25Ranked.push({ id: r.id, score: r.score });
      }

      // ── Dense Path A: Key batch matching ──
      const denseScores: Record<string, number> = {};

      const keyIds = Object.keys(this.keys);
      const keySims =
        keyIds.length > 0
          ? batchCosineSim(
              qEmb,
              keyIds.map((kid) => this.keys[kid].embedding)
            )
          : [];

      // Best concept-key cosine for the query — the key-proximity gate signal.
      // Captured over the raw sims (independent of KEY_RECALL), since keyGate
      // (e.g. e5 0.88) sits above keyRecall (0.85).
      let maxConceptKeySim = 0;
      for (let i = 0; i < keyIds.length; i++) {
        if (this.keys[keyIds[i]].key_type === "concept" && keySims[i] > maxConceptKeySim) {
          maxConceptKeySim = keySims[i];
        }
      }

      keyScores = [];
      for (let i = 0; i < keyIds.length; i++) {
        const kid = keyIds[i];
        const key = this.keys[kid];
        if (key.key_type === "name" || key.key_type === "proper_noun") {
          if (literalKeyMatch(queryLower, key.concept)) {
            keyScores.push([1.0, kid]);
          }
        } else if (keySims[i] >= KEY_RECALL_THRESHOLD) {
          keyScores.push([keySims[i], kid]);
        }
      }
      keyScores.sort((a, b) => b[0] - a[0]);

      for (const [keySim, kid] of keyScores.slice(0, 10)) {
        const idf = this._keyIdf(kid);
        for (const memId of this._keyToMems[kid]?.keys() ?? []) {
          if (skip(memId)) continue;
          const lw = this._getLinkWeight(kid, memId);
          const score = keySim * idf * lw;
          denseScores[memId] = (denseScores[memId] ?? 0) + score;
          bumpRaw(memId, keySim);
          if (!memMatchedKeys[memId]) memMatchedKeys[memId] = [];
          memMatchedKeys[memId].push(this.keys[kid].concept);
          memHop[memId] = 1;
        }
      }

      // ── Dense Path B: Content batch direct matching ──
      const memIds = Object.keys(this.memories);
      if (memIds.length > 0) {
        const contentSims = batchCosineSim(
          qEmb,
          memIds.map((mid) => this.memories[mid].embedding)
        );
        for (let i = 0; i < memIds.length; i++) {
          const mid = memIds[i];
          if (skip(mid)) continue;
          const cSim = contentSims[i];
          allContentSims.push(cSim);
          if (cSim >= CONTENT_RECALL_THRESHOLD) {
            bumpRaw(mid, cSim);
            const contentScore = cSim * 0.8;
            if (mid in denseScores) {
              denseScores[mid] += contentScore * 0.2;
            } else {
              denseScores[mid] = contentScore;
            }
            if (!memMatchedKeys[mid]) memMatchedKeys[mid] = [];
            memMatchedKeys[mid].push("(content)");
            if (!(mid in memHop)) memHop[mid] = 1;
          }
        }
      }

      // ── Build dense ranked list ──
      const denseRanked = Object.entries(denseScores)
        .sort(([, a], [, b]) => b - a)
        .slice(0, DENSE_RESULT_DEPTH)
        .map(([id, score]) => ({ id, score }));

      // ── RRF fusion ──
      const memScores: Record<string, number> = {};

      for (let rank = 0; rank < bm25Ranked.length; rank++) {
        const mid = bm25Ranked[rank].id;
        memScores[mid] = (memScores[mid] ?? 0) + 1 / (RRF_K + rank + 1);
        if (!memMatchedKeys[mid]) memMatchedKeys[mid] = [];
        if (!memMatchedKeys[mid].includes("(bm25)")) memMatchedKeys[mid].push("(bm25)");
        if (!(mid in memHop)) memHop[mid] = 1;
      }

      for (let rank = 0; rank < denseRanked.length; rank++) {
        const mid = denseRanked[rank].id;
        memScores[mid] = (memScores[mid] ?? 0) + 1 / (RRF_K + rank + 1);
      }

      // ── Lexical exact-key boost ──
      // RRF flattens score magnitude, so a memory whose key the query names
      // *literally* ranks no higher than one that merely shares fuzzy content
      // similarity — and with compressed embeddings (e.g. e5) the dense key
      // signal can't break that tie on its own. Give memories an additive bonus
      // when the query literally contains one of their key concepts, so an exact
      // concept hit outranks same-language content noise. IDF-weighted so hub
      // keys don't dominate; on the same RRF scale (~one top-rank contribution).
      // Scan ALL keys, not just keyScores: a literal mention is a strong, model-
      // independent signal that must count even when the key's embedding fell below
      // keyRecall (e.g. "동물" whose cosine to the query is weak but is named outright).
      for (const kid of Object.keys(this.keys)) {
        const concept = this.keys[kid]?.concept;
        if (!concept || concept.length < 2) continue;
        if (!literalKeyMatch(queryLower, concept)) continue;
        const bonus = (1 / (RRF_K + 1)) * this._keyIdf(kid);
        for (const memId of this._keyToMems[kid]?.keys() ?? []) {
          if (skip(memId)) continue;
          memScores[memId] = (memScores[memId] ?? 0) + bonus;
        }
      }

      // ── Apply depth/time modulation to fused scores ──
      for (const mid of Object.keys(memScores)) {
        const mem = this.memories[mid];
        if (!mem) continue;
        const depthFactor = 0.9 + mem.depth * 0.1;
        const tf = this._timeFactor(mem);
        memScores[mid] *= depthFactor * tf;
      }

      // ── Associative traversal: hops 2..maxHops via shared keys + explicit links ──
      // Generalized N-hop drill-down. Each round expands ONLY the frontier
      // discovered in the previous round, so a memory's hop is its shortest
      // distance from a directly-matched memory and nothing is expanded twice.
      // Score decays by HOP_DECAY per hop, so deeper associations contribute less.
      const reverseLinks: Record<string, string[]> = {};
      for (const [mid2, mem2] of Object.entries(this.memories)) {
        for (const l of mem2.links) (reverseLinks[l] ??= []).push(mid2);
      }

      // A "(via)"/"(linked)" tag means "graph-connected to what the query is about" — it should be
      // recorded only when the hop ORIGINATES from a genuine anchor (dense/key match), not from a
      // memory that itself only rode in on a coincidental BM25 token. Otherwise a cluster of
      // lexical-noise docs that share keys among themselves cross-tag each other "(via)" and look
      // graph-supported when their only tie to the query is BM25 (the remote-control-wiki failure).
      // graphAnchored tracks query-genuine reachability: seeded by hop-1 dense/key hits, and
      // extended transitively as legitimate hops land. Scores are NOT gated (plain-recall ranking is
      // unchanged); only provenance tags are — so the inject filter (drop matched_via ⊆ {bm25}) can
      // tell a real associative neighbour from lexical noise.
      const graphAnchored = new Set<string>(
        Object.keys(memScores).filter((mid) => (memMatchedKeys[mid] ?? []).some((v) => v !== "(bm25)"))
      );
      let frontier = new Set<string>(Object.keys(memScores)); // hop-1 set
      for (let h = 2; h <= maxHops && frontier.size > 0; h++) {
        const next = new Set<string>();
        for (const mid of frontier) {
          const baseScore = memScores[mid];
          const sourceAnchored = graphAnchored.has(mid);
          // shared-key neighbors
          for (const kid of this._memToKeys[mid]?.keys() ?? []) {
            if (!(kid in this.keys)) continue;
            const concept = this.keys[kid].concept;
            const idf = this._keyIdf(kid);
            for (const otherMid of this._keyToMems[kid]?.keys() ?? []) {
              if (otherMid === mid || skip(otherMid)) continue;
              const lw = this._getLinkWeight(kid, otherMid);
              memScores[otherMid] = (memScores[otherMid] ?? 0) + baseScore * MemoryGraph.HOP_DECAY * idf * lw;
              if (!memMatchedKeys[otherMid]) memMatchedKeys[otherMid] = [];
              if (sourceAnchored) {
                memMatchedKeys[otherMid].push(`${concept}(via)`);
                graphAnchored.add(otherMid); // legit reachability propagates to deeper hops
              }
              if (!(otherMid in memHop)) { memHop[otherMid] = h; next.add(otherMid); }
            }
          }
          // explicit links (bidirectional)
          const memObj = this.memories[mid];
          if (memObj) {
            const linkedIds = new Set([...memObj.links, ...(reverseLinks[mid] ?? [])]);
            for (const linkedId of linkedIds) {
              if (linkedId === mid || skip(linkedId)) continue;
              memScores[linkedId] = (memScores[linkedId] ?? 0) + baseScore * MemoryGraph.HOP_DECAY;
              if (!memMatchedKeys[linkedId]) memMatchedKeys[linkedId] = [];
              if (sourceAnchored) {
                memMatchedKeys[linkedId].push("(linked)");
                graphAnchored.add(linkedId);
              }
              if (!(linkedId in memHop)) { memHop[linkedId] = h; next.add(linkedId); }
            }
          }
        }
        frontier = next;
      }

      if (expand) {
        for (const mid of Object.keys(memScores)) {
          if ((memHop[mid] ?? 1) >= 2) memScores[mid] *= 0.7;
        }
      }

      const sorted = Object.entries(memScores).sort(([, a], [, b]) => b - a);
      // Absolute score gate (anchor-based): the query counts as "found" only if at
      // least one candidate has a direct dense similarity >= minScore. With no such
      // anchor, every hit is BM25/associative noise, so return nothing. When an anchor
      // exists, keep the full fused/traversed set (anchors + their associative and
      // lexical neighbors); the relative floor below still trims within-result noise.
      // This preserves N-hop/expand results, which by design have low direct similarity.
      // Anchor: the query is "found" iff a definite literal-key hit exists, OR a
      // candidate clears the absolute gate AND the top content similarity is a
      // robust-z outlier of the similarity distribution. The distribution gate
      // catches the e5 failure mode where every cosine is uniformly high so the
      // absolute gate false-positives. minZ (gateZ) = 0 disables it, leaving the
      // 0.7.0 absolute-only behavior unchanged for bge-m3 and other profiles.
      const candidateIds = Object.keys(memScores);
      definiteAnchor = candidateIds.some((mid) => (memRawSim[mid] ?? 0) >= 0.999);
      const absoluteAnchor = candidateIds.some(
        (mid) => passesAbsoluteGate(memRawSim[mid] ?? 0, minScore)
      );
      let maxContentSim = 0;
      for (const s of allContentSims) if (s > maxContentSim) maxContentSim = s;
      const distOK = passesDistributionGate(maxContentSim, allContentSims, minZ, GATE_MIN_POPULATION);
      // Key-proximity anchor: the query is close enough to a curated concept key to be
      // about a known topic. On e5 this is the PRIMARY found/not-found signal — it
      // separates cleanly (found keySim >=0.883, not-found <=0.875) where the content
      // distribution overlaps badly. keyGate=0 disables it (non-e5 profiles).
      const keyAnchor = minKeyGate > 0 && maxConceptKeySim >= minKeyGate;
      // Invariant (keyGate=0 AND gateZ=0 byte-identity): with both disabled, keyAnchor is
      // false and distOK is always true, so hasAnchor collapses to absoluteAnchor — identical
      // to the pre-gate 0.7.0 behavior. definiteAnchor (memRawSim >= 0.999, a literal
      // name/proper-noun match) and keyAnchor short-circuit the distOK check, so real hits on
      // e5 survive a flat content distribution that would otherwise gate them out.
      const hasAnchor = definiteAnchor || keyAnchor || (absoluteAnchor && distOK);
      // Relative score floor: drop results scoring below minRelScore × the top
      // hit. Deep traversal through hub keys pulls in many associations that
      // HOP_DECAY+IDF already score near the noise floor (~2% of top); a floor
      // (e.g. 0.05) trims that flood while keeping genuine associations (~15%+).
      // Default 0 = keep everything (no behavior change).
      const floor = sorted.length ? sorted[0][1] * minRelScore : 0;
      gated = (hasAnchor ? sorted : [])
        .filter(([, score]) => score >= floor)
        .filter(([mid]) => minDepth <= 0 || (this.memories[mid]?.depth ?? 0) >= minDepth);
    });

    // ── Phase 2 (UNLOCKED) ── cross-encoder rerank (opt-in). Model inference is the
    // only slow, I/O-like await in recall; running it outside the lock lets other
    // recalls and writes proceed meanwhile. It only READS immutable memory content
    // (all reads happen synchronously before the await) and mutates nothing shared.
    let ranked: [string, number][] = gated.slice(0, actualTopK);
    if (rerankEnabled() && gated.length > 0) {
      const pool = gated.slice(0, Math.max(actualTopK, RERANK_POOL));
      const scores = await rerankScores(
        query,
        pool.map(([mid]) => this.memories[mid]?.content ?? "")
      );
      if (scores) {
        const reordered = pool
          .map((entry, i) => ({ entry, s: scores[i] }))
          .sort((a, b) => b.s - a.s);
        // Not-found gate (opt-in): a low top relevance logit means nothing answers the
        // query → []. Trusted only when the query and the top candidate share script —
        // cross-lingual logits run low even when relevant, so on a script mismatch we keep
        // the result (the cosine/key gate already vouched). This catches same-language
        // distractors; cross-lingual not-found stays a known limitation (use bilingual keys).
        const topContent = this.memories[reordered[0]?.entry[0]]?.content ?? "";
        const sameScript = hasHangul(query) === hasHangul(topContent);
        if (RERANK_MIN_SCORE !== null && sameScript && reordered[0].s < RERANK_MIN_SCORE) {
          ranked = [];
        } else {
          ranked = reordered.map((x) => x.entry).slice(0, actualTopK);
        }
      }
    }

    // ── Phase 3 (locked, fully synchronous) ── commit reinforcement + assemble the
    // result payload. Re-validate every id with skip(): a concurrent forget/supersede/
    // expiry may have landed during the unlocked rerank above.
    await this._lock.runExclusive(async () => {
      for (const [mid, score] of ranked) {
        if (skip(mid)) continue;
        const mem = this.memories[mid];
        if (reinforce) {
          mem.depth = Math.min(mem.depth + DEPTH_INCREMENT, DEPTH_MAX);
          mem.access_count += 1;
          mem.last_accessed = Date.now() / 1000;
        }
        results.push({
          id: mid,
          content: mem.content,
          keys: this.getKeysForMemory(mid),
          matched_via: [...new Set(memMatchedKeys[mid] ?? [])],
          hop: memHop[mid] ?? 1,
          score: Math.round(score * 1000) / 1000,
          depth: Math.round(mem.depth * 1000) / 1000,
          access_count: mem.access_count,
          source: mem.source,
          supersedes: mem.supersedes,
          superseded_by: this._supersededBy[mid] ?? null,
          created_at: mem.created_at,
          namespace: mem.namespace,
          links: mem.links,
          contradicts: mem.contradicts ?? [],
        });
      }

      // ── Hebbian link reinforcement / decay ── (skipped on a pure read)
      if (reinforce) {
        const returnedSet = new Set(ranked.map(([mid]) => mid));
        const matchedKeyIds = new Set(keyScores.slice(0, 10).map(([, kid]) => kid));

        // Strengthen ONLY the top-ranked result's matched-key links. Reinforcing the whole
        // returned set would inflate weak, frequently-co-retrieved-but-unused tail links over
        // time (the rank-N tail scores near the RRF noise floor). Scoping to the single
        // strongest hit captures the real query→memory association without that pollution.
        // Also scoped to matched keys only, mirroring the decay side below.
        const topMid = ranked.find(([mid]) => !skip(mid))?.[0];
        if (topMid) {
          for (const kid of this._memToKeys[topMid]?.keys() ?? []) {
            if (!matchedKeyIds.has(kid)) continue;
            this._setLinkWeight(kid, topMid, this._getLinkWeight(kid, topMid) + LINK_REINFORCE_AMOUNT);
          }
        }

        // Weaken: explored but not returned
        for (const [, kid] of keyScores.slice(0, 10)) {
          for (const [memId, cw] of this._keyToMems[kid] ?? new Map()) {
            if (skip(memId)) continue;
            if (!returnedSet.has(memId)) {
              this._setLinkWeight(kid, memId, cw - LINK_DECAY_RATE);
            }
          }
        }

        this.markDirty();
      }
    });

    await this.flush(); // outside lock; save() is serialized + atomic (see _saveLock)
    return results;
  }

  // ── Related ──

  getRelated(memoryId: string): object[] {
    if (!(memoryId in this.memories)) return [];

    const related: Record<
      string,
      {
        id: string;
        content: string;
        shared_keys: string[];
        link_type: string;
        depth: number;
        contradicts: string[];
        _score: number;
      }
    > = {};

    // Key-sharing — accumulate a relevance score from key specificity (IDF). A neighbor
    // linked via a rare/specific shared key scores far higher than one linked only by a
    // hub key, so hubs sink to the bottom (and out, after the cap).
    for (const kid of this._memToKeys[memoryId]?.keys() ?? []) {
      const concept = this.keys[kid]?.concept ?? "?";
      const idf = this._keyIdf(kid);
      for (const mid of this._keyToMems[kid]?.keys() ?? []) {
        if (mid === memoryId || !(mid in this.memories)) continue;
        const mem = this.memories[mid];
        if (this._isExpired(mem) || mid in this._supersededBy) continue;
        if (!related[mid]) {
          related[mid] = {
            id: mid,
            content: mem.content,
            shared_keys: [],
            link_type: "key",
            depth: Math.round(mem.depth * 1000) / 1000,
            contradicts: mem.contradicts ?? [],
            _score: 0,
          };
        }
        if (!related[mid].shared_keys.includes(concept)) {
          related[mid].shared_keys.push(concept);
        }
        related[mid]._score += idf;
      }
    }

    // Explicit links (→)
    const sourceMem = this.memories[memoryId];
    for (const linkedId of sourceMem.links) {
      if (!(linkedId in this.memories) || linkedId === memoryId) continue;
      const mem = this.memories[linkedId];
      if (this._isExpired(mem) || linkedId in this._supersededBy) continue;
      if (!related[linkedId]) {
        related[linkedId] = {
          id: linkedId,
          content: mem.content,
          shared_keys: ["(explicit →)"],
          link_type: "explicit",
          depth: Math.round(mem.depth * 1000) / 1000,
          contradicts: mem.contradicts ?? [],
          _score: 0,
        };
      } else {
        related[linkedId].link_type = "both";
        if (!related[linkedId].shared_keys.includes("(explicit →)")) {
          related[linkedId].shared_keys.push("(explicit →)");
        }
      }
      related[linkedId]._score += RELATED_EXPLICIT_BONUS;
    }

    // Reverse links (←)
    for (const [mid, mem] of Object.entries(this.memories)) {
      if (mid === memoryId || this._isExpired(mem) || mid in this._supersededBy) continue;
      if (mem.links.includes(memoryId)) {
        if (!related[mid]) {
          related[mid] = {
            id: mid,
            content: mem.content,
            shared_keys: ["(explicit ←)"],
            link_type: "explicit",
            depth: Math.round(mem.depth * 1000) / 1000,
            contradicts: mem.contradicts ?? [],
            _score: 0,
          };
        } else if (!related[mid].shared_keys.includes("(explicit ←)")) {
          related[mid].shared_keys.push("(explicit ←)");
        }
        related[mid]._score += RELATED_EXPLICIT_BONUS;
      }
    }

    // Rank by specificity score, cap so a hub can't flood the chain, drop internal score.
    return Object.values(related)
      .sort((a, b) => b._score - a._score)
      .slice(0, RELATED_LIMIT)
      .map(({ _score, ...rest }) => rest);
  }

  // ── Delete ──

  async delete(memoryId: string): Promise<boolean> {
    return this._lock.runExclusive(async () => {
      if (!(memoryId in this.memories)) return false;
      delete this.memories[memoryId];
      this._unlinkMemory(memoryId);
      this._removeMemoryReferences([memoryId]);
      this._pruneOrphanKeys();
      try { this._bm25.discard(memoryId); } catch { /* already removed */ }
      delete this._supersededBy[memoryId];
      for (const [oldId, newId] of Object.entries(this._supersededBy)) {
        if (newId === memoryId) delete this._supersededBy[oldId];
      }
      await this.save();
      return true;
    });
  }

  // ── List all ──

  listAll(namespace?: string | null): object[] {
    return Object.entries(this.memories)
      .filter(([mid, mem]) => {
        if (this._isExpired(mem)) return false;
        if (mid in this._supersededBy) return false;
        if (namespace && mem.namespace !== namespace) return false;
        return true;
      })
      .map(([mid, mem]) => ({
        id: mid,
        content: mem.content,
        keys: this.getKeysForMemory(mid),
        depth: Math.round(mem.depth * 1000) / 1000,
        access_count: mem.access_count,
        supersedes: mem.supersedes,
        created_at: mem.created_at,
        namespace: mem.namespace,
        expires_at: mem.ttl,
        links: mem.links,
      }));
  }

  // ── Cleanup expired ──

  async cleanupExpired(): Promise<number> {
    return this._lock.runExclusive(async () => {
      const expired = Object.entries(this.memories)
        .filter(([, mem]) => this._isExpired(mem))
        .map(([mid]) => mid);

      for (const mid of expired) {
        delete this.memories[mid];
        this._unlinkMemory(mid);
        try { this._bm25.discard(mid); } catch { /* already removed */ }
        delete this._supersededBy[mid];
        for (const [oldId, newId] of Object.entries(this._supersededBy)) {
          if (newId === mid) delete this._supersededBy[oldId];
        }
      }
      this._removeMemoryReferences(expired);
      this._pruneOrphanKeys();

      let pruned = false;
      const now = Date.now() / 1000;
      for (const key of Object.values(this.keys)) {
        if (!key.learnedAliases?.length) continue;
        const keep = key.learnedAliases.filter(
          (l) => l.hits > 0 || now - l.addedAt < AUTOKEY_PRUNE_AGE_SECONDS
        );
        if (keep.length === key.learnedAliases.length) continue;
        const dropped = new Set(
          key.learnedAliases.filter((l) => !keep.includes(l)).map((l) => l.alias.toLowerCase())
        );
        key.learnedAliases = keep;
        key.aliases = (key.aliases ?? []).filter((a) => !dropped.has(a.toLowerCase()));
        pruned = true;
      }

      // Drop stale alias candidates — heat that never reached promotion (e.g. long
      // non-promotable queries that fail isShortConcept) — so the persisted ledger
      // cannot grow without bound on a long-lived key.
      for (const key of Object.values(this.keys)) {
        if (!key.aliasCandidates) continue;
        for (const [norm, cand] of Object.entries(key.aliasCandidates)) {
          if (now - cand.lastSeen >= AUTOKEY_PRUNE_AGE_SECONDS) {
            delete key.aliasCandidates[norm];
            pruned = true;
          }
        }
        if (Object.keys(key.aliasCandidates).length === 0) delete key.aliasCandidates;
      }

      if (expired.length > 0 || pruned) await this.save();
      return expired.length;
    });
  }
}

// ── Conversation store ──

export async function saveTurn(
  sessionId: string,
  role: string,
  content: string
): Promise<number> {
  await mkdir(CONVERSATIONS_DIR, { recursive: true });
  const path = conversationPath(sessionId);
  let turn = 0;
  try {
    const text = await readFile(path, "utf-8");
    turn = text.split("\n").filter((l) => l.trim()).length;
  } catch {
    // file does not exist yet
  }
  const entry = JSON.stringify({
    turn,
    role,
    content,
    ts: Date.now() / 1000,
  });
  await appendFile(path, entry + "\n", "utf-8");
  return turn;
}

export async function loadConversation(
  sessionId: string,
  turn?: number | null
): Promise<object[]> {
  const path = conversationPath(sessionId);
  let text: string;
  try {
    text = await readFile(path, "utf-8");
  } catch {
    return [];
  }
  const lines: object[] = [];
  text.split("\n").forEach((line, idx) => {
    if (!line.trim()) return;
    try {
      lines.push(JSON.parse(line) as object);
    } catch (err) {
      throw new Error(
        `Invalid conversation log ${sessionId} at line ${idx + 1}: ${errorMessage(err)}`
      );
    }
  });
  if (turn != null) {
    const start = Math.max(0, turn - 2);
    const end = Math.min(lines.length, turn + 3);
    return lines.slice(start, end);
  }
  return lines;
}
