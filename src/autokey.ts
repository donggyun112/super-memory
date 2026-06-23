import { isShortConcept } from "./embedding.js";
import { cfgRaw } from "./env.js";

export interface RecallBufferEntry {
  queryText: string;
  weakKeyScores: Map<string, number>;
  ts: number;
}

// Runtime-only ring buffer of recent recalls that matched one or more keys only
// *weakly* (semantic, not literal). Never persisted. Bounded by capacity + TTL so
// it can never grow without bound or attribute a confirmation to a stale query.
export class RecallBuffer {
  private _entries: RecallBufferEntry[] = [];
  private readonly _capacity: number;
  private readonly _ttl: number;
  private readonly _now: () => number;

  constructor(opts: { capacity?: number; ttlSeconds?: number; now?: () => number } = {}) {
    this._capacity = Math.max(1, Math.floor(opts.capacity ?? 32));
    this._ttl = Math.max(1, opts.ttlSeconds ?? 300);
    this._now = opts.now ?? (() => Date.now() / 1000);
  }

  push(entry: { queryText: string; weakKeyScores: Map<string, number> }): void {
    this._entries.push({
      queryText: entry.queryText,
      weakKeyScores: new Map(entry.weakKeyScores),
      ts: this._now(),
    });
    if (this._entries.length > this._capacity) {
      this._entries.splice(0, this._entries.length - this._capacity);
    }
  }

  // Most-recent fresh entry that weakly matched keyId; removes keyId from that
  // entry so a single recall confirms a given key at most once.
  consumeWeakMatch(keyId: string): RecallBufferEntry | null {
    const cutoff = this._now() - this._ttl;
    for (let i = this._entries.length - 1; i >= 0; i--) {
      const e = this._entries[i];
      if (e.ts < cutoff) continue;
      if (e.weakKeyScores.has(keyId)) {
        const result: RecallBufferEntry = {
          queryText: e.queryText,
          weakKeyScores: new Map(e.weakKeyScores),
          ts: e.ts,
        };
        e.weakKeyScores.delete(keyId);
        return result;
      }
    }
    return null;
  }

  size(): number {
    return this._entries.length;
  }
}

function envInt(suffix: string, fallback: number, min: number): number {
  const raw = cfgRaw(suffix);
  if (raw === undefined || raw.trim() === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= min ? Math.floor(n) : fallback;
}

// Feature flag. Default ON; set KEYMEM_AUTOKEY=false to disable (mirrors
// KEYMEM_AUTO_MIGRATE). Read once at import.
export const AUTOKEY_ENABLED = cfgRaw("AUTOKEY") !== "false";
export const AUTOKEY_PROMOTE_N = envInt("AUTOKEY_PROMOTE_N", 3, 1);
// Lower cosine bound for the confirmation path (sub-recall near-misses). bge-m3 paraphrase
// near-misses cluster ~0.40–0.55; 0.45 admits the recoverable band while excluding genuinely
// unrelated keys. Set KEYMEM_AUTOKEY_CONFIRM_FLOOR to tune; >= recall threshold disables it.
export const AUTOKEY_CONFIRM_FLOOR = (() => {
  const raw = cfgRaw("AUTOKEY_CONFIRM_FLOOR");
  if (raw === undefined || raw.trim() === "") return 0.45;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 && n <= 1 ? n : 0.45;
})();
export const AUTOKEY_MAX_ALIASES = envInt("AUTOKEY_MAX_ALIASES", 8, 0);
export const AUTOKEY_BUFFER_CAPACITY = 32;
export const AUTOKEY_BUFFER_TTL_SECONDS = 300;
export const AUTOKEY_PRUNE_AGE_SECONDS = envInt(
  "AUTOKEY_PRUNE_AGE", 30 * 24 * 3600, 0
);

// Pure policy: given accumulated heat and the recall-time query↔key cosine, decide
// whether to fold the query into the key space and how. Short-concept gate keeps
// natural-language queries out of the alias set; the content path already serves those.
export function decidePromotion(args: {
  count: number;
  query: string;
  cosine: number;
  learnedAliasCount: number;
  aliasThreshold: number;
  newKeyThreshold: number;
  promoteN: number;
  maxAliases: number;
  // Optional lower cosine bound for the confirmation path: a query that fell BELOW the
  // recall gate (cosine < newKeyThreshold) but was confirmed `promoteN` times by the agent
  // reading the right memory via this key. Repeated confirmation is stronger evidence than
  // a single cosine, so we fold the query in as a learned alias. Omitted = legacy behavior.
  confirmFloor?: number;
}): "alias" | "newKey" | "none" {
  if (args.count < args.promoteN) return "none";
  if (!isShortConcept(args.query)) return "none";
  if (args.cosine >= args.aliasThreshold) {
    return args.learnedAliasCount < args.maxAliases ? "alias" : "none";
  }
  if (args.cosine >= args.newKeyThreshold) return "newKey";
  if (args.confirmFloor !== undefined && args.cosine >= args.confirmFloor) {
    return args.learnedAliasCount < args.maxAliases ? "alias" : "none";
  }
  return "none";
}
