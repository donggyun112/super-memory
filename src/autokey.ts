export interface RecallBufferEntry {
  queryText: string;
  queryEmbedding: number[];
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

  push(entry: { queryText: string; queryEmbedding: number[]; weakKeyScores: Map<string, number> }): void {
    this._entries.push({ ...entry, ts: this._now() });
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
          queryEmbedding: e.queryEmbedding,
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
