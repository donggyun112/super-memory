export interface AliasCandidate {
  count: number;
  lastSeen: number;
  queryText: string;
}

export interface LearnedAlias {
  alias: string;
  addedAt: number;
  hits: number;
}

export interface Key {
  id: string;
  concept: string;
  aliases: string[];
  embedding: number[];
  key_type: "concept" | "name" | "proper_noun";
  // Heat ledger for auto-key self-healing: normalized recall query -> confirmation count.
  // Persisted with the key; cleared when a candidate is promoted. Optional/absent on
  // legacy graphs and on keys that have never received a weak-confirmed read.
  aliasCandidates?: Record<string, AliasCandidate>;
  // Aliases added by auto-key promotion (not authored at remember() time). Provenance for
  // read_key output and the basis for stale-alias pruning.
  learnedAliases?: LearnedAlias[];
}

export interface Memory {
  id: string;
  content: string;
  embedding: number[];
  created_at: number;
  source: Record<string, unknown> | null;
  supersedes: string | null;
  depth: number;
  access_count: number;
  last_accessed: number;
  namespace: string;
  ttl: number | null;
  links: string[];
  contradicts: string[];
}

export interface GraphData {
  keys: Record<string, Key>;
  memories: Record<string, Memory>;
  links: Array<{ key_id: string; memory_id: string; weight?: number }>;
  // Provenance for the embedding vector space (see embeddingFingerprint). Absent
  // in graphs written before fingerprinting; treated as "unknown" on load.
  meta?: { embeddingFingerprint?: string };
}
