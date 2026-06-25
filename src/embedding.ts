import { config } from "dotenv";
config();

import OpenAI from "openai";
import { existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { KNOWN_MODELS, defaultModelDir, ensureModelFiles, type Fetcher } from "./modelDownload.js";
import { cfgRaw, cfgName, homeBaseDir } from "./env.js";

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
export const OPENAI_EMBEDDING_MODEL =
  process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small";

const _DEFAULT_BACKEND = OPENAI_API_KEY ? "openai" : "local";
export const EMBEDDING_BACKEND =
  process.env.EMBEDDING_BACKEND ?? _DEFAULT_BACKEND;
export const LOCAL_EMBEDDING_MODEL =
  process.env.LOCAL_EMBEDDING_MODEL ?? "fast-multilingual-e5-large";
// One deterministic home for ALL downloaded models. Defaults to <keymem home>/models
// (i.e. ~/.keymem/models) — the same parent bge-m3 uses — so e5 and bge-m3 land together instead of
// in a relative ./local_cache that scatters per launch dir. Override (e.g. Docker → /data/models).
export const LOCAL_EMBEDDING_CACHE_DIR =
  process.env.LOCAL_EMBEDDING_CACHE_DIR ?? join(homeBaseDir(), "models");

const EMBED_RETRIES = 3;
type EmbeddingInputType = "passage" | "query";

const LOCAL_MODEL_ALIASES: Record<string, string> = {
  "fast-multilingual-e5-large": "MLE5Large",
  "multilingual-e5-large": "MLE5Large",
  mle5large: "MLE5Large",
  "fast-bge-base-en-v1.5": "BGEBaseENV15",
  "bge-base-en-v1.5": "BGEBaseENV15",
  bgebaseenv15: "BGEBaseENV15",
  "fast-bge-small-en-v1.5": "BGESmallENV15",
  "bge-small-en-v1.5": "BGESmallENV15",
  bgesmallenv15: "BGESmallENV15",
  "fast-all-minilm-l6-v2": "AllMiniLML6V2",
  "all-minilm-l6-v2": "AllMiniLML6V2",
  allminilml6v2: "AllMiniLML6V2",
  "bge-m3": "CUSTOM",
  bgem3: "CUSTOM",
  "baai/bge-m3": "CUSTOM",
  "fast-bge-m3": "CUSTOM",
};

let _openaiClient: OpenAI | null = null;

function getOpenAIClient(): OpenAI {
  if (!_openaiClient) {
    _openaiClient = new OpenAI({ apiKey: OPENAI_API_KEY });
  }
  return _openaiClient;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function normalizeModelName(modelName: string): string {
  return modelName
    .trim()
    .replace(/^EmbeddingModel\./i, "")
    .toLowerCase()
    .replace(/[\s_]/g, "-");
}

function resolveLocalModel(
  modelName: string,
  embeddingModel: Record<string, string>
): string {
  const normalized = normalizeModelName(modelName);
  const aliasKey = LOCAL_MODEL_ALIASES[normalized];
  if (aliasKey && aliasKey in embeddingModel) return embeddingModel[aliasKey];

  for (const [key, value] of Object.entries(embeddingModel)) {
    if (key.toLowerCase() === modelName.trim().toLowerCase()) return value;
    if (value.toLowerCase() === normalized) return value;
  }

  const supported = Object.values(embeddingModel)
    .filter((value) => value !== embeddingModel.CUSTOM)
    .join(", ");
  throw new Error(
    `Unsupported LOCAL_EMBEDDING_MODEL "${modelName}". Supported values: ${supported}`
  );
}

// ── Threshold profiles ──
// Embedding backends have very different cosine distributions, so a single
// threshold set cannot serve all of them. e5 in particular packs unrelated
// short concept keys into the same 0.86–0.92 band as related ones, which
// collapses the key graph at BGE-tuned thresholds. For e5 we therefore raise
// merge/auto-link/key-recall high enough to avoid spurious matching and lean on
// the content path (which separates cleanly), BM25, name-exact keys, and
// explicit key links for retrieval.
export interface ThresholdProfile {
  keyMerge: number;
  memoryDedup: number;
  keyAutoLink: number;
  keyRecall: number;
  contentRecall: number;
  // Absolute cosine floor: a recalled memory must have raw similarity (best of
  // content-sim / matched key-sim) >= minScore, else it is dropped. Lets recall
  // return [] for truly-unrelated queries instead of topK noise.
  minScore: number;
  // Lower bound of the contradiction band [contradiction, memoryDedup). New
  // memories whose best similarity to an existing one falls in this band AND
  // share a key are flagged (not deduped) as potential contradictions.
  contradiction: number;
  // Robust-z (median/MAD) distribution gate threshold: the top content similarity
  // must be at least this many MAD-sigmas above the median of the query's
  // similarity distribution for the query to count as "found". 0 disables the gate.
  // Built for e5, whose narrow packed cosine band defeats the absolute minScore gate.
  gateZ: number;
  // Key-proximity gate: a query counts as "found" if its best concept-key cosine
  // is >= keyGate. On e5 this separates found/not-found far more cleanly than the
  // content distribution (curated concept keys are reliable topic anchors): real
  // queries land >=0.88 against some key, unrelated queries stay <=0.875. 0 disables.
  keyGate: number;
  // Write-time semantic merge for SHORT concept keys: an incoming short key folds into
  // an existing concept key when their cosine >= this (a clear synonym), reconciling the
  // state-blind key choices an LLM makes at remember() time. Model-specific (e5 packs
  // cosines higher than bge-m3, so it needs a higher value). 0 = exact-match-only (off).
  shortKeyMerge: number;
}

export const THRESHOLD_PROFILES: Record<string, ThresholdProfile> = {
  openai: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.5, keyRecall: 0.28, contentRecall: 0.28, minScore: 0.28, contradiction: 0.85, gateZ: 0, keyGate: 0, shortKeyMerge: 0 },
  bge: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.6, keyRecall: 0.6, contentRecall: 0.5, minScore: 0.5, contradiction: 0.85, gateZ: 0, keyGate: 0, shortKeyMerge: 0 },
  // e5: query↔key is asymmetric (query/passage prefixes) and separates well —
  // exact-term matches land ~0.886+, distinct words ≤0.82, so keyRecall 0.85
  // catches real key hits. But key↔key and content↔key are both passage-embedded
  // and do NOT separate, so keyMerge/keyAutoLink stay high to prevent collapse.
  // memoryDedup 0.985: e5 packs distinct-but-similar facts ("A uses Postgres" vs
  // "B uses Mongo" ≈0.96) dangerously close to true paraphrases (≈0.99). Dedup
  // wrongly below 0.985 would silently supersede distinct memories → data loss.
  // gateZ 2.5: calibrated on fast-multilingual-e5-large (8-memory realistic set).
  // Measured FOUND z: [2.80, 3.33, 4.33, 4.81] (min 2.80).
  // Measured NOT-FOUND z: [0.87, 0.98, 1.51, 2.28] (max 2.28).
  // Gap [2.28, 2.80] — chose 2.5 (biased toward lower edge to avoid blocking real matches).
  // Margin is narrow (~0.52σ gap, n=8 fixture); KEYMEM_GATE_Z is the per-deployment
  // escape hatch for re-tuning if the gap shifts on a different corpus or model variant.
  // Known e5 limitation: the gate keys off maxContentSim (content cosine only), so a
  // genuinely-relevant hit that anchors solely via a fuzzy (non-literal) key match but
  // produces a flat content distribution may be gated out — the gate intentionally overrides
  // weak fuzzy-key anchors on e5; only literal key matches (definiteAnchor, memRawSim>=0.999)
  // are protected from the distribution gate.
  // not-found gate DISABLED on e5 by default (gateZ 0, keyGate 0). Empirically, no
  // similarity heuristic reliably separates found/not-found on e5: held-out validation
  // showed both the content-distribution gate AND the key-proximity gate overfit
  // (12-query tuning set 100%, but a fresh held-out set only 63%). e5 packs everything
  // into ~0.83–0.92, and person-attribute queries ("동균 나이", "전화번호") are
  // inseparable from real facts that share the subject — a not-found query can score a
  // higher content-z (전화번호 z=4.70) or topContentCos (동균 나이 0.891) than a real
  // match. Since hiding a real memory (false negative) is a worse failure than returning
  // noise, e5 defaults to no gate (0.7.0 behavior: never hides). The gate machinery is
  // env-opt-in (KEYMEM_GATE_Z / _KEY_GATE) for users who accept the precision/recall
  // tradeoff; for RELIABLE not-found detection use bge-m3, which separates cleanly via the
  // absolute minScore gate (verified found→hit / not-found→[] end-to-end).
  e5: { keyMerge: 0.97, memoryDedup: 0.985, keyAutoLink: 0.93, keyRecall: 0.85, contentRecall: 0.8, minScore: 0.8, contradiction: 0.95, gateZ: 0, keyGate: 0, shortKeyMerge: 0 },
  minilm: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.6, keyRecall: 0.5, contentRecall: 0.45, minScore: 0.45, contradiction: 0.85, gateZ: 0, keyGate: 0, shortKeyMerge: 0 },
  // bge-m3: multilingual, 1024-dim, well-separated (closer to bge than e5).
  // dedup lowered to 0.94 so real duplicates are caught without fragmenting.
  // contradiction 0.80 calibrated against real bge-m3: same-subject conflicting
  // facts ("uses Postgres" vs "uses Mongo") land ~0.81–0.86, while merely-related
  // facts top out ~0.80, so 0.80 separates them. Note a known limit — conflicts
  // differing by a single token ("월요일" vs "금요일") can land ~0.95, above
  // memoryDedup, and are silently superseded rather than flagged. Env-overridable.
  bgem3: { keyMerge: 0.86, memoryDedup: 0.94, keyAutoLink: 0.62, keyRecall: 0.62, contentRecall: 0.55, minScore: 0.55, contradiction: 0.80, gateZ: 0, keyGate: 0, shortKeyMerge: 0 },
};

export function familyForModel(
  modelName: string
): "e5" | "bge" | "minilm" | "bgem3" | "unknown" {
  const normalized = normalizeModelName(modelName);
  if (["bge-m3", "bgem3", "baai/bge-m3", "fast-bge-m3", "bge_m3"].includes(normalized)) {
    return "bgem3";
  }
  const alias = LOCAL_MODEL_ALIASES[normalized];
  if (alias === "MLE5Large") return "e5";
  if (alias === "AllMiniLML6V2") return "minilm";
  if (alias === "BGEBaseENV15" || alias === "BGESmallENV15") return "bge";
  return "unknown";
}

export function usesE5Prefix(family: string): boolean {
  return family === "e5";
}

// Identifies the embedding vector space the graph was built in. Two models of the
// same dimension (e5-large ↔ bge-m3, both 1024-d) produce INCOMPATIBLE vectors, so
// a dimension check alone cannot detect a backend swap. The graph stores this
// fingerprint; on load a mismatch triggers a re-embed. Read live from env so a
// per-process backend change is reflected even though the module-level consts were
// snapshotted at import. OpenAI variants differ by model name; local by model id.
export function embeddingFingerprint(): string {
  const backend = process.env.EMBEDDING_BACKEND ?? _DEFAULT_BACKEND;
  if (backend !== "local") {
    return `openai:${process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small"}`;
  }
  return `local:${normalizeModelName(process.env.LOCAL_EMBEDDING_MODEL ?? "fast-multilingual-e5-large")}`;
}

export function customModelConfig(): { dir: string; file: string } {
  const dir = process.env.LOCAL_EMBEDDING_MODEL_PATH ?? "";
  if (!dir.trim()) {
    throw new Error(
      `LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}" resolves to a CUSTOM model, ` +
        `so LOCAL_EMBEDDING_MODEL_PATH (absolute dir containing model.onnx + tokenizer ` +
        `files) is required. Optionally set LOCAL_EMBEDDING_MODEL_FILE (default model.onnx).`
    );
  }
  return { dir, file: process.env.LOCAL_EMBEDDING_MODEL_FILE ?? "model.onnx" };
}

function modelFilePresent(dir: string, file: string): boolean {
  try {
    return existsSync(join(dir, file)) && statSync(join(dir, file)).size > 0;
  } catch {
    return false;
  }
}

// Resolve the local CUSTOM model directory, auto-downloading a known model (bge-m3) when
// its files are missing. Backward compatible: an existing LOCAL_EMBEDDING_MODEL_PATH with
// the model present is used unchanged with NO download; an unknown custom model with no
// usable path still throws the original guidance. Only the local backend's CUSTOM path
// calls this — OpenAI and fastembed built-ins never do.
export async function ensureCustomEmbeddingModel(fetcher?: Fetcher): Promise<{ dir: string; file: string }> {
  const fileEnv = process.env.LOCAL_EMBEDDING_MODEL_FILE;
  const file = fileEnv ?? "model.onnx";
  const envPath = (process.env.LOCAL_EMBEDDING_MODEL_PATH ?? "").trim();

  // 1. A custom model FILENAME means the user manages this dir themselves → trust it
  //    entirely with no download (preserves the original behavior for bespoke setups).
  if (fileEnv && envPath && modelFilePresent(envPath, file)) return { dir: envPath, file };

  // 2. A known model (bge-m3) → ensure all files, downloading ONLY the missing ones. A
  //    complete dir downloads nothing (backward compatible); a partial/empty one heals.
  if (localModelFamily() === "bgem3") {
    const dir = envPath || defaultModelDir("bge-m3");
    await ensureModelFiles(KNOWN_MODELS["bge-m3"], dir, fetcher);
    return { dir, file: "model.onnx" };
  }

  // 3. Unknown custom model → original path-or-throw guidance.
  return customModelConfig();
}

let _warnedUncalibrated = false;
function localModelFamily(): "e5" | "bge" | "minilm" | "bgem3" {
  const fam = familyForModel(LOCAL_EMBEDDING_MODEL);
  if (fam !== "unknown") return fam;
  // Unknown model: thresholds are NOT calibrated for it. Make the miscalibration
  // loud and point at the env override escape hatch (see original warning).
  if (!_warnedUncalibrated) {
    _warnedUncalibrated = true;
    console.error(
      `[keymem] WARNING: no calibrated threshold profile for ` +
        `LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}". Falling back to BGE ` +
        `thresholds — the graph may mis-cluster. Override per-threshold with ` +
        `KEYMEM_KEY_MERGE / _MEMORY_DEDUP / _KEY_AUTOLINK / _KEY_RECALL / _CONTENT_RECALL / _MIN_SCORE / _CONTRADICTION.`
    );
  }
  return "bge";
}

function envThreshold(suffix: string): number | undefined {
  const raw = cfgRaw(suffix);
  if (raw === undefined || raw.trim() === "") return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0 || n > 1) {
    console.error(`[keymem] WARNING: ignoring ${cfgName(suffix)}="${raw}" (must be a number in [0,1]).`);
    return undefined;
  }
  return n;
}

// Like envThreshold but for non-negative unbounded values (e.g. a z-score gate),
// which legitimately exceed 1. Rejects negative / non-finite input.
function envNonNegative(suffix: string): number | undefined {
  const raw = cfgRaw(suffix);
  if (raw === undefined || raw.trim() === "") return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) {
    console.error(`[keymem] WARNING: ignoring ${cfgName(suffix)}="${raw}" (must be a number >= 0).`);
    return undefined;
  }
  return n;
}

// Per-threshold env overrides are an escape hatch for when the model or the
// character of stored data drifts away from what the built-in profiles assume.
export function getThresholdProfile(): ThresholdProfile {
  const base =
    EMBEDDING_BACKEND !== "local"
      ? THRESHOLD_PROFILES.openai
      : THRESHOLD_PROFILES[localModelFamily()];
  return {
    keyMerge: envThreshold("KEY_MERGE") ?? base.keyMerge,
    memoryDedup: envThreshold("MEMORY_DEDUP") ?? base.memoryDedup,
    keyAutoLink: envThreshold("KEY_AUTOLINK") ?? base.keyAutoLink,
    keyRecall: envThreshold("KEY_RECALL") ?? base.keyRecall,
    contentRecall: envThreshold("CONTENT_RECALL") ?? base.contentRecall,
    minScore: envThreshold("MIN_SCORE") ?? base.minScore,
    contradiction: envThreshold("CONTRADICTION") ?? base.contradiction,
    gateZ: envNonNegative("GATE_Z") ?? base.gateZ,
    keyGate: envThreshold("KEY_GATE") ?? base.keyGate,
    shortKeyMerge: envThreshold("SHORT_KEY_MERGE") ?? base.shortKeyMerge,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _localModel: any = null;

let _testEmbedder:
  | ((text: string, inputType: EmbeddingInputType) => number[])
  | null = null;

// Test-only seam. Lets tests drive embedding-dependent code paths with crafted
// vectors so cosine similarities are deterministic and no ONNX model is loaded.
// Never set this in production code.
export function __setTestEmbedder(
  fn: (text: string, inputType: EmbeddingInputType) => number[]
): void {
  _testEmbedder = fn;
}

export function __clearTestEmbedder(): void {
  _testEmbedder = null;
}

async function getLocalModel() {
  if (!_localModel) {
    try {
      const { FlagEmbedding, EmbeddingModel } = await import("fastembed");
      const model = resolveLocalModel(LOCAL_EMBEDDING_MODEL, EmbeddingModel);
      const initOpts: Record<string, unknown> = {
        model: model as never,
        cacheDir: LOCAL_EMBEDDING_CACHE_DIR,
      };
      if (model === EmbeddingModel.CUSTOM) {
        const { dir, file } = await ensureCustomEmbeddingModel();
        initOpts.modelAbsoluteDirPath = dir;
        initOpts.modelName = file;
      }
      _localModel = await FlagEmbedding.init(initOpts as never);
    } catch (err) {
      throw new Error(
        "Failed to initialize local fastembed model.\n" +
          "Install with: npm install fastembed\n" +
          "Or set OPENAI_API_KEY to use OpenAI embeddings.\n" +
          `Cause: ${errorMessage(err)}`
      );
    }
  }
  return _localModel;
}

async function embedLocal(
  text: string,
  inputType: EmbeddingInputType
): Promise<number[]> {
  const model = await getLocalModel();
  const noPrefix = localModelFamily() === "bgem3";
  if (noPrefix) {
    for await (const batch of model.embed([text])) {
      return Array.from(batch[0]) as number[];
    }
    throw new Error("fastembed returned no embeddings");
  }
  if (inputType === "query" && typeof model.queryEmbed === "function") {
    return Array.from(await model.queryEmbed(text)) as number[];
  }
  const gen =
    typeof model.passageEmbed === "function"
      ? model.passageEmbed([text], 256)
      : model.embed([text]);
  for await (const batch of gen) {
    return Array.from(batch[0]) as number[];
  }
  throw new Error("fastembed returned no embeddings");
}

// Short concept keys (e.g. "Agent A" / "Agent B") embed almost identically and
// would over-merge under semantic matching, conflating distinct entities. Treat
// them like proper nouns: merge only on exact string match. Tunable.
export const SHORT_CONCEPT_MAX_TOKENS = 2;
export const SHORT_CONCEPT_MAX_CHARS = 15;

// Contradiction band: similar enough to be about the same subject, but below the
// dedup threshold so it is a distinct (possibly conflicting) fact, not a paraphrase.
export function inContradictionBand(sim: number, floor: number, dedup: number): boolean {
  return sim >= floor && sim < dedup;
}

export function isShortConcept(concept: string): boolean {
  const trimmed = concept.trim();
  const tokens = trimmed.split(/\s+/).filter(Boolean);
  return tokens.length <= SHORT_CONCEPT_MAX_TOKENS || trimmed.length <= SHORT_CONCEPT_MAX_CHARS;
}

export async function embedTextAsync(
  text: string,
  inputType: EmbeddingInputType = "passage"
): Promise<number[]> {
  if (_testEmbedder) return _testEmbedder(text, inputType);
  if (EMBEDDING_BACKEND === "local") {
    return embedLocal(text, inputType);
  }

  const client = getOpenAIClient();
  for (let attempt = 0; attempt < EMBED_RETRIES; attempt++) {
    try {
      const resp = await client.embeddings.create({
        model: OPENAI_EMBEDDING_MODEL,
        input: text,
      });
      return resp.data[0].embedding;
    } catch (err) {
      if (attempt === EMBED_RETRIES - 1) throw err;
      await new Promise((resolve) =>
        setTimeout(resolve, Math.pow(2, attempt) * 1000)
      );
    }
  }
  throw new Error("unreachable");
}
