import { config } from "dotenv";
config();

import OpenAI from "openai";

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
export const OPENAI_EMBEDDING_MODEL =
  process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small";

const _DEFAULT_BACKEND = OPENAI_API_KEY ? "openai" : "local";
export const EMBEDDING_BACKEND =
  process.env.EMBEDDING_BACKEND ?? _DEFAULT_BACKEND;
export const LOCAL_EMBEDDING_MODEL =
  process.env.LOCAL_EMBEDDING_MODEL ?? "fast-multilingual-e5-large";
export const LOCAL_EMBEDDING_CACHE_DIR =
  process.env.LOCAL_EMBEDDING_CACHE_DIR ?? "local_cache";

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
}

const THRESHOLD_PROFILES: Record<string, ThresholdProfile> = {
  openai: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.5, keyRecall: 0.28, contentRecall: 0.28 },
  bge: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.6, keyRecall: 0.6, contentRecall: 0.5 },
  // e5: query↔key is asymmetric (query/passage prefixes) and separates well —
  // exact-term matches land ~0.886+, distinct words ≤0.82, so keyRecall 0.85
  // catches real key hits. But key↔key and content↔key are both passage-embedded
  // and do NOT separate, so keyMerge/keyAutoLink stay high to prevent collapse.
  // memoryDedup 0.985: e5 packs distinct-but-similar facts ("A uses Postgres" vs
  // "B uses Mongo" ≈0.96) dangerously close to true paraphrases (≈0.99). Dedup
  // wrongly below 0.985 would silently supersede distinct memories → data loss.
  e5: { keyMerge: 0.97, memoryDedup: 0.985, keyAutoLink: 0.93, keyRecall: 0.85, contentRecall: 0.8 },
  minilm: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.6, keyRecall: 0.5, contentRecall: 0.45 },
};

let _warnedUncalibrated = false;
function localModelFamily(): "e5" | "bge" | "minilm" {
  const alias = LOCAL_MODEL_ALIASES[normalizeModelName(LOCAL_EMBEDDING_MODEL)];
  if (alias === "MLE5Large") return "e5";
  if (alias === "AllMiniLML6V2") return "minilm";
  if (alias === "BGEBaseENV15" || alias === "BGESmallENV15") return "bge";
  // Unknown model: thresholds are NOT calibrated for it. A wrong profile can
  // silently collapse (over-merge) or fragment (under-recall) the graph, so make
  // the miscalibration loud and point at the env override escape hatch.
  if (!_warnedUncalibrated) {
    _warnedUncalibrated = true;
    console.error(
      `[super-memory] WARNING: no calibrated threshold profile for ` +
        `LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}". Falling back to BGE ` +
        `thresholds — the graph may mis-cluster. Override per-threshold with ` +
        `SUPER_MEMORY_KEY_MERGE / _MEMORY_DEDUP / _KEY_AUTOLINK / _KEY_RECALL / _CONTENT_RECALL.`
    );
  }
  return "bge";
}

function envThreshold(name: string): number | undefined {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0 || n > 1) {
    console.error(`[super-memory] WARNING: ignoring ${name}="${raw}" (must be a number in [0,1]).`);
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
    keyMerge: envThreshold("SUPER_MEMORY_KEY_MERGE") ?? base.keyMerge,
    memoryDedup: envThreshold("SUPER_MEMORY_MEMORY_DEDUP") ?? base.memoryDedup,
    keyAutoLink: envThreshold("SUPER_MEMORY_KEY_AUTOLINK") ?? base.keyAutoLink,
    keyRecall: envThreshold("SUPER_MEMORY_KEY_RECALL") ?? base.keyRecall,
    contentRecall: envThreshold("SUPER_MEMORY_CONTENT_RECALL") ?? base.contentRecall,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _localModel: any = null;

async function getLocalModel() {
  if (!_localModel) {
    try {
      const { FlagEmbedding, EmbeddingModel } = await import("fastembed");
      const model = resolveLocalModel(LOCAL_EMBEDDING_MODEL, EmbeddingModel);
      _localModel = await FlagEmbedding.init({
        model: model as never,
        cacheDir: LOCAL_EMBEDDING_CACHE_DIR,
      });
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

export async function embedTextAsync(
  text: string,
  inputType: EmbeddingInputType = "passage"
): Promise<number[]> {
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
