// Cross-encoder reranker (opt-in via SUPER_MEMORY_RERANK=true).
//
// The retriever (recall) finds candidate memories; this re-scores the top of that list by
// JOINT (query, memory) relevance — fixing cases where the right memory is in the result
// set but ranked too low under fused BM25/dense scores. Unlike query decomposition (which
// needs an LLM and is the caller's job), a reranker is a MODEL, so it runs in-process.
//
// Loaded lazily on first use. If the model files or fastembed's native deps are missing,
// rerank silently no-ops and recall falls back to its fused ranking — so enabling the flag
// without the model never breaks recall.
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { KNOWN_MODELS, defaultModelDir, ensureModelFiles } from "./modelDownload.js";

let _testReranker:
  | ((query: string, texts: string[]) => number[] | Promise<number[]>)
  | null = null;
// Test seam: drive rerank ordering with crafted scores, no model load. Never set in prod.
export function __setTestReranker(fn: (q: string, t: string[]) => number[] | Promise<number[]>): void {
  _testReranker = fn;
}
export function __clearTestReranker(): void {
  _testReranker = null;
}

export function rerankEnabled(): boolean {
  return _testReranker !== null || process.env.SUPER_MEMORY_RERANK === "true";
}

const MODEL_DIR = process.env.SUPER_MEMORY_RERANK_MODEL_PATH ?? defaultModelDir("reranker");
const MAX_CHARS = 512; // truncate memory text fed to the cross-encoder

let _loaded = false;
let _failed = false;
let _ort: any = null;
let _session: any = null;
let _tok: any = null;

async function ensureLoaded(): Promise<boolean> {
  if (_loaded) return true;
  if (_failed) return false;
  try {
    // Auto-download the reranker model + (shared bge-m3) tokenizer if missing — one-time.
    if (!existsSync(join(MODEL_DIR, "model.onnx")) || !existsSync(join(MODEL_DIR, "tokenizer.json"))) {
      await ensureModelFiles(KNOWN_MODELS.reranker, MODEL_DIR);
    }
    // onnxruntime-node and the tokenizer are fastembed's native deps; resolve through it.
    const feReq = createRequire(createRequire(import.meta.url).resolve("fastembed"));
    _ort = feReq("onnxruntime-node");
    const tk = feReq("@anush008/tokenizers");
    _tok = tk.Tokenizer.fromFile(`${MODEL_DIR}/tokenizer.json`);
    _session = await _ort.InferenceSession.create(`${MODEL_DIR}/model.onnx`);
    _loaded = true;
    return true;
  } catch (err) {
    console.error(
      `[reranker] disabled — could not load model at ${MODEL_DIR}: ${err instanceof Error ? err.message : String(err)}`
    );
    _failed = true;
    return false;
  }
}

/**
 * Relevance score per candidate text for the query (higher = more relevant).
 * Returns null when reranking is unavailable (caller keeps its existing order).
 */
export async function rerankScores(query: string, texts: string[]): Promise<number[] | null> {
  if (_testReranker) return await _testReranker(query, texts);
  if (!(await ensureLoaded())) return null;
  const hasType = _session.inputNames.includes("token_type_ids");
  const scores: number[] = [];
  for (const text of texts) {
    const enc = await _tok.encode(query, (text ?? "").slice(0, MAX_CHARS));
    const ids = enc.getIds() as number[];
    const mask = enc.getAttentionMask() as number[];
    const n = ids.length;
    const feeds: Record<string, unknown> = {
      input_ids: new _ort.Tensor("int64", BigInt64Array.from(ids.map((x) => BigInt(x))), [1, n]),
      attention_mask: new _ort.Tensor("int64", BigInt64Array.from(mask.map((x) => BigInt(x))), [1, n]),
    };
    if (hasType) {
      const typeIds = enc.getTypeIds() as number[];
      feeds.token_type_ids = new _ort.Tensor("int64", BigInt64Array.from(typeIds.map((x) => BigInt(x))), [1, n]);
    }
    const out = await _session.run(feeds);
    scores.push(Number(out[_session.outputNames[0]].data[0]));
  }
  return scores;
}
