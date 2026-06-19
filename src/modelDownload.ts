// Auto-download for LOCAL custom ONNX models (bge-m3 embedder, bge-reranker-v2-m3).
//
// Scope: this is ONLY the local-custom adapter's concern. Online-API backends (OpenAI)
// and fastembed built-ins (e5, bge-base, …) never reach here, and an existing local model
// directory is used as-is — so enabling this changes nothing for those paths (backward
// compatible). It only fills in a model that would otherwise be missing.
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { existsSync, mkdirSync, createWriteStream, renameSync, statSync } from "node:fs";
import { Readable } from "node:stream";

export interface DownloadSpec {
  repo: string; // default source for files that don't override it
  files: { repo?: string; src: string; dest: string }[];
}

// Known auto-downloadable models. The ONNX graphs come from onnx-community (int8 quantized,
// ~570MB vs ~2.2GB fp32), but the tokenizer/config come from BAAI/bge-m3 — onnx-community's
// tokenizer.json uses a serialization fastembed's bundled tokenizer can't parse, while
// BAAI's loads cleanly. bge-m3 and bge-reranker-v2-m3 share the same XLM-RoBERTa tokenizer.
export const KNOWN_MODELS: Record<string, DownloadSpec> = {
  "bge-m3": {
    repo: "BAAI/bge-m3",
    files: [
      { repo: "onnx-community/bge-m3-ONNX", src: "onnx/model_quantized.onnx", dest: "model.onnx" },
      { src: "tokenizer.json", dest: "tokenizer.json" },
      { src: "tokenizer_config.json", dest: "tokenizer_config.json" },
      { src: "config.json", dest: "config.json" },
      { src: "special_tokens_map.json", dest: "special_tokens_map.json" }, // fastembed custom-model requires this
    ],
  },
  reranker: {
    repo: "BAAI/bge-m3", // tokenizer source (shared XLM-RoBERTa)
    files: [
      { repo: "onnx-community/bge-reranker-v2-m3-ONNX", src: "onnx/model_quantized.onnx", dest: "model.onnx" },
      { src: "tokenizer.json", dest: "tokenizer.json" },
      { src: "tokenizer_config.json", dest: "tokenizer_config.json" },
    ],
  },
};

export function defaultModelDir(name: string): string {
  return join(homedir(), ".super-memory", "models", name);
}

export type Fetcher = (url: string, destPath: string) => Promise<void>;

// Stream a URL to a file atomically (.tmp → rename) so a crash never leaves a half file.
async function httpDownload(url: string, destPath: string): Promise<void> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`download failed: HTTP ${res.status} for ${url}`);
  mkdirSync(dirname(destPath), { recursive: true });
  const tmp = `${destPath}.tmp`;
  await new Promise<void>((resolve, reject) => {
    const ws = createWriteStream(tmp);
    Readable.fromWeb(res.body as never).pipe(ws).on("finish", () => resolve()).on("error", reject);
  });
  renameSync(tmp, destPath);
}

function present(path: string): boolean {
  try {
    return existsSync(path) && statSync(path).size > 0;
  } catch {
    return false;
  }
}

/**
 * Ensure every file of a spec exists in `dir`, downloading only the missing ones.
 * `fetcher` is injectable for tests (no network). Returns the directory.
 */
export async function ensureModelFiles(spec: DownloadSpec, dir: string, fetcher: Fetcher = httpDownload): Promise<string> {
  for (const f of spec.files) {
    const dest = join(dir, f.dest);
    if (present(dest)) continue;
    const repo = f.repo ?? spec.repo;
    const url = `https://huggingface.co/${repo}/resolve/main/${f.src}`;
    console.error(`[super-memory] model file missing — downloading ${f.dest} from ${repo} (one-time)…`);
    await fetcher(url, dest);
  }
  return dir;
}
