import { config } from "dotenv";
config();

import OpenAI from "openai";

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
export const OPENAI_EMBEDDING_MODEL =
  process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small";

const _DEFAULT_BACKEND = OPENAI_API_KEY ? "openai" : "local";
export const EMBEDDING_BACKEND =
  process.env.EMBEDDING_BACKEND ?? _DEFAULT_BACKEND;

const EMBED_RETRIES = 3;

let _openaiClient: OpenAI | null = null;

function getOpenAIClient(): OpenAI {
  if (!_openaiClient) {
    _openaiClient = new OpenAI({ apiKey: OPENAI_API_KEY });
  }
  return _openaiClient;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _localModel: any = null;

async function getLocalModel() {
  if (!_localModel) {
    try {
      const { FlagEmbedding, EmbeddingModel } = await import("fastembed");
      _localModel = await FlagEmbedding.init({
        model: EmbeddingModel.BGEBaseENV15,
      });
    } catch {
      throw new Error(
        "fastembed is not installed.\n" +
          "Install with: npm install fastembed\n" +
          "Or set OPENAI_API_KEY to use OpenAI embeddings."
      );
    }
  }
  return _localModel;
}

async function embedLocal(text: string): Promise<number[]> {
  const model = await getLocalModel();
  const gen = model.embed([text]);
  for await (const batch of gen) {
    return Array.from(batch[0]) as number[];
  }
  throw new Error("fastembed returned no embeddings");
}

export async function embedTextAsync(text: string): Promise<number[]> {
  if (EMBEDDING_BACKEND === "local") {
    return embedLocal(text);
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
