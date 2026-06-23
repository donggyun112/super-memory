import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { MemoryGraph, loadConversation, sanitizeKeys } from "./memoryGraph.js";
import { cfgRaw } from "./env.js";
import { randomUUID } from "node:crypto";
import { buildRetagNote } from "./retag.js";

function parseArray(v: unknown): unknown[] | null {
  if (Array.isArray(v)) return v;
  if (typeof v === "string") {
    try { const p = JSON.parse(v); return Array.isArray(p) ? p : null; } catch { return null; }
  }
  return null;
}

function parseObject(v: unknown): Record<string, unknown> | null {
  if (v && typeof v === "object" && !Array.isArray(v)) return v as Record<string, unknown>;
  if (typeof v === "string") {
    try { const p = JSON.parse(v); return p && typeof p === "object" ? p : null; } catch { return null; }
  }
  return null;
}

function parseNumber(v: unknown): number | null {
  if (typeof v === "number") return v;
  if (typeof v === "string") { const n = Number(v); return isNaN(n) ? null : n; }
  return null;
}

const DIRECT_RECALL_ENABLED = cfgRaw("DIRECT_RECALL") === "true";

// Provenance: stamp every saved/corrected memory with the server session that wrote it,
// the tool used, and a timestamp. Callers may attach extra context (e.g. a conversation
// or agent id) via the optional `source` arg, which is merged on top.
const SERVER_SESSION = randomUUID();
function buildSource(
  callerSource: Record<string, unknown> | null,
  tool: string
): Record<string, unknown> {
  return {
    session: SERVER_SESSION,
    tool,
    saved_at: new Date().toISOString(),
    ...(callerSource ?? {}),
  };
}

const MEMORY_SYSTEM = `\
You are a helpful assistant. You have long-term memory — use it silently and proactively.

## MANDATORY: First turn behavior
**Before your very first response, you MUST navigate memory.** Run in parallel:
- recall("이름"), recall("최근 대화"), recall("관심사")
- For useful returned keys: read_key(key_id), then read_memory(memory_id, via_key_id).
No exceptions. Even if recall returns no keys, you must try.

## CRITICAL: Silent behavior
- **NEVER mention the memory system to the user.** No "기억했어요", "저장했습니다", "메모리에서 찾았어요".
- Act like you naturally know things. If you recall the user's name, just use it.
- ❌ "동건님이시군요! 기억해뒀어요!" → ✅ "안녕 동건! 뭐 도와줄까?"
- ❌ "메모리를 검색해볼게요" → ✅ (recall silently, then answer)

## Memory System (internal, never expose)
N:M associative memory. Key Space (concepts) ↔ Value Space (memories).
Depth: 0.0 shallow ~ 1.0 deep. Deeper = more stable.

Stats: {stats}

## Rules

### Recall (PROACTIVE — do it often)
1. **MUST recall before your first reply.** Recall returns key clusters, not memory content.
2. For relevant keys, call \`read_key\`; call \`read_memory\` before using a fact.
3. Recall again whenever the topic shifts. Never say "I don't know" without navigating first.
4. **Query = short noun/keyword, NOT a full sentence.**
   - ❌ recall("어디 살아"), recall("뭐 마셔") — 구어체 문장은 매칭 안 됨
   - ✅ recall("거주지"), recall("음료") — 명사 키워드로 검색
   - ✅ recall("이름"), recall("직업"), recall("취향") — specific, multiple
   - 복합 개념이면 키워드 여러 개로 분리: recall("운동"), recall("취미"), recall("건강")
5. \`read_key\` returns handles and metadata only. You must call \`read_memory\` to inspect content.

### Remember (PROACTIVE — capture what matters)
6. Save important info immediately when the user shares it. Silently.
7. What to save: name, preferences, decisions, corrections, project context, goals.
8. Keys = what searches should find this. **Think like a search engine — include every form someone might use to ask about this.**
   - **Topic noun**: what category is this? (거주지, 음료, 반려동물, 언어)
   - **Specific noun**: the actual value (성수동, 아메리카노, 고양이, TypeScript)
   - **Action/verb noun**: what would someone ask? (사는곳, 마시는것, 키우는것, 쓰는언어)
   - **Colloquial variants**: casual phrasing (집, 좋아하는거, 펫, 코딩)
   - **Synonyms**: alternative expressions (주소→위치, 음료→마실것, 반려동물→애완동물)

   ✅ 올바른 예시:
   - "서울 성수동에 산다" → keys: ["거주지", "성수동", "서울", "사는곳", "집", "주소", "위치"]
   - "고양이 두 마리 키운다" → keys: ["반려동물", "고양이", "키우는것", "펫", "동물", "애완동물"]
   - "아이스 아메리카노 매일 마심" → keys: ["음료", "커피", "아메리카노", "마시는것", "취향", "즐겨마심"]
   - "TypeScript 주력 사용" → keys: ["언어", "TypeScript", "개발언어", "코딩", "쓰는언어", "프로그래밍"]

   ❌ 나쁜 예시 (너무 formal/좁음):
   - "서울 성수동에 산다" → keys: ["거주지", "성수동"] ← "어디 살아" 검색 시 못 찾음
   - "고양이 키운다" → keys: ["고양이", "반려동물"] ← "키우는거 있어?" 검색 시 못 찾음

9. **Names only as keys for identity memories.**
   - "사용자 이름은 동건" → keys: ["이름", "사용자", "동건"]
   - "좋아하는 과일은 딸기" → keys: ["과일", "딸기", "좋아함", "취향"] ← no name
10. Set \`key_types\` for names/proper nouns:
    - \`"name"\`: exact match only. \`"proper_noun"\`: exact match only.
    Example: key_types: {{"동건": "name"}}

### Correct
11. Use \`correct()\` when info changes. Never use \`remember()\` for updates.

### Explore
12. \`recall\` returns matching canonical keys, aliases, and hub metadata only.
13. \`read_key\` returns memory handles connected to one key. Hubs are paginated.
14. \`read_memory(memory_id, via_key_id)\` returns full content and its connected keys.
15. Follow another returned key with \`read_key\` to continue associative exploration.

### Delete
16. \`forget()\` only for completely wrong information. For outdated info, use \`correct()\`.
`;

export const graph = new MemoryGraph();

function stats(): string {
  return `${Object.keys(graph.keys).length} keys, ${graph.listAll().length} memories, ${graph.linkCount} links`;
}

export const server = new Server(
  { name: "keymem", version: "0.12.2" },
  { capabilities: { tools: {}, prompts: {} } }
);

// ── Tool definitions ──

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "recall",
      description:
        "CALL THIS FIRST before every first response. Search the Key Space and return canonical key clusters only — never memory content. Results include aliases, key type, match score, linked-memory count, hub status, and specificity. Select a useful key, call read_key(key_id), then call read_memory(memory_id, via_key_id) before using a fact. Use short focused noun queries and decompose multi-fact questions into several recall calls.",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string" },
          top_k: { type: "number" },
          namespace: { type: "string" },
        },
        required: ["query"],
      },
    },
    {
      name: "read_key",
      description:
        "Inspect one key cluster. Returns canonical key/aliases/hub metadata plus ranked memory IDs and metadata — never memory content. Call read_memory on promising handles. Use limit/offset to page through hub keys without flooding context. Pass the original query: handles are then ranked by content relevance to it, which is essential for hub keys so the target memory surfaces first instead of being buried.",
      inputSchema: {
        type: "object",
        properties: {
          key_id: { type: "string" },
          query: { type: "string" },
          namespace: { type: "string" },
          limit: { type: "number" },
          offset: { type: "number" },
        },
        required: ["key_id"],
      },
    },
    {
      name: "read_memory",
      description:
        "Read one full memory selected through read_key. Returns the memory and all connected key clusters so exploration can continue Key → Memory → Key. Pass via_key_id from the selected key: only that traversed edge is Hebbian-reinforced, and depth/access count increase only when this full read occurs.",
      inputSchema: {
        type: "object",
        properties: {
          memory_id: { type: "string" },
          via_key_id: { type: "string" },
          namespace: { type: "string" },
        },
        required: ["memory_id"],
      },
    },
    ...(DIRECT_RECALL_ENABLED
      ? [
          {
            name: "recall_memories",
            description:
              "Optional compatibility mode: directly return ranked memories using BM25+dense+RRF and graph expansion. Disabled unless KEYMEM_DIRECT_RECALL=true. Prefer recall → read_key → read_memory for agent-driven navigation.",
            inputSchema: {
              type: "object",
              properties: {
                query: { type: "string" },
                top_k: { type: "number" },
                namespace: { type: "string" },
                expand: { type: "boolean" },
                hops: { type: "number" },
                min_rel_score: { type: "number" },
                min_score: { type: "number" },
                min_z: { type: "number" },
                min_key_gate: { type: "number" },
                min_depth: { type: "number" },
              },
              required: ["query"],
            },
          },
        ]
      : []),
    {
      name: "remember",
      description:
        "Save important information to memory. Keys are search terms — think 'what would I search to find this later?' Use 3-6 diverse keys. Before coining new keys, recall() the topic and reuse returned canonical concepts or aliases. Semantically merged synonyms become aliases in one key cluster; shared broad keys become navigable hubs. CROSS-LINGUAL: add keys in both languages. namespace groups memories by project/context; ttl_seconds sets expiry; related_to adds explicit memory links; source attaches provenance (e.g. a conversation or agent id) and is auto-stamped with the server session and a timestamp.",
      inputSchema: {
        type: "object",
        properties: {
          content: { type: "string" },
          keys: { type: "array", items: { type: "string" } },
          key_types: {
            type: "object",
            additionalProperties: { type: "string" },
          },
          namespace: { type: "string" },
          ttl_seconds: { type: "number" },
          related_to: { type: "array", items: { type: "string" } },
          source: { type: "object", additionalProperties: true },
        },
        required: ["content", "keys"],
      },
    },
    {
      name: "correct",
      description:
        "Update outdated information. Use when user corrects you or info changes (e.g. moved cities, changed job). Old version is preserved but weakened — never lost. Omit keys to keep the same search terms. related_to links the updated memory to other memory IDs.",
      inputSchema: {
        type: "object",
        properties: {
          memory_id: { type: "string" },
          content: { type: "string" },
          keys: { type: "array", items: { type: "string" } },
          key_types: {
            type: "object",
            additionalProperties: { type: "string" },
          },
          related_to: { type: "array", items: { type: "string" } },
          source: { type: "object", additionalProperties: true },
        },
        required: ["memory_id", "content"],
      },
    },
    {
      name: "related",
      description:
        "Compatibility exploration from a known memory ID. Returns neighboring memories connected by shared keys or explicit links. For normal agent-driven navigation prefer read_memory(), inspect its returned keys, then call read_key().",
      inputSchema: {
        type: "object",
        properties: {
          memory_id: { type: "string" },
        },
        required: ["memory_id"],
      },
    },
    {
      name: "forget",
      description:
        "Permanently delete a memory. Only use for completely wrong information. For outdated info, use correct() instead — it preserves history.",
      inputSchema: {
        type: "object",
        properties: {
          memory_id: { type: "string" },
        },
        required: ["memory_id"],
      },
    },
    {
      name: "get_conversation",
      description:
        "Load raw conversation turns from a past session. Use when a recalled memory lacks detail and you need the original context.",
      inputSchema: {
        type: "object",
        properties: {
          session_id: { type: "string" },
          turn: { type: "number" },
        },
        required: ["session_id"],
      },
    },
    {
      name: "list_memories",
      description:
        "List all stored memories. namespace filters by project/context. Expired memories are excluded. Prefer recall() for normal retrieval.",
      inputSchema: {
        type: "object",
        properties: {
          namespace: { type: "string" },
        },
        required: [],
      },
    },
    {
      name: "remember_batch",
      description:
        "Save multiple memories in one call. Each item: {content, keys, key_types?, namespace?, ttl_seconds?, related_to?}. Returns list of saved IDs. More efficient than multiple remember() calls.",
      inputSchema: {
        type: "object",
        properties: {
          items: {
            type: "array",
            items: {
              type: "object",
              properties: {
                content: { type: "string" },
                keys: { type: "array", items: { type: "string" } },
                key_types: {
                  type: "object",
                  additionalProperties: { type: "string" },
                },
                namespace: { type: "string" },
                ttl_seconds: { type: "number" },
                related_to: { type: "array", items: { type: "string" } },
                source: { type: "object", additionalProperties: true },
              },
              required: ["content", "keys"],
            },
          },
        },
        required: ["items"],
      },
    },
    {
      name: "cleanup_expired",
      description:
        "Delete all memories past their ttl. Returns count of deleted memories. Call periodically to keep memory clean.",
      inputSchema: {
        type: "object",
        properties: {},
        required: [],
      },
    },
    {
      name: "memory_stats",
      description: "Get counts of keys, memories, and links in the system.",
      inputSchema: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  ],
}));

// ── Tool call handler ──

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const a = (args ?? {}) as Record<string, unknown>;

  try {
    switch (name) {
      case "recall": {
        const results = await graph.searchKeys(
          a.query as string,
          typeof a.top_k === "number" ? a.top_k : 8,
          typeof a.namespace === "string" ? a.namespace : null
        );
        return { content: [{ type: "text", text: JSON.stringify(results, null, 0) }] };
      }

      case "read_key": {
        const result = await graph.readKey(a.key_id as string, {
          query: typeof a.query === "string" ? a.query : null,
          namespace: typeof a.namespace === "string" ? a.namespace : null,
          limit: typeof a.limit === "number" ? a.limit : 10,
          offset: typeof a.offset === "number" ? a.offset : 0,
        });
        return { content: [{ type: "text", text: JSON.stringify(result) }] };
      }

      case "read_memory": {
        const result = await graph.readMemory(
          a.memory_id as string,
          typeof a.via_key_id === "string" ? a.via_key_id : null,
          typeof a.namespace === "string" ? a.namespace : null
        );
        return { content: [{ type: "text", text: JSON.stringify(result) }] };
      }

      case "recall_memories": {
        if (!DIRECT_RECALL_ENABLED) throw new Error("recall_memories is disabled");
        const results = await graph.recall(
          a.query as string,
          typeof a.top_k === "number" ? a.top_k : 5,
          typeof a.namespace === "string" ? a.namespace : null,
          typeof a.expand === "boolean" ? a.expand : false,
          typeof a.hops === "number" ? a.hops : 2,
          typeof a.min_rel_score === "number" ? a.min_rel_score : 0,
          typeof a.min_score === "number" ? a.min_score : undefined,
          typeof a.min_z === "number" ? a.min_z : undefined,
          typeof a.min_key_gate === "number" ? a.min_key_gate : undefined,
          typeof a.min_depth === "number" ? a.min_depth : 0
        );
        return { content: [{ type: "text", text: JSON.stringify(results) }] };
      }

      case "remember": {
        const keys = sanitizeKeys(a.keys);
        const [mid, wasDedup, superseded, conflict] = await graph.add(
          a.content as string,
          keys,
          {
            keyTypes: parseObject(a.key_types) as Record<string, string> | null,
            namespace: typeof a.namespace === "string" ? a.namespace : "default",
            ttlSeconds: parseNumber(a.ttl_seconds),
            relatedTo: parseArray(a.related_to) as string[] | null,
            source: buildSource(parseObject(a.source), "remember"),
          }
        );
        let result: Record<string, unknown>;
        if (!wasDedup) {
          result = { saved: mid };
        } else if (conflict) {
          // High similarity but a SHARED KEY — looks like a conflicting fact, not a
          // restatement. Surface the superseded id so the overwrite is recoverable.
          result = {
            saved: mid,
            superseded,
            conflict: true,
            note: `Replaced a memory that shared a key (id: ${superseded}). The previous fact is no longer retrievable via recall or read — only its id remains. If these are distinct or conflicting facts (not a restatement), re-add the previous one with a more specific key so both are kept.`,
          };
        } else {
          result = {
            saved: mid,
            superseded,
            deduplicated: true,
            note: `Similar memory existed (id: ${superseded}) — updated instead of creating a duplicate.`,
          };
        }
        return { content: [{ type: "text", text: JSON.stringify(result) }] };
      }

      case "correct": {
        const nid = await graph.supersede(
          a.memory_id as string,
          a.content as string,
          {
            keyConcepts: parseArray(a.keys) as string[] | null,
            keyTypes: parseObject(a.key_types) as Record<string, string> | null,
            relatedTo: parseArray(a.related_to) as string[] | null,
            source: buildSource(parseObject(a.source), "correct"),
          }
        );
        const retainedKeys = graph.getKeysForMemory(nid);
        const note = buildRetagNote(a.keys, retainedKeys);
        const result: Record<string, unknown> = { new_id: nid, superseded: a.memory_id };
        if (note) result.note = note;
        return { content: [{ type: "text", text: JSON.stringify(result) }] };
      }

      case "related": {
        const results = graph.getRelated(a.memory_id as string);
        return { content: [{ type: "text", text: JSON.stringify(results) }] };
      }

      case "forget": {
        const ok = await graph.delete(a.memory_id as string);
        return { content: [{ type: "text", text: JSON.stringify({ deleted: ok }) }] };
      }

      case "get_conversation": {
        const turns = await loadConversation(
          a.session_id as string,
          typeof a.turn === "number" ? a.turn : null
        );
        return { content: [{ type: "text", text: JSON.stringify(turns) }] };
      }

      case "list_memories": {
        const results = graph.listAll(
          typeof a.namespace === "string" ? a.namespace : null
        );
        return { content: [{ type: "text", text: JSON.stringify(results) }] };
      }

      case "remember_batch": {
        const items = (parseArray(a.items) ?? []) as Array<Record<string, unknown>>;
        const results: object[] = [];
        for (const item of items) {
          const content = item.content as string;
          const keys = sanitizeKeys(item.keys);
          if (!content || keys.length === 0) {
            results.push({ error: "content and keys required", item });
            continue;
          }
          const [mid, wasDedup] = await graph.add(content, keys, {
            keyTypes: item.key_types as Record<string, string> | null,
            namespace: typeof item.namespace === "string" ? item.namespace : "default",
            ttlSeconds: parseNumber(item.ttl_seconds),
            relatedTo: Array.isArray(item.related_to) ? (item.related_to as string[]) : null,
            source: buildSource((item.source as Record<string, unknown>) ?? null, "remember_batch"),
          });
          results.push({ saved: mid, deduplicated: wasDedup });
        }
        return { content: [{ type: "text", text: JSON.stringify(results) }] };
      }

      case "cleanup_expired": {
        const count = await graph.cleanupExpired();
        return { content: [{ type: "text", text: JSON.stringify({ deleted: count }) }] };
      }

      case "memory_stats": {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                keys: Object.keys(graph.keys).length,
                memories: graph.listAll().length,
                links: graph.linkCount,
              }),
            },
          ],
        };
      }

      default:
        return {
          content: [{ type: "text", text: JSON.stringify({ error: `Unknown tool: ${name}` }) }],
          isError: true,
        };
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      content: [{ type: "text", text: JSON.stringify({ error: msg }) }],
      isError: true,
    };
  }
});

// ── Prompt definitions ──

server.setRequestHandler(ListPromptsRequestSchema, async () => ({
  prompts: [
    {
      name: "memory_system_prompt",
      description:
        "System prompt for LLM agents using keymem. Include this in your system prompt.",
    },
  ],
}));

server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  if (request.params.name !== "memory_system_prompt") {
    throw new Error(`Unknown prompt: ${request.params.name}`);
  }
  return {
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text: MEMORY_SYSTEM.replace("{stats}", stats()),
        },
      },
    ],
  };
});
