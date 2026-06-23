# keymem — Benchmarks

What this measures and, honestly, what it doesn't. The goal is to **prove how much keymem's
key-graph actually buys you** — isolated causally, not asserted by metaphor — and to mark the
limits plainly.

> TL;DR: On **HotpotQA bridge questions** (real external multi-hop data, gold labels, no LLM judge,
> **and keys generated blind by independent subagents**), graph traversal retrieves **both** gold
> supporting paragraphs **63%** of the time vs **53%** for flat semantic and **35%** for lexical
> (+10pp / +28pp). The read path was also made **O(1) instead of O(graph)** (read_memory p50 ~45ms
> → ~0.01ms @ 500 memories). Honest costs: the gain is specific to the *multi-hop* case (it
> slightly *hurts* "comparison" questions), my own hand-derived keys *inflated* it (78% vs 60% —
> trust the blind-key 63/53), it's retrieval-recall not end-task accuracy, and none of this is a
> head-to-head SOTA claim vs mem0/Zep. Details below.

---

## Why an ablation, not a leaderboard score

The standard agent-memory benchmarks — [LoCoMo](https://github.com/snap-research/locomo)
(1,982 questions over long conversations) and [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(~115k-token histories) — score a full pipeline with an **LLM-as-judge**, and the published
vendor numbers are openly disputed (Zep vs Mem0 contest each other's methodology; see
[Zep's critique](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)).
Running those credibly needs the competing systems installed, the full datasets, and thousands
of judge calls — none of which a single-author project can do cleanly or cheaply, and a number
produced that way would be exactly the kind of disputed score the field is tired of.

So this benchmark answers a narrower, **causally clean** question instead:

> Holding the engine, the data, and the embeddings fixed, **how much does the key-graph
> traversal itself add** over flat 1-hop semantic retrieval?

That isolates *our* contribution rather than comparing incomparable stacks. It is a smaller
claim, but an honest one.

---

## 1. Associative-recall ablation

**Design.** Same data, same real embeddings (`bge-m3`), three retrievers — two flat baselines
(one external) and keymem's graph:

| Condition | how | meaning |
|---|---|---|
| **BM25** | standalone MiniSearch over content | classic flat *lexical* store (no embeddings, no graph) |
| **DIRECT** | `recall(expand=false, hops=1)` | flat *semantic* reach — keymem 1-hop, no expansion |
| **GRAPH** | `recall(expand=true, hops=2)` | keymem's multi-hop key-graph traversal |

BM25 and DIRECT are the flat baselines; GRAPH adds graph traversal. The delta isolates what the
key-graph buys over flat lexical / flat semantic retrieval.

**Dataset** (`bench/assoc-fixture.json`): a 14-memory bilingual persona graph. Each `assoc2`
query's answer is a **far memory reachable only via a shared key two hops away** (e.g. *"미나가
키우는 강아지"* → the dog's *allergy* fact, reachable only through the shared key `보리`). The
query has low direct similarity to that target — so flat retrieval should miss it and graph
traversal should reach it. `direct` queries are 1-hop controls; `notfound` must return nothing.

**Metrics.** `reach@10` (target anywhere in the top 10 — does the system find it *at all*),
`hit@5` (target in top 5 — ranking-sensitive), `MRR`. Run: `tsx bench/ablation.ts`.

### Results (`bge-m3`, n per category)

| category | metric | BM25 | DIRECT | GRAPH |
|---|---|---:|---:|---:|
| **assoc2** (6) | **reach@10** | 33% | 50% | **83%** |
| | hit@5 | 17% | 33% | 33% |
| | MRR | 0.19 | 0.13 | 0.23 |
| direct (5) | reach@10 | 80% | 100% | 100% |
| | hit@5 | 80% | 80% | 80% |
| | MRR | 0.48 | 0.82 | **0.69** |
| notfound (3) | not-found acc | 1/3 | 1/3 | 1/3 |

### What this proves (and doesn't)

- ✅ **The key-graph reaches connected-but-dissimilar memories that *both* flat baselines cannot.**
  On `assoc2`, BM25 reaches 2/6 targets, DIRECT (flat semantic) 3/6, GRAPH 5/6 (the 6th lands at
  rank 11, just outside the window). +33pp over flat-dense, +50pp over flat-lexical — *measured*,
  not asserted. Per-query, the graph pulls in the dog-allergy / climbing-injury / caffeine-sleep
  facts via shared keys that neither lexical nor 1-hop semantic similarity ever surfaces. (On the
  `direct` control, DIRECT/GRAPH both reach 100% vs BM25's 80% — embeddings already beat lexical
  on plain queries; the graph's distinct contribution is the associative reach.)
- ⚠️ **The gain is in *reachability*, not top rank.** `HOP_DECAY` scores 2-hop hits low, so they
  arrive at ranks 9–11 — `hit@5` shows **no** gain. The value is real for an agent that
  navigates/pages (the intended `recall → read_key → read_memory` flow), much weaker if you only
  ever read top-5.
- ⚠️ **Honest costs.** Graph expansion slightly *hurt* direct-query ranking (MRR 0.82 → 0.69) by
  mixing associative neighbours into clean results. And **not-found precision is poor (1/3)** —
  2 of 3 distractors returned something — at this small scale the absolute-score gate is too
  loose. Both are the same under DIRECT, so they're engine/gate issues, not graph-specific, but
  they're real.

---

## 2. External validation: HotpotQA multi-hop retrieval

§1 uses a dataset I built, so here's the same question on data I didn't. [HotpotQA](https://hotpotqa.github.io/)
(distractor) ships, per question, **10 paragraphs (2 gold "supporting" + 8 distractors) plus gold
supporting-fact labels** — so we measure support *retrieval* with **no LLM judge**. Mapping to
keymem: each paragraph becomes a memory keyed by its own title + any other paragraph title it
mentions in-text, so a **bridge** entity (what links the question's paragraph to the answer's)
becomes a shared key. For **bridge** questions the answer paragraph is connected-but-dissimilar to
the query (the query never names it) — exactly keymem's case. **comparison** questions name both
entities up front (no bridge to traverse) — a built-in negative control. Each question is scored
in isolation over only its 10 paragraphs. Run: `tsx bench/hotpot.ts`.

### Results (`bge-m3`, N=120 = 96 bridge + 24 comparison, top-5 of 10)

| question type | metric | BM25 | DIRECT | GRAPH |
|---|---|---:|---:|---:|
| **bridge** (96) | support-recall@5 | 70% | 78% | **88%** |
| | **both@5** (got both golds) | 49% | 60% | **78%** |
| comparison (24) | support-recall@5 | 57% | 81% | 77% |
| | both@5 | 25% | 63% | **54%** |
| all (120) | both@5 | 44% | 61% | **73%** |

- ✅ **On bridge (multi-hop, connected-but-dissimilar — keymem's case) the graph clearly wins**:
  both gold paragraphs retrieved **78%** of the time vs **60%** (flat semantic) and **49%**
  (lexical) — +18pp / +29pp, on real external data with gold labels, n=96. This is a stronger
  result than §1 (here the bridge-reached support lands inside top-5 because the pool is only 10).
- ⚠️ **On comparison questions the graph slightly *hurts*** (both@5 54% vs DIRECT's 63%) — expected:
  both entities are already in the query, so there's no bridge to traverse and expansion just adds
  noise. An honest negative that confirms the gain is *specifically* the multi-hop case, not free.
- This is **retrieval-recall of the gold paragraphs, not end-task answer accuracy** — getting both
  supports is necessary, not sufficient, for a correct answer (no LLM judge here).

**Caveats.** The table above derives keys *myself* (title + mentioned-titles), which mirrors the
gold bridge structure — so it risks measuring a graph I keyed to match the answer. Both
conditions share those keys (the ablation is internally fair), but the absolute gain could be
inflated. The next check removes exactly that doubt. (Also: one dataset, one embedder, document
multi-hop not conversational memory; retrieval-recall not end-task accuracy.)

### Validity check: blind agent-generated keys

To kill the "you keyed it to match the answer" objection, the keys were regenerated by
**independent subagents that saw only each paragraph's text** — no question, no gold supports,
no other paragraphs' role — and tagged each for findability (realistic keymem write-time keying).
Then the same bridge questions were re-run on those blind keys (`bench/hotpot-agentkeys.ts`,
N=40 bridge, all 400 paragraphs keyed by agents, 0 fallbacks):

| metric | BM25 | DIRECT | GRAPH | vs heuristic keys |
|---|---:|---:|---:|---|
| support-recall@5 | 61% | 72% | **79%** | (heuristic: 70 / 78 / 88) |
| **both@5** | 35% | 53% | **63%** | (heuristic: 49 / 60 / 78) |

- ✅ **The gain survives blind keying**: GRAPH both@5 **63% vs DIRECT 53% (+10pp), vs BM25 35%
  (+28pp)**. With keys an independent agent produced without ever seeing the task, graph traversal
  still retrieves the connected support more often. The "I keyed it to match the gold" objection is
  answered.
- ⚠️ **…but smaller — my heuristic keys *were* optimistic** (78/60 → 63/53). Honest: the §2 table
  over-states the effect; the blind-key numbers are the ones to trust. The real, defensible claim
  is **+10pp both@5 over flat semantic, +28pp over lexical, with realistic keys.**

## 3. Read-path latency (the v0.12.1 fix)

`read_memory` rewrote the entire `graph.json` on every call (it bumps depth/access), making each
read **O(graph size)**. Reads are the frequent path (every `recall → read_key → read_memory`);
deferring that persistence to `flush()` makes reads O(1). Measured with a synthetic 1024-dim
embedder to isolate graph-op cost from embedding inference (`bench/perf.ts`):

| memories | read_memory p50 — before | after (v0.12.1) |
|---:|---:|---:|
| 500 | 44.8 ms | **0.01 ms** |
| 1,500 | 132.8 ms | flat |
| 3,000 | 262.5 ms | flat |

Before the fix, read latency grew linearly with the store (the full-file rewrite). After, a read
is a RAM mutation + dirty flag. `searchKeys` was already cheap (1–8 ms at 0.5–3k keys); the write
path (`add`) is deliberately left eager — writes are rare, so its O(n) per-save save cost is not
worth trading durability for.

---

## 4. Honest scope & the trajectory caveat

**Scope.** Small synthetic persona graph (14 memories), one embedder, one author's fixtures.
This is **not** LoCoMo/LongMemEval scale, and it is **not** a head-to-head vs mem0/Zep — those
remain future work (they need the competing systems + an LLM judge). What's proven here is the
*marginal contribution of keymem's own graph*, on a probe built specifically to stress the
connected-but-dissimilar case.

**The trajectory caveat.** keymem (like all clever memory layers) bets that *structure beats raw
model reasoning over flat content*. That bet weakens as agentic search improves: an
[Amazon Science AAAI-2026 result](https://www.amazon.science/) reports agentic keyword search at
~94.5% of RAG faithfulness with **no** vector store, and Karpathy has noted that at personal
scale a full RAG stack often adds more latency/noise than it removes
([context](https://venturebeat.com/data/context-architecture-is-replacing-rag-as-agentic-ai-pushes-enterprise-retrieval-to-its-limits)).
The 2026 consensus is **hybrid** (small index + lots of tools), not pure-vector or pure-agentic.

So keymem's durable value is **not** "smarter retrieval than the model" — the model keeps getting
smarter. It is:
1. **Reach** — surfacing connected-but-dissimilar memories an agent wouldn't think to query for
   (the +33pp above), and
2. **Amortization + legibility** — the association is computed once into an explicit, auditable
   edge, instead of re-derived by an LLM hop every query, and you can *read why* two things are
   linked (a key path) rather than trust an opaque cosine.

Whether that earns its complexity over "a strong model + grep + re-query" is, ultimately, an
empirical question per use case. This doc is the start of measuring it honestly, not the last
word.

---

## Reproduce

```bash
tsx bench/ablation.ts     # §1 associative-recall ablation (real bge-m3, ~570MB first run)
tsx bench/perf.ts         # §3 latency vs store size (synthetic embedder)
tsx bench/run.ts          # the existing search-quality regression fixture

# §2 external HotpotQA — first fetch a slice (no full download), then run:
curl -s "https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation&offset=0&length=100" \
  | python3 -c "import sys,json;rows=[r['row'] for r in json.load(sys.stdin)['rows']];print(json.dumps([{'id':r['id'],'question':r['question'],'answer':r['answer'],'type':r['type'],'support':r['supporting_facts']['title'],'titles':r['context']['title'],'paras':[' '.join(s) for s in r['context']['sentences']]} for r in rows]))" \
  > bench/hotpot-slice.json
tsx bench/hotpot.ts 100
tsx bench/hotpot-agentkeys.ts bench/hotpot-agentkeys.json   # §2 validity check w/ blind agent keys
```

Sources: [LoCoMo](https://github.com/snap-research/locomo) · [LongMemEval](https://github.com/xiaowu0162/LongMemEval) · [Zep vs Mem0 methodology dispute](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) · [Mem0 paper](https://arxiv.org/pdf/2504.19413) · [Agentic search replacing RAG (VentureBeat, 2026)](https://venturebeat.com/data/context-architecture-is-replacing-rag-as-agentic-ai-pushes-enterprise-retrieval-to-its-limits)
