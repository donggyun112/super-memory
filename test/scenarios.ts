// End-to-end behavioral suite for the long-term memory + association system.
// Runs against the real local e5 backend. One model load, scenarios isolated by namespace.
// Run: tsx test/scenarios.ts
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
const dataDir = await mkdtemp(join(tmpdir(), "sm-scn-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { LOCAL_EMBEDDING_MODEL, getThresholdProfile } = await import("../src/embedding.ts");

const g = new MemoryGraph();
await g.load();

type R = { id: string; content: string; hop: number; score: number; keys: string[]; matched_via: string[]; depth: number; access_count: number };
const recall = (q: string, ns?: string, expand = false, k = 5) =>
  g.recall(q, k, ns ?? null, expand) as Promise<R[]>;
const remember = (c: string, keys: string[], ns: string, opts: any = {}) =>
  g.add(c, keys, { namespace: ns, ...opts });

let pass = 0, fail = 0;
const fails: string[] = [];
function check(name: string, cond: boolean, detail = "") {
  if (cond) { pass++; console.log(`  ✅ ${name}`); }
  else { fail++; fails.push(name); console.log(`  ❌ ${name}  ${detail}`); }
}
const has = (rs: R[], frag: string) => rs.some(r => r.content.includes(frag));
const top = (rs: R[]) => rs[0]?.content ?? "(none)";
const rank = (rs: R[], frag: string) => rs.findIndex(r => r.content.includes(frag)) + 1;

console.log(`model=${LOCAL_EMBEDDING_MODEL}`);
console.log(`thresholds=${JSON.stringify(getThresholdProfile())}\n`);

// ── S1: Direct recall via native-language key ──
console.log("S1 direct key recall (KO)");
{
  const ns = "s1";
  await remember("사용자는 매운 음식을 잘 못 먹는다", ["매운음식", "spicy", "음식취향"], ns);
  await remember("사용자는 커피를 하루 세 잔 마신다", ["커피", "coffee", "습관"], ns);
  const r = await recall("매운음식", ns);
  check("native key 'matched' → spicy memory top1", rank(r, "매운 음식") === 1, `top=${top(r)}`);
}

// ── S2: Cross-lingual recall via multilingual keys ──
console.log("S2 cross-lingual recall via multilingual keys");
{
  const ns = "s2";
  await remember("사용자는 딸기를 가장 좋아한다", ["딸기", "strawberry", "fruit", "음식취향"], ns);
  await remember("사용자는 등산을 즐긴다", ["등산", "hiking", "취미"], ns);
  const en = await recall("strawberry", ns);
  check("EN query 'strawberry' → KO 딸기 memory found", has(en, "딸기"), `top=${top(en)}`);
  check("EN query 'strawberry' → 딸기 memory top1", rank(en, "딸기") === 1, `top=${top(en)}`);
  const ko = await recall("hiking 좋아함?", ns);
  check("mixed query 'hiking' → 등산 memory found", has(ko, "등산"), `top=${top(ko)}`);
}

// ── S3: Associative leap — reach a memory unreachable by similarity alone ──
console.log("S3 associative key-graph leap");
{
  const ns = "s3";
  // mem2 shares ONLY the 'radioactivity' key with mem1, and its content (Korean,
  // about danger) has no lexical/semantic overlap with the query "Marie Curie".
  // So reaching it proves the key graph — not embedding similarity — found it.
  await remember("Marie Curie won two Nobel Prizes", ["Marie Curie", "scientist", "radioactivity"], ns);
  await remember("방사능 물질은 취급에 주의가 필요하다", ["radioactivity", "위험", "안전"], ns);
  const r = await recall("Marie Curie", ns, true);
  const target = r.find(x => x.content.includes("방사능"));
  check("query 'Marie Curie' reaches 방사능 memory via shared key", !!target, `got=${r.map(x=>x.content+"#"+x.hop).join(" / ")}`);
  check("reached via key-graph path (matched_via has a '(via)' marker)",
    !!target && target.matched_via.some(v => v.endsWith("(via)")),
    `via=${target?.matched_via.join(",")}`);
}

// ── S4: Versioning / belief change ──
console.log("S4 versioning (supersede)");
{
  const ns = "s4";
  const [oldId] = await remember("user lives in Seoul", ["거주지", "Seoul", "location"], ns);
  await g.recall("where does the user live", 3, ns); // raise depth a bit
  const newId = await g.supersede(oldId, "user moved to Busan", { keyConcepts: ["거주지", "Busan", "location"] });
  const r = await recall("거주지", ns);
  check("after supersede: Busan present", has(r, "Busan"), `top=${top(r)}`);
  check("after supersede: Seoul NOT in active results", !has(r, "Seoul"), `got=${r.map(x=>x.content).join(" / ")}`);
  check("old memory preserved (history intact)", oldId in g.memories, "");
  check("new memory supersedes old", g.memories[newId]?.supersedes === oldId, "");
}

// ── S5: Depth growth on repeated recall ──
console.log("S5 depth growth");
{
  const ns = "s5";
  const [id] = await remember("사용자의 직업은 소프트웨어 엔지니어다", ["직업", "engineer", "job"], ns);
  const d0 = g.memories[id].depth;
  for (let i = 0; i < 5; i++) await g.recall("직업", 3, ns);
  const d1 = g.memories[id].depth;
  check("depth increases with repeated recall", d1 > d0, `${d0} -> ${d1}`);
  check("depth stays within [0,1]", d1 <= 1.0 && d1 >= 0, `d1=${d1}`);
}

// ── S6: name-type key — exact, no semantic bleed ──
console.log("S6 name-type exact matching");
{
  const ns = "s6";
  await remember("사용자의 이름은 동건이다", ["이름", "동건"], ns, { keyTypes: { 동건: "name" } });
  await remember("뉴턴은 만유인력을 발견했다", ["뉴턴", "물리"], ns, { keyTypes: { 뉴턴: "name" } });
  const r1 = await recall("동건", ns);
  check("'동건' → 동건 memory top1", rank(r1, "동건이다") === 1, `top=${top(r1)}`);
  const r2 = await recall("동건", ns);
  check("'동건' does NOT surface 뉴턴 memory as name-match", !(r2.find(x=>x.content.includes("뉴턴"))?.matched_via.includes("뉴턴")), "");
}

// ── S7: near-duplicate dedup (supersede instead of duplicate) ──
console.log("S7 near-duplicate dedup");
{
  const ns = "s7";
  const before = Object.keys(g.memories).length;
  await remember("사용자는 고양이 두 마리를 키운다", ["고양이", "cat", "반려동물"], ns);
  const [, wasDup] = await remember("사용자는 고양이 두 마리를 기른다", ["고양이", "cat"], ns);
  const r = await recall("고양이", ns);
  check("near-duplicate detected as dup", wasDup === true, `wasDup=${wasDup}`);
  check("only one active 고양이 memory", r.filter(x => x.content.includes("고양이")).length === 1, `n=${r.filter(x=>x.content.includes("고양이")).length}`);
}

// ── S8: TTL expiry + cleanup ──
console.log("S8 TTL expiry");
{
  const ns = "s8";
  await remember("임시 메모: 오늘 회의는 3시", ["회의", "meeting"], ns, { ttlSeconds: -1 });
  await remember("사용자는 아침형 인간이다", ["수면", "habit"], ns);
  const removed = await g.cleanupExpired();
  check("expired memory cleaned up", removed >= 1, `removed=${removed}`);
  const r = await recall("회의", ns);
  check("expired memory not recalled", !has(r, "회의는 3시"), `got=${r.map(x=>x.content).join(" / ")}`);
}

// ── S9: Hebbian link reinforcement ──
console.log("S9 Hebbian reinforcement");
{
  const ns = "s9";
  const [id] = await remember("사용자는 재즈 음악을 좋아한다", ["재즈", "jazz", "음악"], ns);
  // find a key linked to this memory and read its weight
  const keyOf = (g as any)._memToKeys[id] as Map<string, number>;
  const [firstKey, w0] = [...keyOf][0];
  for (let i = 0; i < 3; i++) await g.recall("재즈", 3, ns);
  const w1 = ((g as any)._memToKeys[id] as Map<string, number>).get(firstKey);
  check("link weight reinforced after repeated co-recall", w1 > w0, `${w0} -> ${w1}`);
}

// ── S10: Namespace isolation ──
console.log("S10 namespace isolation");
{
  await remember("project-A uses PostgreSQL", ["database", "postgres", "projectA"], "nsA");
  await remember("project-B uses MongoDB", ["database", "mongo", "projectB"], "nsB");
  const a = await recall("database", "nsA");
  check("nsA recall returns only nsA memory", has(a, "PostgreSQL") && !has(a, "MongoDB"), `got=${a.map(x=>x.content).join(" / ")}`);
}

// ── S11: related() associative exploration ──
console.log("S11 related() exploration");
{
  const ns = "s11";
  const [id] = await remember("사용자는 파이썬으로 백엔드를 개발한다", ["파이썬", "python", "backend", "개발"], ns);
  await remember("사용자는 FastAPI 프레임워크를 쓴다", ["python", "fastapi", "backend"], ns);
  const rel = g.getRelated(id) as any[];
  check("related() finds memory sharing keys", rel.some(x => x.content.includes("FastAPI")), `rel=${rel.map(x=>x.content).join(" / ")}`);
}

// ── S12: exact concept-key query outranks same-language content noise ──
console.log("S12 exact-key ranking (no lexical content overlap)");
{
  const ns = "s12";
  // The answer shares NO content words with the query "취미" — only a key does.
  await remember("사용자는 주말마다 클라이밍장에 간다", ["클라이밍", "취미", "운동", "주말활동"], ns);
  // Same-language Korean distractors with high e5 content similarity, no '취미' key.
  await remember("배포 파이프라인은 컨테이너 기반으로 구성한다", ["배포", "컨테이너"], ns);
  await remember("서버 설정 파일은 yaml 형식으로 관리한다", ["서버설정", "yaml"], ns);
  await remember("데이터베이스는 인덱스를 주기적으로 점검한다", ["데이터베이스", "인덱스"], ns);
  const r = await recall("취미", ns);
  check("exact concept-key hit ranks the right memory #1", rank(r, "클라이밍장") === 1, `top=${top(r)}`);
}

console.log(`\n${"=".repeat(50)}`);
console.log(`RESULT: ${pass} passed, ${fail} failed`);
if (fail) console.log(`FAILED: ${fails.join(", ")}`);
await rm(dataDir, { recursive: true, force: true });
// Set exit code without process.exit() — the fastembed native worker aborts if
// the process is force-exited mid-flight. Let the event loop drain naturally.
process.exitCode = fail ? 1 : 0;
