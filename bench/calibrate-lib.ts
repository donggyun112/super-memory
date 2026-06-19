export type Relation = "duplicate" | "contradiction" | "independent";

export interface Pair {
  id: string;
  a: string;
  b: string;
  keys_a: string[];
  keys_b: string[];
  relation: Relation;
  confidence: "high" | "low";
  split: "train" | "held-out";
}

export interface ScoredPair {
  pair: Pair;
  simAB: number;
  sharedKey: boolean;
}

export function sharedKey(keysA: string[], keysB: string[]): boolean {
  const a = new Set(keysA.map((k) => k.toLowerCase()));
  return keysB.some((k) => a.has(k.toLowerCase()));
}

export function classifyPair(
  simAB: number,
  shared: boolean,
  floor: number,
  dedupCut: number
): Relation {
  if (simAB >= dedupCut) return "duplicate";
  if (simAB >= floor && shared) return "contradiction";
  return "independent";
}

const RELATIONS: Relation[] = ["duplicate", "contradiction", "independent"];

export interface PRF { p: number; r: number; f1: number; }
export interface Scorecard { perClass: Record<Relation, PRF>; macroF1: number; }

export function prf(scored: ScoredPair[], floor: number, dedupCut: number): Scorecard {
  const tp: Record<Relation, number> = { duplicate: 0, contradiction: 0, independent: 0 };
  const fp: Record<Relation, number> = { duplicate: 0, contradiction: 0, independent: 0 };
  const fn: Record<Relation, number> = { duplicate: 0, contradiction: 0, independent: 0 };
  for (const s of scored) {
    const pred = classifyPair(s.simAB, s.sharedKey, floor, dedupCut);
    const act = s.pair.relation;
    if (pred === act) tp[act]++;
    else { fp[pred]++; fn[act]++; }
  }
  const perClass = {} as Record<Relation, PRF>;
  let macro = 0;
  for (const c of RELATIONS) {
    const p = tp[c] + fp[c] === 0 ? 0 : tp[c] / (tp[c] + fp[c]);
    const r = tp[c] + fn[c] === 0 ? 0 : tp[c] / (tp[c] + fn[c]);
    const f1 = p + r === 0 ? 0 : (2 * p * r) / (p + r);
    perClass[c] = { p, r, f1 };
    macro += f1;
  }
  return { perClass, macroF1: macro / RELATIONS.length };
}

export function range(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

export function priorWeightedFP(
  scored: ScoredPair[],
  floor: number,
  dedupCut: number,
  indepPrior: number
): number {
  const indeps = scored.filter((s) => s.pair.relation === "independent");
  if (indeps.length === 0) return 0;
  const falseFlags = indeps.filter(
    (s) => classifyPair(s.simAB, s.sharedKey, floor, dedupCut) !== "independent"
  ).length;
  return indepPrior * (falseFlags / indeps.length);
}

export function splitPairs(scored: ScoredPair[]): { train: ScoredPair[]; heldOut: ScoredPair[] } {
  return {
    train: scored.filter((s) => s.pair.split === "train"),
    heldOut: scored.filter((s) => s.pair.split === "held-out"),
  };
}
