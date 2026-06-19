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
