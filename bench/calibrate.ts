// bench/calibrate.ts
// Dedup/contradiction threshold calibration. Builds simAB per labeled pair with the
// real embedder, sweeps thresholds on the train split, validates on held-out, and
// prints the 1순위 separability signal. Run: npm run bench:calibrate
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

import { sharedKey, calibrate, range, type Pair, type ScoredPair } from "./calibrate-lib.ts";
const { embedTextAsync } = await import("../src/embedding.ts");
const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na === 0 || nb === 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

const raw = JSON.parse(await readFile(resolve("bench/pairs.json"), "utf-8")) as { pairs: Pair[] };

const scored: ScoredPair[] = [];
for (const pair of raw.pairs) {
  const ea = await embedTextAsync(pair.a);
  const eb = await embedTextAsync(pair.b);
  scored.push({ pair, simAB: cosine(ea, eb), sharedKey: sharedKey(pair.keys_a, pair.keys_b) });
}

const res = calibrate(scored, {
  floors: range(0.75, 0.90, 0.05),
  cuts: range(0.88, 0.99, 0.01),
  indepPrior: 0.95,
});

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
console.log(`\nsuper-memory calibration — model=${LOCAL_EMBEDDING_MODEL} | train=${res.trainN} held-out=${res.heldOutN}`);
console.log("─".repeat(64));
console.log(`BEST (train macro-F1): floor=${res.best.floor} cut=${res.best.dedupCut} -> ${res.best.macroF1.toFixed(2)}`);
console.log(`HELD-OUT macro-F1: ${res.heldOut.macroF1.toFixed(2)}  Δ=${res.overfitDelta.toFixed(2)} ${res.overfitDelta > 0.10 ? "⚠ OVERFIT" : "OK"}`);
console.log(`prior-weighted FP (indep prior 0.95): ${pct(res.priorFP)}`);
console.log("─".repeat(64));
const s = res.separability;
console.log(`best joint dup/contra F1: dup=${s.dupF1.toFixed(2)} contra=${s.contraF1.toFixed(2)} min=${s.minF1.toFixed(2)} @ floor=${s.floor} cut=${s.dedupCut}`);
if (s.minF1 < 0.75) {
  console.log(`⚠ duplicate and contradiction are not separable by cosine alone (min joint F1 ${s.minF1.toFixed(2)}).`);
  console.log(`  This is the 1순위 evidence: a token-level discriminator (harness v2) is needed.`);
}
