# Dedup / Contradiction Calibration Corpus — Design

**Date:** 2026-06-19
**Status:** Approved (brainstorming) → ready for implementation plan
**Scope:** Diagnostic only. Does NOT change the supersede decision or add a recovery path — those are follow-up cycles.

## Problem

The dedup and contradiction thresholds (`memoryDedup`, `contradiction`) in
`src/embedding.ts` are calibrated on tiny fixtures (the gateZ comment cites an
n=8 set; the committed `bench/fixture.json` is 14 memories / 16 queries). There
is no way to trust them, and the existing benchmark (`bench/run.ts`) measures
**retrieval quality only** (recall@1/@5, MRR, not-found) — it cannot measure
dedup or contradiction behavior at all.

Concretely, this blocks fixing the known 1순위 bug: a single-token conflict
("회의는 월요일" vs "회의는 금요일") can land at cosine ≥ `memoryDedup` and is
silently superseded, because the contradiction band is `[contradiction, dedup)`
— it structurally cannot fire above the dedup ceiling. We cannot safely fix that
(distinguishing a paraphrase from a high-similarity conflict) without labeled
data, or we repeat the overfitting that already burned gateZ (tuning set 100%,
held-out 63%).

## Goal

A labeled corpus of memory **pairs** plus a calibration harness that:
1. Scores the current dedup/contradiction thresholds (P/R/F1 per class).
2. Sweeps the thresholds and shows the performance curve, so the optimal cut is
   visible in the data rather than guessed.
3. Quantifies the 1순위 bug (contradiction recall ceiling above dedup).
4. Guards against overfitting via a train/held-out split.

## Non-Goals (explicit follow-ups)

- Changing the supersede **decision** (e.g. a token-level discriminator). The
  harness v1 sweeps `floor`/`dedup_cut` only; a token discriminator is a v2
  lever requiring a harness extension. The corpus labels are reusable for it.
- A read-superseded **recovery path** (`read_memory` rejects superseded ids).
- Filling all ~250 pairs (this spec ships the schema, harness, and ~30 seed
  pairs to prove the harness; full labeling is a separate effort).
- `remember_batch` parity for the already-shipped observability fix.

## Design

### ① Corpus schema — `bench/pairs.json`

dedup and contradiction share one schema (both are "relation between two facts"):

```jsonc
{
  "_doc": "Calibration corpus for dedup/contradiction thresholds. Each pair labels the relation between two facts. Ground truth drives threshold sweeps in bench/calibrate.ts.",
  "pairs": [
    {
      "id": "contra-001",
      "a": "회의는 월요일이다",
      "b": "회의는 금요일이다",
      "keys_a": ["회의", "일정"],
      "keys_b": ["회의", "일정"],
      "relation": "contradiction",   // duplicate | contradiction | independent
      "confidence": "high",          // high | low  (low = boundary/ambiguous)
      "split": "held-out"            // train | held-out
    }
  ]
}
```

**Labels (3):**
- `duplicate` — same fact restated (paraphrase, reordering). Correct system
  behavior: supersede (dedup).
- `contradiction` — same subject, conflicting fact. Correct behavior: flag, keep
  both. This is the class the 1순위 bug fails on.
- `independent` — unrelated/coexisting facts. Correct behavior: keep both, no flag.

**`confidence` field (refinement 1):** boundary pairs where reasonable labelers
disagree (e.g. "서울 산다" vs "마포구 산다") are marked `low`. The harness can
report metrics with and without `low`-confidence pairs so the threshold is not
fit to subjective judgment calls.

### ② Harness — `bench/calibrate.ts`

```
1. Load pairs.json; split into train / held-out.
2. For each pair: simAB = cosine(embed(a), embed(b)) — ONE computation per pair,
   embedded the same way the graph does (reuse embedTextAsync). Also record
   whether keys_a and keys_b share a concept string.
3. Sweep (contradiction_floor, dedup_cut) over 0.70–0.99:
   classify each pair by mirroring the real MemoryGraph decision path:
     simAB >= dedup_cut                                  -> "duplicate"
     contradiction_floor <= simAB < dedup_cut AND shared -> "contradiction"
     else                                                -> "independent"
4. Confusion matrix vs labels -> per-class P/R/F1 + macro.
5. Pick best macro-F1 on TRAIN -> rescore on HELD-OUT.
6. Print |train macro-F1 - held-out macro-F1| (overfit warning if > 0.10).
```

**Fidelity note:** pair-isolation is valid for calibrating the *scalar*
threshold (the dedup cut is the cosine boundary where duplicate-pair and
independent-pair distributions separate — graph size does not move it). The
harness mirrors `_findDuplicate` (≥dedup), `_findContradiction` (band + shared
key), and the independent fallthrough, so a chosen threshold transfers to
`THRESHOLD_PROFILES`.

### ③ Metrics & output

- Per-class P/R/F1 + macro; dedup_cut sweep table at a fixed floor, then narrow
  by crossing floor × cut.
- Train→held-out macro-F1 delta as the overfit signal.
- **Prior-aware false-positive (refinement 2):** the corpus is class-balanced
  for stable F1, but production sees mostly `independent` (conflicts are rare).
  Output the false-positive rate re-weighted to a configurable real prior, so an
  operating point picked on balanced data is not misleading for the live mix.
- **1순위 evidence:** surface the contradiction-recall ceiling — if raising
  `dedup_cut` cannot lift contradiction recall (because conflicts sit above the
  cut), print it as the explicit signal that thresholds alone are insufficient
  and a token discriminator (v2) is required.

```
super-memory calibration — model=bge-m3 | train=175 held-out=75
[dedup_cut sweep @ floor=0.80]
 cut    dup-F1  contra-F1  indep-F1  macro
 0.90    .88      .61        .97      .82
 0.94    .90      .58        .97      .82   <- current
 0.97    .85      .54        .96      .78
BEST (train macro-F1): cut=0.93 floor=0.81 -> .84
HELD-OUT @ best:                            -> .79   Δ=.05  OK (>.10 warns)
prior-weighted FP @ best (indep prior 0.95): .03
⚠ contra-F1 ceiling .61 — unaffected by dedup_cut. Token discriminator (v2) needed.
```

### ④ File layout

```
bench/
  pairs.json        # new corpus (~250 target; ~30 seed pairs this cycle)
  calibrate.ts      # new harness (②③)
  LABELING.md       # labeling guideline (refinement 1)
  fixture.json      # unchanged
  run.ts            # unchanged
package.json: "bench:calibrate": "tsx bench/calibrate.ts"
```

Independent from the retrieval benchmark — two regression guards that run
separately.

### Corpus construction (out of scope to complete here, defined for the plan)

- **Generation:** LLM-generated pairs, human-verified labels, ~250 target,
  KO + EN mixed, reusing the Mina persona style from `fixture.json`.
- **Diversity (refinement 3):** generation prompts must force natural,
  conversational phrasing and cap minimal-pairs (single-token edits), whose
  artificially high cosine would bias the threshold.
- **Split (refinement 4):** stratified by `relation` so held-out keeps class
  balance. Note honestly: 250 pairs is a floor, not luxury — held-out per class
  is ~25, so the train→held-out delta is the trustworthy signal, not absolute F1.

## This-cycle deliverables

1. `bench/pairs.json` schema + ~30 seed pairs (covers all 3 labels, KO+EN).
2. `bench/calibrate.ts` (sweep, split, prior-aware FP, 1순위 ceiling signal).
3. `bench/LABELING.md` (label definitions + boundary rules + confidence).
4. `package.json` script `bench:calibrate`.

## Risks (from objective re-analysis)

- **Label subjectivity at boundaries** — mitigated by `confidence` + LABELING.md;
  ambiguous pairs are inherently low-stakes (no single right answer to lose).
- **Class base-rate mismatch** — mitigated by prior-weighted FP reporting.
- **Minimal-pair cosine artifacts** — mitigated by generation diversity rules.
- **Thin held-out** — accepted; delta (not absolute F1) is the signal; 250 is a floor.
- **Pair-isolation vs populated graph** — low: the scalar threshold is graph-size
  invariant; a sample full-graph cross-check can be added if a discrepancy appears.
