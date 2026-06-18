# Distribution-Based "Not-Found" Gate (robust-z / MAD) — Design

**Date:** 2026-06-18
**Status:** Approved (design)
**Area:** `src/embedding.ts`, `src/memoryGraph.ts`, `src/server.ts`, `test/`
**Builds on:** the absolute score gate shipped in 0.7.0 (`docs/superpowers/specs/2026-06-18-retriever-quality-design.md` §2)

## Problem

The absolute cosine gate (`minScore`) added in 0.7.0 works on well-separated models
(bge-m3) but is **inert on the default multilingual-e5 backend**, confirmed live: an
unrelated query ("블록체인 합의 알고리즘…") with `min_score=0.85` still returned 3
unrelated memories instead of `[]`. e5 packs all cosine similarities into a narrow
~0.86–0.99 band, so any positive absolute threshold produces a false "anchor" — the
"not-found" detection (request: distinguish 못 찾음) does not work on the shipped default.

Within-result noise trimming is already handled by the relative floor `min_rel_score`
(verified working on e5). This design addresses only the *query-level* question: **does
this query have any genuine match, or is everything uniform noise?**

## Key insight

Even when e5 compresses all similarities into a high narrow band, a *genuine* match
stands out as a **right-tail outlier** of that query's similarity distribution. An
absolute threshold can't see this; a **distribution-relative** measure can, and it
auto-adapts to each model's band. We use a **robust z-score** (median + MAD) because the
packed e5 band is skewed and non-normal, where mean/std are unreliable.

## Design

### Signal & computation (in `recall()`, `src/memoryGraph.ts`)

- Dense Path B already computes `contentSims` for **all** memories (the background
  population). Collect the full array as `allContentSims` (before the `contentRecall`
  threshold filter).
- Compute over `allContentSims`:
  - `median`
  - `MAD = median(|x − median|)`
  - **robust z of the top hit**: `(maxContentSim − median) / (1.4826 × MAD)`
    (the 1.4826 factor scales MAD to a σ-equivalent so `gateZ` reads in sigma-like units).

### Anchor condition (per-profile AND composition)

Replace the current binary anchor check with:

```
hasAnchor = definiteAnchor || (absoluteAnchor && passesDistributionGate(...))
```

- **`definiteAnchor`** — any candidate with a literal exact key match
  (`name`/`proper_noun`, `rawSim ≈ 1.0`, i.e. `memRawSim[mid] >= 0.999`). A literal hit
  is relevant regardless of the content distribution (protects e.g. name queries like
  "동균" from being gated out).
- **`absoluteAnchor`** — at least one candidate passes the existing
  `passesAbsoluteGate(memRawSim[mid] ?? 0, minScore)`. (Unchanged from 0.7.0.)
- **`passesDistributionGate(maxContentSim, allContentSims, gateZ, GATE_MIN_POPULATION)`**
  returns **true (skip / pass)** when any of:
  - `gateZ <= 0` (gate disabled for this profile/call),
  - `allContentSims.length < GATE_MIN_POPULATION` (too few memories for reliable stats),
  - `MAD` is degenerate (≈ 0 — uniform distribution, can't compute z);
  otherwise returns `robustZScore(maxContentSim, allContentSims) >= gateZ`.

When no anchor exists, `recall()` returns `[]` (same mechanism as the 0.7.0 anchor gate).

### Per-model behavior

| Model | `minScore` | `gateZ` | Effective gate |
|-------|-----------|---------|----------------|
| e5 | low/0 | **on** (calibrated) | distribution gate decides "not found" |
| bge-m3 | on (0.55) | **0** (off) | absolute gate decides (already well-separated) |
| bge / minilm / openai | on | 0 (off) | absolute gate (unchanged) |

Both gates can be enabled together (AND) for a model where that helps. A profile with
`gateZ = 0` has **identical behavior to 0.7.0** — no regression.

### Config & interfaces

- `ThresholdProfile` gains `gateZ: number` (0 = disabled). Every profile row defines it.
- Env override `SUPER_MEMORY_GATE_Z` (wired via existing `envThreshold`, but note: z is
  not in [0,1] — see Open detail below).
- `recall()` gains a parameter `minZ` (default = profile `gateZ`); MCP tool exposes
  `min_z`.
- Constant `GATE_MIN_POPULATION = 8`.

**`envThreshold` range:** the existing `envThreshold` validates `[0,1]`. A z threshold
is typically 2–5, so `SUPER_MEMORY_GATE_Z` must use a separate parser that accepts a
non-negative finite number (e.g. `envNonNegative(name)`), not `envThreshold`. The
`recall(minZ)` param and profile values are likewise plain non-negative numbers.

### Pure helpers (testable in isolation)

In `src/memoryGraph.ts` (or a small stats helper), exported:

- `robustZScore(top: number, values: number[]): number`
  — returns `(top − median) / (1.4826 × MAD)`; returns `Infinity` when `MAD` is 0 and
  `top > median` (clear outlier vs a degenerate distribution), `0` when `top == median`.
  Documented behavior must be deterministic and unit-tested.
- `passesDistributionGate(top: number, values: number[], gateZ: number, minCount: number): boolean`
  — the skip/pass logic above.

## Calibration (empirical, like the contradiction-floor calibration)

Measure `robustZScore(maxContentSim, allContentSims)` on the **real graph + e5** for:
- FOUND queries (e.g. "이름", "고양이", "super-memory 목표")
- NOT-FOUND queries (e.g. "블록체인 합의 알고리즘", "양자역학 블랙홀")

Pick e5 `gateZ` between the two clusters (favoring no false "not-found" on real matches).
Record the measured values and chosen `gateZ` in the profile comment. bge-m3 `gateZ = 0`.
A manual `*.live.ts` script performs the measurement; the chosen value is committed.

## Testing

**Unit (deterministic, via the test embedder seam + crafted vectors):**
- `robustZScore`: known arrays → known z; `top == median` → 0; degenerate `MAD == 0` with
  `top > median` → `Infinity`.
- `passesDistributionGate`: disabled (`gateZ <= 0`) → true; population `< minCount` → true;
  `MAD == 0` → true; clear outlier ≥ gateZ → true; non-outlier < gateZ → false.
- `recall()` end-to-end (pass `min_z` to activate): a uniform-similarity candidate set
  (no outlier) returns `[]` even though the absolute gate would pass; a set with one clear
  outlier returns hits. A literal name-key match returns hits even with no content outlier
  (`definiteAnchor`). `min_z = 0` reproduces 0.7.0 behavior.

**Manual real-model (`*.live.ts`, not in `npm test`):** the calibration measurement above,
plus an end-to-end e5 check that a known unrelated query now returns `[]` and a known
relevant query still returns hits.

## Non-goals

- In-result noise trimming (already `min_rel_score`).
- Changing behavior for profiles with `gateZ = 0` (bge-m3 etc.) — must be byte-for-byte
  the 0.7.0 anchor behavior.
- Per-result z filtering — this is a single query-level found/not-found decision.
- Parametric (mean/std) gating or percentile gating — robust-z (median/MAD) chosen.
