# Calibration Pair Labeling Guide

Each pair (a, b) gets one `relation`:

- **duplicate** — same fact restated (paraphrase, reorder, synonym). System should
  supersede (dedup). e.g. "라떼를 마신다" / "마시는 건 라떼야".
- **contradiction** — same subject, conflicting value. System should flag, keep both.
  MUST share a key. e.g. "회의는 월요일" / "회의는 금요일".
- **independent** — unrelated or coexisting facts. Keep both, no flag.

## Boundary rules
- Refinement (general vs specific: "서울 산다" / "마포구 산다") → `independent`, `confidence: "low"`.
- If two reasonable people would disagree on the label → set `confidence: "low"`.
- Different subjects can never be `contradiction` (no shared key) → `independent`.

## Generation rules (avoid bias)
- Use natural, conversational phrasing. Vary sentence structure between a and b.
- Cap single-token minimal-pairs (e.g. only day-of-week swapped) at ~1/3 of
  contradictions; their artificially high cosine biases the threshold.
- Mix KO and EN; reuse the Mina persona for realism.
