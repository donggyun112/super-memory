// Selection policy for inject-mode recall. Given associatively-expanded candidates (already in
// relevance order), choose which top-K to inject:
//   preferDepth   — within the relevance-selected window, surface confirmed (deep) memories first
//                   so the more reliable ones lead. RELEVANCE decides membership; depth only
//                   reorders inside it. It must NEVER reach past the window to pull a deeper but
//                   less-relevant candidate in — a frequently-read yet barely-relevant memory
//                   (e.g. a giant doc matched only by stray BM25 terms) would otherwise displace a
//                   genuinely relevant but shallow new memory.
//   exploreShallow — reserve one slot for the SHALLOWEST candidate in the whole pool, so weak/recent
//                   memories occasionally resurface (and can be reinforced) instead of being
//                   permanently buried by deep ones. An ε-exploration on memory — the one
//                   deliberate reach past the relevance window, bounded to a single slot.
// Both are opt-in. NOTE: their *value* is longitudinal (it only shows once memories sit at
// different depths over real use) and is not captured by a one-shot benchmark — this module just
// makes the policy correct and testable.
export interface InjectCandidate {
  id: string;
  depth: number;
}

export function selectInject(
  candidates: InjectCandidate[],
  topK: number,
  opts: { preferDepth?: boolean; exploreShallow?: boolean } = {}
): string[] {
  if (topK <= 0 || candidates.length === 0) return [];
  // Membership is fixed by relevance: take the top-K window first. preferDepth then reorders only
  // WITHIN that window (depth desc, relevance order as the stable tiebreak) — it cannot widen the
  // set, so a deep-but-irrelevant candidate ranked below the window can never get pulled in.
  let window = candidates.slice(0, topK);
  if (opts.preferDepth) {
    window = window
      .map((c, i) => ({ c, i }))
      .sort((x, y) => y.c.depth - x.c.depth || x.i - y.i)
      .map((x) => x.c);
  }
  const pick = window.map((c) => c.id);
  if (opts.exploreShallow && candidates.length > topK) {
    const shallowest = candidates.reduce((a, b) => (b.depth < a.depth ? b : a));
    if (!pick.includes(shallowest.id)) pick[pick.length - 1] = shallowest.id; // give a weak memory a slot
  }
  return pick;
}
