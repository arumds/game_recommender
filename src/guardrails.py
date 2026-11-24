"""Responsible-engagement guardrail layer applied after retrieval (ILLUSTRATIVE ANALOGY).

This is NOT a real responsible-gaming policy. Steam is a video-game store, not gambling. The layer shows
how to put business/safety constraints between a model and the user, and how to measure what they cost:

  1. High-intensity players (top quartile of last-two-weeks playtime among players who played recently)
     do not get games with in-app purchases (a stand-in for "spend-trigger" content).
  2. Everyone gets a diversity re-rank (MMR on genre/tag similarity) so the list is not ten near-copies.
"""
import numpy as np


def heavy_user_mask(recent_hours, quantile=0.75):
    active = recent_hours[recent_hours > 0]
    thr = np.quantile(active, quantile) if len(active) else np.inf
    return recent_hours >= thr, float(thr)


def _cosine_matrix(mh):
    x = mh / np.maximum(np.linalg.norm(mh, axis=1, keepdims=True), 1e-9)
    return x @ x.T


def apply_guardrails(candidates, heavy, iap, mh, k=10, lam=0.7):
    """candidates: [n, C] item idx in rank order. Returns [n, k]."""
    sim = _cosine_matrix(mh)
    n, c = candidates.shape
    rel_base = 1.0 - np.arange(c) / c
    out = np.zeros((n, k), dtype=np.int64)
    for r in range(n):
        items = candidates[r]
        rel = rel_base
        if heavy[r]:
            ok = ~iap[items]
            items, rel = items[ok], rel[ok]
        chosen = [0]
        remaining = list(range(1, len(items)))
        while len(chosen) < min(k, len(items)) and remaining:
            s = sim[np.ix_(items[remaining], items[chosen])].max(1)
            j = int(np.argmax(lam * rel[remaining] - (1 - lam) * s))
            chosen.append(remaining.pop(j))
        row = items[chosen]
        out[r, :len(row)] = row
        if len(row) < k:  # not enough candidates left after filtering: pad with the filtered-out tail
            pad = [x for x in candidates[r] if x not in set(row)][:k - len(row)]
            out[r, len(row):] = pad
    return out


def intra_list_diversity(recs, mh):
    sim = _cosine_matrix(mh)
    vals = []
    for row in recs:
        s = sim[np.ix_(row, row)]
        vals.append(1.0 - (s.sum() - len(row)) / (len(row) * (len(row) - 1)))
    return float(np.mean(vals))


def iap_share(recs, iap):
    return float(iap[recs].mean())
