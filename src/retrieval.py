"""Dot-product retrieval and candidate merging. Numpy only, so training and serving share it exactly."""
import numpy as np


def retrieve_topk(user_vecs, item_vecs, seen, k, chunk=4096):
    """user_vecs [n,d], item_vecs [m,d], seen: CSR [n,m] of items to exclude. Returns [n,k] item idx, best first."""
    out = []
    for s in range(0, len(user_vecs), chunk):
        scores = user_vecs[s:s + chunk] @ item_vecs.T
        scores[seen[s:s + chunk].toarray() > 0] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.take_along_axis(scores, top, 1).argsort(1)[:, ::-1]
        out.append(np.take_along_axis(top, order, 1))
    return np.vstack(out)


def build_candidates(als_top, tt_top, absent_rank=99):
    """Union of the two retrievers' lists, ALS items first. Returns cand [n,C] (-1 padded) and each source's
    rank for every candidate (absent_rank when the source did not return it)."""
    n, k = als_top.shape
    c = 2 * k
    cand = np.full((n, c), -1, dtype=np.int64)
    ar = np.full((n, c), absent_rank, dtype=np.float32)
    tr = np.full((n, c), absent_rank, dtype=np.float32)
    for r in range(n):
        pos = {}
        for rank, i in enumerate(als_top[r]):
            pos[int(i)] = len(pos)
            cand[r, pos[int(i)]] = i
            ar[r, pos[int(i)]] = rank
        for rank, i in enumerate(tt_top[r]):
            i = int(i)
            if i not in pos:
                pos[i] = len(pos)
                cand[r, pos[i]] = i
            tr[r, pos[i]] = rank
    return cand, ar, tr
