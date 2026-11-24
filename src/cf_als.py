"""Collaborative filtering: ALS for implicit feedback (library: `implicit`).

Confidence = 1 + alpha * log1p(hours played). Log damps the heavy tail (the 95th percentile of playtime
is 61 hours, the max is 10,000+), so one obsessive-hours game does not dominate a user's row.
"""
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


def build_matrix(train, n_users, n_items, alpha=8.0):
    vals = (1.0 + alpha * np.log1p(train["playtime"].values / 60.0)).astype(np.float32)
    return sp.csr_matrix((vals, (train["u"].values, train["i"].values)), shape=(n_users, n_items))


def train_als(matrix, factors=64, regularization=0.05, iterations=15, seed=0):
    model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                    iterations=iterations, random_state=seed, use_gpu=False)
    model.fit(matrix, show_progress=False)
    return model


def recommend_als(model, matrix, user_idx, k, chunk=8192):
    out = []
    for s in range(0, len(user_idx), chunk):
        ids = user_idx[s:s + chunk]
        items, _ = model.recommend(ids, matrix[ids], N=k, filter_already_liked_items=True)
        out.append(items)
    return np.vstack(out)


def similar_items(model, item_idx, k=5):
    """Cross-sell: games whose ALS vectors are closest to this one."""
    ids, scores = model.similar_items(item_idx, N=k + 1)
    return [(int(i), float(s)) for i, s in zip(ids[1:], scores[1:])]
