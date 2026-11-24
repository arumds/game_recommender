"""LightGBM LambdaMART re-ranker over merged ALS + two-tower candidates."""
import numpy as np
import lightgbm as lgb
import scipy.sparse as sp

from featurize import FEATURES, pair_features


def grade_matrix(test, n_users, n_items):
    """Graded relevance for held-out positives by hours played: 1 (<10 h), 2 (<50 h), 3 (50 h+)."""
    hours = test["playtime"].values / 60.0
    grade = np.where(hours < 10, 1, np.where(hours < 50, 2, 3)).astype(np.int8)
    g = np.zeros((n_users, n_items), dtype=np.int8)
    g[test["u"].values, test["i"].values] = grade
    return g


def training_frame(users, cand, ar, tr, T, grades):
    X, r, c = pair_features(users, cand, ar, tr, T)
    y = grades[users[r], cand[r, c]]
    group = np.bincount(r, minlength=len(users))
    return X, y, group[group > 0]


def train_ranker(X, y, group, rounds=150, seed=0):
    ds = lgb.Dataset(X, label=y, group=group, feature_name=FEATURES)
    params = dict(objective="lambdarank", metric="ndcg", eval_at=[10], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambdarank_truncation_level=20, verbose=-1, seed=seed, num_threads=2)
    return lgb.train(params, ds, num_boost_round=rounds)


def rank_candidates(ranker, users, cand, ar, tr, T, top=50):
    """Score every candidate, return [n, top] item idx sorted by ranker score."""
    X, r, c = pair_features(users, cand, ar, tr, T)
    scores = np.full(cand.shape, -np.inf, dtype=np.float64)
    scores[r, c] = ranker.predict(X)
    order = np.argsort(-scores, axis=1)[:, :top]
    return np.take_along_axis(cand, order, 1)
