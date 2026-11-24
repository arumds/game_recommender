"""Metrics via ranx (Recall@K, NDCG@K, MAP@K) + paired bootstrap on per-user scores."""
import numpy as np
from ranx import Qrels, Run, evaluate

METRICS = ["recall@10", "recall@20", "ndcg@10", "map@10"]


def to_run(recs, user_idx, k):
    return Run.from_dict({str(u): {str(int(i)): float(k - r) for r, i in enumerate(row[:k])}
                          for row, u in zip(recs, user_idx)})


def to_qrels(test, user_idx):
    wanted = set(user_idx)
    d = {}
    for u, i in zip(test["u"].values, test["i"].values):
        if u in wanted:
            d.setdefault(str(u), {})[str(int(i))] = 1
    return Qrels.from_dict(d)


def evaluate_run(run, qrels, metrics=METRICS):
    return evaluate(qrels, run, metrics)


def per_user(run, qrels, metric="ndcg@10"):
    return np.asarray(evaluate(qrels, run, metric, return_mean=False))


def paired_bootstrap(a, b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
