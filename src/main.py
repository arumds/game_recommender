"""Train and evaluate the whole stack on UCSD Steam, then save the artifacts the API loads.

  retrieval: popularity, ALS, two-tower  ->  merge  ->  LightGBM ranker  ->  guardrails
  plus: cross-sell examples

Players are split 70/30: the ranker learns from the first group's held-out games, and every number below is
measured on the second group, which the ranker never trained on.

Run from the project root:  python src/main.py
"""
import json
import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import build_dataset
from features import build_item_features
from cf_als import build_matrix, train_als, similar_items
from two_tower import train_two_tower, TwoTowerRetriever
from retrieval import retrieve_topk, build_candidates
from featurize import FEATURES, build_tables, derive
from ranker import grade_matrix, training_frame, train_ranker
from pipeline import recommend, N_RETRIEVE
from guardrails import heavy_user_mask, intra_list_diversity, iap_share
from evaluate import to_run, to_qrels, evaluate_run, per_user, paired_bootstrap

DATA_DIR = os.environ.get("STEAM_DATA_DIR", "data")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
t0 = time.time()
tick = lambda msg: print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

ds = build_dataset(DATA_DIR)
train, test = ds["train"], ds["test"]
n_users, n_items = len(ds["user_index"]), len(ds["item_index"])
tick(f"users={n_users:,} items={n_items:,} train_pos={len(train):,} test_pos={len(test):,}")
feats = build_item_features(ds["games"], ds["item_index"])
user_ids = sorted(ds["user_index"], key=ds["user_index"].get)

# ---- retrieval models -------------------------------------------------------------------------------------
matrix = build_matrix(train, n_users, n_items)
seen = (matrix > 0).tocsr()
als = train_als(matrix)
tick("ALS done")
model, losses, profiles = train_two_tower(train, n_users, n_items, feats)
tt = TwoTowerRetriever(model, feats, profiles)
tick("two-tower done")

heavy, heavy_thr = heavy_user_mask(ds["recent_hours"])
T = build_tables(train, n_users, n_items, feats, als.user_factors, als.item_factors,
                 tt.all_user_vecs(), tt.item_vecs.numpy(), ds["recent_hours"], matrix)
derive(T, heavy_thr)

# ---- user split, candidates, ranker -----------------------------------------------------------------------
rng = np.random.default_rng(0)
perm = rng.permutation(n_users)
cut = int(0.7 * n_users)
rank_users, eval_users = np.sort(perm[:cut]), np.sort(perm[cut:])

als_top = retrieve_topk(T["als_user"], T["als_item"], seen, N_RETRIEVE)
tt_top = retrieve_topk(T["tt_user"], T["tt_item"], seen, N_RETRIEVE)
cand, ar, tr = build_candidates(als_top, tt_top)
tick(f"candidates built (avg {np.mean((cand >= 0).sum(1)):.0f} per player)")

grades = grade_matrix(test, n_users, n_items)
X, y, group = training_frame(rank_users, cand[rank_users], ar[rank_users], tr[rank_users], T, grades)
tick(f"ranker training rows={len(X):,}, positive rate={np.mean(y > 0):.3%}")
ranker = train_ranker(X, y, group)
imp = dict(zip(FEATURES, ranker.feature_importance(importance_type="gain")))
tick("ranker done")
print("feature importance (gain share):")
tot = sum(imp.values())
for f, v in sorted(imp.items(), key=lambda kv: -kv[1]):
    print(f"  {f:<16}{v / tot:7.1%}")

# ---- evaluation on held-out players ----------------------------------------------------------------------
qrels = to_qrels(test, eval_users)
K = 10
pop = T["pop"]
order = np.argsort(-pop)
pop_recs = np.vstack([order[~(seen[u].toarray().ravel() > 0)[order]][:K] for u in eval_users])
als_recs, tt_recs = als_top[eval_users, :K], tt_top[eval_users, :K]
# non-learned fusion baseline: reciprocal-rank fusion of the two lists
rrf_recs = []
for u in eval_users:
    sc = {}
    for lst in (als_top[u], tt_top[u]):
        for r, i in enumerate(lst):
            sc[int(i)] = sc.get(int(i), 0.0) + 1.0 / (60 + r)
    rrf_recs.append(sorted(sc, key=sc.get, reverse=True)[:K])
rrf_recs = np.array(rrf_recs)

final, ranked = recommend(eval_users, T, ranker, seen, k=K)
recs = {"popularity": pop_recs, "als": als_recs, "two_tower": tt_recs, "rrf_als_tt": rrf_recs,
        "ranker": ranked[:, :K], "ranker+guardrails": final}
runs = {n: to_run(r, eval_users, K) for n, r in recs.items()}
metrics = ["recall@10", "ndcg@10", "map@10"]
results = {n: evaluate_run(r, qrels, metrics) for n, r in runs.items()}
print(f"\nHeld-out players: {len(eval_users):,}  (top-{K})")
print(f"{'model':<20}" + "".join(f"{m:>12}" for m in metrics))
for n, res in results.items():
    print(f"{n:<20}" + "".join(f"{res[m]:>12.4f}" for m in metrics))

cand_recall = np.mean([len(set(cand[u][cand[u] >= 0]) & set(test.loc[test["u"] == u, "i"])) /
                       max(1, (test["u"] == u).sum()) for u in eval_users[:2000]])
print(f"candidate-set recall (ceiling for any ranker, first 2000 players): {cand_recall:.3f}")

nd = {n: per_user(r, qrels) for n, r in runs.items()}
print("\nPaired bootstrap on per-user NDCG@10 (mean diff, 95% CI):")
comparisons = {}
for a, b in [("ranker", "als"), ("ranker", "two_tower"), ("ranker", "rrf_als_tt"), ("two_tower", "als"),
             ("ranker+guardrails", "ranker")]:
    d, lo, hi = paired_bootstrap(nd[a], nd[b])
    verdict = "significant" if lo > 0 or hi < 0 else "not distinguishable from zero"
    comparisons[f"{a}_vs_{b}"] = {"diff": d, "ci95": [lo, hi]}
    print(f"  {a} - {b}: {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")

hv = heavy[eval_users]
guard = {
    "heavy_players_eval": int(hv.sum()), "heavy_threshold_hours_2w": heavy_thr,
    "diversity": [intra_list_diversity(ranked[:, :K], T["mh"]), intra_list_diversity(final, T["mh"])],
    "iap_share_heavy": [iap_share(ranked[hv, :K], T["iap"]), iap_share(final[hv], T["iap"])],
    "iap_share_all": [iap_share(ranked[:, :K], T["iap"]), iap_share(final, T["iap"])],
}
print("\nGuardrails, ranker output before -> after:")
for k_, v in guard.items():
    if isinstance(v, list):
        print(f"  {k_:<16} {v[0]:.4f} -> {v[1]:.4f}")

inv = {v: k for k, v in ds["item_index"].items()}
name = lambda i: ds["names"].get(inv[i], inv[i])
examples = {name(int(i)): [name(j) for j, _ in similar_items(als, int(i), 5)] for i in np.argsort(-pop)[:3]}
print("\nCross-sell examples:")
for k_, v in examples.items():
    print(f"  {k_}: {v}")

# ---- save artifacts for the API ---------------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)
skip = {"mh_norm", "sim", "heavy"}  # derived at load time
np.savez_compressed(os.path.join(MODELS_DIR, "tables.npz"), **{k: v for k, v in T.items() if k not in skip})
sp.save_npz(os.path.join(MODELS_DIR, "seen.npz"), seen)
ranker.save_model(os.path.join(MODELS_DIR, "ranker.txt"))
with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
    json.dump({"user_ids": user_ids, "item_ids": [inv[i] for i in range(n_items)],
               "item_names": [name(i) for i in range(n_items)], "heavy_threshold": heavy_thr}, f)
os.makedirs("outputs", exist_ok=True)
with open("outputs/results.json", "w") as f:
    json.dump({"metrics": results, "ndcg10_bootstrap": comparisons, "guardrails": guard,
               "candidate_recall_first2000": cand_recall,
               "feature_importance": {k: float(v) for k, v in imp.items()}, "cross_sell": examples,
               "n_users": n_users, "n_items": n_items, "eval_players": int(len(eval_users))}, f, indent=2)
tick(f"saved artifacts to {MODELS_DIR}/ and outputs/results.json")
