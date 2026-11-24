"""Train and evaluate on UCSD Steam: popularity, ALS, two-tower, then the guardrail layer.

Run from the project root:  python src/main.py
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import build_dataset
from features import build_item_features
from cf_als import build_matrix, train_als, recommend_als, similar_items
from two_tower import train_two_tower, TwoTowerRetriever
from evaluate import to_run, to_qrels, evaluate_run, per_user, paired_bootstrap
from guardrails import heavy_user_mask, apply_guardrails, intra_list_diversity, iap_share

DATA_DIR = os.environ.get("STEAM_DATA_DIR", "data")
K = 20
t0 = time.time()

ds = build_dataset(DATA_DIR)
train, test = ds["train"], ds["test"]
n_users, n_items = len(ds["user_index"]), len(ds["item_index"])
print(f"users={n_users:,} items={n_items:,} train_pos={len(train):,} test_pos={len(test):,}  ({time.time()-t0:.0f}s)")

feats = build_item_features(ds["games"], ds["item_index"])
print(f"item metadata coverage: {feats['mh'][:, -1].mean():.1%}")

eval_users = np.arange(n_users)
qrels = to_qrels(test, eval_users)
matrix = build_matrix(train, n_users, n_items)

# 1) Popularity
pop = np.asarray((matrix > 0).sum(0)).ravel().astype(float)
# most-liked games, skipping what each user already liked
def pop_for_users(chunk=8192):
    out = []
    order = np.argsort(-pop)
    for s in range(0, n_users, chunk):
        ids = eval_users[s:s + chunk]
        seen = matrix[ids].toarray() > 0
        ranked = np.where(seen[:, order], -1, 0)  # -1 = seen
        recs = []
        for r in range(len(ids)):
            recs.append(order[ranked[r] == 0][:K])
        out.append(np.vstack(recs))
    return np.vstack(out)
pop_recs = pop_for_users()

# 2) ALS
als = train_als(matrix)
als_recs = recommend_als(als, matrix, eval_users, K)
print(f"ALS done ({time.time()-t0:.0f}s)")

# 3) Two-tower
model, losses, profiles = train_two_tower(train, n_users, n_items, feats)
tt = TwoTowerRetriever(model, feats, profiles)
tt_recs = tt.recommend(eval_users, matrix, K)
print(f"two-tower done ({time.time()-t0:.0f}s)")

runs = {n: to_run(r, eval_users, K) for n, r in
        [("popularity", pop_recs), ("als", als_recs), ("two_tower", tt_recs)]}
results = {n: evaluate_run(r, qrels) for n, r in runs.items()}
print(f"\n{'model':<12}" + "".join(f"{m:>12}" for m in results["als"]))
for n, res in results.items():
    print(f"{n:<12}" + "".join(f"{v:>12.4f}" for v in res.values()))

print("\nPaired bootstrap on per-user NDCG@10 (mean diff, 95% CI):")
nd = {n: per_user(r, qrels) for n, r in runs.items()}
comparisons = {}
for a, b in [("als", "popularity"), ("two_tower", "popularity"), ("als", "two_tower")]:
    d, lo, hi = paired_bootstrap(nd[a], nd[b])
    verdict = "significant" if lo > 0 or hi < 0 else "not distinguishable from zero"
    comparisons[f"{a}_vs_{b}"] = {"diff": d, "ci95": [lo, hi]}
    print(f"  {a} - {b}: {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")

# 4) Guardrails on ALS top-50 candidates
cand = recommend_als(als, matrix, eval_users, 50)
heavy, thr = heavy_user_mask(ds["recent_hours"])
before = cand[:, :10]
after = apply_guardrails(cand, heavy, feats["iap"], feats["mh"], k=10)
rb, ra = to_run(before, eval_users, 10), to_run(after, eval_users, 10)
gm = ["recall@10", "ndcg@10"]
mb, ma = evaluate_run(rb, qrels, gm), evaluate_run(ra, qrels, gm)
guard = {
    "heavy_users": int(heavy.sum()), "heavy_threshold_hours_2w": thr,
    "recall@10": [mb["recall@10"], ma["recall@10"]], "ndcg@10": [mb["ndcg@10"], ma["ndcg@10"]],
    "diversity": [intra_list_diversity(before, feats["mh"]), intra_list_diversity(after, feats["mh"])],
    "iap_share_heavy": [iap_share(before[heavy], feats["iap"]), iap_share(after[heavy], feats["iap"])],
    "iap_share_all": [iap_share(before, feats["iap"]), iap_share(after, feats["iap"])],
}
print(f"\nGuardrails (before -> after), heavy users={guard['heavy_users']:,} (>= {thr:.1f} h in last 2 weeks):")
for k_, v in guard.items():
    if isinstance(v, list):
        print(f"  {k_:<16} {v[0]:.4f} -> {v[1]:.4f}")

# 5) Cross-sell examples from ALS item vectors
inv = {v: k for k, v in ds["item_index"].items()}
name = lambda i: ds["names"].get(inv[i], inv[i])
print("\nCross-sell examples (ALS item similarity):")
examples = {}
popular = np.argsort(-pop)[:3]
for it in popular:
    sims = [name(j) for j, _ in similar_items(als, int(it), 5)]
    examples[name(int(it))] = sims
    print(f"  {name(int(it))}: {sims}")

os.makedirs("outputs", exist_ok=True)
with open("outputs/results.json", "w") as f:
    json.dump({"metrics": results, "ndcg10_bootstrap": comparisons, "guardrails": guard,
               "cross_sell": examples, "n_users": n_users, "n_items": n_items,
               "train_positives": len(train), "test_positives": len(test)}, f, indent=2)
print(f"\nSaved outputs/results.json ({time.time()-t0:.0f}s total)")
