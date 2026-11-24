"""Recent-inactivity ("lapse") model.

Honest framing: the data is one snapshot with no dates, so there is no true churn event. The label is
"no playtime at all in the last two weeks" (playtime_2weeks == 0 for every game) among players with at least
10 lifetime hours. Features use LIFETIME playtime only, never the two-week column, so the label is not a feature.
Scores are cross-fitted (each player is scored by a model that never saw them), so reported AUC is honest.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

from features import _price

MIN_LIFETIME_HOURS = 10.0
CHURN_FEATURES = ["n_owned", "n_played", "n_positive", "log_total_hours", "top1_share", "top3_share",
                  "median_hours_played", "unplayed_share", "f2p_share"]


def build_frame(items_df, games):
    df = items_df.copy()
    df["hours"] = df["playtime"] / 60.0
    price_bucket = games.set_index("id")["price"].map(_price)
    df["f2p"] = (df["item_id"].map(price_bucket) == 0).astype(float)
    df = df.sort_values(["user_id", "hours"], ascending=[True, False])
    g = df.groupby("user_id")
    total = g["hours"].sum()
    top1 = g["hours"].max()
    top3 = df.groupby("user_id").head(3).groupby("user_id")["hours"].sum()
    played = df[df["hours"] > 0]
    frame = pd.DataFrame({
        "n_owned": g.size(),
        "n_played": played.groupby("user_id").size(),
        "n_positive": df[df["playtime"] >= 120].groupby("user_id").size(),
        "log_total_hours": np.log1p(total),
        "top1_share": top1 / total.clip(lower=1e-9),
        "top3_share": top3 / total.clip(lower=1e-9),
        "median_hours_played": played.groupby("user_id")["hours"].median(),
        "unplayed_share": 1 - played.groupby("user_id").size() / g.size(),
        "f2p_share": g["f2p"].mean(),
        "recent_hours": g["playtime_2w"].sum() / 60.0,
        "total_hours": total,
    }).fillna(0.0)
    frame = frame[frame["total_hours"] >= MIN_LIFETIME_HOURS]
    frame["inactive"] = (frame["recent_hours"] == 0).astype(int)
    return frame


def crossfit(frame, seed=0, folds=5):
    X, y = frame[CHURN_FEATURES].values, frame["inactive"].values
    oof = np.zeros(len(frame))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                           feature_fraction=0.9, verbose=-1, seed=seed, num_threads=2),
                      lgb.Dataset(X[tr], label=y[tr], feature_name=CHURN_FEATURES), num_boost_round=200)
        oof[te] = m.predict(X[te])
    report = {
        "players": int(len(frame)), "inactive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, oof)), "pr_auc": float(average_precision_score(y, oof)),
        "auc_total_hours_only": float(roc_auc_score(y, -frame["log_total_hours"].values)),
    }
    return pd.Series(oof, index=frame.index), report
