"""Load the UCSD Steam files and build the train/test split.

Files (put them in data/):
  australian_users_items.json.gz   one Python-literal dict per line: user_id, items[{item_id, item_name, playtime_forever, playtime_2weeks}]
  steam_games.json.gz              one Python-literal dict per line: id, genres, tags, specs, price, release_date, early_access ...

Data facts that shape the design (checked on the real files):
  * Playtime is in minutes. There is NO timestamp, so a time-based split is impossible; we hold out
    a random 20% of each user's positives instead. This is weaker than a time split and is stated in the README.
  * Some user_ids appear on more than one line. Duplicates are merged (max playtime per user+item).
  * 36% of owned games were never played (playtime 0). Owning is not engagement, so a positive is
    playtime_forever >= 120 minutes.
"""
import ast
import gzip
import os
import numpy as np
import pandas as pd

POSITIVE_MINUTES = 120
MIN_USER_POSITIVES = 10
MIN_ITEM_POSITIVES = 20
TEST_FRACTION = 0.2


def _read_literals(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield ast.literal_eval(line)  # files are Python literals (single quotes, u'' prefixes), not JSON


def load_items(path):
    rows, names = [], {}
    for d in _read_literals(path):
        for it in d["items"]:
            rows.append((d["user_id"], it["item_id"], it["playtime_forever"], it["playtime_2weeks"]))
            names[it["item_id"]] = it["item_name"]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "playtime", "playtime_2w"])
    # merge duplicate user lines
    df = df.groupby(["user_id", "item_id"], as_index=False).max()
    return df, names


def load_games(path):
    return pd.DataFrame(list(_read_literals(path))).drop_duplicates("id")


def build_dataset(data_dir="data", seed=0):
    items_df, names = load_items(os.path.join(data_dir, "australian_users_items.json.gz"))
    games = load_games(os.path.join(data_dir, "steam_games.json.gz"))

    recent_hours = items_df.groupby("user_id")["playtime_2w"].sum() / 60.0

    pos = items_df[items_df["playtime"] >= POSITIVE_MINUTES]
    keep_items = pos.groupby("item_id").size()
    keep_items = keep_items[keep_items >= MIN_ITEM_POSITIVES].index
    pos = pos[pos["item_id"].isin(keep_items)]
    ucount = pos.groupby("user_id").size()
    pos = pos[pos["user_id"].isin(ucount[ucount >= MIN_USER_POSITIVES].index)].copy()

    user_ids = sorted(pos["user_id"].unique())
    item_ids = sorted(pos["item_id"].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {m: i for i, m in enumerate(item_ids)}
    pos["u"] = pos["user_id"].map(user_index)
    pos["i"] = pos["item_id"].map(item_index)

    # Random per-user holdout of 20% of positives.
    rng = np.random.default_rng(seed)
    pos["rand"] = rng.random(len(pos))
    pos["rank"] = pos.groupby("u")["rand"].rank(method="first")
    n = pos.groupby("u")["rand"].transform("size")
    n_test = np.maximum(1, np.round(n * TEST_FRACTION)).astype(int)
    is_test = pos["rank"] <= n_test
    train, test = pos[~is_test].reset_index(drop=True), pos[is_test].reset_index(drop=True)

    return {
        "train": train, "test": test, "user_index": user_index, "item_index": item_index,
        "games": games, "names": names,
        "recent_hours": recent_hours.reindex(user_ids).fillna(0.0).values,
    }
