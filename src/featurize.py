"""Feature tables and (user, item) pair features. Numpy only, shared by training and the API,
so the ranker sees identical features online and offline (no train/serve skew)."""
import numpy as np

FEATURES = ["als_score", "tt_score", "als_rank", "tt_rank", "item_pop", "item_price", "item_year",
            "item_iap", "item_early", "item_n_genres", "item_has_meta", "user_n_pos", "user_recent_h",
            "user_iap_share", "user_mean_price", "affinity", "price_gap"]


def build_tables(train, n_users, n_items, feats, als_user, als_item, tt_user, tt_item,
                 recent_hours, matrix):
    """All per-user and per-item arrays needed after training. User stats come from TRAIN positives only."""
    import scipy.sparse as sp
    mh = feats["mh"]
    m = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train["u"].values, train["i"].values)),
                      shape=(n_users, n_items))
    n_pos = np.asarray(m.sum(1)).ravel().astype(np.float32)
    psum = np.asarray(m @ mh, dtype=np.float32)
    profile = psum / np.maximum(np.linalg.norm(psum, axis=1, keepdims=True), 1e-9)
    price = feats["price"].astype(np.float32)
    iap = feats["iap"].astype(np.float32)
    denom = np.maximum(n_pos, 1)
    return {
        "als_user": np.asarray(als_user, dtype=np.float32), "als_item": np.asarray(als_item, dtype=np.float32),
        "tt_user": np.asarray(tt_user, dtype=np.float32), "tt_item": np.asarray(tt_item, dtype=np.float32),
        "mh": mh, "price": feats["price"], "year": feats["year"], "iap": feats["iap"],
        "pop": np.asarray(m.sum(0)).ravel().astype(np.float32),
        "n_pos": n_pos, "profile": profile.astype(np.float32),
        "user_mean_price": np.asarray(m @ price).ravel() / denom,
        "user_iap_share": np.asarray(m @ iap).ravel() / denom,
        "recent_hours": recent_hours.astype(np.float32),
        "n_genres": np.array(feats["n_genres"]),
    }


def derive(T, heavy_threshold):
    """Arrays computed from the saved ones; run after build_tables and after loading artifacts."""
    from guardrails import cosine_matrix
    mh = T["mh"]
    T["mh_norm"] = (mh / np.maximum(np.linalg.norm(mh, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    T["sim"] = cosine_matrix(mh).astype(np.float32)
    T["heavy"] = T["recent_hours"] >= heavy_threshold
    return T


def pair_features(users, cand, als_rank, tt_rank, T, chunk=2000):
    """Returns X [n_pairs, F], and (row, col) positions of each pair in `cand`. cand is -1 padded."""
    Xs, rows, cols = [], [], []
    n_genres = int(T["n_genres"])
    early_col, has_meta_col = T["mh"].shape[1] - 3, T["mh"].shape[1] - 1
    for s in range(0, len(users), chunk):
        u, c = users[s:s + chunk], cand[s:s + chunk]
        r, cc = np.nonzero(c >= 0)
        uu, ii = u[r], c[r, cc]
        f = np.column_stack([
            (T["als_user"][uu] * T["als_item"][ii]).sum(1),
            (T["tt_user"][uu] * T["tt_item"][ii]).sum(1),
            als_rank[s:s + chunk][r, cc], tt_rank[s:s + chunk][r, cc],
            np.log1p(T["pop"][ii]), T["price"][ii], T["year"][ii], T["iap"][ii],
            T["mh"][ii, early_col], T["mh"][ii, :n_genres].sum(1), T["mh"][ii, has_meta_col],
            np.log1p(T["n_pos"][uu]), np.log1p(T["recent_hours"][uu]),
            T["user_iap_share"][uu], T["user_mean_price"][uu],
            (T["profile"][uu] * T["mh_norm"][ii]).sum(1),
            T["price"][ii] - T["user_mean_price"][uu],
        ]).astype(np.float32)
        Xs.append(f)
        rows.append(r + s)
        cols.append(cc)
    return np.vstack(Xs), np.concatenate(rows), np.concatenate(cols)
