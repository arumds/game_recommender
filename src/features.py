"""Item side-information from steam_games.json.gz (genres, tags, price, release year, flags)."""
import numpy as np

N_TAGS = 60
PRICE_BINS = [0.0, 5.0, 10.0, 20.0, 40.0]  # bucket 0 = free, then <5, <10, <20, <40, >=40 ; 6 = unknown
N_YEARS = 15  # 2005..2018 + unknown


def _price(v):
    if isinstance(v, (int, float)) and not np.isnan(v):
        if v == 0:
            return 0
        for b, edge in enumerate(PRICE_BINS[1:], start=1):
            if v < edge:
                return b
        return 5
    if isinstance(v, str) and v.lower().startswith("free"):
        return 0
    return 6


def _year(s):
    try:
        y = int(str(s)[:4])
        return int(np.clip(y - 2005, 0, N_YEARS - 2))
    except ValueError:
        return N_YEARS - 1


def build_item_features(games, item_index):
    """Returns dict: mh [n_items, F] float32 (genres + top tags + early_access + iap + has_meta),
    price [n_items] int64, year [n_items] int64, genre_cols (indices of genre columns), iap [n_items] bool."""
    g = games.set_index("id")
    n = len(item_index)
    genres = sorted({x for lst in g["genres"].dropna() for x in lst})
    tag_counts = {}
    for item_id in item_index:
        if item_id in g.index and isinstance(g.at[item_id, "tags"], list):
            for t in g.at[item_id, "tags"]:
                tag_counts[t] = tag_counts.get(t, 0) + 1
    tags = [t for t, _ in sorted(tag_counts.items(), key=lambda kv: -kv[1])[:N_TAGS]]
    vocab = {("g", x): j for j, x in enumerate(genres)}
    vocab.update({("t", x): len(genres) + j for j, x in enumerate(tags)})
    base = len(vocab)
    mh = np.zeros((n, base + 3), dtype=np.float32)
    price = np.full(n, 6, dtype=np.int64)
    year = np.full(n, N_YEARS - 1, dtype=np.int64)
    iap = np.zeros(n, dtype=bool)
    for item_id, i in item_index.items():
        if item_id not in g.index:
            continue
        r = g.loc[item_id]
        mh[i, base + 2] = 1.0  # has metadata
        for x in (r["genres"] if isinstance(r["genres"], list) else []):
            mh[i, vocab[("g", x)]] = 1.0
        for x in (r["tags"] if isinstance(r["tags"], list) else []):
            if ("t", x) in vocab:
                mh[i, vocab[("t", x)]] = 1.0
        mh[i, base] = float(bool(r["early_access"]))
        specs = r["specs"] if isinstance(r["specs"], list) else []
        iap[i] = "In-App Purchases" in specs
        mh[i, base + 1] = float(iap[i])
        price[i] = _price(r["price"])
        year[i] = _year(r["release_date"])
    return {"mh": mh, "price": price, "year": year, "iap": iap, "n_genres": len(genres),
            "genres": genres, "tags": tags}
