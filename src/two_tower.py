"""Two-tower retrieval model (PyTorch).

User tower: user-id embedding + the user's taste profile (mean of the item feature vectors of their
            training positives) + a bucket for how many games they play -> MLP -> L2-normalised vector.
Item tower: item-id embedding + feature vector (genres, tags, early access, in-app purchases, has-metadata)
            + price bucket + release-year bucket -> MLP -> L2-normalised vector.
Training: in-batch sampled softmax with logQ correction for popularity bias; duplicate items in a batch masked.

Leakage guard: the taste profile is built from the same positives used as training labels, so for a
(user, item) training pair the profile is recomputed WITHOUT that item. At inference the full profile is used.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_COUNT_BUCKETS = 8


def count_bucket(n):
    return np.minimum(np.log2(np.maximum(n, 1)).astype(np.int64), N_COUNT_BUCKETS - 1)


class Tower(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, feat_dim, emb=32, out_dim=32, hidden=128):
        super().__init__()
        self.user_id = nn.Embedding(n_users, emb)
        self.count = nn.Embedding(N_COUNT_BUCKETS, 8)
        self.item_id = nn.Embedding(n_items, emb)
        self.price = nn.Embedding(7, 8)
        self.year = nn.Embedding(15, 8)
        self.user_tower = Tower(emb + feat_dim + 8, hidden, out_dim)
        self.item_tower = Tower(emb + feat_dim + 16, hidden, out_dim)

    def user_vec(self, u, profile, bucket):
        return self.user_tower(torch.cat([self.user_id(u), profile, self.count(bucket)], -1))

    def item_vec(self, i, mh, price, year):
        return self.item_tower(torch.cat([self.item_id(i), mh, self.price(price), self.year(year)], -1))


def user_profiles(train, n_users, mh):
    """Sum of item feature vectors per user, and positive counts."""
    import scipy.sparse as sp
    m = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train["u"].values, train["i"].values)),
                      shape=(n_users, mh.shape[0]))
    return np.asarray(m @ mh, dtype=np.float32), np.asarray(m.sum(1)).ravel().astype(np.float32)


def train_two_tower(train, n_users, n_items, feats, epochs=8, batch_size=2048, lr=2e-3,
                    temperature=0.1, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mh = torch.tensor(feats["mh"])
    price, year = torch.tensor(feats["price"]), torch.tensor(feats["year"])
    psum, pcnt = user_profiles(train, n_users, feats["mh"])
    psum, pcnt = torch.tensor(psum), torch.tensor(pcnt)
    bucket = torch.tensor(count_bucket(pcnt.numpy()))

    u = train["u"].values.astype(np.int64)
    i = train["i"].values.astype(np.int64)
    counts = np.bincount(i, minlength=n_items).astype(np.float64) + 1.0
    log_q = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)

    model = TwoTower(n_users, n_items, mh.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(u))
        total, steps = 0.0, 0
        for s in range(0, len(u), batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:
                continue
            ub, ib = torch.tensor(u[idx]), torch.tensor(i[idx])
            prof = (psum[ub] - mh[ib]) / (pcnt[ub] - 1).clamp(min=1).unsqueeze(1)  # profile without target
            uv = model.user_vec(ub, prof, bucket[ub])
            iv = model.item_vec(ib, mh[ib], price[ib], year[ib])
            logits = uv @ iv.T / temperature - log_q[ib][None, :]
            same = ib[:, None] == ib[None, :]
            same.fill_diagonal_(False)
            logits = logits.masked_fill(same, -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(idx)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        losses.append(total / max(steps, 1))
        print(f"  two-tower epoch {ep + 1}/{epochs} loss {losses[-1]:.3f}", flush=True)
    return model, losses, (psum, pcnt)


class TwoTowerRetriever:
    def __init__(self, model, feats, profiles):
        model.eval()
        self.model = model
        self.psum, self.pcnt = profiles
        self.bucket = torch.tensor(count_bucket(self.pcnt.numpy()))
        with torch.no_grad():
            self.item_vecs = model.item_vec(torch.arange(feats["mh"].shape[0]), torch.tensor(feats["mh"]),
                                            torch.tensor(feats["price"]), torch.tensor(feats["year"]))

    def all_user_vecs(self):
        u = torch.arange(len(self.pcnt))
        prof = self.psum / self.pcnt.clamp(min=1).unsqueeze(1)
        with torch.no_grad():
            return self.model.user_vec(u, prof, self.bucket).numpy()

    def recommend(self, user_idx, seen_matrix, k, chunk=4096):
        out = []
        for s in range(0, len(user_idx), chunk):
            ids = np.asarray(user_idx[s:s + chunk])
            u = torch.tensor(ids.astype(np.int64))
            prof = self.psum[u] / self.pcnt[u].clamp(min=1).unsqueeze(1)
            with torch.no_grad():
                scores = (self.model.user_vec(u, prof, self.bucket[u]) @ self.item_vecs.T).numpy()
            scores[seen_matrix[ids].toarray() > 0] = -np.inf
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            order = np.take_along_axis(scores, top, 1).argsort(1)[:, ::-1]
            out.append(np.take_along_axis(top, order, 1))
        return np.vstack(out)


# Entry point for two_tower_proc.vectors_in_subprocess (see there for why this runs in its own process).

if __name__ == "__main__":
    import sys
    import pandas as pd
    with np.load(sys.argv[1]) as d:
        _train = pd.DataFrame({"u": d["u"], "i": d["i"]})
        _feats = {k: d[k] for k in ("mh", "price", "year")}
        _n_users, _n_items = int(d["n_users"]), int(d["n_items"])
    _model, _, _profiles = train_two_tower(_train, _n_users, _n_items, _feats)
    _tt = TwoTowerRetriever(_model, _feats, _profiles)
    np.savez(sys.argv[2], user_vecs=_tt.all_user_vecs(), item_vecs=_tt.item_vecs.numpy())
