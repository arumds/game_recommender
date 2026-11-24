"""FastAPI service: loads the artifacts written by main.py, no training and no torch.

  POST /recommend {"user_id": "...", "k": 10}   personalised list (retrieval -> ranker -> guardrails)
  GET  /similar/{item_id}?k=5                   cross-sell neighbours
  GET  /health

Unknown players get a popularity list (cold start). Run:  uvicorn serve:app --host 0.0.0.0 --port 8000
"""
import json
import os
import sys
from contextlib import asynccontextmanager

import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from featurize import derive
from pipeline import recommend

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
STATE = {}


@asynccontextmanager
async def lifespan(app):
    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        meta = json.load(f)
    T = dict(np.load(os.path.join(MODELS_DIR, "tables.npz")))
    derive(T, meta["heavy_threshold"])
    item_vecs = T["als_item"] / np.maximum(np.linalg.norm(T["als_item"], axis=1, keepdims=True), 1e-9)
    STATE.update(
        T=T, meta=meta, seen=sp.load_npz(os.path.join(MODELS_DIR, "seen.npz")).tocsr(),
        ranker=lgb.Booster(model_file=os.path.join(MODELS_DIR, "ranker.txt")),
        user_index={u: i for i, u in enumerate(meta["user_ids"])},
        item_index={m: i for i, m in enumerate(meta["item_ids"])},
        item_unit=item_vecs, popular=np.argsort(-T["pop"]),
    )
    yield
    STATE.clear()


app = FastAPI(title="Steam recommender", lifespan=lifespan)


class Request(BaseModel):
    user_id: str
    k: int = Field(10, ge=1, le=10)  # guardrail re-rank selects from the top 50 and is built for k <= 10


def _items(idx, scores=None):
    names = STATE["meta"]["item_names"]
    ids = STATE["meta"]["item_ids"]
    return [{"rank": r + 1, "item_id": ids[i], "name": names[i]} for r, i in enumerate(idx)]


@app.get("/health")
def health():
    return {"status": "ok", "players": len(STATE["user_index"]), "games": len(STATE["item_index"])}


@app.post("/recommend")
def recommend_endpoint(req: Request):
    T = STATE["T"]
    u = STATE["user_index"].get(req.user_id)
    if u is None:
        return {"user_id": req.user_id, "source": "popularity_fallback", "items": _items(STATE["popular"][:req.k])}
    final, _ = recommend(np.array([u]), T, STATE["ranker"], STATE["seen"], k=req.k)
    churn = float(T["churn"][u])
    return {
        "user_id": req.user_id, "source": "personalized",
        "heavy_recent_player": bool(T["heavy"][u]),
        "inactivity_risk": None if np.isnan(churn) else round(churn, 3),
        "items": _items(final[0]),
    }


@app.get("/similar/{item_id}")
def similar(item_id: str, k: int = 5):
    i = STATE["item_index"].get(item_id)
    if i is None:
        raise HTTPException(404, "unknown item_id")
    sims = STATE["item_unit"] @ STATE["item_unit"][i]
    sims[i] = -np.inf
    top = np.argsort(-sims)[:k]
    return {"item_id": item_id, "name": STATE["meta"]["item_names"][i], "similar": _items(top)}
