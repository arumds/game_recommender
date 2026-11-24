"""FastAPI service: loads the artifacts written by main.py (no training, no torch) and serves recommendations.

  GET /recommend?user_id=...&k=10&guardrails=true   retrieval -> ranker -> guardrails, with per-stage latency
  GET /similar/{item_id}?k=5                         cross-sell neighbours (ALS item similarity)
  GET /health
  GET /monitoring                                    latency percentiles, feature drift (PSI), model age, alerts
  GET /dashboard                                     the same, as a page that refreshes every 5s

Unknown players get a popularity list (cold start).
Run (any working directory):  python src/app.py     then open http://127.0.0.1:8000/docs or /dashboard
"""
import json
import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import FEATURES, derive
from retrieval import retrieve_topk, build_candidates
from ranker import rank_candidates
from guardrails import apply_guardrails
from pipeline import N_RETRIEVE
from monitoring import Monitor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))
DASHBOARD_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
MONITORED_PATHS = {"/recommend", "/similar", "/health"}  # /similar/{id} is grouped as /similar

STATE = {}  # populated at startup: model artifacts + lookups


def load_artifacts():
    required = ["meta.json", "tables.npz", "seen.npz", "ranker.txt"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing trained model artifact(s): " + ", ".join(missing) + "\n\n"
            "Run this first:\n\n    python src/main.py\n\n"
            f"Expected models directory: {MODELS_DIR}"
        )

    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        meta = json.load(f)
    T = dict(np.load(os.path.join(MODELS_DIR, "tables.npz")))
    derive(T, meta["heavy_threshold"])
    als_unit = T["als_item"] / np.maximum(np.linalg.norm(T["als_item"], axis=1, keepdims=True), 1e-9)
    return {
        "T": T,
        "meta": meta,
        "seen": sp.load_npz(os.path.join(MODELS_DIR, "seen.npz")).tocsr(),
        "ranker": lgb.Booster(model_file=os.path.join(MODELS_DIR, "ranker.txt")),
        "user_index": {u: i for i, u in enumerate(meta["user_ids"])},
        "item_index": {m: i for i, m in enumerate(meta["item_ids"])},
        "item_unit": als_unit,
        "popular": np.argsort(-T["pop"]),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading trained artifacts...")
    STATE.update(load_artifacts())
    STATE["monitor"] = Monitor(MODELS_DIR)
    print(f"Ready: {len(STATE['user_index']):,} players, {len(STATE['item_index']):,} games loaded.")
    yield
    STATE.clear()


app = FastAPI(title="Steam game recommender API", lifespan=lifespan)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        monitor = STATE.get("monitor")
        path = request.url.path
        if path.startswith("/similar/"):
            path = "/similar"
        if monitor and path in MONITORED_PATHS:
            monitor.record_request(path, status, (time.perf_counter() - t0) * 1000)


def _items(idx):
    T, meta = STATE["T"], STATE["meta"]
    return [{"rank": r + 1, "item_id": meta["item_ids"][i], "name": meta["item_names"][i],
             "in_app_purchases": bool(T["iap"][i])} for r, i in enumerate(idx)]


@app.get("/monitoring")
def monitoring():
    if not STATE:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return STATE["monitor"].snapshot()


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(DASHBOARD_HTML)


@app.get("/health")
def health():
    if not STATE:
        return {"status": "not_ready"}
    return {
        "status": "ok",
        "n_players": len(STATE["user_index"]),
        "n_games": len(STATE["item_index"]),
        "feature_cols": FEATURES,
    }


@app.get("/recommend")
def recommend(
    user_id: str = Query(..., description="Steam user id, e.g. one from models/meta.json"),
    k: int = Query(10, ge=1, le=10, description="Number of games (guardrail re-rank is built for k <= 10)"),
    guardrails: bool = Query(True, description="Apply the in-app-purchase filter for heavy players + diversity re-rank"),
):
    if not STATE:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    T, monitor = STATE["T"], STATE["monitor"]
    u = STATE["user_index"].get(user_id)
    if u is None:
        monitor.count("cold_start_fallback")
        return {"user_id": user_id, "source": "popularity_fallback", "items": _items(STATE["popular"][:k])}

    timings = {}
    t_start = time.perf_counter()
    users = np.array([u])

    # ---------------- 1. RETRIEVAL: ALS + two-tower, merged ----------------
    t0 = time.perf_counter()
    seen = STATE["seen"][users]
    als_top = retrieve_topk(T["als_user"][users], T["als_item"], seen, N_RETRIEVE)
    tt_top = retrieve_topk(T["tt_user"][users], T["tt_item"], seen, N_RETRIEVE)
    cand, ar, tr = build_candidates(als_top, tt_top)
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # ---------------- 2. RANKING: LightGBM LambdaMART ----------------
    t0 = time.perf_counter()
    ranked, X = rank_candidates(STATE["ranker"], users, cand, ar, tr, T, top=N_RETRIEVE, return_features=True)
    monitor.record_features(X, FEATURES)
    timings["ranking_ms"] = (time.perf_counter() - t0) * 1000

    # ---------------- 3. GUARDRAILS ----------------
    t0 = time.perf_counter()
    heavy = bool(T["heavy"][u])
    if guardrails:
        final = apply_guardrails(ranked, T["heavy"][users], T["iap"], T["mh"], k=k, sim=T["sim"])[0]
        if heavy:
            monitor.count("heavy_player_iap_filtered")
    else:
        final = ranked[0, :k]
    timings["guardrails_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = (time.perf_counter() - t_start) * 1000
    monitor.record_stage_timings("/recommend", timings)

    return {
        "user_id": user_id,
        "source": "personalized",
        "heavy_recent_player": heavy,
        "guardrails": guardrails,
        "n_candidates": int((cand >= 0).sum()),
        "items": _items(final),
        "latency_ms": {k_: round(v, 2) for k_, v in timings.items()},
    }


@app.get("/similar/{item_id}")
def similar(item_id: str, k: int = Query(5, ge=1, le=50)):
    if not STATE:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    i = STATE["item_index"].get(item_id)
    if i is None:
        raise HTTPException(status_code=404, detail="unknown item_id")
    sims = STATE["item_unit"] @ STATE["item_unit"][i]
    sims[i] = -np.inf
    top = np.argsort(-sims)[:k]
    return {"item_id": item_id, "name": STATE["meta"]["item_names"][i], "similar": _items(top)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
