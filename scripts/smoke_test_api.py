"""End-to-end API check (run after training):  python scripts/smoke_test_api.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fastapi.testclient import TestClient
import app as api

with TestClient(api.app) as c:
    print("health:", {k: v for k, v in c.get("/health").json().items() if k != "feature_cols"})
    meta = json.load(open(os.path.join(api.MODELS_DIR, "meta.json")))
    uid = meta["user_ids"][123]
    r = c.get("/recommend", params={"user_id": uid, "k": 5}).json()
    print("personalized:", r["source"], "| heavy:", r["heavy_recent_player"], "| latency:", r["latency_ms"])
    for it in r["items"]:
        print("  ", it["rank"], it["name"])
    cold = c.get("/recommend", params={"user_id": "no-such-player", "k": 3}).json()
    print("cold start:", cold["source"], [i["name"] for i in cold["items"]])
    item = meta["item_ids"][0]
    print("similar:", [i["name"] for i in c.get(f"/similar/{item}", params={"k": 3}).json()["similar"]])
    assert c.get("/similar/does-not-exist").status_code == 404
    assert c.get("/recommend", params={"user_id": uid, "k": 50}).status_code == 422

    # heavy players must never receive in-app-purchase games (with guardrails on)
    T = api.STATE["T"]
    heavy_ids = [meta["user_ids"][i] for i in np.nonzero(T["heavy"])[0][:50]]
    for h in heavy_ids:
        items = c.get("/recommend", params={"user_id": h, "k": 10}).json()["items"]
        assert not any(i["in_app_purchases"] for i in items), h
    print("guardrail check passed for 50 heavy players")

    # enough traffic for drift numbers, then read the monitoring snapshot
    for u in meta["user_ids"][:30]:
        c.get("/recommend", params={"user_id": u})
    m = c.get("/monitoring").json()
    print("monitoring: requests", m["requests_total"], "| p95 /recommend",
          m["latency_ms"]["/recommend:total_ms"]["p95"], "ms | counters", m["counters"], "| alerts", m["alerts"])
    drift = {f: d["status"] for f, d in m["feature_drift"].items()}
    print("drift status:", drift)
    assert m["model"].get("trained_at"), "model_metadata.json missing: rerun main.py"
    assert c.get("/dashboard").status_code == 200
print("smoke test passed")
