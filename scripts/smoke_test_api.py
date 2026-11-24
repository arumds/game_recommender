"""End-to-end API check (run after training):  python scripts/smoke_test_api.py"""
import json
import os
import sys

sys.path.insert(0, "src")
from fastapi.testclient import TestClient
import serve

with TestClient(serve.app) as c:
    print("health:", c.get("/health").json())
    meta = json.load(open(os.path.join(serve.MODELS_DIR, "meta.json")))
    uid = meta["user_ids"][123]
    r = c.post("/recommend", json={"user_id": uid, "k": 5}).json()
    print("personalized:", r["source"], "| heavy:", r["heavy_recent_player"], "| inactivity_risk:", r["inactivity_risk"])
    for it in r["items"]:
        print("  ", it["rank"], it["name"])
    cold = c.post("/recommend", json={"user_id": "no-such-player", "k": 3}).json()
    print("cold start:", cold["source"], [i["name"] for i in cold["items"]])
    item = meta["item_ids"][0]
    print("similar:", [i["name"] for i in c.get(f"/similar/{item}?k=3").json()["similar"]])
    assert c.get("/similar/does-not-exist").status_code == 404
    assert c.post("/recommend", json={"user_id": uid, "k": 50}).status_code == 422
    # heavy player must never receive in-app-purchase games
    import numpy as np
    T = serve.STATE["T"]
    heavy_ids = [meta["user_ids"][i] for i in np.nonzero(T["heavy"])[0][:50]]
    iap_names = {meta["item_names"][i] for i in np.nonzero(T["iap"])[0]}
    for h in heavy_ids:
        names = {i["name"] for i in c.post("/recommend", json={"user_id": h, "k": 10}).json()["items"]}
        assert not (names & iap_names), h
    print("guardrail check passed for 50 heavy players")
