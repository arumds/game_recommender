# UCSD Steam: Retrieval, Ranking, Guardrails and Serving

Personalization stack on the real UCSD Steam data (`australian_users_items`, `steam_games`).

```
ALS (collaborative filtering) ─┐
                               ├─ merge ─ LightGBM LambdaMART ranker ─ guardrails ─ top-10  ── FastAPI
Two-tower (PyTorch) ───────────┘
```

| Part | File | Library |
|---|---|---|
| Collaborative filtering (confidence = 1 + 8 x log1p(hours)) | `src/cf_als.py` | `implicit` |
| Two-tower, logQ-corrected in-batch softmax, leakage-free taste profile | `src/two_tower.py` | PyTorch |
| Candidate merge + dot-product retrieval | `src/retrieval.py` | numpy |
| Shared feature code (training and serving) | `src/featurize.py` | numpy |
| LambdaMART re-ranker, graded labels by hours played | `src/ranker.py` | LightGBM |
| Guardrails (no in-app-purchase games for intense recent players, diversity re-rank) | `src/guardrails.py` | numpy |
| Metrics + paired bootstrap | `src/evaluate.py` | `ranx` |
| API | `src/serve.py`, `Dockerfile` | FastAPI |

## Run
Put `australian_users_items.json.gz` and `steam_games.json.gz` in `data/`.
```bash
pip install -r requirements.txt
python src/main.py                    # ~6.5 min on 2 CPUs: trains, evaluates, writes models/ and outputs/results.json
python scripts/smoke_test_api.py      # end-to-end API check
uvicorn serve:app --app-dir src --port 8000     # then open http://127.0.0.1:8000/docs
```
Docker (serving only; train first so `models/` exists; the Dockerfile has not been built in the authoring environment):
```bash
docker build -t steam-reco-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" steam-reco-api
```
```bash
curl -X POST localhost:8000/recommend -H 'content-type: application/json' -d '{"user_id":"<a steam id>","k":10}'
```

## Data facts and choices (checked on the real files)
- 88,310 lines but 70,912 distinct players: duplicate lines are merged (max playtime per player+game).
- 36% of owned games were never played, so **a positive is playtime >= 120 minutes**.
- Kept players with >= 10 positives and games with >= 20 positives: 45,791 players, 3,750 games, 1.82M positives.
- **No timestamps**, so a random 20% of each player's positives is held out instead of a time split. Numbers are
  more optimistic than a time split would give.
- Players are split 70/30: the ranker trains on the first group, all results below are on the other 13,738 players.
- User and item statistics (popularity, taste profile, mean price) come from training positives only.
- Known leak, small: `user_recent_h` sums last-two-weeks playtime over all games, including held-out ones.

## Results (one run, seed 0; top-10, 13,738 held-out players; `outputs/results.json`)
| Model | Recall@10 | NDCG@10 | MAP@10 |
|---|---|---|---|
| Popularity | 0.207 | 0.230 | 0.119 |
| ALS | 0.247 | 0.248 | 0.129 |
| Two-tower | 0.246 | 0.259 | 0.130 |
| Rank fusion of ALS + two-tower (no learning) | 0.266 | 0.278 | 0.144 |
| **LightGBM ranker** | **0.293** | **0.321** | **0.175** |
| Ranker + guardrails | 0.274 | 0.293 | 0.152 |

Paired bootstrap on per-user NDCG@10 (95% CI): ranker beats ALS by +0.073 [+0.070, +0.075], the two-tower by
+0.062, and the non-learned fusion by +0.042 [+0.040, +0.044]. Candidate-set recall is 0.61, the ceiling for any ranker.
Guardrails cost 0.027 NDCG (CI excludes zero), cut in-app-purchase games in intense players' lists from 22.4% to
0% (2,375 intense players in the evaluation group), and raise list diversity from 0.62 to 0.76.

## Limits
- Offline accuracy on ownership data does not predict online lift. An A/B test is the real test.
- The guardrail layer illustrates the pattern. Steam is not gambling and "in-app purchases" stands in for spend-triggering
  content. A real responsible-gaming policy needs domain rules and review.
- One seed, no tuning. The API latency (about 3 ms in-process on 2 CPUs) was not load-tested.
- Cold start is a popularity list. New games have no ALS vector.
