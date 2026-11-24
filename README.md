# UCSD Steam: Collaborative Filtering + Two-Tower Retrieval + Guardrails

Personalization project on the real UCSD Steam data (`australian_users_items`, `steam_games`), built around
what a personalization data-scientist role asks for: retrieval models, offline evaluation with significance,
cross-sell, and a constraint layer between model and player.

| Part | What it does | Library |
|---|---|---|
| Popularity | most-played games, excluding the player's own | numpy |
| ALS | collaborative filtering, confidence = 1 + 8 x log1p(hours) | `implicit` |
| Two-tower | user tower (id + taste profile) and item tower (id + genres/tags/price/year/flags), logQ-corrected in-batch softmax | PyTorch |
| Guardrails | no in-app-purchase games for high-intensity players, diversity re-rank for everyone | numpy |
| Cross-sell | "players of X also play" from ALS item vectors | `implicit` |
| Evaluation | Recall@10/20, NDCG@10, MAP@10 via `ranx`; paired bootstrap CI on NDCG@10 | `ranx` |

## Run
Put `australian_users_items.json.gz` and `steam_games.json.gz` in `data/`, then:
```bash
pip install -r requirements.txt
python src/main.py        # ~6 min on 2 CPUs; writes outputs/results.json
```

## Data facts and choices (all checked on the real files)
- 88,310 lines but only 70,912 distinct users: duplicate user lines are merged (max playtime per user+game).
- 36% of owned games were never played, so **a positive is playtime >= 120 minutes**, not ownership.
- Kept players with >= 10 positives and games with >= 20 positives: 45,791 players, 3,750 games, 1.82M positives.
- Playtime is in minutes. **There is no timestamp**, so a time-based split is impossible. A random 20% of each
  player's positives is held out. This overstates real-world accuracy compared with a time split.
- Items a player liked in training are never recommended. Owned-but-unplayed games stay recommendable.
- Two-tower leakage guard: the taste profile is recomputed without the target game for each training pair.

## Results (single run, seed 0; see outputs/results.json)
See the table printed by `main.py`. Differences are judged by the bootstrap CI, not by the means.

## Limits
- Ownership data reflects what people bought, not what a recommender showed them. Offline accuracy here does not
  predict online lift.
- The guardrail layer is an **illustration**. Steam is not gambling, and "in-app purchases" is a stand-in for
  spend-triggering content. A real responsible-gaming policy needs domain rules and review.
- No tuning was done. One seed. Only NDCG@10 has a significance test.
