"""Full online recommendation path: two retrievers -> merge -> LightGBM ranker -> guardrails."""
import numpy as np
from retrieval import retrieve_topk, build_candidates
from ranker import rank_candidates
from guardrails import apply_guardrails

N_RETRIEVE = 50


def recommend(users, T, ranker, seen, k=10):
    """users: array of user indices. Returns (final [n,k] after guardrails, ranked [n,50] before guardrails)."""
    als_top = retrieve_topk(T["als_user"][users], T["als_item"], seen[users], N_RETRIEVE)
    tt_top = retrieve_topk(T["tt_user"][users], T["tt_item"], seen[users], N_RETRIEVE)
    cand, ar, tr = build_candidates(als_top, tt_top)
    ranked = rank_candidates(ranker, users, cand, ar, tr, T, top=N_RETRIEVE)
    final = apply_guardrails(ranked, T["heavy"][users], T["iap"], T["mh"], k=k, sim=T["sim"])
    return final, ranked
