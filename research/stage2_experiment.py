# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Stage-2 selection experiment: score-order vs MMR vs activation-coverage.

All three selectors receive the SAME top-N candidate set from one stage-1+PPR
pass (the ORIGINAL LinearRAG stage 1), so any metric difference is attributable
to the SELECTION RULE alone:

  score  - take the top-k by PPR score (what LinearRAG does today)
  mmr    - Carbonell&Goldstein MMR: relevance minus max embedding-cosine to the
           already-chosen set (generic diversity)
  cover  - greedy submodular coverage: relevance plus the activation mass of
           the graph entities this passage newly covers (query-conditioned)

mmr and cover are the SAME greedy skeleton with a different redundancy signal,
so mmr is the controlled baseline that isolates whether the graph-activation
signal (not generic diversity) is what helps. 'oracle' is the ceiling: could a
perfect selector fit all gold into top-k from these candidates.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".claude/skills/linearrag/scripts")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever, _residual_query  # noqa: E402


def _minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def select_score(n_cand, k, **_):
    return list(range(min(k, n_cand)))


def select_cover(ent, rel_norm, w_ent, k, alpha):
    """Greedy: alpha*relevance + newly-covered (weighted) entity mass.

    w_ent maps entity id -> coverage weight (already rarity-adjusted if desired,
    and normalized so a passage's coverage gain is comparable to rel_norm)."""
    covered, chosen, remaining = set(), [], set(range(len(ent)))
    while len(chosen) < k and remaining:
        best, bg = None, -1e18
        for i in remaining:
            new = [e for e in ent[i] if e not in covered]
            cg = float(w_ent[new].sum()) if new else 0.0
            g = alpha * rel_norm[i] + cg
            if g > bg:
                bg, best = g, i
        chosen.append(best)
        covered.update(int(e) for e in ent[best])
        remaining.discard(best)
    return chosen


def rrf_order(order_a, order_b, k, kk=60):
    """RRF fuse two rankings (lists of candidate indices)."""
    score = {}
    for order in (order_a, order_b):
        for rank, i in enumerate(order):
            score[i] = score.get(i, 0.0) + 1.0 / (kk + rank + 1)
    return sorted(score, key=lambda i: -score[i])[:k]


def select_cover_soft(ent, rel_norm, w_ent, k, alpha, gamma):
    """A: saturating (soft) coverage. Covering an entity the m-th time yields
    w_e * gamma**m instead of 0. gamma=0 reduces to hard coverage; gamma=1 to no
    dedup. Keeps a bridge entity worth *some* credit on the 2nd gold so a shared
    bridge no longer zeroes out the answer-side passage (fixes pair-breaking).
    Submodular for gamma in [0,1] (concave saturation)."""
    from collections import defaultdict
    count = defaultdict(int)
    chosen, remaining = [], set(range(len(ent)))
    while len(chosen) < k and remaining:
        best, bg = None, -1e18
        for i in remaining:
            cg = float(sum(w_ent[e] * (gamma ** count[e]) for e in ent[i]))
            g = alpha * rel_norm[i] + cg
            if g > bg:
                bg, best = g, i
        chosen.append(best)
        for e in ent[best]:
            count[int(e)] += 1
        remaining.discard(best)
    return chosen


def select_residual(emb_cand, q_base, k, strength, ent=None, w_ent=None,
                    alpha=0.0):
    """C: residual-query slot selection — the Stage-1 residual math applied to
    passage slots. Pick slot 1 by dense sim; project the chosen passages'
    directions out of the query; re-score the rest by the RESIDUAL query; repeat.
    A passage that merely paraphrases an already-chosen one goes orthogonal to
    the residual and sinks; a passage answering the unmet part of the query
    rises. If ent/w_ent given, add activation coverage on top (C+cover)."""
    chosen, remaining, covered = [], list(range(len(emb_cand))), set()
    while len(chosen) < k and remaining:
        if chosen and strength > 0:
            q_res = _residual_query(q_base, emb_cand[chosen], strength)
        else:
            q_res = q_base
        dscore = emb_cand @ q_res
        if ent is None:
            best = max(remaining, key=lambda i: dscore[i])
        else:
            dmm = _minmax(dscore)
            best, bg = None, -1e18
            for i in remaining:
                new = [e for e in ent[i] if e not in covered]
                cg = float(w_ent[new].sum()) if new else 0.0
                g = alpha * dmm[i] + cg
                if g > bg:
                    bg, best = g, i
            covered.update(int(e) for e in ent[best])
        chosen.append(best)
        remaining.remove(best)
    return chosen


def select_mmr(sims, rel_norm, k, lam):
    """Greedy MMR: lam*relevance - (1-lam)*max cosine to chosen."""
    chosen, remaining = [], set(range(sims.shape[0]))
    while len(chosen) < k and remaining:
        best, bv = None, -1e18
        for i in remaining:
            red = max((sims[i, j] for j in chosen), default=0.0)
            v = lam * rel_norm[i] - (1.0 - lam) * red
            if v > bv:
                bv, best = v, i
        chosen.append(best)
        remaining.discard(best)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--top-n", type=int, default=30, help="candidate depth")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alphas", default="0.3,0.5,1.0,2.0")
    ap.add_argument("--mmr-lams", default="0.5,0.7")
    ap.add_argument("--gammas", default="0.3,0.5", help="soft-coverage saturation")
    ap.add_argument("--res-strengths", default="1.0", help="residual projection")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--type", default=None)
    ap.add_argument("--focus", action="store_true",
                    help="only the fix-candidate comparison (A/B/C vs refs)")
    ap.add_argument("--s1", default="baseline", choices=["baseline", "resirag"],
                    help="stage-1 that generates candidates+activation")
    a = ap.parse_args()

    # stage-1 params fed to stage2_candidates: baseline (original LinearRAG) or
    # ResiRAG (residual-query + relative threshold + adaptive gate).
    s1_kw = ({} if a.s1 == "baseline" else
             dict(residual_strength=1.0, residual_k=3, threshold_mode="relative",
                  delta_rel=0.4, adaptive=True, adapt_lo=2, adapt_hi=20))

    index = TriGraphIndex.load(a.index)
    embed = Embedder(index.meta["embedding_model"])
    retr = ResiRetriever(index, embed, load_nlp(index.meta["spacy_model"]))
    C = index.C.tocsr()
    embP = np.asarray(index.emb_passages)

    qs = json.load(open(a.questions, encoding="utf-8"))
    if a.type:
        qs = [q for q in qs if q.get("type") == a.type]
    qs = qs[: a.limit]
    alphas = [float(x) for x in a.alphas.split(",")]
    lams = [float(x) for x in a.mmr_lams.split(",")]
    gammas = [float(x) for x in a.gammas.split(",")]
    res_s = [float(x) for x in a.res_strengths.split(",")]

    # A/B/C fix candidates (alpha fixed at 1.0, the coverDense reference point)
    abc = [f"coverSoft@{g}" for g in gammas] + ["coverDual@1.0", "dualAnchor"] + \
          [f"residual@{s}" for s in res_s] + [f"residualCover@{s}" for s in res_s]
    if a.focus:
        conds = ["score", "oracle", "dense", "coverDense@1.0"] + abc
    else:
        conds = ["score", "oracle", "dense", "rrf_ppr_dense"] + \
                [f"mmr@{l}" for l in lams] + [f"denseMMR@{l}" for l in lams] + \
                [f"cover@{al}" for al in alphas] + \
                [f"coverIDF@{al}" for al in alphas] + \
                [f"coverDense@{al}" for al in alphas] + \
                [f"coverUniform@{al}" for al in alphas] + \
                [f"coverAll@{al}" for al in alphas] + abc
    rec = {c: 0.0 for c in conds}
    allg = {c: 0 for c in conds}
    n_gold = 0

    retr.stage2_candidates(qs[0]["question"], top_n=a.top_n, **s1_kw)  # warm
    for q in qs:
        gold = set(q.get("gold_ids", []))
        if not gold:
            continue
        n_gold += 1
        cd = retr.stage2_candidates(q["question"], top_n=a.top_n, **s1_kw)
        cand_ids = cd["cand_ids"]
        rel_norm = _minmax(cd["rel"])
        ent = [C[r].indices for r in cd["cand_rows"]]
        w = cd["activation"] / np.maximum(cd["levels"], 1)
        # per-candidate coverage gain normalized to ~[0,1] so alpha balances it
        # against rel_norm: scale by the single best passage's raw coverage.
        best_cov = max((float(w[e].sum()) for e in ent), default=0.0) or 1.0
        w_cov = w / best_cov
        # rarity (IDF) variant: an entity in many candidates (e.g. the hop-0
        # seed that every crowding passage shares) is downweighted, so covering
        # a RARE activated entity (the discriminative bridge/answer side) wins.
        df = np.zeros_like(w)
        for e_arr in ent:
            df[e_arr] += 1.0
        idf = np.log(1.0 + len(ent) / np.maximum(df, 1.0))
        w_idf = w * idf
        best_idf = max((float(w_idf[e].sum()) for e in ent), default=0.0) or 1.0
        w_idf = w_idf / best_idf
        # CONTROLS to isolate what the coverage dedup owes to the activation
        # signal vs generic entity-overlap dedup:
        #  uniform : cover ACTIVATED entities with equal weight (ignore magnitude)
        #  all     : cover ALL entities with equal weight (drop the activation
        #            restriction entirely -> generic entity-overlap dedup)
        w_uni = (cd["levels"] > 0).astype(np.float64)
        w_uni = w_uni / (max((float(w_uni[e].sum()) for e in ent), default=0.0) or 1.0)
        w_all = np.ones_like(w)
        w_all = w_all / (max((float(w_all[e].sum()) for e in ent), default=0.0) or 1.0)
        emb_cand = embP[cd["cand_rows"]]
        sims = emb_cand @ emb_cand.T
        dsim = emb_cand @ cd["query_vec"]
        dense_norm = _minmax(dsim)
        dense_order = list(np.argsort(dsim)[::-1])
        ppr_order = list(range(len(cand_ids)))  # candidates already sorted by PPR

        def score_set(chosen_idx):
            got = {cand_ids[i] for i in chosen_idx} & gold
            return len(got) / len(gold), (got == gold)

        q_base = cd["query_vec"] / (np.linalg.norm(cd["query_vec"]) + 1e-12)
        rel_dual = 0.5 * rel_norm + 0.5 * dense_norm  # B: PPR+dense fused anchor
        dual_order = list(np.argsort(rel_dual)[::-1])

        picks = {"score": select_score(len(cand_ids), a.k),
                 "dense": dense_order[:a.k],
                 "coverDense@1.0": select_cover(ent, dense_norm, w_idf, a.k, 1.0)}
        # A: soft/saturating coverage (dense anchor, alpha=1.0)
        for g in gammas:
            picks[f"coverSoft@{g}"] = select_cover_soft(
                ent, dense_norm, w_idf, a.k, 1.0, g)
        # B: dual anchor (PPR+dense) with/without activation coverage
        picks["coverDual@1.0"] = select_cover(ent, rel_dual, w_idf, a.k, 1.0)
        picks["dualAnchor"] = dual_order[:a.k]
        # C: residual-query slot selection (pure) and + activation coverage
        for s in res_s:
            picks[f"residual@{s}"] = select_residual(emb_cand, q_base, a.k, s)
            picks[f"residualCover@{s}"] = select_residual(
                emb_cand, q_base, a.k, s, ent=ent, w_ent=w_idf, alpha=1.0)
        if not a.focus:
            picks["rrf_ppr_dense"] = rrf_order(ppr_order, dense_order, a.k)
            for l in lams:
                picks[f"mmr@{l}"] = select_mmr(sims, rel_norm, a.k, l)
                picks[f"denseMMR@{l}"] = select_mmr(sims, dense_norm, a.k, l)
            for al in alphas:
                picks[f"cover@{al}"] = select_cover(ent, rel_norm, w_cov, a.k, al)
                picks[f"coverIDF@{al}"] = select_cover(ent, rel_norm, w_idf, a.k, al)
                picks[f"coverDense@{al}"] = select_cover(ent, dense_norm, w_idf, a.k, al)
                picks[f"coverUniform@{al}"] = select_cover(ent, dense_norm, w_uni, a.k, al)
                picks[f"coverAll@{al}"] = select_cover(ent, dense_norm, w_all, a.k, al)
        # oracle: best achievable from candidates with k slots
        gold_in_cand = [i for i, cid in enumerate(cand_ids) if cid in gold]
        picks["oracle"] = gold_in_cand[:a.k]

        for c, idx in picks.items():
            if c not in rec:  # not in the selected conds set (focus mode)
                continue
            r, ag = score_set(idx)
            rec[c] += r
            allg[c] += ag

    print(f"index={a.index}  n_gold={n_gold}  top_n={a.top_n}  k={a.k}")
    print(f"{'selector':16s} {'GoldRecall':>11s} {'AllGoldHit':>11s}")
    for c in conds:
        print(f"{c:16s} {rec[c]/n_gold:>11.1%} {allg[c]/n_gold:>11.1%}")


if __name__ == "__main__":
    main()
