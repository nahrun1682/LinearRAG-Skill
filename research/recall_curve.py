# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""LLM-free 'how much garbage / how few chunks' analysis.

Answer accuracy (e2e) confounds retrieval quality with the reader LLM's own
competence, so it is a poor isolation of the retriever. This measures the
retrieval side directly, without any LLM:

  Recall@k / AllGoldHit@k for k in {1,2,3,5,10}  -> does the selector pack gold
      into the TOP slots (fewer distractors to read) rather than merely into
      the top-5 somewhere?
  min-k(all gold)  -> how deep must the reader go to have BOTH bridge passages.
      Smaller = tighter context = less noise. inf if not within the depth.
  precision@k = |gold in top-k| / k  -> reported for completeness (at fixed k it
      tracks recall since |gold| is small, but the low-k end shows density).

Compares the PPR baseline (score) against coverSoft (soft-coverage selection).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".claude/skills/linearrag/scripts")
sys.path.insert(0, "research")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever  # noqa: E402
from stage2_experiment import _minmax, select_cover_soft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=500)
a = ap.parse_args()

index = TriGraphIndex.load(a.index)
embed = Embedder(index.meta["embedding_model"])
retr = ResiRetriever(index, embed, load_nlp(index.meta["spacy_model"]))
C = index.C.tocsr()
embP = np.asarray(index.emb_passages)
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]

KS = [1, 2, 3, 5, 10]
DEPTH = a.top_n  # order the FULL candidate pool so min-k is comparable across
#                  methods (both search the same depth; earlier bug compared
#                  score over 50 vs coverSoft over 10).
methods = ["score", "coverSoft"]
recall = {m: {k: 0.0 for k in KS} for m in methods}
allg = {m: {k: 0 for k in KS} for m in methods}
prec = {m: {k: 0.0 for k in KS} for m in methods}
mink = {m: [] for m in methods}
n = 0

retr.stage2_candidates(qs[0]["question"], top_n=a.top_n)
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    n += 1
    cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
    cand_ids = cd["cand_ids"]
    ent = [C[r].indices for r in cd["cand_rows"]]
    w = cd["activation"] / np.maximum(cd["levels"], 1)
    df = np.zeros_like(w)
    for e_arr in ent:
        df[e_arr] += 1.0
    idf = np.log(1.0 + len(ent) / np.maximum(df, 1.0))
    w_idf = w * idf
    w_idf = w_idf / (max((float(w_idf[e].sum()) for e in ent), default=0.0) or 1.0)
    dsim = embP[cd["cand_rows"]] @ cd["query_vec"]
    dense_norm = _minmax(dsim)

    orders = {
        "score": list(range(len(cand_ids))),                       # PPR order
        "coverSoft": select_cover_soft(ent, dense_norm, w_idf, DEPTH, 1.0, a.gamma),
    }
    for m, order in orders.items():
        ids = [cand_ids[i] for i in order]
        # min-k for all gold
        mk = np.inf
        for kk in range(1, len(ids) + 1):
            if gold <= set(ids[:kk]):
                mk = kk
                break
        mink[m].append(mk)
        for k in KS:
            got = len(gold & set(ids[:k]))
            recall[m][k] += got / len(gold)
            allg[m][k] += (got == len(gold))
            prec[m][k] += got / k

print(f"index={a.index}  n={n}  top_n={a.top_n}  gamma={a.gamma}")
for m in methods:
    print(f"\n--- {m} ---")
    print("  k        :  " + "  ".join(f"{k:>5}" for k in KS))
    print("  Recall@k :  " + "  ".join(f"{recall[m][k]/n:>5.1%}" for k in KS))
    print("  AllGold@k:  " + "  ".join(f"{allg[m][k]/n:>5.1%}" for k in KS))
    print("  Prec@k   :  " + "  ".join(f"{prec[m][k]/n:>5.1%}" for k in KS))
    mk = np.array(mink[m], dtype=float)
    solved = np.isfinite(mk)
    print(f"  min-k(all gold): median={np.median(mk[solved]):.0f} "
          f"mean={mk[solved].mean():.1f} (over {solved.mean():.0%} solved within {DEPTH})")
