# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Metric alignment: report OUR token-free rankings under the metrics that
published competitors use, so numbers can be compared apples-to-apples.

  GoldRecall@k  fraction of a question's gold passages in top-k (SPRIG/HippoRAG-style R@k)
  LastHop@5     the answer-bearing gold passage is in top-5 (Calibrated Fusion's
                LastHop@5, approximated: last-hop gold = gold whose text contains
                the answer string; questions with no answer-bearing gold skipped)
  AllGoldHit@5  all golds in top-5 (our strict headline metric, sanity check)

Rankings compared (all token-free): bm25, rrf, entPPR (pure entity-seeded PPR),
minrank (parameter-free merge), slot23 (tuned slots).
"""
import argparse
import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, ".claude/skills/linearrag/scripts")
from common import Embedder, TriGraphIndex, load_nlp, normalize_entity  # noqa: E402
from resirag import ResiRetriever  # noqa: E402
import query_graph as qg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--limit", type=int, default=1000)
a = ap.parse_args()

_tok = re.compile(r"\w+", re.UNICODE)
tok = lambda s: _tok.findall(s.lower())
corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
ids = [p["id"] for p in corpus]
id2row = {p["id"]: i for i, p in enumerate(corpus)}
index = TriGraphIndex.load(a.index)
bm25 = BM25Okapi([tok(p.get("title", "") + " " + p["text"]) for p in corpus])
embed = Embedder(index.meta["embedding_model"])
nlp = load_nlp(index.meta["spacy_model"])
retr = ResiRetriever(index, embed, nlp)
embP = np.asarray(index.emb_passages)
embE = np.asarray(index.emb_entities)
W, n_p, n_e = retr._W, retr._n_p, embE.shape[0]
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]


def rrf_order(oa, ob, kk=60):
    sc = {}
    for order in (oa, ob):
        for r, row in enumerate(order):
            sc[row] = sc.get(row, 0.0) + 1.0 / (kk + r + 1)
    return sorted(sc, key=lambda r: -sc[r])


def min_rank_merge(vis, hid, n):
    rv = {r: i for i, r in enumerate(vis)}
    rh = {r: i for i, r in enumerate(hid)}
    BIG = 10 ** 9
    cand = set(rv) | set(rh)
    return sorted(cand, key=lambda r: (min(rv.get(r, BIG), rh.get(r, BIG)),
                                       max(rv.get(r, BIG), rh.get(r, BIG))))[:n]


def slots(vis, hid, n_vis, n_hid, rest):
    out, seen = [], set()
    for r in vis:
        if len(out) >= n_vis: break
        if r not in seen: seen.add(r); out.append(r)
    for r in hid:
        if len(out) >= n_vis + n_hid: break
        if r not in seen: seen.add(r); out.append(r)
    for r in rest:
        if len(out) >= 50: break
        if r not in seen: seen.add(r); out.append(r)
    return out


K_DEPTH = 200
LINRAG_LAM = 0.05  # faithful LinearRAG (paper's optimal ~0.05), NOT our old 1.5 default
methods = ["bm25", "rrf", "entPPR", "linrag05", "minrank", "fuseLin", "slot23"]
agg = {m: {"gr5": [], "gr10": [], "agh5": [], "lh5": []} for m in methods}
n_lh = 0
retr.stage2_candidates(qs[0]["question"], top_n=50)
for q in qs:
    gold = [g for g in q.get("gold_ids", []) if g in id2row]
    if not gold:
        continue
    Q = q["question"]
    qv = embed([Q])[0]
    bm_scores = bm25.get_scores(tok(Q))
    bm_order = list(np.argsort(bm_scores)[::-1][:K_DEPTH * 3])
    dense_order = list(np.argsort(embP @ qv)[::-1][:K_DEPTH * 3])
    rrf_ord = rrf_order(bm_order, dense_order)[:K_DEPTH]

    q_ents = [normalize_entity(e.text) for e in nlp(Q).ents
              if not qg._is_numeric_label(e.label_)]
    q_ents = [e for e in q_ents if e]
    eseed = np.zeros(n_e)
    if q_ents:
        sims = embed(q_ents) @ embE.T
        for j in range(sims.shape[0]):
            ge = int(np.argmax(sims[j]))
            eseed[ge] = max(eseed[ge], float(sims[j, ge]))
    ps, _ = qg._ppr_iterate(W, n_p, np.zeros(n_p), eseed, damping=0.5)
    eppr = list(np.argsort(ps)[::-1][:K_DEPTH])
    hid_list = eppr if q_ents else []

    # faithful LinearRAG (lam=0.05) full retriever ranking
    lin = [int(r) for r in retr.stage2_candidates(Q, top_n=K_DEPTH, lam=LINRAG_LAM)["cand_rows"]]

    rankings = {
        "bm25": bm_order[:50],
        "rrf": rrf_ord[:50],
        "entPPR": eppr[:50],
        "linrag05": lin[:50],
        "minrank": min_rank_merge(rrf_ord, hid_list, 50),         # ours: RRF ∪ pureEnt
        "fuseLin": min_rank_merge(rrf_ord, lin, 50),              # RRF ∪ faithful LinearRAG
        "slot23": slots(rrf_ord, hid_list, 2, 3, rrf_ord),
    }
    goldset = set(gold)
    # last-hop gold ~ the gold whose text contains the answer string
    ans = (q.get("answer") or "").strip().lower()
    lh_gold = None
    if ans and len(ans) >= 2:
        for g in gold:
            t = (corpus[id2row[g]].get("title", "") + " " + corpus[id2row[g]]["text"]).lower()
            if ans in t:
                lh_gold = g
                break
    if lh_gold:
        n_lh += 1
    for m, order in rankings.items():
        top5 = {ids[r] for r in order[:5]}
        top10 = {ids[r] for r in order[:10]}
        agg[m]["gr5"].append(len(goldset & top5) / len(goldset))
        agg[m]["gr10"].append(len(goldset & top10) / len(goldset))
        agg[m]["agh5"].append(goldset <= top5)
        if lh_gold:
            agg[m]["lh5"].append(lh_gold in top5)

n = len(agg["bm25"]["gr5"])
print(f"index={a.index}  n={n}  (LastHop defined for {n_lh} questions: answer-bearing gold)")
print(f"\n{'method':>9s} {'GoldR@5':>9s} {'GoldR@10':>9s} {'LastHop@5':>10s} {'AllGold@5':>10s}")
for m in methods:
    print(f"{m:>9s} {np.mean(agg[m]['gr5']):>8.1%} {np.mean(agg[m]['gr10']):>8.1%} "
          f"{np.mean(agg[m]['lh5']):>9.1%} {np.mean(agg[m]['agh5']):>9.1%}")
print("\nreference (published, different setups — indicative only):")
print("  SPRIG GraphRRF 2wiki R@10 79.4 | Calibrated Fusion LastHop@5 2wiki 53.6 musique 76.5")
print("  (Calibrated Fusion's LastHop conditions on prior hops — ours is unconditioned, harder)")
