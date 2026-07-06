# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Hop-depth stratification (MuSiQue 2/3/4-hop): does the method scale to harder
tasks? Tests three predictions from the d^k PPR decay + query-invisibility story:

  (1) hiddenness cliff SHARPENS with hop count (deeper hops are further from the
      query, so the hardest gold is more hidden).
  (2) entity-PPR REACH decays with hop count (mass at k hops ~ d^k; the deepest
      gold needs more graph steps → worse PPR rank → lower pool recall).
  (3) query-based methods (BM25/RRF) collapse faster than entity-PPR/fusion as
      depth grows (they never propagate across bridges at all).

MuSiQue `type` field ('2hop','3hop1','4hop2',...) gives the hop count; gold count
matches it. Reports, per hop group: n, mean worst-gold hiddenness, all-gold pool
recall@50 for bm25/rrf/entPPR/fusion, and entity-PPR reach (median rank of the
deepest gold, %gold within top-50).
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
ap.add_argument("--index", default="linearrag_index/musique_hpr_trf")
ap.add_argument("--corpus", default="dataset/musique_hpr/corpus.jsonl")
ap.add_argument("--questions", default="dataset/musique_hpr/questions.json")
ap.add_argument("--pool-n", type=int, default=50)
ap.add_argument("--base-n", type=int, default=30)
ap.add_argument("--add-n", type=int, default=20)
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


def dedup(seq, n):
    out, seen = [], set()
    for r in seq:
        if r not in seen:
            seen.add(r); out.append(r)
        if len(out) >= n:
            break
    return out


retr.stage2_candidates(qs[0]["question"], top_n=a.pool_n)
rows = []
for q in qs:
    gold = [g for g in q.get("gold_ids", []) if g in id2row]
    if not gold:
        continue
    hop = int(q.get("type", "2hop")[0])  # '2hop'->2, '3hop1'->3, '4hop2'->4
    Q = q["question"]
    qv = embed([Q])[0]
    bm_scores = bm25.get_scores(tok(Q))
    bm_order = list(np.argsort(bm_scores)[::-1])
    dense = embP @ qv
    rrf_ord = rrf_order(bm_order, list(np.argsort(dense)[::-1]))

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
    eppr = list(np.argsort(ps)[::-1])
    eppr_rank = {row: i for i, row in enumerate(eppr)}

    base_rrf = rrf_ord[:a.base_n]
    inj = [r for r in eppr if r not in set(base_rrf)][:a.add_n]
    pools = {
        "bm25": set(ids[r] for r in bm_order[:a.pool_n]),
        "rrf": set(ids[r] for r in rrf_ord[:a.pool_n]),
        "entPPR": set(ids[r] for r in eppr[:a.pool_n]),
        "fusion": set(ids[r] for r in dedup(list(base_rrf) + inj, a.pool_n)),
    }
    goldset = set(gold)

    def hidden(gid):
        r = id2row[gid]
        return 1.0 - 0.5 * ((dense < dense[r]).mean() + (bm_scores < bm_scores[r]).mean())
    worst_h = max(hidden(g) for g in gold)
    # entity-PPR reach: rank of the DEEPEST (hardest) gold = worst PPR rank among golds
    gold_pranks = [eppr_rank.get(id2row[g], len(eppr)) for g in gold]
    rows.append({"hop": hop, "worst_h": worst_h,
                 "deep_prank": max(gold_pranks), "best_prank": min(gold_pranks),
                 "gold_in_ppr50": np.mean([r < 50 for r in gold_pranks]),
                 **{m: (goldset <= pset) for m, pset in pools.items()}})

print(f"index={a.index}  n={len(rows)}  (MuSiQue by hop count)")
print(f"\n{'hop':>4s} {'n':>5s} {'mean_hidden':>12s} | "
      f"{'bm25':>7s} {'rrf':>7s} {'entPPR':>7s} {'fusion':>7s} | "
      f"{'deepGoldPPRrank(med)':>21s} {'gold∈PPR50':>11s}")
for hop in (2, 3, 4):
    sel = [r for r in rows if r["hop"] == hop]
    if not sel:
        continue
    mh = np.mean([r["worst_h"] for r in sel])
    pr = {m: np.mean([r[m] for r in sel]) for m in ("bm25", "rrf", "entPPR", "fusion")}
    med_deep = np.median([r["deep_prank"] for r in sel])
    reach = np.mean([r["gold_in_ppr50"] for r in sel])
    print(f"{hop:>4d} {len(sel):>5d} {mh:>12.3f} | "
          f"{pr['bm25']:>6.0%} {pr['rrf']:>6.0%} {pr['entPPR']:>6.0%} {pr['fusion']:>6.0%} | "
          f"{med_deep:>21.0f} {reach:>10.0%}")

# hiddenness by hop AND gold position not available (order unknown); worst-gold is
# the AllGold bottleneck, which is what matters.
print("\nnote: deepGoldPPRrank = median over questions of the WORST gold's rank in")
print("the entity-PPR list (lower=better reach). gold∈PPR50 = avg fraction of a")
print("question's golds that entity-PPR puts in its top-50.")
