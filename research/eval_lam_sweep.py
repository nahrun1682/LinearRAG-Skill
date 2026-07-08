# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""INTEGRITY CHECK: is LinearRAG's dense-mixing term actually harmful, or did OUR
reimplementation over-weight it?

LinearRAG eq.7 seeds passage nodes with (lam*sim(q,p) + entity_bonus)*w_p, then
runs PPR. The paper's sensitivity analysis reports optimal lam ~= 0.05 (small).
Our resirag.py defaulted to lam=1.5. We had claimed "LinearRAG sabotages its own
propagation via dense mixing" — but if lam=0.05 (faithful) performs like lam=0,
that claim is an artifact of OUR lam=1.5, not LinearRAG's design.

Sweep lam in {0, 0.05, 0.5, 1.5} on the real LinearRAG retriever (stage2_candidates)
and compare to pure entity-seeded PPR (no passage seed at all). Report AllGoldHit@5
and all-gold pool recall@50, overall and by hiddenness. Verdict:
  - if lam=0.05 ~ lam=0 and both >> lam=1.5  -> our claim was OUR bug; retract.
  - if pure-entity still >> lam=0             -> passage-seeding itself (not dense) is the issue.
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
ap.add_argument("--limit", type=int, default=300)
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

LAMS = [0.0, 0.05, 0.5, 1.5]
cols = [f"lam={l:g}" for l in LAMS] + ["pureEnt"]
rows = []
retr.stage2_candidates(qs[0]["question"], top_n=50)
for q in qs:
    gold = set(g for g in q.get("gold_ids", []) if g in id2row)
    if not gold:
        continue
    Q = q["question"]
    qv = embed([Q])[0]
    bm_scores = bm25.get_scores(tok(Q))
    dense = embP @ qv

    pools = {}
    for l in LAMS:
        cd = retr.stage2_candidates(Q, top_n=50, lam=l)
        pools[f"lam={l:g}"] = [int(r) for r in cd["cand_rows"]]
    # pure entity-seeded PPR (no passage seed)
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
    pools["pureEnt"] = list(np.argsort(ps)[::-1][:50])

    def hidden(gid):
        r = id2row[gid]
        return 1.0 - 0.5 * ((dense < dense[r]).mean() + (bm_scores < bm_scores[r]).mean())
    worst_h = max(hidden(g) for g in gold)
    rec = {"worst_h": worst_h}
    for c, pool in pools.items():
        pset = {ids[r] for r in pool}
        rec[c + "_pool"] = gold <= pset
        rec[c + "_ag5"] = gold <= {ids[r] for r in pool[:5]}
    rows.append(rec)

n = len(rows)
print(f"index={a.index}  n={n}")
print("\n=== AllGoldHit@5 ===")
for c in cols:
    print(f"  {c:10s} {np.mean([r[c+'_ag5'] for r in rows]):>6.1%}")
print("\n=== all-gold POOL recall@50 ===")
for c in cols:
    print(f"  {c:10s} {np.mean([r[c+'_pool'] for r in rows]):>6.1%}")
print("\n=== POOL recall@50 by hiddenness quartile ===")
hs = np.array([r["worst_h"] for r in rows])
edges = np.quantile(hs, [0, .25, .5, .75, 1.0])
print(f"{'bin':14s} {'n':>4s} " + " ".join(f"{c:>9s}" for c in cols))
for b in range(4):
    lo, hi = edges[b], edges[b + 1]
    sel = [r for r in rows if (lo <= r["worst_h"] <= hi if b == 3 else lo <= r["worst_h"] < hi)]
    if not sel:
        continue
    cells = " ".join(f"{np.mean([r[c+'_pool'] for r in sel]):>8.0%} " for c in cols)
    print(f"[{lo:.2f},{hi:.2f}] {len(sel):>4d} {cells}")
