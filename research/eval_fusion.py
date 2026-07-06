# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Experiment C: does the BM25 (union) GRAPH candidate pool lift the FINAL answer?

Diagnosis (A+B) showed hidden 2nd-hop gold is missing from the BM25 pool but the
graph's PPR pool reaches it. Here we test whether fusing pools translates the
pool-recall gain into actual AllGoldHit@5, once a selector reads/ranks it.

Candidate pools (each capped so the reranker sees ~top-N):
  bm25    : BM25 top-N
  fusion  : round-robin merge of BM25 top-N and PPR top-N (dedup), capped to N
Selectors on each pool: crossenc (cheap reading) and coverSoft (token-free).
If fusion+crossenc >> bm25+crossenc, hidden-hop injection is real end-to-end.
"""
import argparse
import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, ".claude/skills/linearrag/scripts")
sys.path.insert(0, "research")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever  # noqa: E402
from stage2_experiment import _minmax, select_cover_soft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
a = ap.parse_args()

_tok = re.compile(r"\w+", re.UNICODE)
tok = lambda s: _tok.findall(s.lower())
corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
ids = [p["id"] for p in corpus]
index = TriGraphIndex.load(a.index)
assert all(corpus[i]["id"] == index.passages[i]["id"] for i in range(0, len(corpus), 500))
bm25 = BM25Okapi([tok(p.get("title", "") + " " + p["text"]) for p in corpus])
embed = Embedder(index.meta["embedding_model"])
retr = ResiRetriever(index, embed, load_nlp(index.meta["spacy_model"]))
C = index.C.tocsr()
embP = np.asarray(index.emb_passages)
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]

from sentence_transformers import CrossEncoder
ce = CrossEncoder(a.ce_model, max_length=512)


def clip(t, w=160):
    return " ".join(t.split()[:w])


def merge(a_rows, b_rows, n):
    """round-robin dedup merge (keeps both sources represented in top-n)."""
    out, seen = [], set()
    for x, y in zip(a_rows, b_rows):
        for r in (x, y):
            if r not in seen:
                seen.add(r)
                out.append(r)
            if len(out) >= n:
                return out
    for r in a_rows + b_rows:
        if r not in seen:
            seen.add(r)
            out.append(r)
        if len(out) >= n:
            break
    return out[:n]


def pick_ce(q, rows):
    pairs = [[q, corpus[r].get("title", "") + ". " + clip(corpus[r]["text"])] for r in rows]
    order = np.argsort(ce.predict(pairs))[::-1][:a.k]
    return {ids[rows[i]] for i in order}


def pick_cs(cd, rows):
    ent = [C[r].indices for r in rows]
    w = cd["activation"] / np.maximum(cd["levels"], 1)
    df = np.zeros_like(w)
    for e in ent:
        df[e] += 1.0
    w_idf = w * np.log(1.0 + len(ent) / np.maximum(df, 1.0))
    w_idf = w_idf / (max((float(w_idf[e].sum()) for e in ent), default=0.0) or 1.0)
    dn = _minmax(embP[rows] @ cd["query_vec"])
    return {ids[rows[i]] for i in select_cover_soft(ent, dn, w_idf, a.k, 1.0, a.gamma)}


agg = {c: [0.0, 0] for c in
       ["bm25pool_recall", "fusionpool_recall",
        "bm25+crossenc", "fusion+crossenc", "bm25+coverSoft", "fusion+coverSoft"]}
n = 0
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    n += 1
    cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
    bm_rows = list(np.argsort(bm25.get_scores(tok(q["question"])))[::-1][:a.top_n])
    ppr_rows = list(cd["cand_rows"])
    fu_rows = merge(bm_rows, ppr_rows, a.top_n)

    def rec(pool):
        return len(gold) and (gold <= {ids[r] for r in pool})
    picks = {
        "bm25pool_recall": gold <= {ids[r] for r in bm_rows},
        "fusionpool_recall": gold <= {ids[r] for r in fu_rows},
        "bm25+crossenc": gold <= pick_ce(q["question"], bm_rows),
        "fusion+crossenc": gold <= pick_ce(q["question"], fu_rows),
        "bm25+coverSoft": gold <= pick_cs(cd, bm_rows),
        "fusion+coverSoft": gold <= pick_cs(cd, fu_rows),
    }
    for key, hit in picks.items():
        agg[key][0] += float(hit)
        agg[key][1] += 1

name = a.corpus.split("/")[-2]
print(f"\n{name}  n={n}  (AllGoldHit@5 unless *_recall = all-gold-in-pool)")
for key in agg:
    print(f"  {key:22s} {agg[key][0]/n:>6.1%}")
