# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Lever-2: does coverSoft (our Stage-2 selection) generalize to BM25 candidates?

If BM25+coverSoft > plain BM25, the selection layer is a general token-free
component that improves even lexical retrieval — turning the "BM25 beats us on
hotpot" threat into "our layer improves BM25 too". This is the key result that
elevates the work from "a LinearRAG patch" to "a general selection layer".

Pipeline: BM25(title+text) generates top-N candidates; coverSoft reranks them
using the graph's query-activation (coverage weights) + dense anchor. Still
token-free (activation is graph-based; embeddings precomputed). Compared against
plain BM25 and, for reference, LinearRAG-PPR + coverSoft.
"""
import argparse
import json
import re

import numpy as np
import sys

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
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=1000)
a = ap.parse_args()

_tok = re.compile(r"\w+", re.UNICODE)
tok = lambda s: _tok.findall(s.lower())

corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
index = TriGraphIndex.load(a.index)
# safety: BM25 row indices must line up with index passages/embeddings
assert len(corpus) == len(index.passages) and all(
    corpus[i]["id"] == index.passages[i]["id"] for i in range(0, len(corpus), 500)), \
    "corpus/index passage order mismatch"

bm25 = BM25Okapi([tok(p.get("title", "") + " " + p["text"]) for p in corpus])
ids = [p["id"] for p in corpus]

embed = Embedder(index.meta["embedding_model"])
retr = ResiRetriever(index, embed, load_nlp(index.meta["spacy_model"]))
C = index.C.tocsr()
embP = np.asarray(index.emb_passages)
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]


def coversoft_over(rows, activation, levels, query_vec, k):
    ent = [C[r].indices for r in rows]
    w = activation / np.maximum(levels, 1)
    df = np.zeros_like(w)
    for e_arr in ent:
        df[e_arr] += 1.0
    idf = np.log(1.0 + len(ent) / np.maximum(df, 1.0))
    w_idf = w * idf
    w_idf = w_idf / (max((float(w_idf[e].sum()) for e in ent), default=0.0) or 1.0)
    dense_norm = _minmax(embP[rows] @ query_vec)
    return select_cover_soft(ent, dense_norm, w_idf, k, 1.0, a.gamma)


conds = ["bm25", "bm25+coverSoft", "linearrag", "linearrag+coverSoft"]
rec = {c: 0.0 for c in conds}
allg = {c: 0 for c in conds}
n = 0
retr.stage2_candidates(qs[0]["question"], top_n=a.top_n)
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    n += 1
    cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
    act, lev, qv = cd["activation"], cd["levels"], cd["query_vec"]

    bm_rows = list(np.argsort(bm25.get_scores(tok(q["question"])))[::-1][:a.top_n])
    ppr_rows = list(cd["cand_rows"])

    picks = {
        "bm25": [ids[r] for r in bm_rows[:a.k]],
        "bm25+coverSoft": [ids[bm_rows[i]] for i in
                           coversoft_over(bm_rows, act, lev, qv, a.k)],
        "linearrag": [ids[r] for r in ppr_rows[:a.k]],
        "linearrag+coverSoft": [cd["cand_ids"][i] for i in
                                coversoft_over(ppr_rows, act, lev, qv, a.k)],
    }
    for c, pid in picks.items():
        got = len(gold & set(pid))
        rec[c] += got / len(gold)
        allg[c] += (got == len(gold))

name = a.corpus.split("/")[-2]
print(f"{name}  n={n}  gamma={a.gamma}  top_n={a.top_n}")
print(f"{'condition':22s} {'GoldRecall@5':>13s} {'AllGoldHit@5':>13s}")
for c in conds:
    print(f"{c:22s} {rec[c]/n:>13.1%} {allg[c]/n:>13.1%}")
