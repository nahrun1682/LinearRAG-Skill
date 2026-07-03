# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""JEMHopQA (Japanese multi-hop) gold-passage evaluation, grouped by hop.

Evidence is (subject, relation, value) triples, not passage annotations, and the
corpus is chunked (many chunks per Wikipedia title). So gold is defined PER HOP:
  gold_group(hop) = chunks of `subject`'s title that CONTAIN `value`  (precise),
                    else ALL chunks of `subject`'s title  (article-level fallback).
A hop is satisfied if top-k retrieves >=1 passage from its group.
  AllGoldHit@k = every hop satisfied (both bridge passages present)
  GoldRecall@k = fraction of hops satisfied
This mirrors the English gold_ids eval but with hop-grouped gold, and — unlike
Contain@5 — can score the YES/NO answers too (retrieval side).

Compares score (original LinearRAG) vs coverSoft (our Stage-2 selection).
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".claude/skills/linearrag/scripts")
sys.path.insert(0, "research")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever  # noqa: E402
from stage2_experiment import _minmax, select_cover_soft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="linearrag_index/jemhopqa")
ap.add_argument("--questions", default="dataset/jemhopqa/questions.json")
ap.add_argument("--corpus", default="dataset/jemhopqa/corpus.jsonl")
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--precise-only", action="store_true",
                help="keep only value-in-chunk gold hops; drop title-fallback hops")
a = ap.parse_args()

corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
by_title = defaultdict(list)
for p in corpus:
    by_title[p["title"]].append((p["id"], p["text"]))

qs = json.load(open(a.questions, encoding="utf-8"))
YESNO = {"YES", "NO", "はい", "いいえ"}


def gold_groups(q):
    """List of (group_ids:set, precise:bool) per evidence hop."""
    groups = []
    for e in q.get("evidence") or []:
        if not (isinstance(e, (list, tuple)) and len(e) >= 3):
            continue
        subj, val = e[0], str(e[2])
        chunks = by_title.get(subj, [])
        if not chunks:
            continue
        precise = [cid for cid, t in chunks if val in t]
        if precise:
            groups.append((set(precise), True))
        elif not a.precise_only:
            groups.append(({cid for cid, _ in chunks}, False))
    return groups


index = TriGraphIndex.load(a.index)
embed = Embedder(index.meta["embedding_model"])
retr = ResiRetriever(index, embed, load_nlp(index.meta["spacy_model"]))
C = index.C.tocsr()
embP = np.asarray(index.emb_passages)


def coversoft_order(cd, k):
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
    return select_cover_soft(ent, dense_norm, w_idf, k, 1.0, a.gamma)


def evalset(sel):
    buckets = defaultdict(lambda: [0.0, 0, 0])  # name -> [recall, allgold, n]
    for q in qs:
        gg = gold_groups(q)
        if not gg:
            continue
        cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
        order = (list(range(len(cd["cand_ids"]))) if sel == "score"
                 else coversoft_order(cd, a.k))
        top = {cd["cand_ids"][i] for i in order[:a.k]}
        sat = sum(1 for grp, _ in gg if top & grp)
        rec = sat / len(gg)
        allg = int(sat == len(gg))
        yn = q["answer"] in YESNO
        for name in ("all", q.get("type", "?"),
                     "substantive" if not yn else "yesno"):
            b = buckets[name]
            b[0] += rec
            b[1] += allg
            b[2] += 1
    return buckets


retr.stage2_candidates(qs[0]["question"], top_n=a.top_n)  # warm
print(f"index={a.index}  n={len(qs)}  gamma={a.gamma}  k={a.k}")
res = {sel: evalset(sel) for sel in ("score", "coverSoft")}
order = ["all", "substantive", "comparison", "compositional", "yesno"]
print(f"{'bucket':14s} {'n':>4s}  {'score R/AGH':>14s}  {'coverSoft R/AGH':>16s}")
for name in order:
    b0, b1 = res["score"].get(name), res["coverSoft"].get(name)
    if not b0:
        continue
    n = b0[2]
    print(f"{name:14s} {n:>4d}  {b0[0]/n:>6.1%}/{b0[1]/n:>6.1%}  "
          f"{b1[0]/n:>7.1%}/{b1[1]/n:>7.1%}")
