# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Layer-2 bet: feed the GRAPH bridge signal into the selector (token-free-ish).

Diagnosis: the hidden 2nd-hop gold is query-invisible AND text-dissimilar to the
chosen 1st hop (they share a bridge ENTITY, not similar prose). Text-chain CE was
weak because it tried to cross the bridge with TEXT. Here we cross it with the
GRAPH: a candidate B is promoted if it shares (activated) entities with an
already-chosen passage A. Greedy, one CE pass + sparse entity lookups, no LLM.

  slot 1 : argmax CE(query, cand)
  slot t : argmax over remaining of
             (1-w)*CE_norm(cand) + w*bridge(cand | chosen)
           bridge(B|S) = sum over entities e in B that also appear in some chosen
             A in S, weighted by activation[e]*IDF[e]  (the shared-bridge mass).
Compared on the BM25|graph fusion pool against pointwise CE (the strong cheap
baseline that could NOT convert the injected hidden hops).
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

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--ws", default="0.3,0.5,0.7", help="bridge weights to sweep")
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
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]
ws = [float(x) for x in a.ws.split(",")]

from sentence_transformers import CrossEncoder
ce = CrossEncoder(a.ce_model, max_length=512)


def clip(t, w=160):
    return " ".join(t.split()[:w])


def doc(r):
    return corpus[r].get("title", "") + ". " + clip(corpus[r]["text"])


def merge(a_rows, b_rows, n):
    out, seen = [], set()
    for x, y in zip(a_rows, b_rows):
        for r in (x, y):
            if r not in seen:
                seen.add(r); out.append(r)
            if len(out) >= n:
                return out
    for r in a_rows + b_rows:
        if r not in seen:
            seen.add(r); out.append(r)
        if len(out) >= n:
            break
    return out[:n]


def pointwise(cescore, rows):
    return {ids[rows[i]] for i in np.argsort(cescore)[::-1][:a.k]}


def graph_bridge(cescore, rows, ewt):
    """ewt: per-entity weight = activation*IDF (0 for inactive). Greedy select."""
    ce_n = (cescore - cescore.min()) / (cescore.max() - cescore.min() + 1e-9)
    ent = [set(C[r].indices.tolist()) for r in rows]
    chosen, remaining = [int(np.argmax(cescore))], set(range(len(rows)))
    remaining.discard(chosen[0])
    picks_by_w = {}
    for w in ws:
        ch = list(chosen)
        rem = set(remaining)
        covered = set(ent[ch[0]])
        while len(ch) < a.k and rem:
            best, bg = None, -1e18
            for i in rem:
                shared = ent[i] & covered
                bridge = float(sum(ewt[e] for e in shared))
                g = (1 - w) * ce_n[i] + w * bridge
                if g > bg:
                    bg, best = g, i
            ch.append(best)
            covered |= ent[best]
            rem.discard(best)
        picks_by_w[w] = {ids[rows[i]] for i in ch}
    return picks_by_w


def gated_bridge(cescore, rows, ewt, m=12):
    """Keep only CE top-m (readable) candidates, then greedy set-select k of them
    by CE + bridge-to-chosen. Avoids promoting low-CE distractors."""
    top_m = list(np.argsort(cescore)[::-1][:m])
    ce_n = (cescore - cescore.min()) / (cescore.max() - cescore.min() + 1e-9)
    ent = {i: set(C[rows[i]].indices.tolist()) for i in top_m}
    out = {}
    for w in ws:
        chosen = [int(top_m[0])]
        rem = set(top_m[1:])
        covered = set(ent[chosen[0]])
        while len(chosen) < a.k and rem:
            best, bg = None, -1e18
            for i in rem:
                bridge = float(sum(ewt[e] for e in (ent[i] & covered)))
                g = (1 - w) * ce_n[i] + w * bridge
                if g > bg:
                    bg, best = g, i
            chosen.append(best); covered |= ent[best]; rem.discard(best)
        out[w] = {ids[rows[i]] for i in chosen}
    return out


tags = ["fusionpool", "fusion+pointwise"] + [f"fusion+bridge@{w}" for w in ws] + \
       [f"fusion+gated@{w}" for w in ws]
agg = {t: 0.0 for t in tags}
n = 0
retr.stage2_candidates(qs[0]["question"], top_n=a.top_n)
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    n += 1
    cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
    bm_rows = list(np.argsort(bm25.get_scores(tok(q["question"])))[::-1][:a.top_n])
    fu_rows = merge(bm_rows, list(cd["cand_rows"]), a.top_n)
    # entity weight over full vocab: activation*IDF(among candidates)
    act = cd["activation"] / np.maximum(cd["levels"], 1)
    ent_lists = [C[r].indices for r in fu_rows]
    df = np.zeros_like(act)
    for e in ent_lists:
        df[e] += 1.0
    ewt = act * np.log(1.0 + len(fu_rows) / np.maximum(df, 1.0))
    cescore = np.asarray(ce.predict([[q["question"], doc(r)] for r in fu_rows]))

    agg["fusionpool"] += gold <= {ids[r] for r in fu_rows}
    agg["fusion+pointwise"] += gold <= pointwise(cescore, fu_rows)
    bw = graph_bridge(cescore, fu_rows, ewt)
    gw = gated_bridge(cescore, fu_rows, ewt)
    for w in ws:
        agg[f"fusion+bridge@{w}"] += gold <= bw[w]
        agg[f"fusion+gated@{w}"] += gold <= gw[w]

name = a.corpus.split("/")[-2]
print(f"\n{name}  n={n}  (AllGoldHit@5)")
for t in tags:
    print(f"  {t:22s} {agg[t]/n:>6.1%}")
