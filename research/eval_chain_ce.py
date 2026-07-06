# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Experiment D: chain-conditional cross-encoder on the graph-fusion pool.

Diagnosis: the hidden 2nd-hop gold is invisible to QUERY-based scoring at both
retrieval (fixed by graph fusion into the pool) and selection (why fusion+
pointwise-CE didn't convert the pool gain). To select it we must score it
against the ALREADY-CHOSEN hop, not the query.

chainCE (greedy, token-free-ish; the same 22M cross-encoder, no LLM/training):
  slot 1 : pick argmax CE(query, cand)                    [query-relevance]
  slot t : re-score remaining by
             (1-w)*CE(query, cand) + w*max_j CE(query + chosen_j, cand)
           where "query + chosen_j" prepends the chosen passage's text so the CE
           reads "given A, is B relevant" -> the bridge/hidden hop surfaces.

Compares, on the SAME BM25|graph fusion pool:
  pointwiseCE  - CE(query,·) top-k  (the strong baseline that matched GPT-4o)
  chainCE      - the chain-conditional greedy selection
Also reports on the bm25-only pool for reference.
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
ap.add_argument("--w", type=float, default=0.5, help="chain weight")
ap.add_argument("--chain-words", type=int, default=60, help="words of chosen psg to prepend")
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
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]

from sentence_transformers import CrossEncoder
ce = CrossEncoder(a.ce_model, max_length=512)


def clip(t, w):
    return " ".join(t.split()[:w])


def doc(r, w=160):
    return corpus[r].get("title", "") + ". " + clip(corpus[r]["text"], w)


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


def pointwise(q, rows):
    s = np.asarray(ce.predict([[q, doc(r)] for r in rows]))
    order = np.argsort(s)[::-1][:a.k]
    return {ids[rows[i]] for i in order}


def chain(q, rows):
    base = np.asarray(ce.predict([[q, doc(r)] for r in rows]))  # CE(query, cand)
    base_n = (base - base.min()) / (base.max() - base.min() + 1e-9)
    chosen, remaining = [], list(range(len(rows)))
    # slot 1: pure query relevance
    first = int(np.argmax(base))
    chosen.append(first); remaining.remove(first)
    while len(chosen) < a.k and remaining:
        # chain score: how relevant is cand GIVEN each chosen passage
        cond_q = [f"{q} {clip(corpus[rows[c]]['text'], a.chain_words)}" for c in chosen]
        best_cond = np.full(len(rows), -1e9)
        for cq in cond_q:
            s = np.asarray(ce.predict([[cq, doc(r)] for r in [rows[i] for i in remaining]]))
            for idx, i in enumerate(remaining):
                best_cond[i] = max(best_cond[i], s[idx])
        bc = best_cond.copy()
        fin = bc[remaining]
        fin_n = (fin - fin.min()) / (fin.max() - fin.min() + 1e-9)
        score = {i: (1 - a.w) * base_n[i] + a.w * fin_n[j]
                 for j, i in enumerate(remaining)}
        nxt = max(remaining, key=lambda i: score[i])
        chosen.append(nxt); remaining.remove(nxt)
    return {ids[rows[i]] for i in chosen}


agg = {c: 0.0 for c in ["bm25pool+pointwise", "bm25pool+chain",
                        "fusion+pointwise", "fusion+chain", "fusionpool_recall"]}
n = 0
fails = 0
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    try:
        cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
        bm_rows = list(np.argsort(bm25.get_scores(tok(q["question"])))[::-1][:a.top_n])
        fu_rows = merge(bm_rows, list(cd["cand_rows"]), a.top_n)
        Q = q["question"]
        row = {
            "bm25pool+pointwise": gold <= pointwise(Q, bm_rows),
            "bm25pool+chain": gold <= chain(Q, bm_rows),
            "fusion+pointwise": gold <= pointwise(Q, fu_rows),
            "fusion+chain": gold <= chain(Q, fu_rows),
            "fusionpool_recall": gold <= {ids[r] for r in fu_rows},
        }
    except Exception as e:  # noqa: BLE001
        fails += 1
        print(f"  skip q ({str(e)[:70]})")
        continue
    n += 1
    for k, v in row.items():
        agg[k] += v

name = a.corpus.split("/")[-2]
print(f"\n{name}  n={n}  w={a.w}  fails={fails}  (AllGoldHit@5; *_recall = all-gold-in-pool)")
for key in ["fusionpool_recall", "bm25pool+pointwise", "bm25pool+chain",
            "fusion+pointwise", "fusion+chain"]:
    print(f"  {key:22s} {agg[key]/n:>6.1%}")
