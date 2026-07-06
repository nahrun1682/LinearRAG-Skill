# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=2.0","rank-bm25>=0.2.2","sentence-transformers>=3.0"]
# ///
"""Cross-encoder reranker rung on the SAME BM25 candidates + gold metric.

Places the "reads, but pointwise" rung between coverSoft (doesn't read, sees the
set) and GPT-4o (reads AND sees the set). Same BM25 top-N pool; the cross-encoder
scores each (query, passage) pair and we take top-k. Local, no API, no training
(pretrained bge-reranker). Latency reported so it lands on the cost ladder.
"""
import argparse
import json
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
ap.add_argument("--words", type=int, default=160)
a = ap.parse_args()

_tok = re.compile(r"\w+", re.UNICODE)
tok = lambda s: _tok.findall(s.lower())
corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
ids = [p["id"] for p in corpus]
bm25 = BM25Okapi([tok(p.get("title", "") + " " + p["text"]) for p in corpus])
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]


def clip(t, w):
    return " ".join(t.split()[:w])


from sentence_transformers import CrossEncoder
reranker = CrossEncoder(a.model, max_length=512)

items = []
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    rows = list(np.argsort(bm25.get_scores(tok(q["question"])))[::-1][:a.top_n])
    items.append({"q": q["question"], "gold": gold, "rows": rows})

reranker.predict([[items[0]["q"], corpus[items[0]["rows"][0]]["text"][:400]]])  # warm

t0 = time.time()
ce_picks = []
for it in items:
    pairs = [[it["q"], (corpus[r].get("title", "") + ". " + clip(corpus[r]["text"], a.words))]
             for r in it["rows"]]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1][:a.k]
    ce_picks.append([ids[it["rows"][i]] for i in order])
dt = time.time() - t0


def score(picklist):
    rec = ag = 0
    for it, pid in zip(items, picklist):
        got = len(it["gold"] & set(pid))
        rec += got / len(it["gold"])
        ag += (got == len(it["gold"]))
    n = len(items)
    return rec / n, ag / n


name = a.corpus.split("/")[-2]
print(f"\n{name}  n={len(items)}  model={a.model}  "
      f"{1000*dt/len(items):.0f} ms/query (CPU, {a.top_n} pairs)")
print(f"{'selector':18s} {'GoldRecall@5':>13s} {'AllGoldHit@5':>13s}")
bm25_pick = [[ids[r] for r in it["rows"][:a.k]] for it in items]
for tag, pl in [("bm25", bm25_pick), ("bm25+crossenc", ce_picks)]:
    r, g = score(pl)
    print(f"{tag:18s} {r:>13.1%} {g:>13.1%}")
