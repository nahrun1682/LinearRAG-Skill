# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Stage-2 headroom probe: where do the gold passages actually RANK?

AllGoldHit@5 is a set-coverage metric but LinearRAG's final step is a pointwise
top-k cut of the PPR score. If the missing gold passages mostly sit at ranks
6..50, a set-aware selection (coverage/rerank) can recover them without
touching stage 1; if they sit beyond 50, stage 2 reranking cannot help and the
problem is upstream. This probe measures exactly that on the canonical corpora.

Reports, per dataset:
  - AllGoldHit@k and GoldRecall@k for k in {5,10,20,50}
  - distribution of the WORST gold rank (the max rank over a question's golds)
  - near-miss share: >=1 gold in top-5 AND all golds within top-{20,50}
    (the population a top-50 -> top-5 reranker could fix)
  - redundancy at the cut: among near-misses, how many top-5 passages share an
    entity with an already higher-ranked top-5 passage (crowding-out evidence)
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".claude/skills/linearrag/scripts")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--depth", type=int, default=50)
a = ap.parse_args()

index = TriGraphIndex.load(a.index)
embed = Embedder(index.meta["embedding_model"])
retr = ResiRetriever(index, embed, load_nlp(index.meta["spacy_model"]))
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]

id2row = {p["id"]: i for i, p in enumerate(index.passages)}
C = index.C.tocsr()

KS = (5, 10, 20, 50)
allgold = {k: 0 for k in KS}
recall = {k: 0.0 for k in KS}
worst_ranks = []           # max gold rank per question (inf if beyond depth)
near_miss_20 = near_miss_50 = 0
partial_at_5 = 0           # >=1 but not all golds in top5
crowd_pairs = crowd_total = 0

retr(qs[0]["question"], residual_strength=0.0, threshold_mode="absolute")
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    res = retr(q["question"], top_k=a.depth, residual_strength=0.0,
               threshold_mode="absolute")
    ids = [p["id"] for p in res["passages"]]
    rank = {g: (ids.index(g) + 1 if g in ids else np.inf) for g in gold}
    worst = max(rank.values())
    worst_ranks.append(worst)
    for k in KS:
        got = sum(1 for g in gold if rank[g] <= k)
        recall[k] += got / len(gold)
        allgold[k] += got == len(gold)
    in5 = sum(1 for g in gold if rank[g] <= 5)
    if 0 < in5 < len(gold):
        partial_at_5 += 1
        if worst <= 20:
            near_miss_20 += 1
        if worst <= 50:
            near_miss_50 += 1
        # crowding: count top-5 passages sharing an entity with a higher-ranked one
        rows = [id2row[i] for i in ids[:5]]
        seen = set()
        for r in rows:
            ents = set(C[r].indices)
            if ents & seen:
                crowd_pairs += 1
            seen |= ents
        crowd_total += 4  # comparisons among 5 (first has no predecessor)

n = len(worst_ranks)
print(f"index={a.index}  n={n}  depth={a.depth}")
print(f"{'k':>4} {'GoldRecall@k':>13} {'AllGoldHit@k':>13}")
for k in KS:
    print(f"{k:>4} {recall[k]/n:>13.1%} {allgold[k]/n:>13.1%}")
wr = np.array([w for w in worst_ranks if np.isfinite(w)])
print(f"\nworst-gold-rank present within depth: {len(wr)}/{n}")
if len(wr):
    print(f"  median={np.median(wr):.0f}  p75={np.percentile(wr,75):.0f}  "
          f"p90={np.percentile(wr,90):.0f}")
print(f"\npartial@5 (some but not all gold in top5): {partial_at_5}/{n} "
      f"= {partial_at_5/n:.1%}")
print(f"  ... of which ALL golds within top-20: {near_miss_20} "
      f"({near_miss_20/max(partial_at_5,1):.0%} of partials)")
print(f"  ... of which ALL golds within top-50: {near_miss_50} "
      f"({near_miss_50/max(partial_at_5,1):.0%} of partials)")
print(f"\nceiling if a perfect top-20->top-5 reranker existed: "
      f"AllGoldHit@5 could reach {(allgold[5]+near_miss_20)/n:.1%} "
      f"(now {allgold[5]/n:.1%})")
print(f"crowding at the cut (top-5 passages sharing an entity with a "
      f"higher-ranked one): {crowd_pairs}/{crowd_total} "
      f"= {crowd_pairs/max(crowd_total,1):.0%}")
