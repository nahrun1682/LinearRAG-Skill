# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rank-bm25>=0.2.2"]
# ///
"""BM25 sparse-retrieval baseline on the SAME canonical corpora + gold_ids.

The urgent sanity check: does plain lexical BM25 already beat the graph method?
BM25 is notoriously strong on Wikipedia-style multi-hop sets because bridge
entities are lexically distinctive named entities. We score it with the exact
same gold-passage metrics (GoldRecall@5 / AllGoldHit@5 / Recall@1..3) so the
numbers drop straight into the comparison tables. No LLM, no NER, no embeddings.

Reports BM25 over passage TEXT and over TITLE+TEXT (titles are very informative
for these datasets, so both are worth seeing).
"""
import argparse
import json
import re

from rank_bm25 import BM25Okapi

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--limit", type=int, default=1000)
ap.add_argument("--use-title", action="store_true", help="index title+text")
a = ap.parse_args()

_tok = re.compile(r"\w+", re.UNICODE)


def tok(s):
    return _tok.findall(s.lower())


corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
ids = [p["id"] for p in corpus]
docs = [(p.get("title", "") + " " + p["text"]) if a.use_title else p["text"]
        for p in corpus]
bm25 = BM25Okapi([tok(d) for d in docs])

qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]
KS = [1, 2, 3, 5, 10]
recall = {k: 0.0 for k in KS}
allg = {k: 0 for k in KS}
contain = 0
n = 0
import numpy as np

for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    n += 1
    scores = bm25.get_scores(tok(q["question"]))
    order = np.argsort(scores)[::-1]
    top_ids = [ids[i] for i in order[:max(KS)]]
    # Contain@5
    ans = [q["answer"].casefold()] + [x.casefold() for x in q.get("answer_aliases", [])]
    blob = " ".join(docs[i].casefold() for i in order[:5])
    if any(x in blob for x in ans if x):
        contain += 1
    for k in KS:
        got = len(gold & set(top_ids[:k]))
        recall[k] += got / len(gold)
        allg[k] += (got == len(gold))

tag = "title+text" if a.use_title else "text"
print(f"BM25({tag})  {a.corpus.split('/')[-2]}  n={n}")
print("  k        : " + "  ".join(f"{k:>5}" for k in KS))
print("  Recall@k : " + "  ".join(f"{recall[k]/n:>5.1%}" for k in KS))
print("  AllGold@k: " + "  ".join(f"{allg[k]/n:>5.1%}" for k in KS))
print(f"  Contain@5: {contain/n:.1%}")
