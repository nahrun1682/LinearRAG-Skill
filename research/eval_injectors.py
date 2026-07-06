# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2","openai>=1.40",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Injector shoot-out: which cheap candidate INJECTOR reaches hidden 2nd-hop gold?

The graph's proven job is injecting query-invisible hidden hops into the pool.
But is graph-PPR uniquely capable, or do cheaper injectors (dense-NN, dual-entity
ANN a la BridgeRAG) reach the same hidden hops? All augment the SAME BM25 base
pool; all are query-time cheap (matrix products, no LLM, no training). We compare:

  bm25            base
  +dense          add nearest passages to the QUERY embedding
  +dualEntity     add nearest passages to each QUERY-ENTITY embedding (1-jump;
                  BridgeRAG-style entity expansion, but pool-level)
  +graphPPR       add LinearRAG PPR top candidates (multi-hop bridge propagation)

Metrics: (1) all-gold POOL recall overall + by hiddenness quartile (where the
hidden hops live), (2) POOL recall of the single MOST-HIDDEN gold per question.
The bet: graphPPR wins in the HIDDEN bins because it propagates across bridges
several hops, while dense/ANN see only a 1-jump neighborhood.
"""
import argparse
import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, ".claude/skills/linearrag/scripts")
sys.path.insert(0, "research")
from common import Embedder, TriGraphIndex, load_nlp, normalize_entity  # noqa: E402
from resirag import ResiRetriever  # noqa: E402
import query_graph as qg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--base-n", type=int, default=30, help="BM25 base pool size")
ap.add_argument("--add-n", type=int, default=20, help="candidates each injector adds")
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--rm3-fb", type=int, default=10, help="RM3 feedback docs")
ap.add_argument("--rm3-terms", type=int, default=20, help="RM3 expansion terms")
ap.add_argument("--llm", action="store_true", help="also run GPT-4o selection per pool")
ap.add_argument("--model", default="gpt-4o-2024-08-06")
ap.add_argument("--workers", type=int, default=10)
a = ap.parse_args()

_tok = re.compile(r"\w+", re.UNICODE)
tok = lambda s: _tok.findall(s.lower())
corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
ids = [p["id"] for p in corpus]
id2row = {p["id"]: i for i, p in enumerate(corpus)}
index = TriGraphIndex.load(a.index)
assert all(corpus[i]["id"] == index.passages[i]["id"] for i in range(0, len(corpus), 500))
bm25 = BM25Okapi([tok(p.get("title", "") + " " + p["text"]) for p in corpus])
embed = Embedder(index.meta["embedding_model"])
nlp = load_nlp(index.meta["spacy_model"])
retr = ResiRetriever(index, embed, nlp)
embP = np.asarray(index.emb_passages)          # (P,d) L2-normalized
embE = np.asarray(index.emb_entities)          # (E,d)
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]


def toprows(scores, n, exclude):
    order = np.argsort(scores)[::-1]
    out = []
    for r in order:
        if r not in exclude:
            out.append(int(r))
        if len(out) >= n:
            break
    return out


injectors = ["bm25", "+dense", "+dualEntity", "+rm3", "+graphPPR"]
from collections import Counter
STOP = set("the a an of to in and or is are was were be been being by for on at as with that "
           "this it its from which who whom whose what when where how why not no s t "
           "he she they we you his her their our your".split())
rows = []
retr.stage2_candidates(qs[0]["question"], top_n=a.base_n)
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    Q = q["question"]
    qv = embed([Q])[0]
    base = list(np.argsort(bm25.get_scores(tok(Q)))[::-1][:a.base_n])
    base_set = set(base)

    # dense injector: nearest passages to query
    dense_add = toprows(embP @ qv, a.add_n, base_set)

    # dual-entity injector: union of nearest passages to each query-entity embedding
    q_ents = [normalize_entity(e.text) for e in nlp(Q).ents
              if not qg._is_numeric_label(e.label_)]
    q_ents = [e for e in q_ents if e]
    de_add = []
    if q_ents:
        evecs = embed(q_ents)
        # map each query entity to its nearest graph entity, then that entity's passages
        sims = evecs @ embE.T
        de_scores = np.zeros(len(corpus))
        for j in range(sims.shape[0]):
            ge = int(np.argmax(sims[j]))
            de_scores = np.maximum(de_scores, np.asarray(embP @ embE[ge]))
        de_add = toprows(de_scores, a.add_n, base_set)

    # RM3 injector: pseudo-relevance feedback query expansion (classic TEXT bridge)
    qset = set(tok(Q))
    tf = Counter()
    for r in base[:a.rm3_fb]:
        for t in tok(corpus[r].get("title", "") + " " + corpus[r]["text"]):
            if t not in qset and t not in STOP and not t.isdigit() and len(t) > 2:
                tf[t] += 1
    exp_terms = [t for t, _ in tf.most_common(a.rm3_terms)]
    exp_query = tok(Q) * 3 + exp_terms  # RM3-lite: original kept, expansion added
    rm3_add = toprows(bm25.get_scores(exp_query), a.add_n, base_set)

    # graph PPR injector
    cd = retr.stage2_candidates(Q, top_n=a.base_n + a.add_n)
    ppr_rows = [int(r) for r in cd["cand_rows"] if int(r) not in base_set][:a.add_n]

    def dedup(seq):
        out, seen = [], set()
        for r in seq:
            if r not in seen:
                seen.add(r); out.append(r)
        return out
    pools_ord = {
        "bm25": base,
        "+dense": dedup(base + dense_add),
        "+dualEntity": dedup(base + de_add),
        "+rm3": dedup(base + rm3_add),
        "+graphPPR": dedup(base + ppr_rows),
    }
    # hiddenness of the most-hidden gold (query dense+lexical percentile)
    dcorpus = embP @ qv
    bmall = bm25.get_scores(tok(Q))

    def hidden(gid):
        r = id2row[gid]
        return 1.0 - 0.5 * ((dcorpus < dcorpus[r]).mean() + (bmall < bmall[r]).mean())
    worst_h = max(hidden(g) for g in gold)
    rows.append({"q": Q, "gold": gold, "worst_h": worst_h,
                 "pools": pools_ord,
                 **{k: (gold <= set(v)) for k, v in {kk: [ids[r] for r in vv]
                    for kk, vv in pools_ord.items()}.items()}})

n = len(rows)
print(f"index={a.index}  n={n}  base=BM25 top{a.base_n}  each injector +{a.add_n}")
print("\n=== all-gold POOL recall (overall) ===")
for k in injectors:
    print(f"  {k:14s} {np.mean([r[k] for r in rows]):>6.1%}")

print("\n=== all-gold POOL recall by hiddenness quartile (of most-hidden gold) ===")
hs = np.array([r["worst_h"] for r in rows])
edges = np.quantile(hs, [0, .25, .5, .75, 1.0])
print(f"{'bin':16s} {'n':>4s} " + " ".join(f"{k:>12s}" for k in injectors))
for b in range(4):
    lo, hi = edges[b], edges[b + 1]
    sel = [r for r in rows if (lo <= r["worst_h"] <= hi if b == 3 else lo <= r["worst_h"] < hi)]
    if not sel:
        continue
    cells = " ".join(f"{np.mean([r[k] for r in sel]):>11.0%} " for k in injectors)
    print(f"[{lo:.2f},{hi:.2f}]{'':4s} {len(sel):>4d} {cells}")

# === answer-level: GPT-4o selection per injector pool ===
if a.llm:
    import concurrent.futures as cf
    import os
    import time
    from openai import OpenAI

    def key():
        for l in open(".env"):
            if l.strip().startswith("OPENAI_API_KEY"):
                return l.split("=", 1)[1].strip().strip('"').strip("'")
    client = OpenAI(api_key=key())
    sel_re = re.compile(r"\[(\d+)\]")
    PROMPT = ("I will provide you with {num} passages, each indicated by a numerical "
              "identifier []. Select the passages based on their relevance to the search "
              "query: {q}.\n\n{ctx}\n\nSearch Query: {q}\n\nPlease follow the steps below:\n"
              "Step 1. Please list up the information requirements to answer the query.\n"
              "Step 2. for each requirement in Step 1, find the passages that has the "
              "information of the requirement.\n"
              "Step 3. Choose the passages that mostly covers clear and diverse information "
              "to answer the query. Number of passages is unlimited. The format of final "
              "output should be '### Final Selection: [] []', e.g., ### Final Selection: [2] [1].")
    name = a.corpus.split("/")[-2]

    def clip(t, w=110):
        return " ".join(t.split()[:w])

    def gpt_pick(item, inj, cache):
        if item["q"] in cache:
            picks = cache[item["q"]]
        else:
            prows = item["pools"][inj]
            ctx = "\n".join(f"[{j+1}] ({corpus[r].get('title','')}) {clip(corpus[r]['text'])}"
                            for j, r in enumerate(prows))
            prompt = PROMPT.format(num=len(prows), q=item["q"], ctx=ctx)
            picks = None
            for att in range(5):
                try:
                    r = client.chat.completions.create(model=a.model, temperature=0,
                        messages=[{"role": "user", "content": prompt}])
                    after = r.choices[0].message.content.split("Final Selection")[-1]
                    picks = []
                    for p in [int(x) for x in sel_re.findall(after)]:
                        if 1 <= p <= len(prows) and ids[prows[p-1]] not in picks:
                            picks.append(ids[prows[p-1]])
                    break
                except Exception:
                    if att == 4:
                        picks = None
                    else:
                        time.sleep(2 ** att)
        cache[item["q"]] = picks
        return picks

    print("\n=== answer-level: injector pool + GPT-4o selection (AllGoldHit@5) ===")
    print(f"{'injector':14s} {'gpt4o':>8s}")
    for inj in injectors:
        safe = inj.replace("+", "")
        cfile = f"research/cache_inj_{name}_{safe}_{a.model}_{a.limit}.json"
        cache = json.load(open(cfile, encoding="utf-8")) if os.path.exists(cfile) else {}
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            picks = list(ex.map(lambda it: gpt_pick(it, inj, cache), rows))
        json.dump(cache, open(cfile, "w", encoding="utf-8"), ensure_ascii=False)
        hit = np.mean([it["gold"] <= set((p or [])[:5]) for it, p in zip(rows, picks)])
        print(f"  {inj:12s} {hit:>8.1%}")
