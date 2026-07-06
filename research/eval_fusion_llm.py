# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2","openai>=1.40",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Strong-selector phase: does a strong selector convert the GRAPH-injected
hidden hops that cheap selectors couldn't?

Diagnosis so far: graph fusion adds hidden 2nd-hop gold to the pool (+12-27pt
pool recall), but pointwise CE (MiniLM +0.3, bge +3.4) barely converts it —
because it scores each candidate vs the query, and the hidden hop is
query-invisible. A SET-aware reader (GPT-4o) reads all candidates together and
can connect A->B, so it should convert the injected hidden hops.

Same questions, k=5, AllGoldHit@5. For BM25 pool and BM25|graph FUSION pool:
  poolrecall - all-gold present in pool (ceiling)
  crossenc   - MiniLM pointwise (cheap reader, for reference)
  gpt4o      - SETR-verbatim set selection (strong reader). bm25-pool calls are
               reused from the existing cache; only fusion-pool is newly called.
The number we want: fusion+gpt4o vs bm25+gpt4o, and how close to fusion poolrecall.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, ".claude/skills/linearrag/scripts")
sys.path.insert(0, "research")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever  # noqa: E402


def load_env_key(path=".env"):
    for line in open(path, encoding="utf-8"):
        if line.strip().startswith("OPENAI_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no OPENAI_API_KEY")


ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--model", default="gpt-4o-2024-08-06")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--words", type=int, default=110)
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
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {q}.

{ctx}

Search Query: {q}

Please follow the steps below:
Step 1. Please list up the information requirements to answer the query.
Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.
Step 3. Choose the passages that mostly covers clear and diverse information to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [2] [1]."""


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


# ---- pass 1: pools + cross-encoder ----
items = []
retr.stage2_candidates(qs[0]["question"], top_n=a.top_n)
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
    bm_rows = list(np.argsort(bm25.get_scores(tok(q["question"])))[::-1][:a.top_n])
    fu_rows = merge(bm_rows, list(cd["cand_rows"]), a.top_n)

    def ce_pick(rows):
        s = np.asarray(ce.predict([[q["question"], doc(r)] for r in rows]))
        return {ids[rows[i]] for i in np.argsort(s)[::-1][:a.k]}
    items.append({"q": q["question"], "gold": gold,
                  "bm_rows": bm_rows, "fu_rows": fu_rows,
                  "bm_ce": ce_pick(bm_rows), "fu_ce": ce_pick(fu_rows)})
print(f"pass1 done: {len(items)} q")

# ---- pass 2: GPT-4o on each pool (bm25 reuses old cache) ----
from openai import OpenAI
client = OpenAI(api_key=load_env_key())
sel_re = re.compile(r"\[(\d+)\]")
name = a.corpus.split("/")[-2]
caches = {
    "bm": (f"research/cache_llm_{name}_{a.model}_{a.limit}.json", "bm_rows"),
    "fu": (f"research/cache_llm_fusion_{name}_{a.model}_{a.limit}.json", "fu_rows"),
}


def gpt_select(it, rows_key, cache):
    if it["q"] in cache:
        return cache[it["q"]]
    rows = it[rows_key]
    ctx = "\n".join(f"[{j+1}] ({corpus[r].get('title','')}) {clip(corpus[r]['text'], a.words)}"
                    for j, r in enumerate(rows))
    prompt = PROMPT.format(num=len(rows), q=it["q"], ctx=ctx)
    for attempt in range(5):
        try:
            r = client.chat.completions.create(model=a.model, temperature=0,
                                                messages=[{"role": "user", "content": prompt}])
            after = r.choices[0].message.content.split("Final Selection")[-1]
            out = []
            for p in [int(x) for x in sel_re.findall(after)]:
                if 1 <= p <= len(rows) and ids[rows[p - 1]] not in out:
                    out.append(ids[rows[p - 1]])
            return out
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                print(f"  llm fail [{it['q'][:40]}]: {type(e).__name__} {str(e)[:120]}")
                return None
            time.sleep(2 ** attempt)


gpt = {}
for tag, (cfile, rows_key) in caches.items():
    cache = json.load(open(cfile, encoding="utf-8")) if os.path.exists(cfile) else {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        raw = list(ex.map(lambda it: gpt_select(it, rows_key, cache), items))
    fails = sum(x is None for x in raw)
    cache.update({it["q"]: sel for it, sel in zip(items, raw) if sel is not None})
    json.dump(cache, open(cfile, "w", encoding="utf-8"), ensure_ascii=False)
    gpt[tag] = raw
    print(f"gpt4o[{tag}] {time.time()-t0:.0f}s fails={fails}")


def ratehit(picks):
    return np.mean([it["gold"] <= (set(p[:a.k]) if p is not None else set())
                    for it, p in zip(items, picks)])


def poolrec(key):
    return np.mean([it["gold"] <= {ids[r] for r in it[key]} for it in items])


N = len(items)
print(f"\n{name}  n={N}  (AllGoldHit@5; poolrecall = all-gold-in-pool)")
print(f"  {'bm25 poolrecall':22s} {poolrec('bm_rows'):>6.1%}")
print(f"  {'fusion poolrecall':22s} {poolrec('fu_rows'):>6.1%}")
print(f"  {'bm25+crossenc':22s} {np.mean([it['gold']<=it['bm_ce'] for it in items]):>6.1%}")
print(f"  {'fusion+crossenc':22s} {np.mean([it['gold']<=it['fu_ce'] for it in items]):>6.1%}")
print(f"  {'bm25+gpt4o':22s} {ratehit(gpt['bm']):>6.1%}")
print(f"  {'fusion+gpt4o':22s} {ratehit(gpt['fu']):>6.1%}")
