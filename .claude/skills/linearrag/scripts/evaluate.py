# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.14",
#     "spacy>=3.8,<3.9",
#     "sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
# NOTE: standalone script; wider Python range than the project on purpose
"""Batch retrieval evaluation: Retrieval-Contain and Evidence-Entity Recall.

Expects a questions.json in the dataset format: a list of objects with at
least {"question", "answer"} and optionally "evidence" (list of
[subject, relation, object] triples whose endpoints are entity surfaces).
"""
from __future__ import annotations

import argparse
import json
import time

from common import Embedder, TriGraphIndex, load_nlp, normalize_entity
from query_graph import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LinearRAG retrieval")
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True, help="questions.json path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.8)
    parser.add_argument("--lam", type=float, default=1.5)
    parser.add_argument("--w-p", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--sigma-top-n", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N questions only")
    args = parser.parse_args()

    index = TriGraphIndex.load(args.index)
    retriever = Retriever(index, Embedder(index.meta["embedding_model"]),
                          load_nlp(index.meta["spacy_model"]))
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    contain_hits, ev_hits, ev_total = 0, 0, 0
    t0 = time.time()
    for q in questions:
        result = retriever(q["question"], top_k=args.top_k, delta=args.delta,
                           max_iterations=args.max_iterations,
                           lam=args.lam, w_p=args.w_p, damping=args.damping,
                           sigma_top_n=args.sigma_top_n)
        joined = " ".join(p["text"] for p in result["passages"]).casefold()
        contained = q["answer"].casefold() in joined
        contain_hits += contained

        activated = {r["entity"] for r in result["activated_entities"]}
        evidence_entities = set()
        for step in q.get("evidence", []):
            if not isinstance(step, (list, tuple)):
                raise TypeError(
                    f"evidence step must be a [subject, relation, object] list, "
                    f"got {type(step).__name__}")
            evidence_entities.add(normalize_entity(str(step[0])))
            evidence_entities.add(normalize_entity(str(step[-1])))
        evidence_entities.discard("")
        ev_total += len(evidence_entities)
        ev_hits += len(evidence_entities & activated)

        mark = "o" if contained else "x"
        print(f"[{mark}] {q['question'][:70]}  ans={q['answer'][:30]}")

    n = len(questions)
    if n == 0:
        print("no questions to evaluate")
        return
    elapsed = time.time() - t0
    print(f"\nRetrieval-Contain: {contain_hits}/{n} = {contain_hits / n:.1%}")
    if ev_total:
        print(f"Evidence-Entity Recall: {ev_hits}/{ev_total} = {ev_hits / ev_total:.1%}")
    print(f"avg retrieval time: {elapsed / n:.3f}s/query (total {elapsed:.1f}s incl. model load)")


if __name__ == "__main__":
    main()
