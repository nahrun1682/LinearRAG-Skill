# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.14",
#     "spacy>=3.8,<3.9",
#     "sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
# ]
# ///
# NOTE: standalone script; wider Python range than the project on purpose
"""Build a LinearRAG Tri-Graph index from a corpus (paper §3.1, token-free)."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from common import (
    DEFAULT_EMBEDDING_MODEL,
    SPACY_MODELS,
    TriGraphIndex,
    detect_language,
    normalize_entity,
)


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Merge budget: paragraphs are merged up to ~max_chars; a single
    paragraph longer than max_chars is kept intact (never hard-split)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 1 > max_chars:
            merged.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        merged.append(buf)
    return merged


def load_corpus(path: str | Path, max_chars: int = 1000) -> list[dict]:
    """Load passages from a directory (.txt/.md), a .jsonl file, or a
    dataset-style chunks.json ("id:text" string list)."""
    path = Path(path)
    passages: list[dict] = []

    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() not in {".txt", ".md"}:
                continue
            rel = file.relative_to(path).as_posix()
            text = file.read_text(encoding="utf-8")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            for i, chunk in enumerate(_split_paragraphs(text, max_chars)):
                passages.append({"id": f"{rel}#{i}", "title": rel, "text": chunk})
        return passages

    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                passages.append({"id": str(row["id"]), "title": row.get("title", ""),
                                 "text": row["text"]})
        return passages

    # dataset-style chunks.json: list of "id:text" strings
    data = json.loads(path.read_text(encoding="utf-8"))
    for i, item in enumerate(data):
        if not isinstance(item, str):
            raise TypeError(
                f"chunks.json item {i}: expected str, got {type(item).__name__}")
        pid, sep, text = item.partition(":")
        if not sep:
            raise ValueError(
                f"chunks.json item {i}: missing ':' separator in {item!r}")
        passages.append({"id": pid, "title": "", "text": text})
    return passages


def assemble_tri_graph(passages, analyze, embed,
                       meta: dict | None = None) -> TriGraphIndex:
    """Build the Tri-Graph (paper §3.1): sentence/entity nodes plus the
    contain matrix C (counts) and mention matrix M (binary).

    ``analyze(texts) -> [[(sentence, [entity surface, ...]), ...], ...]`` splits
    each passage into sentences and yields each sentence's entity surface forms
    in a single batched pass.
    """
    sentences: list[dict] = []
    entity_ids: dict[str, int] = {}
    m_rows, m_cols = [], []
    c_rows, c_cols, c_vals = [], [], []

    analyzed = analyze([p["text"] for p in passages])

    for p_idx, sents in enumerate(analyzed):
        occurrence = Counter()
        for sent, surfaces in sents:
            s_idx = len(sentences)
            sentences.append({"text": sent, "passage": p_idx})
            seen_in_sentence = set()
            for surface in surfaces:
                name = normalize_entity(surface)
                if not name:
                    continue
                e_idx = entity_ids.setdefault(name, len(entity_ids))
                occurrence[e_idx] += 1
                if e_idx not in seen_in_sentence:
                    seen_in_sentence.add(e_idx)
                    m_rows.append(s_idx)
                    m_cols.append(e_idx)
        for e_idx, count in occurrence.items():
            c_rows.append(p_idx)
            c_cols.append(e_idx)
            c_vals.append(count)

    entities = list(entity_ids)
    n_p, n_s, n_e = len(passages), len(sentences), len(entities)
    M = sparse.csr_matrix(
        (np.ones(len(m_rows), dtype=np.float32), (m_rows, m_cols)), shape=(n_s, n_e))
    C = sparse.csr_matrix(
        (np.asarray(c_vals, dtype=np.float32), (c_rows, c_cols)), shape=(n_p, n_e))

    return TriGraphIndex(
        passages=passages,
        sentences=sentences,
        entities=entities,
        C=C,
        M=M,
        emb_passages=embed([p["text"] for p in passages]),
        emb_sentences=embed([s["text"] for s in sentences]),
        emb_entities=embed(entities) if entities else np.zeros((0, 1), dtype=np.float32),
        meta=meta or {},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LinearRAG Tri-Graph index")
    parser.add_argument("--input", required=True, help="corpus dir / .jsonl / chunks.json")
    parser.add_argument("--output", required=True, help="index output directory")
    parser.add_argument("--lang", choices=["auto", "en", "ja"], default="auto")
    parser.add_argument("--model", default=None, help="sentence-transformers model name")
    parser.add_argument("--max-chars", type=int, default=1000)
    args = parser.parse_args()

    from common import Embedder, load_nlp, make_analyzer

    t0 = time.time()
    passages = load_corpus(args.input, max_chars=args.max_chars)
    if not passages:
        raise SystemExit(f"no passages found in {args.input}")

    lang = args.lang if args.lang != "auto" else detect_language(
        [p["text"] for p in passages[:50]])
    model_name = args.model or DEFAULT_EMBEDDING_MODEL
    print(f"passages={len(passages)} lang={lang} model={model_name}")

    nlp = load_nlp(lang)
    embed = Embedder(model_name)

    meta = {"language": lang, "embedding_model": model_name,
            "spacy_model": SPACY_MODELS[lang], "num_passages": len(passages)}
    index = assemble_tri_graph(passages, make_analyzer(nlp), embed, meta)
    index.save(args.output)

    print(f"sentences={len(index.sentences)} entities={len(index.entities)}")
    print(f"edges: C(nnz)={index.C.nnz} M(nnz)={index.M.nnz}")
    print(f"done in {time.time() - t0:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
