# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.14",
#     "spacy>=3.8",
#     "sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""Build a LinearRAG Tri-Graph index from a corpus (paper §3.1, token-free)."""
from __future__ import annotations

import json
from pathlib import Path


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Merge blank-line-separated paragraphs up to max_chars per passage."""
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
    for item in data:
        pid, _, text = item.partition(":")
        passages.append({"id": pid, "title": "", "text": text})
    return passages
