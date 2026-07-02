"""LinearRAG Tri-Graph index: shared data structures and I/O.

Implements the indexing layer of LinearRAG (arXiv:2510.10114) directly from
the paper's equations. Written independently of the reference implementation.
"""
from __future__ import annotations

import re
import unicodedata

_JA_CHARS = re.compile(r"[぀-ヿ一-鿿]")

SPACY_MODELS = {"en": "en_core_web_sm", "ja": "ja_core_news_sm"}
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def detect_language(texts: list[str], sample_size: int = 50) -> str:
    """Return "ja" when Japanese characters dominate a sample of the corpus."""
    sample = "".join(texts[:sample_size])
    if not sample:
        return "en"
    ratio = len(_JA_CHARS.findall(sample)) / len(sample)
    return "ja" if ratio > 0.1 else "en"


def normalize_entity(surface: str) -> str:
    """NFKC-normalize, casefold, and collapse whitespace so mentions align."""
    return " ".join(unicodedata.normalize("NFKC", surface).casefold().split())
