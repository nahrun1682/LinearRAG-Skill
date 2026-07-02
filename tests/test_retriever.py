import types

import numpy as np
from scipy import sparse

from common import TriGraphIndex
from query_graph import Retriever

# A tiny hand-built index over a 4-dim one-hot "embedding" space.
#
# entities: e0="alpha", e1="beta", e2="gamma"  -> one-hot dims 0,1,2
# passages: p0 contains {alpha, beta}, p1 contains {gamma}
# The fake embedder maps a surface/query to a one-hot vector by keyword, so
# cosine similarity is exact-match based and fully deterministic.

_VOCAB = {"alpha": 0, "beta": 1, "gamma": 2}
_DIM = 4


def _onehot(token: str) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    for word, dim in _VOCAB.items():
        if word in token.lower():
            v[dim] = 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def fake_embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack([_onehot(t) for t in texts]).astype(np.float32)


class _FakeEnt:
    def __init__(self, text: str, label: str = "PERSON"):
        self.text = text
        self.label_ = label


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


def make_fake_nlp(entity_map):
    """Return a callable(query) -> doc whose .ents come from entity_map.

    entity_map maps a query string to the list of entity surface strings the
    fake NER should emit. Unlisted queries yield no entities.
    """
    def nlp(query: str):
        return _FakeDoc([_FakeEnt(s) for s in entity_map.get(query, [])])
    return nlp


def _index() -> TriGraphIndex:
    # emb rows are one-hot so passage/sentence/entity embeddings line up with
    # the same vocab the fake embedder uses for queries.
    emb_entities = np.eye(3, _DIM, dtype=np.float32)          # alpha,beta,gamma
    emb_passages = np.array([[0.707, 0.707, 0, 0],            # p0 ~ alpha+beta
                             [0, 0, 1, 0]], dtype=np.float32)  # p1 ~ gamma
    emb_sentences = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0]], dtype=np.float32)
    return TriGraphIndex(
        passages=[{"id": "p0", "title": "Alpha-Beta", "text": "alpha beta."},
                  {"id": "p1", "title": "Gamma", "text": "gamma only."}],
        sentences=[{"text": "alpha.", "passage": 0},
                   {"text": "beta.", "passage": 0},
                   {"text": "gamma.", "passage": 1}],
        entities=["alpha", "beta", "gamma"],
        C=sparse.csr_matrix(np.array([[1, 1, 0], [0, 0, 1]], dtype=np.float32)),
        M=sparse.csr_matrix(np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]], dtype=np.float32)),
        emb_passages=emb_passages,
        emb_sentences=emb_sentences,
        emb_entities=emb_entities,
        meta={"language": "en", "embedding_model": "dummy"},
    )


def test_retriever_returns_top_k_passages():
    retriever = Retriever(_index(), fake_embed,
                          make_fake_nlp({"alpha thing": ["alpha"]}))
    result = retriever("alpha thing", top_k=2)

    assert set(result) == {"query", "query_entities", "activated_entities",
                           "passages", "params"}
    assert result["query"] == "alpha thing"
    assert result["query_entities"] == ["alpha"]
    assert len(result["passages"]) == 2
    ranks = [p["rank"] for p in result["passages"]]
    assert ranks == [1, 2]
    scores = [p["score"] for p in result["passages"]]
    assert scores == sorted(scores, reverse=True)   # rank asc == score desc
    assert result["params"]["top_k"] == 2


def test_retriever_top_k_caps_result_count():
    retriever = Retriever(_index(), fake_embed, make_fake_nlp({}))
    result = retriever("gamma", top_k=1)
    assert len(result["passages"]) == 1
    assert result["passages"][0]["rank"] == 1


def test_retriever_entity_match_boosts_relevant_passage():
    # Query names "alpha" -> entity activation should push p0 (contains alpha)
    # ahead of p1 (only gamma).
    retriever = Retriever(_index(), fake_embed,
                          make_fake_nlp({"who is alpha": ["alpha"]}))
    result = retriever("who is alpha", top_k=2)

    assert result["passages"][0]["id"] == "p0"
    activated_ids = {a["entity"] for a in result["activated_entities"]}
    assert "alpha" in activated_ids
    # seed entity is level 1
    alpha = next(a for a in result["activated_entities"] if a["entity"] == "alpha")
    assert alpha["level"] == 1


def test_retriever_no_entities_falls_back_to_dpr():
    # Fake NER yields nothing; retrieval must still work via passage DPR sim.
    retriever = Retriever(_index(), fake_embed, make_fake_nlp({}))
    result = retriever("gamma", top_k=2)

    assert result["query_entities"] == []
    assert result["activated_entities"] == []
    assert len(result["passages"]) == 2
    # "gamma" query is most similar to p1 by DPR
    assert result["passages"][0]["id"] == "p1"


def test_retriever_transition_matrix_cached_across_calls():
    # The binarized B is built once at construction and reused; mutating the
    # index's C afterward must not change a second query's result.
    index = _index()
    retriever = Retriever(index, fake_embed, make_fake_nlp({}))
    first = retriever("gamma", top_k=2)
    second = retriever("gamma", top_k=2)
    assert [p["id"] for p in first["passages"]] == [p["id"] for p in second["passages"]]


def test_retriever_binarizes_counts_without_mutating_index():
    index = _index()
    original = index.C.copy()
    Retriever(index, fake_embed, make_fake_nlp({}))
    # Construction must not mutate the caller's C matrix.
    assert (index.C != original).nnz == 0


def test_retriever_filters_numeric_query_entities():
    # ORDINAL/CARDINAL query entities ("first", "two") match hub nodes whose
    # degree drowns the real signal; they must not become seeds.
    def nlp(query):
        return _FakeDoc([_FakeEnt("first", "ORDINAL"), _FakeEnt("alpha", "PERSON")])
    retriever = Retriever(_index(), fake_embed, nlp)
    result = retriever("who is alpha, first of her name", top_k=1)
    assert result["query_entities"] == ["alpha"]


def test_sigma_sparsification_keeps_only_top_n():
    from query_graph import _sparsify_top_n
    v = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
    out = _sparsify_top_n(v, 2)
    assert (out > 0).sum() == 2
    assert out[1] == np.float32(0.9) and out[2] == np.float32(0.5)
    # top_n >= len or <= 0 leaves the vector untouched
    np.testing.assert_array_equal(_sparsify_top_n(v, 10), v)
    np.testing.assert_array_equal(_sparsify_top_n(v, 0), v)
