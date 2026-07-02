import numpy as np
from scipy import sparse

from query_graph import personalized_pagerank

# 3パッセージ × 2エンティティ: p0-{e0}, p1-{e0,e1}, p2-{e1}
# read-only fixture: do not mutate in tests
B = sparse.csr_matrix(np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32))


def test_seeded_entity_lifts_connected_passages():
    passage_seeds = np.zeros(3, dtype=np.float32)
    entity_seeds = np.array([1.0, 0.0], dtype=np.float32)  # e0 のみシード
    p_scores, e_scores = personalized_pagerank(B, passage_seeds, entity_seeds,
                                               damping=0.5)
    assert p_scores[0] > p_scores[2]   # e0 に接続する p0 > 非接続の p2
    assert p_scores[1] > p_scores[2]


def test_scores_form_distribution():
    # sum==1 は B が孤立ノードを持たない場合のみ成立（孤立ノードがあると
    # docstring 記載の通り質量が漏れる）。このfixtureは意図的に全ノード接続。
    p, e = personalized_pagerank(B, np.ones(3, dtype=np.float32),
                                 np.ones(2, dtype=np.float32), damping=0.5)
    assert abs(p.sum() + e.sum() - 1.0) < 1e-6
    assert (p >= 0).all() and (e >= 0).all()


def test_uniform_fallback_when_no_seeds():
    p, e = personalized_pagerank(B, np.zeros(3, dtype=np.float32),
                                 np.zeros(2, dtype=np.float32), damping=0.5)
    assert abs(p.sum() + e.sum() - 1.0) < 1e-6
