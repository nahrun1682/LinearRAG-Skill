import numpy as np
from scipy import sparse

from query_graph import activate_entities, initial_activation

# entities: 0=beatrice, 1=frederick, 2=germany, 3=noise
# sentences: s0={beatrice,frederick}, s1={frederick,germany}, s2={noise}
M = sparse.csr_matrix(np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32))
SIGMA = np.array([0.9, 0.8, 0.1], dtype=np.float32)  # クエリ-文類似度
A0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # シード=beatrice


def test_two_hop_bridging_activates_intermediate_and_target():
    scores, levels, trace = activate_entities(A0, M, SIGMA, delta=0.5, max_iterations=4)
    assert scores[1] > 0.5          # frederick (1-hop bridge)
    assert scores[2] > 0.5          # germany (2-hop)
    assert scores[3] == 0.0         # noise は枝刈りされる
    assert levels[0] == 1           # シードは level 1
    assert levels[1] == 2
    assert levels[2] == 3
    assert levels[3] == 0           # 非活性


def test_pruning_threshold_blocks_weak_paths():
    scores, levels, _ = activate_entities(A0, M, SIGMA, delta=10.0, max_iterations=4)
    # 閾値が高すぎると新規活性はゼロ、シードのみ残る
    assert scores[0] == 1.0
    assert (scores[1:] == 0).all()


def test_terminates_early_when_no_new_entities():
    _, _, trace = activate_entities(A0, M, SIGMA, delta=0.5, max_iterations=10)
    assert len(trace) <= 3  # 3反復以内に自然停止（10回回らない）


def test_initial_activation_matches_most_similar_entity():
    emb_entities = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)
    emb_entities /= np.linalg.norm(emb_entities, axis=1, keepdims=True)
    q_ent_vecs = np.array([[1, 0]], dtype=np.float32)  # entity 0 に最も近い
    a0 = initial_activation(q_ent_vecs, emb_entities)
    assert a0.argmax() == 0
    assert a0[1] == 0.0
