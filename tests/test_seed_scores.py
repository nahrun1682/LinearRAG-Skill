import numpy as np
from scipy import sparse

from query_graph import passage_seed_scores

# 2パッセージ × 3エンティティ。p0 は活性エンティティを多く含む。
C = sparse.csr_matrix(np.array([[3, 1, 0], [0, 0, 5]], dtype=np.float32))
A = np.array([0.9, 0.8, 0.0], dtype=np.float32)       # entity 2 は非活性
LEVELS = np.array([1, 2, 0], dtype=np.int32)
SIM_QP = np.array([0.5, 0.5], dtype=np.float32)       # DPR類似度は同点


def test_activated_entities_boost_containing_passage():
    scores = passage_seed_scores(SIM_QP, C, A, LEVELS, lam=1.5, w_p=0.05)
    assert scores[0] > scores[1]    # 活性エンティティを含む p0 が上
    assert (scores >= 0).all()


def test_deeper_entities_contribute_less():
    # 同じエンティティでも level が深いほど寄与が小さい (L_ei で除算)
    shallow = passage_seed_scores(SIM_QP, C, A, np.array([1, 1, 0]), lam=0.0, w_p=1.0)
    deep = passage_seed_scores(SIM_QP, C, A, np.array([3, 3, 0]), lam=0.0, w_p=1.0)
    assert shallow[0] > deep[0]


def test_no_activation_falls_back_to_dpr_only():
    scores = passage_seed_scores(np.array([0.9, 0.1]), C,
                                 np.zeros(3), np.zeros(3, dtype=np.int32),
                                 lam=1.5, w_p=0.05)
    assert scores[0] > scores[1]
