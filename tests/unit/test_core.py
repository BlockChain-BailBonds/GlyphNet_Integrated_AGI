import numpy as np
from engine.glyphmatics_core import (
    merge_op, entropy_from_singulars, unification_score,
    Ψ_quant, G_μν, Γ_arc, Θ_def, Π_meta, Φ_qc
)

def test_merge_op():
    A = np.eye(3)
    B = np.ones((3, 4))
    M = merge_op(A, B, rank_cap=2)
    assert M.shape == (3, 2)

def test_entropy():
    s = np.array([1.0, 0.0])
    assert abs(entropy_from_singulars(s)) < 1e-10

def test_unification_score():
    score = unification_score(Ψ_quant, G_μν, Γ_arc, Θ_def, Π_meta, Φ_qc)
    assert 0.0 <= score <= 1.0
