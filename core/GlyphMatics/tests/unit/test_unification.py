# tests/unit/test_unification.py
import numpy as np
from glyphmatics.component import (
    Ψ_quant, G_μν, Γ_arc, Θ_def, Π_meta, Φ_qc,
    unification_score, quantum_gravity_bridge
)

def test_unification_score_bounds():
    score = unification_score(Ψ_quant, G_μν, Γ_arc, Θ_def, Π_meta, Φ_qc)
    assert 0.0 <= score <= 1.0

def test_quantum_gravity_bridge():
    link, dE = quantum_gravity_bridge(Ψ_quant, G_μν)
    assert link.shape[0] <= 64  # rank_cap
    assert isinstance(dE, float)
