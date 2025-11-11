# src/glyphmatics/component.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from engine.glyphmatics_core import (
    quantum_gravity_bridge, gravity_arc_bridge, arc_defense_bridge,
    defense_policy_bridge, policy_quantum_bridge, quasicrystal_universal,
    unification_score, arc_meta_policy_solve,
    Ψ_quant, G_μν, Γ_arc, Θ_def, Π_meta, Φ_qc
)

class GlyphMaticsEngine:
    """High-level interface to the unified tensor engine."""

    def run_universe(self) -> Dict[str, float]:
        link, dE = quantum_gravity_bridge(Ψ_quant, G_μν)
        γ = gravity_arc_bridge(G_μν, Γ_arc)
        Θs = arc_defense_bridge(Γ_arc, Θ_def)
        Πu = defense_policy_bridge(Θs, Π_meta, R=γ, η=0.05)
        dΨdt = policy_quantum_bridge(Πu, Ψ_quant)
        Φr = quasicrystal_universal(Φ_qc)
        U = unification_score(Ψ_quant, G_μν, Γ_arc, Θ_def, Πu, Φr)

        return {
            "ΔE_qg": round(dE, 6),
            "Γ_arc_flow": round(γ, 6),
            "Π_trace": round(float(np.trace(Πu)), 6),
            "dΨ_dt_shape": tuple(dΨdt.shape),
            "U_score": round(U, 6)
        }

    def solve_arc(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray]
    ) -> List[np.ndarray]:
        return arc_meta_policy_solve(train_pairs, test_inputs)
