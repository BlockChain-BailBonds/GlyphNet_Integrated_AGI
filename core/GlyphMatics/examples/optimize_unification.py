# examples/optimize_unification.py
import numpy as np
from glyphmatics.component import (
    Ψ_quant, G_μν, Γ_arc, Θ_def, Π_meta, Φ_qc,
    unification_score, quantum_gravity_bridge, gravity_arc_bridge,
    arc_defense_bridge, defense_policy_bridge, policy_quantum_bridge,
    quasicrystal_universal
)

def optimize_unification(iters=200, lr=0.01, seed=918):
    rng = np.random.default_rng(seed)
    vars = [Ψ_quant.copy(), G_μν.copy(), Γ_arc.copy(), Θ_def.copy(), Π_meta.copy(), Φ_qc.copy()]
    names = ["Ψ_quant", "G_μν", "Γ_arc", "Θ_def", "Π_meta", "Φ_qc"]

    def forward():
        link_QG, _ = quantum_gravity_bridge(vars[0], vars[1])
        γ_arc = gravity_arc_bridge(vars[1], vars[2])
        Θ_sec = arc_defense_bridge(vars[2], vars[3])
        Π_upd = defense_policy_bridge(Θ_sec, vars[4], R_defense=γ_arc)
        policy_quantum_bridge(Π_upd, vars[0])
        Φ_res = quasicrystal_universal(vars[5])
        return unification_score(vars[0], vars[1], vars[2], vars[3], Π_upd, Φ_res)

    scores = []
    for i in range(iters):
        score = forward()
        scores.append(score)

        if i % 20 == 0:
            print(f"Iter {i:3d} | U_score = {score:.6f}")

        # Finite diff gradients
        eps = 1e-6
        grads = []
        for j, v in enumerate(vars):
            orig = v.copy()
            v += eps
            score_up = forward()
            grads.append((score_up - score) / eps)
            v[:] = orig

        # SGD
        for j, v in enumerate(vars):
            v -= lr * grads[j]

    print(f"Final U_score: {scores[-1]:.6f}")
    return scores[-1], scores


if __name__ == "__main__":
    optimize_unification()
