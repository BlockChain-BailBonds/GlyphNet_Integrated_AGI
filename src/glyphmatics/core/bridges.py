import numpy as np
from .tensor_ops import kron, merge_op

_rng = np.random.default_rng(918)

# Global symbols
Ψ_quant = _rng.normal(size=(2, 2))
G_μν    = _rng.normal(size=(4, 4))
Γ_arc   = _rng.normal(size=(10, 10))
Θ_def   = _rng.normal(size=(8, 8))
Π_meta  = _rng.normal(size=(6, 6))
Φ_qc    = _rng.normal(size=(5, 5))
comm    = _rng.normal(size=(3, 3))
ent     = _rng.normal(size=(3, 3))

def quantum_gravity_bridge(Ψ, G):
    link = merge_op(kron(Ψ, G), kron(comm, ent))
    ΔE_qg = float(np.tensordot(Ψ, G, axes=2))
    return link, ΔE_qg

def gravity_arc_bridge(G, Γ):
    G = np.asarray(G, dtype=float)
    Γ = np.asarray(Γ, dtype=float)
    Gy, Gx = np.gradient(G, axis=0), np.gradient(G, axis=1)
    grad = np.hypot(Gx, Gy)
    k = min(grad.shape[0], grad.shape[1], Γ.shape[0], Γ.shape[1])
    return float((grad[:k, :k] * Γ[:k, :k]).sum())

def arc_defense_bridge(Γ, Θ):
    return merge_op(Γ, Θ)

def defense_policy_bridge(Θ, Π, R_defense, baseline=0.0, η=0.05):
    return Π + (η * (R_defense - baseline)) * np.eye(Π.shape[0])

def policy_quantum_bridge(Π, Ψ):
    dΠx, dΨx = np.gradient(Π, axis=1), np.gradient(Ψ, axis=1)
    return kron(dΠx, dΨx)

def quasicrystal_universal(Φ, φ_golden=(1 + 5**0.5) / 2):
    return Φ * φ_golden

def unified_tensor(Ψ, G, Γ, Θ, Π, Φ):
    return kron(Ψ, kron(G, kron(Γ, kron(Θ, kron(Π, Φ)))))

def unification_score(Ψ, G, Γ, Θ, Π, Φ) -> float:
    U = unified_tensor(Ψ, G, Γ, Θ, Π, Φ).reshape(Ψ.shape[0], -1)
    H_u = entropy(svd(U, compute_uv=False))
    B = blockdiag(Ψ, G, Γ, Θ, Π, Φ)
    H_b = entropy(svd(B, compute_uv=False))
    return float(1.0 - H_u / max(H_b, 1e-12))
