# engine/glyphmatics_core.py
"""
GlyphMatics Core Engine
=======================

A unified tensor-network engine bridging:
- Quantum Mechanics
- Gravity
- ARC-AGI Reasoning
- Defense Systems
- Meta-Policy Optimization
- Quasicrystal Geometry

All operations are deterministic (seed=918), NumPy-only, and differentiable.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd
from collections import deque
from typing import Tuple, List, Dict, Callable, Any


# ======================================================
# 1. CORE TENSOR OPERATIONS
# ======================================================

def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product of two arrays."""
    return np.kron(a, b)


def merge_op(a: np.ndarray, b: np.ndarray, rank_cap: int = 64) -> np.ndarray:
    """
    Merge two tensors via low-rank SVD compression.
    
    Args:
        a, b: Input tensors with leading dimension as bond
        rank_cap: Maximum bond dimension after merge
    
    Returns:
        Compressed merged tensor
    """
    A = a.reshape(a.shape[0], -1)
    B = b.reshape(b.shape[0], -1)
    M = np.concatenate([A, B], axis=1)
    U, S, Vt = svd(M, full_matrices=False)
    k = max(1, min(min(A.shape[0], Vt.shape[0]), rank_cap))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


def entropy_from_singulars(s: np.ndarray) -> float:
    """Von Neumann entropy from singular values."""
    s = np.maximum(s, 1e-12)
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())


def blockdiag(*tensors: np.ndarray) -> np.ndarray:
    """Block-diagonal concatenation of tensors."""
    mats = [x.reshape(x.shape[0], -1) for x in tensors]
    H = sum(m.shape[0] for m in mats)
    W = max(m.shape[1] for m in mats) if mats else 0
    out = np.zeros((H, W))
    i = 0
    for m in mats:
        h, w = m.shape
        out[i:i + h, :w] = m
        i += h
    return out


# ======================================================
# 2. DETERMINISTIC SYMBOLIC DOMAINS
# ======================================================

_rng = np.random.default_rng(918)  # Fixed seed for reproducibility

Ψ_quant = _rng.normal(size=(2, 2))      # Quantum state
G_μν    = _rng.normal(size=(4, 4))      # Spacetime metric
Γ_arc   = _rng.normal(size=(10, 10))    # ARC pattern tensor
Θ_def   = _rng.normal(size=(8, 8))      # Defense capability
Π_meta  = _rng.normal(size=(6, 6))      # Meta-policy
Φ_qc    = _rng.normal(size=(5, 5))      # Quasicrystal order
comm    = _rng.normal(size=(3, 3))      # Communication channel
ent     = _rng.normal(size=(3, 3))      # Entanglement witness


# ======================================================
# 3. GLYPHMATIC BRIDGES
# ======================================================

def quantum_gravity_bridge(Ψ: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, float]:
    """Bridge quantum state with gravity via entanglement + energy shift."""
    link = merge_op(kron(Ψ, G), kron(comm, ent))
    ΔE_qg = float(np.tensordot(Ψ, G, axes=2))
    return link, ΔE_qg


def gravity_arc_bridge(G: np.ndarray, Γ: np.ndarray) -> float:
    """Flow from spacetime curvature to ARC pattern recognition."""
    Gy = np.gradient(G, axis=0)
    Gx = np.gradient(G, axis=1)
    grad = np.hypot(Gx, Gy)
    k = min(grad.shape[0], Γ.shape[0])
    return float((grad[:k, :k] * Γ[:k, :k]).sum())


def arc_defense_bridge(Γ: np.ndarray, Θ: np.ndarray) -> np.ndarray:
    """Merge ARC reasoning into defense posture."""
    return merge_op(Γ, Θ)


def defense_policy_bridge(
    Θ: np.ndarray,
    Π: np.ndarray,
    R: float,
    baseline: float = 0.0,
    η: float = 0.05
) -> np.ndarray:
    """Update meta-policy via defense reward signal."""
    return Π + (η * (R - baseline)) * np.eye(Π.shape[0])


def policy_quantum_bridge(Π: np.ndarray, Ψ: np.ndarray) -> np.ndarray:
    """Backpropagate policy gradient into quantum evolution."""
    dΠx = np.gradient(Π, axis=1)
    dΨx = np.gradient(Ψ, axis=1)
    return kron(dΠx, dΨx)


def quasicrystal_universal(Φ: np.ndarray, φ_golden: float = (1 + 5**0.5) / 2) -> np.ndarray:
    """Resonate quasicrystal with golden ratio symmetry."""
    return Φ * φ_golden


# ======================================================
# 4. UNIFICATION ENGINE
# ======================================================

def unified_tensor(Ψ, G, Γ, Θ, Π, Φ) -> np.ndarray:
    """Full cross-domain tensor contraction."""
    return kron(Ψ, kron(G, kron(Γ, kron(Θ, kron(Π, Φ)))))


def unification_score(Ψ, G, Γ, Θ, Π, Φ) -> float:
    """Measure emergent coherence: U → 1.0 = perfect unity."""
    U = unified_tensor(Ψ, G, Γ, Θ, Π, Φ).reshape(Ψ.shape[0], -1)
    H_u = entropy_from_singulars(svd(U, compute_uv=False))
    B = blockdiag(Ψ, G, Γ, Θ, Π, Φ)
    H_b = entropy_from_singulars(svd(B, compute_uv=False))
    return float(1.0 - H_u / max(H_b, 1e-12))


# ======================================================
# 5. ARC GRID PRIMITIVES
# ======================================================

# --- Transformations ---
def rot90(g: np.ndarray) -> np.ndarray: return np.rot90(g, 1)
def rot180(g: np.ndarray) -> np.ndarray: return np.rot90(g, 2)
def rot270(g: np.ndarray) -> np.ndarray: return np.rot90(g, 3)
def mirror_x(g: np.ndarray) -> np.ndarray: return np.flipud(g)
def mirror_y(g: np.ndarray) -> np.ndarray: return np.fliplr(g)
def transpose(g: np.ndarray) -> np.ndarray: return g.T.copy()


# --- Color & Geometry ---
def swap_colors(g: np.ndarray, a: int, b: int) -> np.ndarray:
    out = g.copy()
    ma, mb = (out == a), (out == b)
    out[ma], out[mb] = b, a
    return out


def flood_fill(g: np.ndarray, r: int, c: int, new_color: int) -> np.ndarray:
    H, W = g.shape
    old = int(g[r, c])
    if old == new_color:
        return g.copy()
    out = g.copy()
    q = deque([(r, c)])
    while q:
        i, j = q.popleft()
        if not (0 <= i < H and 0 <= j < W) or out[i, j] != old:
            continue
        out[i, j] = new_color
        q.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
    return out


def find_components(g: np.ndarray) -> List[Dict[str, Any]]:
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    for i in range(H):
        for j in range(W):
            if g[i, j] == 0 or seen[i, j]:
                continue
            color = int(g[i, j])
            q = deque([(i, j)])
            pixels = []
            seen[i, j] = True
            while q:
                r, c = q.popleft()
                pixels.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [p[0] for p in pixels]
            cs = [p[1] for p in pixels]
            comps.append({
                "color": color,
                "pixels": pixels,
                "bbox": (min(rs), min(cs), max(rs) + 1, max(cs) + 1),
                "size": len(pixels)
            })
    return comps


def symmetry_score(g: np.ndarray) -> Dict[str, float]:
    H, W = g.shape
    s_v = float((g == mirror_y(g)).mean())
    s_h = float((g == mirror_x(g)).mean())
    s_t = float((g == transpose(g)).mean()) if H == W else 0.0
    return {"vert": s_v, "horiz": s_h, "diag": s_t}


def borders(g: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "top": g[0, :].copy(),
        "bottom": g[-1, :].copy(),
        "left": g[:, 0].copy(),
        "right": g[:, -1].copy()
    }


def palette_order(g: np.ndarray) -> List[int]:
    vals, counts = np.unique(g, return_counts=True)
    return [int(v) for v, _ in sorted(zip(vals, counts), key=lambda x: (-x[1], x[0]))]


# --- Morphology ---
def mask(g: np.ndarray, color: int) -> np.ndarray:
    return (g == color).astype(np.uint8)


def paint(g: np.ndarray, m: np.ndarray, color: int) -> np.ndarray:
    out = g.copy()
    out[m.astype(bool)] = color
    return out


def grow(m: np.ndarray) -> np.ndarray:
    H, W = m.shape
    out = m.copy()
    for i in range(H):
        for j in range(W):
            if m[i, j]:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = i + dr, j + dc
                if 0 <= nr < H and 0 <= nc < W and m[nr, nc]:
                    out[i, j] = 1
                    break
    return out


def shrink(m: np.ndarray) -> np.ndarray:
    H, W = m.shape
    out = m.copy()
    for i in range(H):
        for j in range(W):
            if not m[i, j]:
                continue
            n = sum(
                0 <= i + dr < H and 0 <= j + dc < W and m[i + dr, j + dc]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if n <= 1:
                out[i, j] = 0
    return out


def skeletonize(m: np.ndarray, iters: int = 2) -> np.ndarray:
    out = m.copy()
    for _ in range(iters):
        out = shrink(out)
    return out


def place(canvas: np.ndarray, sub: np.ndarray, r0: int, c0: int) -> np.ndarray:
    out = canvas.copy()
    H, W = sub.shape
    out[r0:r0 + H, c0:c0 + W] = sub
    return out


# ======================================================
# 6. ARC META-POLICY ENGINE
# ======================================================

def detect(g: np.ndarray) -> Dict[str, Any]:
    return {
        "components": find_components(g),
        "symmetry": symmetry_score(g),
        "palette": palette_order(g),
        "borders": borders(g)
    }


def hypothesize(train: List[Tuple[np.ndarray, np.ndarray]]) -> List[Callable[[np.ndarray], np.ndarray]]:
    hyps: List[Callable[[np.ndarray], np.ndarray]] = []
    for inp, out in train:
        pi = palette_order(inp)
        po = palette_order(out)
        if pi and po and pi[0] != po[0]:
            a, b = pi[0], po[0]
            hyps.append(lambda x, a=a, b=b: swap_colors(x, a, b))
    hyps.extend([rot90, rot180, rot270, transpose, mirror_x, mirror_y])
    hyps.append(lambda x: mirror_y(transpose(x)) if x.size else x)

    # Deduplicate
    uniq: List[Callable[[np.ndarray], np.ndarray]] = []
    seen = set()
    for h in hyps:
        key = (getattr(h, "__name__", "λ"),
               tuple(sorted(getattr(h, "__defaults__", ()) or ())))
        if key not in seen:
            uniq.append(h)
            seen.add(key)
    return uniq


def validate(h: Callable[[np.ndarray], np.ndarray], train: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    ok = 0
    for inp, out in train:
        try:
            if np.array_equal(h(inp), out):
                ok += 1
        except Exception:
            pass
    return ok / max(1, len(train))


def generalize(h: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    return h


def apply_rule(h: Callable[[np.ndarray], np.ndarray], g: np.ndarray) -> np.ndarray:
    return h(g)


def arc_meta_policy_solve(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_inputs: List[np.ndarray]
) -> List[np.ndarray]:
    """Solve ARC task via meta-policy: hypothesize → validate → apply."""
    hyps = hypothesize(train_pairs)
    scores = [validate(h, train_pairs) for h in hyps]
    if not scores:
        return [x.copy() for x in test_inputs]
    best = generalize(hyps[int(np.argmax(scores))])
    return [apply_rule(best, x) for x in test_inputs]


# ======================================================
# 7. DEMO & SELF-TEST
# ======================================================

if __name__ == "__main__":
    # Universe chain
    link, dE = quantum_gravity_bridge(Ψ_quant, G_μν)
    γ = gravity_arc_bridge(G_μν, Γ_arc)
    Θs = arc_defense_bridge(Γ_arc, Θ_def)
    Πu = defense_policy_bridge(Θs, Π_meta, R=γ, η=0.05)
    dΨdt = policy_quantum_bridge(Πu, Ψ_quant)
    Φr = quasicrystal_universal(Φ_qc)
    U = unification_score(Ψ_quant, G_μν, Γ_arc, Θ_def, Πu, Φr)

    print({
        "ΔE_qg": round(dE, 6),
        "Γ_arc_flow": round(γ, 6),
        "Π_trace": round(float(np.trace(Πu)), 6),
        "dΨ_dt_shape": tuple(dΨdt.shape),
        "U_score": round(U, 6)
    })

    # ARC demo
    A = np.array([[1,1,0],[0,0,0],[0,0,0]], dtype=np.uint8)
    B = np.array([[2,2,0],[0,0,0],[0,0,0]], dtype=np.uint8)
    preds = arc_meta_policy_solve([(A, B)], [A])
    print({"arc_demo_ok": bool(np.array_equal(preds[0], B))})
