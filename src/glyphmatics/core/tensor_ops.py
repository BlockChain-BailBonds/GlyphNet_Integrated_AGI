import numpy as np
from numpy.linalg import svd

def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)

def merge_op(a: np.ndarray, b: np.ndarray, rank_cap: int = 64) -> np.ndarray:
    A = a.reshape(a.shape[0], -1)
    B = b.reshape(b.shape[0], -1)
    M = np.concatenate([A, B], axis=1)
    U, S, Vt = svd(M, full_matrices=False)
    k = max(1, min(min(A.shape[0], Vt.shape[0]), rank_cap))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]

def entropy(s: np.ndarray) -> float:
    s = np.maximum(s, 1e-12)
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())

def blockdiag(*tensors: np.ndarray) -> np.ndarray:
    mats = [x.reshape(x.shape[0], -1) for x in tensors]
    heights = [m.shape[0] for m in mats]
    widths = [m.shape[1] for m in mats]
    H, W = sum(heights), max(widths)
    out = np.zeros((H, W))
    i = 0
    for m in mats:
        h, w = m.shape
        out[i:i+h, :w] = m
        i += h
    return out
