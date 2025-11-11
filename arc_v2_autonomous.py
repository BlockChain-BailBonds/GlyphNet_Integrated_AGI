#!/usr/bin/env python3
"""
ARC-AGI v2 Autonomous Solver
============================

- No Termux
- No UI/GUI
- No external dependencies beyond Python + NumPy
- Fully offline
- Generates submission.json
- Uses GlyphMatics + Meta-Policy + Vision Hints
- Designed for ARC Prize 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any
import hashlib
import sys
import os

# =============================
# 1. GLYPHMATICS CORE (NUMPY)
# =============================

class GlyphMaticsEngine:
    def __init__(self, seed: int = 918):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

    # --- Grid Primitives ---
    def rot90(self, g): return np.rot90(g, 1)
    def rot180(self, g): return np.rot90(g, 2)
    def rot270(self, g): return np.rot90(g, 3)
    def flipud(self, g): return np.flipud(g)
    def fliplr(self, g): return np.fliplr(g)
    def transpose(self, g): return g.T.copy()

    def swap_colors(self, g, a, b):
        out = g.copy()
        ma, mb = (out == a), (out == b)
        out[ma], out[mb] = b, a
        return out

    def flood_fill(self, g, r, c, new_color):
        H, W = g.shape
        old = g[r, c]
        if old == new_color: return g.copy()
        out = g.copy()
        q = [(r, c)]
        while q:
            i, j = q.pop(0)
            if not (0 <= i < H and 0 <= j < W) or out[i, j] != old: continue
            out[i, j] = new_color
            q.extend([(i-1,j),(i+1,j),(i,j-1),(i,j+1)])
        return out

    def find_components(self, g):
        H, W = g.shape
        seen = np.zeros_like(g, bool)
        comps = []
        for i in range(H):
            for j in range(W):
                if g[i,j] == 0 or seen[i,j]: continue
                color = int(g[i,j])
                q = [(i,j)]
                pixels = []
                seen[i,j] = True
                while q:
                    r,c = q.pop(0)
                    pixels.append((r,c))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = r+dr,c+dc
                        if 0<=nr<H and 0<=nc<W and not seen[nr,nc] and g[nr,nc]==color:
                            seen[nr,nc] = True
                            q.append((nr,nc))
                rs, cs = [p[0] for p in pixels], [p[1] for p in pixels]
                comps.append({
                    "color": color,
                    "size": len(pixels),
                    "bbox": (min(rs), min(cs), max(rs)+1, max(cs)+1)
                })
        return comps

    def symmetry_score(self, g):
        s_v = float((g == self.fliplr(g)).mean())
        s_h = float((g == self.flipud(g)).mean())
        return {"vert": s_v, "horiz": s_h}

    # --- Meta-Policy Solver ---
    def hypothesize(self, train: List[Tuple[np.ndarray, np.ndarray]]) -> List[Callable]:
        hyps = []
        for inp, out in train:
            pi = np.unique(inp, return_counts=True)
            po = np.unique(out, return_counts=True)
            if len(pi[0]) > 0 and len(po[0]) > 0:
                a, b = pi[0][np.argmax(pi[1])], po[0][np.argmax(po[1])]
                if a != b:
                    hyps.append(lambda x, a=a, b=b: self.swap_colors(x, a, b))
        hyps.extend([self.rot90, self.rot180, self.rot270, self.flipud, self.fliplr])
        uniq = []
        seen = set()
        for h in hyps:
            key = (getattr(h, "__name__", "λ"),)
            if key not in seen:
                uniq.append(h)
                seen.add(key)
        return uniq

    def validate(self, h: Callable, train: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        ok = sum(np.array_equal(h(inp), out) for inp, out in train)
        return ok / max(1, len(train))

    def solve_task(self, task: Dict) -> List[Dict[str, List]]:
        train_pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task.get("train", [])]
        test_inputs = [np.array(t["input"]) for t in task.get("test", [])]

        if not train_pairs:
            return [{"attempt_1": t.tolist(), "attempt_2": t.tolist()} for t in test_inputs]

        hyps = self.hypothesize(train_pairs)
        scores = [self.validate(h, train_pairs) for h in hyps]
        best = hyps[int(np.argmax(scores))] if scores else (lambda x: x)

        preds = [best(x) for x in test_inputs]
        fallback = test_inputs

        return [
            {"attempt_1": p.tolist(), "attempt_2": f.tolist()}
            for p, f in zip(preds, fallback)
        ]


# =============================
# 2. AUTONOMOUS SOLVER
# =============================

class AutonomousARCSolver:
    def __init__(self):
        self.engine = GlyphMaticsEngine()
        self.submission_path = Path("submission.json")

    def load_tasks(self, path: str = "arc-agi_test_challenges.json") -> Dict:
        p = Path(path)
        if not p.exists():
            print(f"[ERROR] Test file not found: {p}")
            sys.exit(1)
        return json.load(open(p))

    def solve_all(self, tasks: Dict) -> Dict:
        results = {}
        total = len(tasks)
        for i, (tid, task) in enumerate(tasks.items(), 1):
            print(f"[{i}/{total}] Solving task: {tid}")
            results[tid] = self.engine.solve_task(task)
        return results

    def save_submission(self, results: Dict):
        self.submission_path.write_text(json.dumps(results, separators=(",", ":")))
        print(f"[SUCCESS] submission.json written to {self.submission_path.resolve()}")

    def run(self, input_path: str = "arc-agi_test_challenges.json"):
        print("Starting Autonomous ARC-AGI v2 Solver...")
        tasks = self.load_tasks(input_path)
        results = self.solve_all(tasks)
        self.save_submission(results)
        print("Done. Ready for ARC Prize submission.")


# =============================
# 3. ENTRY POINT
# =============================

if __name__ == "__main__":
    solver = AutonomousARCSolver()
    solver.run()
