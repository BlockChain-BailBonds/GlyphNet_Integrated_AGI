import numpy as np
from src.glyphmatics.component import GlyphMaticsEngine

engine = GlyphMaticsEngine()
print("Universe:", engine.run_universe())

A = np.array([[1,1,0],[0,0,0],[0,0,0]], dtype=np.uint8)
B = np.array([[2,2,0],[0,0,0],[0,0,0]], dtype=np.uint8)
pred = engine.solve_arc([(A,B)], [A])[0]
print("ARC Correct:", np.array_equal(pred, B))
