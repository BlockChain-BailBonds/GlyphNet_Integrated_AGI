# tests/integration/test_arc_solver.py
import numpy as np
from glyphmatics.component import UnifiedARCComponent

def test_arc_color_swap():
    comp = UnifiedARCComponent()
    A = np.array([[1,1,0],[0,0,0],[0,0,0]], dtype=np.uint8)
    B = np.array([[2,2,0],[0,0,0],[0,0,0]], dtype=np.uint8)
    pred = comp.solve_arc([(A,B)], [A])[0]
    assert np.array_equal(pred, B)
