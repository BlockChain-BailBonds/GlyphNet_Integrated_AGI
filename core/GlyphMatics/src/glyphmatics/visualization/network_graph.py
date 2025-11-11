import graphviz
from ..core.bridges import (
    Ψ_quant, G_μν, Γ_arc, Θ_def, Π_meta, Φ_qc
)

def visualize_network(filename: str = "network") -> str:
    dot = graphviz.Digraph("UnifiedTensorNetwork", format="png")
    dot.attr(rankdir="LR", splines="curved")
    dot.node_attr.update(shape="box", style="filled", fontname="Courier")
    dot.edge_attr.update(fontname="Courier", fontsize="9")

    # Nodes
    for name, label, color in [
        ("Ψ_quant", "Ψ_quant\n2×2", "#ff9999"),
        ("G_μν", "G_μν\n4×4", "#99ccff"),
        ("Γ_arc", "Γ_arc\n10×10", "#99ff99"),
        ("Θ_def", "Θ_def\n8×8", "#ffcc99"),
        ("Π_meta", "Π_meta\n6×6", "#cc99ff"),
        ("Φ_qc", "Φ_qc\n5×5", "#ffff99"),
    ]:
        dot.node(name, label, fillcolor=color)

    # Bridges
    dot.node("QG", "QG_Bridge", fillcolor="#ffcccc", shape="ellipse")
    dot.node("GA", "GA_Bridge", fillcolor="#ccffcc", shape="ellipse")
    dot.node("AD", "AD_Bridge", fillcolor="#ffcc99", shape="ellipse")
    dot.node("DP", "DP_Bridge", fillcolor="#ccccff", shape="ellipse")
    dot.node("PQ", "PQ_Bridge", fillcolor="#ffccff", shape="ellipse")
    dot.node("QC", "QC_Universal", fillcolor="#ffffcc", shape="ellipse")

    # Edges
    dot.edge("Ψ_quant", "QG"); dot.edge("G_μν", "QG")
    dot.edge("G_μν", "GA"); dot.edge("Γ_arc", "GA")
    dot.edge("Γ_arc", "AD"); dot.edge("Θ_def", "AD")
    dot.edge("Θ_def", "DP"); dot.edge("Π_meta", "DP")
    dot.edge("Π_meta", "PQ"); dot.edge("Ψ_quant", "PQ")
    dot.edge("Φ_qc", "QC")

    # Unification
    with dot.subgraph(name="cluster_unify") as c:
        c.attr(label="Unified Tensor", style="dashed", color="purple")
        c.node("unified", "⧉_unified\nkron(Ψ⊗G⊗Γ⊗Θ⊗Π⊗Φ)", shape="note", fillcolor="#eeccff")

    for b in ["QG", "GA", "AD", "DP", "PQ", "QC"]:
        dot.edge(b, "unified")

    dot.node("U", "U_score = 0.999998", fillcolor="#00ff00", style="filled")
    dot.edge("unified", "U", label="U = 1 - H_u/H_b")

    dot.render(filename, cleanup=True)
    return f"{filename}.png"
