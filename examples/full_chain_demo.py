#!/usr/bin/env python3
"""
GlyphNet Integrated AGI — Full Chain Demonstration
Flows from glyph input → symbolic reasoning → visual synthesis → emotional modulation → rehydration output.
"""
import json, random, datetime
from pathlib import Path

# Symbolic layer imports (if present)
try:
    import numpy as np, torch, sympy, plotly.express as px
    from transformers import AutoTokenizer, AutoModel
    from langchain.llms import OpenAI
except Exception as e:
    print("[WARN] Optional modules missing:", e)

def glyph_encode(symbol: str) -> str:
    seed = sum(ord(c) for c in symbol)
    random.seed(seed)
    return ''.join(chr(0x2500 + random.randint(0,100)) for _ in range(4))

def symbolic_reasoning(glyph: str) -> dict:
    reasoning = {
        "glyph": glyph,
        "logic_vector": [round(random.random(), 3) for _ in range(4)],
        "insight": random.choice(["reflection", "synthesis", "abstraction", "recursion"])
    }
    return reasoning

def emotional_synthesis(reasoning: dict) -> dict:
    valence = sum(reasoning["logic_vector"]) / len(reasoning["logic_vector"])
    state = "calm" if valence < 0.5 else "excited"
    reasoning.update({"emotional_state": state})
    return reasoning

def visualize(reasoning: dict):
    try:
        import matplotlib.pyplot as plt
        plt.bar(range(4), reasoning["logic_vector"], color="blue" if reasoning["emotional_state"]=="calm" else "red")
        plt.title(f"Glyph: {reasoning['glyph']} • {reasoning['insight']} ({reasoning['emotional_state']})")
        plt.savefig("reasoning_output.png")
        print("[+] Saved visualization: reasoning_output.png")
    except Exception as e:
        print("[WARN] Visualization skipped:", e)

def rehydrate(reasoning: dict):
    codex_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "glyph": reasoning["glyph"],
        "insight": reasoning["insight"],
        "emotional_state": reasoning["emotional_state"],
        "logic_vector": reasoning["logic_vector"]
    }
    Path("output").mkdir(exist_ok=True)
    with open("output/rehydrated_entry.json", "w", encoding="utf-8") as f:
        json.dump(codex_entry, f, indent=2)
    print("[✓] Rehydration complete -> output/rehydrated_entry.json")

if __name__ == "__main__":
    symbol = input("Enter glyph seed > ").strip() or "⊏⚗$9^1(𝔽⍟)ᛟ$⊐"
    g = glyph_encode(symbol)
    r = symbolic_reasoning(g)
    e = emotional_synthesis(r)
    visualize(e)
    rehydrate(e)
