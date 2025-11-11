#!/usr/bin/env python3
# Deterministic generator of full self-contained sigil strings (with deps)
# Emits NDJSON: {"sigil": "...", "L": int, "D": int, "id": "hex16"}
import argparse, hashlib, json
from itertools import product, combinations

COLS = list(range(1,10))
TRANS = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1)]
SCALES = [1,2,3]

def prim_ops():
    ops = ["R90","R180","R270","MX","MY"]
    ops += [f"T({dx},{dy})" for dx,dy in TRANS]
    ops += [f"SC({s})" for s in SCALES]
    ops += [f"FL({c})" for c in COLS]
    for a,b in combinations(COLS,2):
        ops.append(f"SW({a},{b})")
    for a,b in combinations(COLS,2):
        ops.append(f"MV({a},{b})")
    ops += [f"TR({rid})" for rid in range(1,6)]
    return ops

PRIMS = prim_ops()

def body_str(seq): return ";".join(seq)
def dep_block(name, dep_seq): return f"[dep:{name}]{{{body_str(dep_seq)}}}"
def meta_str(kv): return "meta(" + ",".join(f"{k}={v}" for k,v in kv.items()) + ")"
def sigil(meta_kv, body_seq, dep_defs):
    deps = "" if not dep_defs else "::" + ";".join(dep_defs)
    return "{"+ meta_str(meta_kv) + "::" + body_str(body_seq) + deps + "}"
def digest(s): return hashlib.sha256(s.encode()).hexdigest()[:16]

def dep_sequences(max_dep_len):
    base = [
        ["MX"], ["MY"], ["R90"], ["R180"], ["R270"],
        ["FL(1)"], ["FL(2)"], ["SW(1,2)"], ["MV(1,2)"], ["SC(2)"],
        ["T(1,0)"], ["T(0,1)"]
    ]
    out = []
    for L in range(1, max_dep_len+1):
        for combo in product(base, repeat=L):
            seq = [op for chunk in combo for op in chunk]
            out.append(seq)
    return out

def ensure_call(body_seq, dep_count):
    if dep_count == 0: return body_seq
    seq = list(body_seq)
    slots = max(1, len(seq))
    for i in range(dep_count):
        pos = (i * max(1, slots // dep_count)) % (len(seq)+1)
        seq.insert(pos, f"CALL(D{i+1})")
    return seq

def enumerate_sigils(max_len, max_deps, dep_len, limit):
    dep_pool = dep_sequences(dep_len)
    emitted = 0
    for L in range(1, max_len+1):
        for seq in product(PRIMS, repeat=L):
            for D in range(0, max_deps+1):
                deps_defs = []
                for i in range(D):
                    dep_seq = dep_pool[i % len(dep_pool)]
                    deps_defs.append(dep_block(f"D{i+1}", dep_seq))
                body_with_calls = ensure_call(list(seq), D)
                s = sigil({"version":1,"L":L,"D":D}, body_with_calls, deps_defs)
                print(json.dumps({"sigil": s, "L": L, "D": D, "id": digest(s)}, ensure_ascii=False))
                emitted += 1
                if limit and emitted >= limit: return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=3)
    ap.add_argument("--max_deps", type=int, default=2)
    ap.add_argument("--dep_len", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)  # 0 = no cap
    args = ap.parse_args()
    enumerate_sigils(args.max_len, args.max_deps, args.dep_len, args.limit)
