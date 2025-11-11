#!/usr/bin/env python3
# sigils/sigil_interpreter.py — executable, deterministic
import re, json, copy
from typing import List, Dict, Tuple, Optional
from collections import deque, Counter

Grid = List[List[int]]

META_RE = re.compile(r'^\{meta\((?P<meta>[^)]*)\)::(?P<body>[^:]*)?(?:::(?P<deps>.*))?\}$')
DEP_RE  = re.compile(r'\[dep:(?P<name>[A-Za-z0-9_]+)\]\{(?P<body>[^}]*)\}')
TOK_SPLIT = re.compile(r'\s*;\s*')

# ---------- parsing ----------
def parse_meta(meta_str:str)->Dict[str,str]:
    out={}
    if not meta_str: return out
    for kv in meta_str.split(","):
        if not kv: continue
        k,v = kv.split("=",1)
        out[k.strip()] = v.strip()
    return out

def parse_body(body_str:str)->List[str]:
    if not body_str: return []
    return [t for t in TOK_SPLIT.split(body_str) if t]

def parse_deps(deps_str:str)->Dict[str,List[str]]:
    if not deps_str: return {}
    out={}
    for m in DEP_RE.finditer(deps_str):
        out[m.group("name")] = parse_body(m.group("body"))
    return out

def parse_sigil(s: str):
    m = META_RE.match(s)
    if not m: raise ValueError("Invalid SIGIL format")
    meta = parse_meta(m.group("meta"))
    body = parse_body(m.group("body") or "")
    deps = parse_deps(m.group("deps") or "")
    return meta, body, deps

# ---------- grid ops ----------
def clone(g:Grid)->Grid: return [r[:] for r in g]
def rot90(g:Grid)->Grid: return [list(reversed(c)) for c in zip(*g)]
def rot180(g:Grid)->Grid: return rot90(rot90(g))
def rot270(g:Grid)->Grid: return rot90(rot180(g))
def mirror_x(g:Grid)->Grid: return [list(reversed(r)) for r in g]      # horizontal flip
def mirror_y(g:Grid)->Grid: return list(reversed(g))                    # vertical flip
def translate(g:Grid, dx:int, dy:int)->Grid:
    h,w=len(g),len(g[0])
    z=[[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            ny, nx = y+dy, x+dx
            if 0<=ny<h and 0<=nx<w: z[ny][nx]=g[y][x]
    return z
def scale(g:Grid, s:int)->Grid:
    if s<=1: return clone(g)
    h,w=len(g),len(g[0])
    z=[[0]*(w*s) for _ in range(h*s)]
    for y in range(h):
        for x in range(w):
            v=g[y][x]
            for yy in range(s):
                for xx in range(s):
                    z[y*s+yy][x*s+xx]=v
    return z
def flood_fill_color(g:Grid, c:int)->Grid:
    return [[c if v!=0 else 0 for v in r] for r in g]
def swap_color(g:Grid, a:int,b:int)->Grid:
    z=clone(g)
    for r in z:
        for i,v in enumerate(r):
            if v==a: r[i]=b
            elif v==b: r[i]=a
    return z
def move_color(g:Grid, a:int,b:int)->Grid:
    z=clone(g)
    for r in z:
        for i,v in enumerate(r):
            if v==a: r[i]=b
    return z

# ---------- helpers ----------
def size(g:Grid)->Tuple[int,int]: return len(g), len(g[0])
def inside(h,w,y,x)->bool: return 0<=y<h and 0<=x<w
def neighbors4(y,x): return ((y-1,x),(y+1,x),(y,x-1),(y,x+1))

def components(g:Grid)->Tuple[List[List[int]], Dict[int,Tuple[int,int,int]]]:
    """Return label grid and stats: label -> (color, size, first_idx). 0 = background."""
    h,w=size(g)
    lab=[[0]*w for _ in range(h)]
    nxt=1
    stats={}
    for y in range(h):
        for x in range(w):
            if g[y][x]==0 or lab[y][x]!=0: continue
            col=g[y][x]
            q=deque([(y,x)]); lab[y][x]=nxt; cnt=0
            while q:
                cy,cx=q.popleft(); cnt+=1
                for ny,nx in neighbors4(cy,cx):
                    if inside(h,w,ny,nx) and g[ny][nx]==col and lab[ny][nx]==0:
                        lab[ny][nx]=nxt; q.append((ny,nx))
            stats[nxt]=(col,cnt,y*w+x)
            nxt+=1
    return lab, stats

def dominant_border_color(g:Grid)->int:
    h,w=size(g); vals=[]
    vals+=g[0]; vals+=g[h-1]
    for y in range(1,h-1):
        vals.append(g[y][0]); vals.append(g[y][w-1])
    vals=[v for v in vals if v!=0]
    if not vals: return 0
    return Counter(vals).most_common(1)[0][0]

def fill_between_equal_colors_line(arr:List[int])->List[int]:
    """Bridge zeros between equal colors on a 1D line."""
    out=arr[:]; n=len(arr); i=0
    while i<n:
        if out[i]==0: i+=1; continue
        c=out[i]; j=i+1
        while j<n and out[j]==0: j+=1
        if j<n and out[j]==c:
            for k in range(i+1,j): out[k]=c
        i=j
    return out

# ---------- TR rules ----------
def TR1_symmetry_complete(g:Grid)->Grid:
    """Complete to closest symmetry (vertical or horizontal)."""
    h,w=size(g)
    # mismatch counts
    mism_vert=sum(1 for y in range(h) for x in range(w//2) if g[y][x]!=g[y][w-1-x])
    mism_horz=sum(1 for y in range(h//2) for x in range(w) if g[y][x]!=g[h-1-y][x])
    z=clone(g)
    if mism_vert<=mism_horz:
        # mirror left onto right
        for y in range(h):
            for x in range(w//2):
                z[y][w-1-x]=z[y][x]
    else:
        # mirror top onto bottom
        for y in range(h//2):
            for x in range(w):
                z[h-1-y][x]=z[y][x]
    return z

def TR2_border_fill(g:Grid)->Grid:
    """Paint border to dominant border color, then one-pixel inward ring if zeros."""
    h,w=size(g); z=clone(g)
    c=dominant_border_color(g)
    if c==0: return z
    # outer border
    for x in range(w): z[0][x]=c; z[h-1][x]=c
    for y in range(h): z[y][0]=c; z[y][w-1]=c
    # inward ring fill where zeros
    if h>2 and w>2:
        for x in range(1,w-1):
            if z[1][x]==0: z[1][x]=c
            if z[h-2][x]==0: z[h-2][x]=c
        for y in range(1,h-1):
            if z[y][1]==0: z[y][1]=c
            if z[y][w-2]==0: z[y][w-2]=c
    return z

def TR3_denoise_small_islands(g:Grid, thresh:int=2)->Grid:
    """Remove tiny components (< = thresh) by replacing with local mode."""
    h,w=size(g); z=clone(g)
    lab, stats = components(g)
    small=[lid for lid,(col,cnt,_) in stats.items() if cnt<=thresh]
    if not small: return z
    for y in range(h):
        for x in range(w):
            lid=lab[y][x]
            if lid in small:
                # gather neighbor colors
                neigh=[]
                for ny,nx in neighbors4(y,x):
                    if inside(h,w,ny,nx) and g[ny][nx]!=0:
                        neigh.append(g[ny][nx])
                z[y][x] = Counter(neigh).most_common(1)[0][0] if neigh else 0
    return z

def TR4_sequence_propagation(g:Grid)->Grid:
    """Bridge gaps between equal colors along rows and columns."""
    h,w=size(g); z=clone(g)
    # rows
    for y in range(h):
        z[y]=fill_between_equal_colors_line(z[y])
    # cols
    for x in range(w):
        col=[z[y][x] for y in range(h)]
        col=fill_between_equal_colors_line(col)
        for y in range(h): z[y][x]=col[y]
    return z

def TR5_keep_largest_component(g:Grid)->Grid:
    """Keep largest nonzero component. Zero out others."""
    z=clone(g)
    _, stats = components(g)
    if not stats: return z
    # largest by size; tie-break by earliest index
    best=max(stats.items(), key=lambda kv:(kv[1][1], -kv[1][2]))[0]
    h,w=size(g)
    lab,_=components(g)
    for y in range(h):
        for x in range(w):
            if lab[y][x]!=best: z[y][x]=0
    return z

# ---------- executor ----------
def apply_op(g:Grid, tok:str, deps:Dict[str,List[str]])->Grid:
    if tok=="R90": return rot90(g)
    if tok=="R180": return rot180(g)
    if tok=="R270": return rot270(g)
    if tok=="MX": return mirror_x(g)
    if tok=="MY": return mirror_y(g)
    if tok.startswith("T("):
        dx,dy=map(int, tok[2:-1].split(",")); return translate(g,dx,dy)
    if tok.startswith("SC("):
        s=int(tok[3:-1]); return scale(g,s)
    if tok.startswith("FL("):
        c=int(tok[3:-1]); return flood_fill_color(g,c)
    if tok.startswith("SW("):
        a,b=map(int, tok[3:-1].split(",")); return swap_color(g,a,b)
    if tok.startswith("MV("):
        a,b=map(int, tok[3:-1].split(",")); return move_color(g,a,b)
    if tok.startswith("TR("):
        rid=int(tok[3:-1])
        if rid==1: return TR1_symmetry_complete(g)
        if rid==2: return TR2_border_fill(g)
        if rid==3: return TR3_denoise_small_islands(g, thresh=2)
        if rid==4: return TR4_sequence_propagation(g)
        if rid==5: return TR5_keep_largest_component(g)
        return g
    if tok.startswith("CALL("):
        name=tok[5:-1]
        if name not in deps: raise KeyError(f"Missing dependency: {name}")
        z=g
        for sub in deps[name]:
            z=apply_op(z, sub, deps)
        return z
    raise ValueError(f"Unknown token: {tok}")

def run_sigil(s: str, grid: Grid)->Grid:
    _, body, deps = parse_sigil(s)
    z=clone(grid)
    for tok in body:
        z=apply_op(z, tok, deps)
    return z

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--sigil", required=True)
    ap.add_argument("--grid_json", required=True)
    ap.add_argument("--out_json", required=True)
    a=ap.parse_args()
    with open(a.grid_json) as f: grid=json.load(f)
    out=run_sigil(a.sigil, grid)
    with open(a.out_json,"w") as f: json.dump(out,f)
