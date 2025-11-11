#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts"
mkdir -p "$OUT"
python3 "$ROOT/sigils/sigil_enumerator.py" --max_len 3 --max_deps 2 --dep_len 2 > "$OUT/sigils.ndjson"
cp "$ROOT/sigils/sigil_interpreter.py" "$OUT/"
echo '[ [0,0,1,1],[0,2,0,2],[3,0,0,0],[3,3,0,4] ]' > "$OUT/sample_grid.json"
echo "wrote: artifacts/sigils.ndjson, sigil_interpreter.py, sample_grid.json"
