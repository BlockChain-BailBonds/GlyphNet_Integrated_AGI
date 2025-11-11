GLYPHNOTES_GLYPH_STRING_COMBOS/
│
├── README.md
├── LICENSE
├── index.html
│
├── glyphstore.md
├── codex_index.md
│
├── languages/
│   ├── python.md
│   ├── cpp.md
│   ├── bash.md
│   ├── go.md
│   ├── java.md
│   ├── javascript.md
│   ├── rust.md
│   ├── sql.md
│   ├── html.md
│   ├── apl.md
│   ├── kotlin.md
│   ├── ja.md
│   └── zh.md
│
├── sigils/
│   ├── sigil_enumerator.py        # Generates all full self-contained sigil string combos
│   ├── sigil_interpreter.py       # Executes sigil strings on ARC-style grids
│   ├── pack.sh                    # Builds artifact bundles (sigils + interpreter)
│   └── enumerate                  # Convenience launcher (bash wrapper)
│
├── artifacts/
│   ├── sigils.ndjson              # Output of enumerator (auto-generated)
│   ├── sample_grid.json           # Example input grid
│   └── out.json                   # Example transformation output
│
└── README_SIGILS.md               # Usage guide for sigil subsystem

cd GLYPHNOTES_GLYPH_STRING_COMBOS
mkdir -p sigils artifacts languages
python3 sigils/sigil_enumerator.py --max_len 3 --max_deps 2 --dep_len 2 > artifacts/sigils.ndjson
bash sigils/pack.sh

# GlyphNotes Sigil Subsystem

This module extends GlyphNotes into an **executable system** that can
generate and run fully self-contained glyph string + dependency combos.

## Generate Sigils
```bash
python3 sigils/sigil_enumerator.py --max_len 3 --max_deps 2 --dep_len 2 > artifacts/sigils.ndjson
