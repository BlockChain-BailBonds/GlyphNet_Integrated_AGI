#!/data/data/com.termux/files/usr/bin/bash
# GlyphNet Integrated AGI – Environment Bootstrap
set -e
echo "[*] Bootstrapping GlyphNet environment..."

# Update and install core packages
pkg update -y && pkg upgrade -y
pkg install -y git python clang rust binutils jq unzip zstd wget curl

# Python environment
python3 -m ensurepip
pip install --upgrade pip wheel setuptools
pip install numpy pandas matplotlib torch torchvision sympy fastapi plotly requests transformers opencv-python langchain

# Directory verification
dirs=(core vision sigils codex emotion)
for d in "${dirs[@]}"; do
    if [ -d "$d" ]; then
        echo "[+] Verified: $d/"
    else
        echo "[-] Missing directory: $d/"
    fi
done

# Create runtime metadata
cat > runtime_manifest.json <<META
{
  "project": "GlyphNet_Integrated_AGI",
  "version": "1.0.0",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "components": ["GlyphMatics", "Vision-Glyphs", "GLYPH_STRING_COMBOS", "ApexAgentSigilagiGlyphNotes", "Aletheis_Empatheos", "Base_Codex"]
}
META

echo "[✓] Environment ready. Use: python3 examples/full_chain_demo.py"
