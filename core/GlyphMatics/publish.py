#!/usr/bin/env python3
# publish.py – Build & Upload GlyphMatics to PyPI

import subprocess
import sys

def main():
    print("=== Building GlyphMatics ===")
    subprocess.run([sys.executable, "-m", "build"], check=True)
    print("Build complete – dist/ created.")
    
    print("=== Validating ===")
    subprocess.run(["twine", "check", "dist/*"], check=True)
    
    print("=== Uploading to TestPyPI ===")
    subprocess.run(["twine", "upload", "--repository-url", "https://test.pypi.org/legacy/", "dist/*"], check=True)
    
    print("=== Test Install ===")
    subprocess.run(["pip", "install", "--index-url", "https://test.pypi.org/simple/", "glyphmatics"], check=True)
    
    print("=== SUCCESS – GlyphMatics on TestPyPI! ===")
    print("Now upload to live PyPI: twine upload dist/*")
    print("Update version in pyproject.toml for next release.")

if __name__ == "__main__":
    main()
