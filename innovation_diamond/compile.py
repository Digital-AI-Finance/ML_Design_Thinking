#!/usr/bin/env python3
"""
Compile Innovation Diamond presentation.
Runs pdflatex twice for cross-references, archives aux files.
"""

import subprocess
import shutil
from pathlib import Path
from datetime import datetime

def main():
    slides_dir = Path(__file__).parent / "slides"

    # Find the main .tex file (latest timestamped or main.tex)
    tex_files = list(slides_dir.glob("*.tex"))
    if not tex_files:
        print("ERROR: No .tex files found in slides/")
        return 1

    # Prefer timestamped files, else main.tex
    timestamped = [f for f in tex_files if f.stem[0:8].isdigit()]
    if timestamped:
        tex_file = max(timestamped, key=lambda f: f.stem)
    else:
        tex_file = slides_dir / "main.tex"
        if not tex_file.exists():
            tex_file = tex_files[0]

    print(f"Compiling: {tex_file.name}")

    # Run pdflatex twice
    for run in [1, 2]:
        print(f"  Pass {run}/2...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file.name],
            cwd=slides_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0 and run == 2:
            print("ERROR: pdflatex failed")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return 1

    # Move aux files to temp/
    temp_dir = slides_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    for ext in [".aux", ".log", ".nav", ".out", ".snm", ".toc", ".vrb"]:
        for f in slides_dir.glob(f"*{ext}"):
            shutil.move(str(f), str(temp_dir / f.name))

    pdf_file = tex_file.with_suffix(".pdf")
    if pdf_file.exists():
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        print(f"\nSuccess: {pdf_file.name} ({size_mb:.1f} MB)")

    return 0

if __name__ == "__main__":
    exit(main())
