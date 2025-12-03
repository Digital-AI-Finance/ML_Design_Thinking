#!/usr/bin/env python3
"""
Automated LaTeX compilation with cleanup for Week 5 presentation.
Archives auxiliary files to keep workspace clean.
"""

import os
import subprocess
import glob
import shutil
from datetime import datetime
from pathlib import Path
import sys

# Auxiliary file extensions to clean up
AUX_EXTENSIONS = ['.aux', '.log', '.nav', '.out', '.snm', '.toc', '.vrb',
                  '.fls', '.fdb_latexmk', '.synctex.gz', '.bbl', '.blg']

def find_main_tex():
    """Find the main tex file (latest timestamp pattern)."""
    tex_files = glob.glob("*_main.tex")
    if not tex_files:
        tex_files = glob.glob("*main*.tex")
    if not tex_files:
        # If no main file found, look for any .tex file
        tex_files = glob.glob("*.tex")
        # Filter out part files and appendix
        tex_files = [f for f in tex_files if not f.startswith('part') and not f.startswith('appendix')]

    if not tex_files:
        raise FileNotFoundError("No main tex file found")

    # Sort by modification time and get the latest
    tex_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return tex_files[0]

def compile_latex(tex_file):
    """Compile the LaTeX file twice for complete references."""
    print(f"Compiling {tex_file}...")

    # First pass
    print("First compilation pass...")
    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_file],
                          capture_output=True, text=True, encoding='utf-8', errors='ignore')

    if result.stdout and ("Fatal error" in result.stdout or "Emergency stop" in result.stdout):
        print("ERROR: Compilation failed on first pass")
        print("Error output:")
        print(result.stdout[-2000:])  # Print last 2000 chars of output
        return None

    # Second pass for references
    print("Second compilation pass for references...")
    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_file],
                          capture_output=True, text=True, encoding='utf-8', errors='ignore')

    # Check if PDF was created
    pdf_file = tex_file.replace('.tex', '.pdf')
    if os.path.exists(pdf_file):
        print(f"Successfully created {pdf_file}")
        return pdf_file
    else:
        print("WARNING: PDF file was not created. Check for errors in the log.")
        # Print last part of log for debugging
        if os.path.exists(tex_file.replace('.tex', '.log')):
            with open(tex_file.replace('.tex', '.log'), 'r', encoding='utf-8', errors='ignore') as log:
                lines = log.readlines()
                print("Last 50 lines of log:")
                print(''.join(lines[-50:]))
        return None

def cleanup_auxiliary_files():
    """Move auxiliary files to archive folder."""
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_dir = Path("archive") / f"aux_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved_files = []
    for ext in AUX_EXTENSIONS:
        for file in glob.glob(f"*{ext}"):
            try:
                dest = archive_dir / file
                shutil.move(file, dest)
                moved_files.append(file)
            except Exception as e:
                print(f"Could not move {file}: {e}")

    if moved_files:
        print(f"Moved {len(moved_files)} auxiliary files to {archive_dir}")
        print(f"Files moved: {', '.join(moved_files[:5])}" +
              (f" and {len(moved_files)-5} more" if len(moved_files) > 5 else ""))
    else:
        print("No auxiliary files to clean up")

    return archive_dir

def open_pdf(pdf_file):
    """Open the PDF file with default viewer."""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(pdf_file)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', pdf_file])
        else:  # Linux
            subprocess.run(['xdg-open', pdf_file])
        print(f"Opening {pdf_file}...")
    except Exception as e:
        print(f"Could not open PDF automatically: {e}")
        print(f"Please open {pdf_file} manually")

def main():
    """Main compilation workflow."""
    print("=" * 50)
    print("Week 5 LaTeX Compilation Tool")
    print("Topic Modeling & Ideation")
    print("=" * 50)

    try:
        # Find main tex file
        if len(sys.argv) > 1:
            tex_file = sys.argv[1]
            if not os.path.exists(tex_file):
                print(f"ERROR: File {tex_file} not found")
                return 1
        else:
            tex_file = find_main_tex()

        print(f"Found main file: {tex_file}")

        # Compile
        pdf_file = compile_latex(tex_file)

        if pdf_file:
            # Clean up auxiliary files
            print("\nCleaning up auxiliary files...")
            cleanup_auxiliary_files()

            # Open PDF
            open_pdf(pdf_file)

            print("\n" + "=" * 50)
            print("Compilation complete!")
            print(f"PDF location: {os.path.abspath(pdf_file)}")
            print("=" * 50)
            return 0
        else:
            print("\n" + "=" * 50)
            print("Compilation failed. Check the log files for errors.")
            print("=" * 50)
            return 1

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure you're in the Week_05 directory and main.tex exists")
        return 1
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit(main())