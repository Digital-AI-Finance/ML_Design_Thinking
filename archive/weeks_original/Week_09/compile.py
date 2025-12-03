import subprocess
import os
import sys
from datetime import datetime
import shutil
import glob

print("=" * 50)
print("Week 9 LaTeX Compilation Tool")
print("Multi-Metric Validation & Model Selection")
print("=" * 50)

# Find the latest main.tex file
main_files = glob.glob("*_main.tex")
if not main_files:
    print("ERROR: No main.tex file found")
    sys.exit(1)

main_file = sorted(main_files)[-1]
print(f"Found main file: {main_file}")

# Compile twice for references
print(f"Compiling {main_file}...")
print("First compilation pass...")

result1 = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", main_file],
    capture_output=True,
    text=True
)

if result1.returncode != 0:
    print("ERROR: Compilation failed on first pass")
    print("Error output:")
    print(result1.stdout[-2000:])
    sys.exit(1)

print("Second compilation pass (for references)...")
result2 = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", main_file],
    capture_output=True,
    text=True
)

if result2.returncode != 0:
    print("WARNING: Second pass had issues, but PDF may still be usable")

# Create archive directory for auxiliary files
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
archive_dir = f"archive/aux_{timestamp}"
os.makedirs(archive_dir, exist_ok=True)

# Move auxiliary files
aux_extensions = ["*.aux", "*.log", "*.nav", "*.out", "*.snm", "*.toc", "*.vrb"]
moved_count = 0

for pattern in aux_extensions:
    for file in glob.glob(pattern):
        try:
            shutil.move(file, os.path.join(archive_dir, file))
            moved_count += 1
        except Exception as e:
            print(f"Warning: Could not move {file}: {e}")

print(f"\nMoved {moved_count} auxiliary files to {archive_dir}")

# Find and display PDF
pdf_name = main_file.replace('.tex', '.pdf')
if os.path.exists(pdf_name):
    abs_path = os.path.abspath(pdf_name)
    print("\n" + "=" * 50)
    print("SUCCESS: Compilation complete!")
    print(f"PDF location: {abs_path}")
    print("=" * 50)

    # Try to open PDF (Windows)
    try:
        os.startfile(abs_path)
        print("Opened PDF in default viewer")
    except:
        print("Could not auto-open PDF")
else:
    print("\nWARNING: PDF file not found, compilation may have failed")
    sys.exit(1)