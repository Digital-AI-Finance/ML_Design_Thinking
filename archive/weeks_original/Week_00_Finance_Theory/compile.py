#!/usr/bin/env python3
"""
Compile LaTeX presentations with automatic cleanup and archiving.
Usage: python compile.py [filename.tex]
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
import glob

# Configuration
AUX_EXTENSIONS = ['.aux', '.log', '.nav', '.out', '.snm', '.toc', '.vrb', 
                  '.fls', '.fdb_latexmk', '.synctex.gz', '.bbl', '.blg']
ARCHIVE_DIR = 'archive'
AUX_DIR = os.path.join(ARCHIVE_DIR, 'aux')
PREVIOUS_DIR = os.path.join(ARCHIVE_DIR, 'previous')
BUILDS_DIR = os.path.join(ARCHIVE_DIR, 'builds')

def setup_directories():
    """Create archive directory structure if it doesn't exist."""
    for directory in [ARCHIVE_DIR, AUX_DIR, PREVIOUS_DIR, BUILDS_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}")

def compile_tex(filename):
    """Compile LaTeX file to PDF."""
    if not filename.endswith('.tex'):
        print(f"Error: {filename} is not a .tex file")
        return False
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        return False
    
    print(f"\nCompiling {filename}...")
    print("-" * 50)
    
    # Run pdflatex twice for proper references
    for i in range(2):
        print(f"\nPass {i+1} of 2...")
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', filename],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print("\nCompilation warnings/errors:")
                # Show only relevant error lines
                for line in result.stdout.split('\n'):
                    if 'Error' in line or 'Warning' in line or '!' in line:
                        print(f"  {line}")
                
                # Check if PDF was still created
                pdf_name = filename.replace('.tex', '.pdf')
                if not os.path.exists(pdf_name):
                    print(f"\nError: PDF not created. Check {filename} for errors.")
                    return False
                else:
                    print(f"\nWarning: Compilation had issues but PDF was created.")
        
        except subprocess.TimeoutExpired:
            print("Error: Compilation timed out")
            return False
        except Exception as e:
            print(f"Error during compilation: {e}")
            return False
    
    pdf_name = filename.replace('.tex', '.pdf')
    if os.path.exists(pdf_name):
        print(f"\nSuccess! Created: {pdf_name}")
        
        # Get file size
        size = os.path.getsize(pdf_name) / 1024  # KB
        print(f"PDF size: {size:.1f} KB")
        
        # Archive a copy with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        archive_name = os.path.join(BUILDS_DIR, f"{timestamp}_{os.path.basename(pdf_name)}")
        shutil.copy2(pdf_name, archive_name)
        print(f"Archived to: {archive_name}")
        
        return True
    
    return False

def cleanup_auxiliary_files():
    """Move auxiliary files to archive."""
    print("\nCleaning up auxiliary files...")
    moved_count = 0
    
    for ext in AUX_EXTENSIONS:
        for file in glob.glob(f"*{ext}"):
            try:
                destination = os.path.join(AUX_DIR, file)
                shutil.move(file, destination)
                moved_count += 1
                print(f"  Moved: {file} -> {AUX_DIR}/")
            except Exception as e:
                print(f"  Could not move {file}: {e}")
    
    if moved_count > 0:
        print(f"\nMoved {moved_count} auxiliary files to archive")
    else:
        print("No auxiliary files to clean up")

def list_tex_files():
    """List available .tex files in current directory."""
    tex_files = glob.glob("*.tex")
    if not tex_files:
        print("No .tex files found in current directory")
        return None
    
    print("\nAvailable .tex files:")
    for i, file in enumerate(tex_files, 1):
        size = os.path.getsize(file) / 1024
        print(f"  {i}. {file} ({size:.1f} KB)")
    
    # Prioritize main.tex if it exists
    if "main.tex" in tex_files:
        return "main.tex"
    
    # Otherwise look for the most recent timestamped file
    timestamped = [f for f in tex_files if f.split('_')[0].isdigit()]
    if timestamped:
        return sorted(timestamped)[-1]  # Return most recent
    
    return tex_files[0]

def main():
    """Main compilation workflow."""
    print("="*60)
    print("LaTeX Compiler with Automatic Cleanup")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Determine which file to compile
    if len(sys.argv) > 1:
        tex_file = sys.argv[1]
    else:
        tex_file = list_tex_files()
        if tex_file:
            print(f"\nAuto-selected: {tex_file}")
    
    if not tex_file:
        print("\nUsage: python compile.py [filename.tex]")
        print("Or place .tex files in current directory")
        return
    
    # Compile the file
    success = compile_tex(tex_file)
    
    # Clean up regardless of success
    cleanup_auxiliary_files()
    
    # Final summary
    print("\n" + "="*60)
    if success:
        pdf_name = tex_file.replace('.tex', '.pdf')
        print(f"COMPILATION SUCCESSFUL")
        print(f"Output: {pdf_name}")
        print(f"\nTo view: start {pdf_name} (Windows)")
        print(f"         open {pdf_name} (Mac)")
        print(f"         xdg-open {pdf_name} (Linux)")
    else:
        print("COMPILATION FAILED - Check error messages above")
    print("="*60)

if __name__ == "__main__":
    main()