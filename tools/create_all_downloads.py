"""Create download archives for all lecture PDFs.

Generates:
- all-lectures.zip: ZIP containing all 14 individual PDFs
- all-lectures.pdf: Single merged PDF with all lectures
"""

import zipfile
from pathlib import Path

# Try to import PyPDF2, fall back to pypdf if not available
try:
    from PyPDF2 import PdfMerger
except ImportError:
    try:
        from pypdf import PdfMerger
    except ImportError:
        print("ERROR: Please install PyPDF2 or pypdf: pip install PyPDF2")
        exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DOWNLOADS_DIR = PROJECT_ROOT / "static" / "downloads"

# PDF order (matches course topic sequence)
PDF_ORDER = [
    "ml-foundations.pdf",
    "supervised-learning.pdf",
    "unsupervised-learning.pdf",
    "clustering.pdf",
    "classification.pdf",
    "neural-networks.pdf",
    "generative-ai.pdf",
    "nlp-sentiment.pdf",
    "topic-modeling.pdf",
    "responsible-ai.pdf",
    "structured-output.pdf",
    "validation-metrics.pdf",
    "ab-testing.pdf",
    "finance-applications.pdf",
]


def create_zip():
    """Create ZIP archive of all PDFs."""
    zip_path = DOWNLOADS_DIR / "all-lectures.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for pdf_name in PDF_ORDER:
            pdf_path = DOWNLOADS_DIR / pdf_name
            if pdf_path.exists():
                zf.write(pdf_path, pdf_name)
                print(f"  Added: {pdf_name}")
            else:
                print(f"  WARNING: Missing {pdf_name}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nCreated: {zip_path}")
    print(f"Size: {size_mb:.1f} MB")
    return zip_path


def create_merged_pdf():
    """Create single merged PDF from all lectures."""
    merged_path = DOWNLOADS_DIR / "all-lectures.pdf"

    merger = PdfMerger()

    for pdf_name in PDF_ORDER:
        pdf_path = DOWNLOADS_DIR / pdf_name
        if pdf_path.exists():
            merger.append(str(pdf_path))
            print(f"  Merged: {pdf_name}")
        else:
            print(f"  WARNING: Missing {pdf_name}")

    merger.write(str(merged_path))
    merger.close()

    size_mb = merged_path.stat().st_size / (1024 * 1024)
    print(f"\nCreated: {merged_path}")
    print(f"Size: {size_mb:.1f} MB")
    return merged_path


def main():
    print("=" * 50)
    print("Creating Download Archives")
    print("=" * 50)

    # Verify downloads directory exists
    if not DOWNLOADS_DIR.exists():
        print(f"ERROR: Downloads directory not found: {DOWNLOADS_DIR}")
        return

    # Count available PDFs
    available = [p for p in PDF_ORDER if (DOWNLOADS_DIR / p).exists()]
    print(f"\nFound {len(available)}/{len(PDF_ORDER)} PDFs")

    # Create ZIP
    print("\n--- Creating ZIP Archive ---")
    create_zip()

    # Create merged PDF
    print("\n--- Creating Merged PDF ---")
    create_merged_pdf()

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
