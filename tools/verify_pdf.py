"""Verify a compiled PDF matches the download and update resources.md with checkmark."""
import json
import re
import sys
from datetime import date
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF required. Install with: pip install pymupdf")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
MAPPING_FILE = Path(__file__).parent / "pdf_source_mapping.json"
DOWNLOADS = ROOT / "static" / "downloads"
RESOURCES_MD = ROOT / "content" / "resources.md"

# Topic display names mapping
TOPIC_DISPLAY_NAMES = {
    "ml_foundations": "ML Foundations",
    "supervised_learning": "Supervised Learning",
    "unsupervised_learning": "Unsupervised Learning",
    "clustering": "Clustering",
    "nlp_sentiment": "NLP & Sentiment",
    "classification": "Classification",
    "topic_modeling": "Topic Modeling",
    "generative_ai": "Generative AI",
    "neural_networks": "Neural Networks",
    "responsible_ai": "Responsible AI",
    "structured_output": "Structured Output",
    "validation_metrics": "Validation & Metrics",
    "ab_testing": "A/B Testing",
    "finance_applications": "Finance Applications",
}


def get_page_count(pdf_path: Path) -> int:
    """Get page count from a PDF file."""
    if not pdf_path.exists():
        return -1
    try:
        doc = fitz.open(pdf_path)
        pages = len(doc)
        doc.close()
        return pages
    except Exception:
        return -1


def get_mapping_for_topic(topic_name: str) -> dict:
    """Get mapping info for a topic from the JSON file."""
    if not MAPPING_FILE.exists():
        return None

    with open(MAPPING_FILE) as f:
        data = json.load(f)

    # Convert topic name to PDF name (e.g., clustering -> clustering.pdf)
    pdf_name = topic_name.replace("_", "-") + ".pdf"

    for mapping in data["mappings"]:
        if mapping["pdf"] == pdf_name:
            return mapping

    return None


def verify_topic(topic_name: str) -> bool:
    """Verify that the compiled PDF matches the download."""
    mapping = get_mapping_for_topic(topic_name)
    if not mapping:
        print(f"ERROR: No mapping found for topic: {topic_name}")
        return False

    if mapping["type"] == "generated":
        print(f"SKIP: {topic_name} is a generated file")
        return False

    # Get download PDF info
    download_pdf = DOWNLOADS / mapping["pdf"]
    download_pages = get_page_count(download_pdf)

    if download_pages < 0:
        print(f"ERROR: Cannot read download PDF: {download_pdf}")
        return False

    # Get source PDF info (the one in slides/ folder)
    source_tex = Path(mapping["full_path"])
    source_pdf = source_tex.with_suffix(".pdf")

    # Also check for PDF with same name as .tex file
    if not source_pdf.exists():
        # Try finding any PDF in the slides folder
        slides_folder = source_tex.parent
        pdfs = list(slides_folder.glob("*.pdf"))
        if pdfs:
            # Get the most recently modified one
            source_pdf = max(pdfs, key=lambda p: p.stat().st_mtime)

    if not source_pdf.exists():
        print(f"ERROR: Cannot find compiled PDF in: {source_tex.parent}")
        return False

    source_pages = get_page_count(source_pdf)

    if source_pages < 0:
        print(f"ERROR: Cannot read source PDF: {source_pdf}")
        return False

    # Compare
    if download_pages == source_pages:
        print(f"VERIFIED: {topic_name} - {source_pages} pages match")
        return True
    else:
        print(f"MISMATCH: {topic_name} - source has {source_pages} pages, download has {download_pages}")
        return False


def update_resources_md(topic_name: str):
    """Update resources.md with checkmark and date for the topic."""
    if not RESOURCES_MD.exists():
        print(f"ERROR: resources.md not found: {RESOURCES_MD}")
        return False

    display_name = TOPIC_DISPLAY_NAMES.get(topic_name, topic_name)
    today = date.today().isoformat()
    checkmark = f"Y {today}"  # Using Y instead of unicode checkmark for ASCII compatibility

    content = RESOURCES_MD.read_text(encoding="utf-8")

    # Pattern to match the topic row in the table
    # Looking for: | Topic Name | ... | ... | Verified | ... |
    # The Verified column should be the 4th column

    # First check if the row exists
    pattern = rf"(\| {re.escape(display_name)} \|[^|]+\|[^|]+\|)([^|]*)(\|[^|]+\|)"

    match = re.search(pattern, content)
    if match:
        # Replace the Verified column content
        new_content = re.sub(
            pattern,
            rf"\g<1> {checkmark} \g<3>",
            content
        )
        RESOURCES_MD.write_text(new_content, encoding="utf-8")
        print(f"UPDATED: resources.md - {display_name} verified on {today}")
        return True
    else:
        print(f"WARNING: Could not find row for {display_name} in resources.md")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_pdf.py <topic_name>")
        print("Example: python verify_pdf.py clustering")
        print("\nAvailable topics:")
        for topic in TOPIC_DISPLAY_NAMES:
            print(f"  {topic}")
        sys.exit(1)

    topic_name = sys.argv[1]

    # Normalize topic name (convert hyphens to underscores)
    topic_name = topic_name.replace("-", "_")

    if topic_name not in TOPIC_DISPLAY_NAMES:
        print(f"ERROR: Unknown topic: {topic_name}")
        print("Available topics:", ", ".join(TOPIC_DISPLAY_NAMES.keys()))
        sys.exit(1)

    # Verify the PDF
    if verify_topic(topic_name):
        # Update resources.md
        update_resources_md(topic_name)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
