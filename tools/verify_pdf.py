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


def get_pdf_info(pdf_path: Path) -> dict:
    """Get comprehensive info from a PDF file."""
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(pdf_path)
        info = {
            "pages": len(doc),
            "size_kb": pdf_path.stat().st_size / 1024,
            "first_page_text": "",
            "last_page_text": "",
            "text_hash": 0,
            "image_count": 0,
            "link_count": 0,
            "outline_items": 0,
            "fonts": set(),
            "title_slide": "",
            "slide_titles": [],
        }

        if len(doc) > 0:
            # Extract text from first and last pages
            info["first_page_text"] = doc[0].get_text()[:500]
            info["last_page_text"] = doc[-1].get_text()[:500]

            # Create text fingerprint and count elements
            all_text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                all_text += page_text

                # Count images per page
                info["image_count"] += len(page.get_images())

                # Count links
                info["link_count"] += len(page.get_links())

                # Extract slide title (first line of each page, cleaned)
                lines = [l.strip() for l in page_text.split('\n') if l.strip()]
                if lines:
                    title = lines[0][:80]  # First 80 chars of first line
                    info["slide_titles"].append(title)
                    if page_num == 0:
                        info["title_slide"] = title

                # Collect fonts used
                for font in page.get_fonts():
                    if font[3]:  # font name
                        info["fonts"].add(font[3])

            info["text_hash"] = hash(all_text) % (10**10)

            # Get PDF outline/bookmarks count
            toc = doc.get_toc()
            info["outline_items"] = len(toc) if toc else 0

        # Convert set to sorted list for comparison
        info["fonts"] = sorted(info["fonts"])

        doc.close()
        return info
    except Exception as e:
        print(f"ERROR reading {pdf_path}: {e}")
        return None


def get_page_count(pdf_path: Path) -> int:
    """Get page count from a PDF file."""
    info = get_pdf_info(pdf_path)
    return info["pages"] if info else -1


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
    """Verify that the compiled PDF matches the download using multiple criteria."""
    mapping = get_mapping_for_topic(topic_name)
    if not mapping:
        print(f"ERROR: No mapping found for topic: {topic_name}")
        return False

    if mapping["type"] == "generated":
        print(f"SKIP: {topic_name} is a generated file")
        return False

    # Get download PDF info
    download_pdf = DOWNLOADS / mapping["pdf"]
    download_info = get_pdf_info(download_pdf)

    if not download_info:
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

    source_info = get_pdf_info(source_pdf)

    if not source_info:
        print(f"ERROR: Cannot read source PDF: {source_pdf}")
        return False

    # Multi-criteria verification
    checks = {}
    details = []

    # 1. Page count (exact match required - CRITICAL)
    checks["pages"] = download_info["pages"] == source_info["pages"]
    details.append(f"pages: {source_info['pages']}/{download_info['pages']}")

    # 2. File size (within tolerance - recompilation varies)
    size_ratio = source_info["size_kb"] / download_info["size_kb"] if download_info["size_kb"] > 0 else 0
    checks["size"] = 0.20 <= size_ratio <= 5.0  # Wide tolerance for recompilation
    details.append(f"size: {size_ratio:.0%}")

    # 3. Image count (should be similar - IMPORTANT)
    img_diff = abs(source_info["image_count"] - download_info["image_count"])
    checks["images"] = img_diff <= max(5, download_info["image_count"] * 0.1)  # 10% or 5 tolerance
    details.append(f"imgs: {source_info['image_count']}/{download_info['image_count']}")

    # 4. Title slide match (first slide title - IMPORTANT)
    # Normalize whitespace for comparison
    src_title = ' '.join(source_info["title_slide"].split()).lower()
    dl_title = ' '.join(download_info["title_slide"].split()).lower()
    checks["title"] = src_title == dl_title or src_title in dl_title or dl_title in src_title

    # 5. Slide titles similarity (compare first 5 and last 5 slides)
    src_titles = [' '.join(t.split()).lower() for t in source_info["slide_titles"]]
    dl_titles = [' '.join(t.split()).lower() for t in download_info["slide_titles"]]

    # Compare first 5 titles
    first_match = sum(1 for s, d in zip(src_titles[:5], dl_titles[:5]) if s == d or s in d or d in s)
    # Compare last 5 titles
    last_match = sum(1 for s, d in zip(src_titles[-5:], dl_titles[-5:]) if s == d or s in d or d in s)
    title_score = (first_match + last_match) / 10 if len(src_titles) >= 5 else 1.0
    checks["slide_titles"] = title_score >= 0.6  # 60% of sampled titles match
    details.append(f"titles: {title_score:.0%}")

    # 6. Text hash (exact content match - informational)
    checks["hash"] = download_info["text_hash"] == source_info["text_hash"]

    # Calculate verification score
    critical_checks = ["pages", "images", "title", "slide_titles"]
    critical_pass = all(checks[c] for c in critical_checks)
    all_pass = all(checks.values())

    # Determine result
    if all_pass:
        print(f"VERIFIED (exact): {topic_name} - {' | '.join(details)}")
        return True
    elif critical_pass:
        print(f"VERIFIED: {topic_name} - {' | '.join(details)}")
        return True
    else:
        failed = [c for c in critical_checks if not checks[c]]
        print(f"MISMATCH: {topic_name} - failed: {', '.join(failed)} | {' | '.join(details)}")
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
