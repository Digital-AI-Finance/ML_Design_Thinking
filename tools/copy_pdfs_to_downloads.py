"""Copy latest slide PDFs from topics to static/downloads for Hugo website."""

import shutil
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
TOPICS_DIR = ROOT / "topics"
DOWNLOADS_DIR = ROOT / "static" / "downloads"

# Topic to filename mapping (hyphenated for web URLs)
TOPIC_MAPPING = {
    "ab_testing": "ab-testing",
    "classification": "classification",
    "clustering": "clustering",
    "finance_applications": "finance-applications",
    "generative_ai": "generative-ai",
    "ml_foundations": "ml-foundations",
    "neural_networks": "neural-networks",
    "nlp_sentiment": "nlp-sentiment",
    "responsible_ai": "responsible-ai",
    "structured_output": "structured-output",
    "supervised_learning": "supervised-learning",
    "topic_modeling": "topic-modeling",
    "unsupervised_learning": "unsupervised-learning",
    "validation_metrics": "validation-metrics",
}

def find_latest_main_pdf(slides_dir):
    """Find the latest *main*.pdf file in the slides directory."""
    if not slides_dir.exists():
        return None

    # Find all main PDFs (excluding beginner versions)
    pdfs = [p for p in slides_dir.glob("*main*.pdf")
            if "beginner" not in p.name.lower()]

    if not pdfs:
        return None

    # Sort by modification time (newest first) or by name (timestamp prefix)
    pdfs.sort(key=lambda p: p.name, reverse=True)
    return pdfs[0]

def main():
    # Ensure downloads directory exists
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for topic_folder, web_name in TOPIC_MAPPING.items():
        slides_dir = TOPICS_DIR / topic_folder / "slides"
        pdf = find_latest_main_pdf(slides_dir)

        if pdf:
            dest = DOWNLOADS_DIR / f"{web_name}.pdf"
            shutil.copy2(pdf, dest)
            copied.append((topic_folder, pdf.name, dest.name))
            print(f"[OK] {topic_folder}: {pdf.name} -> {dest.name}")
        else:
            missing.append(topic_folder)
            print(f"[MISSING] {topic_folder}: No main PDF found")

    print(f"\n--- Summary ---")
    print(f"Copied: {len(copied)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print(f"\nMissing topics: {', '.join(missing)}")

if __name__ == "__main__":
    main()
