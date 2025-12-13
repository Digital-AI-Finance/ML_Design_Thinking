"""Convert markdown handouts to LaTeX and PDF.

Creates a pdf/ folder structure in each topic's handouts directory:
  handouts/pdf/basic/handout.tex + handout.pdf
  handouts/pdf/intermediate/handout.tex + handout.pdf
  handouts/pdf/advanced/handout.tex + handout.pdf

Also copies PDFs to static/downloads/handouts/ for website downloads.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOPICS_DIR = ROOT / "topics"
TEMPLATE = ROOT / "tools" / "handout_template.tex"
DOWNLOADS_DIR = ROOT / "static" / "downloads" / "handouts"

# Topic display names
TOPIC_NAMES = {
    "ab_testing": "A/B Testing",
    "classification": "Classification",
    "clustering": "Clustering",
    "finance_applications": "Finance Applications",
    "generative_ai": "Generative AI",
    "ml_foundations": "ML Foundations",
    "neural_networks": "Neural Networks",
    "nlp_sentiment": "NLP & Sentiment",
    "responsible_ai": "Responsible AI",
    "structured_output": "Structured Output",
    "supervised_learning": "Supervised Learning",
    "topic_modeling": "Topic Modeling",
    "unsupervised_learning": "Unsupervised Learning",
    "validation_metrics": "Validation & Metrics",
}

LEVEL_NAMES = {
    "1": "Basic",
    "2": "Intermediate",
    "3": "Advanced",
}


def get_level_from_filename(filename: str) -> str:
    """Extract level (1, 2, 3) from handout filename."""
    if "handout_1" in filename or "_basic" in filename:
        return "1"
    elif "handout_2" in filename or "_intermediate" in filename:
        return "2"
    elif "handout_3" in filename or "_advanced" in filename:
        return "3"
    return None


def get_level_folder(level: str) -> str:
    """Get folder name for level."""
    return {"1": "basic", "2": "intermediate", "3": "advanced"}.get(level)


def extract_title_from_md(md_path: Path) -> str:
    """Extract title from first H1 header in markdown."""
    content = md_path.read_text(encoding="utf-8")
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return md_path.stem.replace("_", " ").title()


def convert_md_to_tex(md_path: Path, tex_path: Path, topic: str, level: str) -> bool:
    """Convert markdown to LaTeX using pandoc."""
    topic_name = TOPIC_NAMES.get(topic, topic.replace("_", " ").title())
    level_name = LEVEL_NAMES.get(level, level)
    title = extract_title_from_md(md_path)

    # Ensure output directory exists
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pandoc",
        str(md_path),
        "-o", str(tex_path),
        "--standalone",
        f"--template={TEMPLATE}",
        "--listings",
        f"--variable=topic:{topic_name}",
        f"--variable=level:{level_name}",
        f"--variable=title:{title}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR pandoc: {e.stderr}")
        return False


def compile_tex_to_pdf(tex_path: Path) -> bool:
    """Compile LaTeX to PDF using pdflatex."""
    # Run pdflatex twice for references
    for _ in range(2):
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory", str(tex_path.parent),
            str(tex_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=tex_path.parent)
            # Check if PDF was created
            pdf_path = tex_path.with_suffix(".pdf")
            if not pdf_path.exists():
                print(f"  ERROR: PDF not created")
                return False
        except subprocess.CalledProcessError as e:
            print(f"  ERROR pdflatex: {e}")
            return False

    # Clean up aux files
    for ext in [".aux", ".log", ".out", ".toc", ".nav", ".snm"]:
        aux_file = tex_path.with_suffix(ext)
        if aux_file.exists():
            aux_file.unlink()

    return True


def process_handout(md_path: Path, topic: str) -> Path:
    """Process a single handout: convert to tex, compile to pdf."""
    level = get_level_from_filename(md_path.name)
    if not level:
        print(f"  SKIP: Cannot determine level for {md_path.name}")
        return None

    level_folder = get_level_folder(level)
    pdf_dir = md_path.parent / "pdf" / level_folder
    tex_path = pdf_dir / "handout.tex"
    pdf_path = pdf_dir / "handout.pdf"

    print(f"  {level_folder}: {md_path.name}")

    # Convert to LaTeX
    if not convert_md_to_tex(md_path, tex_path, topic, level):
        return None

    # Compile to PDF
    if not compile_tex_to_pdf(tex_path):
        return None

    print(f"    -> {pdf_path.relative_to(ROOT)}")
    return pdf_path


def copy_to_downloads(pdf_path: Path, topic: str, level: str) -> Path:
    """Copy PDF to static/downloads/handouts/ with standardized name."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Create download filename: topic-slug-level.pdf
    topic_slug = topic.replace("_", "-")
    level_name = get_level_folder(level)
    download_name = f"{topic_slug}-{level_name}.pdf"
    download_path = DOWNLOADS_DIR / download_name

    shutil.copy2(pdf_path, download_path)
    return download_path


def main():
    """Convert all markdown handouts to PDF."""
    print("Converting markdown handouts to PDF...")
    print(f"Template: {TEMPLATE}")
    print()

    if not TEMPLATE.exists():
        print(f"ERROR: Template not found: {TEMPLATE}")
        sys.exit(1)

    results = {"success": [], "failed": []}

    for topic_dir in sorted(TOPICS_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue

        topic = topic_dir.name
        if topic not in TOPIC_NAMES:
            continue

        handouts_dir = topic_dir / "handouts"
        if not handouts_dir.exists():
            continue

        # Find level handouts
        md_files = []
        for md_path in handouts_dir.glob("*.md"):
            level = get_level_from_filename(md_path.name)
            if level:
                md_files.append((md_path, level))

        if not md_files:
            continue

        print(f"{TOPIC_NAMES[topic]} ({topic})")

        for md_path, level in sorted(md_files, key=lambda x: x[1]):
            pdf_path = process_handout(md_path, topic)
            if pdf_path:
                # Copy to downloads
                download_path = copy_to_downloads(pdf_path, topic, level)
                results["success"].append((topic, level, download_path))
            else:
                results["failed"].append((topic, level, md_path))

        print()

    # Summary
    print("=" * 60)
    print(f"SUCCESS: {len(results['success'])} PDFs created")
    print(f"FAILED: {len(results['failed'])} handouts")

    if results["failed"]:
        print("\nFailed handouts:")
        for topic, level, path in results["failed"]:
            print(f"  - {topic} {level}: {path}")

    print(f"\nDownloads: {DOWNLOADS_DIR}")


if __name__ == "__main__":
    main()
