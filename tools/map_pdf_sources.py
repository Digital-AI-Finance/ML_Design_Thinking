"""Map PDF downloads to their source .tex folders and output JSON."""
import json
import re
from datetime import datetime
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
DOWNLOADS = ROOT / "static" / "downloads"
TOPICS = ROOT / "topics"
INNOVATION = ROOT / "innovation_diamond"
OUTPUT = Path(__file__).parent / "pdf_source_mapping.json"

# Special mappings for non-standard PDFs
SPECIAL_MAPPINGS = {
    "innovation-diamond.pdf": ("innovation_diamond/slides", None),
    "innovation-diamond-summary.pdf": ("innovation_diamond/slides/summary", "summary"),
    "innovation-diamond-practical.pdf": ("innovation_diamond/slides/summary", "practical"),
    "innovation-diamond-worksheet.pdf": ("innovation_diamond/handouts", "worksheet"),
    "all-lectures.pdf": ("GENERATED", None),
    "all-lectures.zip": ("GENERATED", None),
}


def pdf_to_topic_name(pdf_name: str) -> str:
    """Convert PDF name to topic folder name: pdf-name.pdf -> pdf_name"""
    return pdf_name.replace(".pdf", "").replace("-", "_")


def find_latest_tex(folder: Path, pattern: str = None) -> tuple:
    """Find the latest timestamped .tex file in a folder."""
    if not folder.exists():
        return None, None

    tex_files = list(folder.glob("*.tex"))
    if not tex_files:
        return None, None

    # Filter by pattern if provided
    if pattern:
        if pattern == "summary":
            # Exclude practical variant
            tex_files = [f for f in tex_files if "summary" in f.name and "practical" not in f.name]
        else:
            tex_files = [f for f in tex_files if pattern in f.name]

    # Sort by timestamp in filename (YYYYMMDD_HHMM format)
    timestamped = []
    for f in tex_files:
        match = re.match(r"(\d{8}_\d{4})", f.name)
        if match:
            timestamped.append((match.group(1), f))

    if timestamped:
        # Sort by timestamp descending
        timestamped.sort(key=lambda x: x[0], reverse=True)
        return timestamped[0][1].name, str(timestamped[0][1])

    # Fallback: return first .tex file found
    return tex_files[0].name, str(tex_files[0])


def map_pdf_to_source(pdf_path: Path) -> dict:
    """Map a single PDF to its source location."""
    pdf_name = pdf_path.name

    # Check special mappings first
    if pdf_name in SPECIAL_MAPPINGS:
        rel_folder, pattern = SPECIAL_MAPPINGS[pdf_name]
        if rel_folder == "GENERATED":
            return {
                "pdf": pdf_name,
                "topic": None,
                "source_folder": "GENERATED",
                "tex_file": None,
                "full_path": None,
                "type": "generated"
            }

        folder = ROOT / rel_folder
        tex_name, full_path = find_latest_tex(folder, pattern)

        return {
            "pdf": pdf_name,
            "topic": "innovation_diamond",
            "source_folder": rel_folder,
            "tex_file": tex_name,
            "full_path": full_path,
            "type": "innovation_diamond"
        }

    # Standard topic mapping
    topic_name = pdf_to_topic_name(pdf_name)
    folder = TOPICS / topic_name / "slides"
    tex_name, full_path = find_latest_tex(folder)

    return {
        "pdf": pdf_name,
        "topic": topic_name,
        "source_folder": f"topics/{topic_name}/slides",
        "tex_file": tex_name,
        "full_path": full_path,
        "type": "topic"
    }


def main():
    """Generate PDF-to-source mapping JSON."""
    print("Scanning PDFs in static/downloads/...")

    pdfs = sorted(DOWNLOADS.glob("*.pdf"))
    mappings = []

    for pdf in pdfs:
        mapping = map_pdf_to_source(pdf)
        mappings.append(mapping)

        # Print status
        if mapping["type"] == "generated":
            print(f"  {mapping['pdf']:40} -> GENERATED")
        elif mapping["tex_file"]:
            print(f"  {mapping['pdf']:40} -> {mapping['source_folder']}/{mapping['tex_file']}")
        else:
            print(f"  {mapping['pdf']:40} -> NOT FOUND")

    # Also check for .zip files
    zips = sorted(DOWNLOADS.glob("*.zip"))
    for zip_file in zips:
        if zip_file.name in SPECIAL_MAPPINGS:
            mappings.append({
                "pdf": zip_file.name,
                "topic": None,
                "source_folder": "GENERATED",
                "tex_file": None,
                "full_path": None,
                "type": "generated"
            })
            print(f"  {zip_file.name:40} -> GENERATED")

    # Create output
    output = {
        "generated": datetime.now().isoformat(),
        "total_pdfs": len([m for m in mappings if m["pdf"].endswith(".pdf")]),
        "mappings": mappings
    }

    # Write JSON
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to: {OUTPUT}")
    print(f"Total mappings: {len(mappings)}")

    # Summary by type
    types = {}
    for m in mappings:
        types[m["type"]] = types.get(m["type"], 0) + 1
    print(f"By type: {types}")


if __name__ == "__main__":
    main()
