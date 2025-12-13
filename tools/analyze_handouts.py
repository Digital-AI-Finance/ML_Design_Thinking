"""Analyze handouts for duplicates, unused files, and verify structure."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOPICS_DIR = ROOT / "topics"

def get_file_hash(filepath: Path) -> str:
    """Get MD5 hash of file content."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def analyze_handouts():
    """Analyze all handouts across topics."""
    results = {
        "topics": {},
        "duplicates": [],
        "potential_unused": [],
        "missing_standard": [],
        "summary": {}
    }

    # Track file hashes for duplicate detection
    file_hashes = defaultdict(list)

    # Standard handout pattern
    standard_names = [
        "handout_1_basic", "handout_2_intermediate", "handout_3_advanced",
        "handout_1_level", "handout_2_level", "handout_3_level"
    ]

    total_files = 0
    total_md = 0
    total_pdf = 0
    total_tex = 0

    for topic_dir in sorted(TOPICS_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue

        handouts_dir = topic_dir / "handouts"
        if not handouts_dir.exists():
            results["topics"][topic_dir.name] = {"exists": False, "files": []}
            continue

        topic_info = {
            "exists": True,
            "files": [],
            "md_files": [],
            "pdf_files": [],
            "tex_files": [],
            "other_files": [],
            "has_standard_3": False
        }

        for f in sorted(handouts_dir.iterdir()):
            if f.is_file():
                total_files += 1
                file_info = {
                    "name": f.name,
                    "size_kb": f.stat().st_size / 1024,
                    "ext": f.suffix.lower()
                }
                topic_info["files"].append(file_info)

                # Categorize by extension
                if f.suffix.lower() == ".md":
                    topic_info["md_files"].append(f.name)
                    total_md += 1
                elif f.suffix.lower() == ".pdf":
                    topic_info["pdf_files"].append(f.name)
                    total_pdf += 1
                elif f.suffix.lower() == ".tex":
                    topic_info["tex_files"].append(f.name)
                    total_tex += 1
                else:
                    topic_info["other_files"].append(f.name)

                # Hash for duplicate detection
                file_hash = get_file_hash(f)
                file_hashes[file_hash].append(str(f.relative_to(ROOT)))

        # Check if has standard 3-level structure
        md_names = [f.rsplit('.', 1)[0] for f in topic_info["md_files"]]
        has_standard = any(
            any(std in name for name in md_names)
            for std in ["handout_1", "handout_2", "handout_3"]
        )
        topic_info["has_standard_3"] = has_standard

        if not has_standard:
            results["missing_standard"].append(topic_dir.name)

        results["topics"][topic_dir.name] = topic_info

    # Find duplicates (files with same hash)
    for hash_val, files in file_hashes.items():
        if len(files) > 1:
            results["duplicates"].append({
                "hash": hash_val[:8],
                "files": files
            })

    # Find potential unused files (PDFs without matching tex, old versions)
    for topic_name, info in results["topics"].items():
        if not info.get("exists"):
            continue

        # Check for PDFs that might be outdated duplicates
        pdf_basenames = [f.rsplit('.', 1)[0] for f in info["pdf_files"]]
        tex_basenames = [f.rsplit('.', 1)[0] for f in info["tex_files"]]

        # PDFs with matching .tex are fine
        # PDFs without .tex might be orphaned
        for pdf in info["pdf_files"]:
            basename = pdf.rsplit('.', 1)[0]
            # Skip if it's clearly a compiled version
            if "_compiled" in basename:
                continue
            # Check for similar names (potential duplicates)
            similar = [p for p in info["pdf_files"] if p != pdf and basename.split('_')[0] in p]
            if similar:
                results["potential_unused"].append({
                    "topic": topic_name,
                    "file": pdf,
                    "reason": f"Similar to: {similar}"
                })

    # Summary
    results["summary"] = {
        "total_topics": len([t for t in results["topics"].values() if t.get("exists")]),
        "total_files": total_files,
        "total_md": total_md,
        "total_pdf": total_pdf,
        "total_tex": total_tex,
        "duplicate_groups": len(results["duplicates"]),
        "potential_unused": len(results["potential_unused"]),
        "missing_standard_structure": len(results["missing_standard"])
    }

    return results

def print_report(results: dict):
    """Print analysis report."""
    print("=" * 70)
    print("HANDOUTS ANALYSIS REPORT")
    print("=" * 70)

    # Summary
    s = results["summary"]
    print(f"\nSUMMARY:")
    print(f"  Topics with handouts: {s['total_topics']}")
    print(f"  Total files: {s['total_files']}")
    print(f"  - Markdown: {s['total_md']}")
    print(f"  - PDF: {s['total_pdf']}")
    print(f"  - TeX: {s['total_tex']}")
    print(f"  Duplicate groups: {s['duplicate_groups']}")
    print(f"  Potential unused: {s['potential_unused']}")

    # Per-topic breakdown
    print(f"\nPER-TOPIC BREAKDOWN:")
    print("-" * 70)
    for topic, info in sorted(results["topics"].items()):
        if not info.get("exists"):
            print(f"  {topic}: NO HANDOUTS FOLDER")
            continue

        md_count = len(info["md_files"])
        pdf_count = len(info["pdf_files"])
        tex_count = len(info["tex_files"])
        standard = "Y" if info["has_standard_3"] else "N"

        print(f"  {topic:25} MD:{md_count:2} PDF:{pdf_count:2} TEX:{tex_count:2} Standard3:{standard}")

    # Duplicates
    if results["duplicates"]:
        print(f"\nDUPLICATE FILES (same content):")
        print("-" * 70)
        for dup in results["duplicates"]:
            print(f"  Hash {dup['hash']}:")
            for f in dup["files"]:
                print(f"    - {f}")

    # Potential unused
    if results["potential_unused"]:
        print(f"\nPOTENTIAL UNUSED FILES:")
        print("-" * 70)
        for item in results["potential_unused"]:
            print(f"  {item['topic']}/{item['file']}")
            print(f"    Reason: {item['reason']}")

    # Missing standard structure
    if results["missing_standard"]:
        print(f"\nTOPICS MISSING STANDARD 3-LEVEL STRUCTURE:")
        print("-" * 70)
        for topic in results["missing_standard"]:
            print(f"  - {topic}")

    print("\n" + "=" * 70)

def main():
    results = analyze_handouts()
    print_report(results)

    # Save JSON report
    report_path = ROOT / "docs" / "handouts_analysis.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON report saved to: {report_path}")

if __name__ == "__main__":
    main()
