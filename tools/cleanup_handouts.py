"""Clean up duplicate and unused handout files."""
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOPICS_DIR = ROOT / "topics"
ARCHIVE_DIR = ROOT / "archive" / "handouts_cleanup"

def archive_file(filepath: Path, reason: str):
    """Move file to archive with reason suffix."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Create topic subfolder in archive
    rel_path = filepath.relative_to(TOPICS_DIR)
    topic = rel_path.parts[0]
    archive_topic = ARCHIVE_DIR / topic
    archive_topic.mkdir(exist_ok=True)

    # Archive with reason
    dest = archive_topic / f"{filepath.stem}__{reason}{filepath.suffix}"
    shutil.move(str(filepath), str(dest))
    print(f"  ARCHIVED: {filepath.name} -> archive/{topic}/ ({reason})")
    return dest

def cleanup_clustering():
    """Clean up clustering handouts - remove duplicate naming conventions."""
    print("\n=== CLUSTERING ===")
    handouts = TOPICS_DIR / "clustering" / "handouts"

    # Keep the descriptive names, archive the generic ones
    duplicates = [
        # (file_to_archive, reason, better_version)
        ("handout_1_basic_advanced_clustering.md", "old_naming", "handout_1_basic_clustering_fundamentals.md"),
        ("handout_2_intermediate_implementation.md", "old_naming", "handout_2_intermediate_clustering_implementation.md"),
        ("handout_3_advanced_theory.md", "old_naming", "handout_3_advanced_clustering_theory.md"),
    ]

    for old_file, reason, better in duplicates:
        old_path = handouts / old_file
        better_path = handouts / better
        if old_path.exists() and better_path.exists():
            archive_file(old_path, reason)
        elif old_path.exists():
            print(f"  SKIP: {old_file} (no better version found)")

def cleanup_generative_ai():
    """Clean up generative_ai handouts - remove placeholder files."""
    print("\n=== GENERATIVE AI ===")
    handouts = TOPICS_DIR / "generative_ai" / "handouts"

    # These are tiny placeholders (< 1KB)
    placeholders = [
        "handout_1_level.md",
        "handout_2_level.md",
        "handout_3_level.md",
    ]

    for placeholder in placeholders:
        path = handouts / placeholder
        if path.exists():
            size = path.stat().st_size
            if size < 1500:  # Less than 1.5KB = placeholder
                archive_file(path, "placeholder")
            else:
                print(f"  SKIP: {placeholder} ({size} bytes - not a placeholder)")

def cleanup_ml_foundations():
    """Clean up ml_foundations handouts - keep only FINAL versions."""
    print("\n=== ML FOUNDATIONS ===")
    handouts = TOPICS_DIR / "ml_foundations" / "handouts"

    # Keep: 20251007_2345_discovery_handout_FINAL.pdf, 20251008_0800_discovery_solutions_v2.pdf
    # Archive: older versions

    old_versions = [
        ("20251007_2200_discovery_handout.pdf", "superseded"),
        ("20251007_2200_discovery_handout.tex", "superseded"),
        ("20251007_2300_discovery_handout_v2_new.pdf", "superseded"),
        ("20251007_2330_discovery_handout_final.pdf", "superseded"),
        ("20251007_2200_discovery_solutions.pdf", "superseded"),
        ("20251007_2200_discovery_solutions.tex", "superseded"),
    ]

    for old_file, reason in old_versions:
        path = handouts / old_file
        if path.exists():
            archive_file(path, reason)

def cleanup_other_topics():
    """Check other topics for _level.md placeholders."""
    print("\n=== OTHER TOPICS ===")

    topics_with_level = ["neural_networks", "unsupervised_learning", "finance_applications"]

    for topic in topics_with_level:
        handouts = TOPICS_DIR / topic / "handouts"
        if not handouts.exists():
            continue

        for level_file in handouts.glob("handout_*_level.md"):
            size = level_file.stat().st_size
            # Check if there's a better-named version
            level_num = level_file.stem.split("_")[1]  # e.g., "1" from "handout_1_level"
            better_patterns = [
                f"handout_{level_num}_basic*.md",
                f"handout_{level_num}_intermediate*.md",
                f"handout_{level_num}_advanced*.md",
            ]

            has_better = False
            for pattern in better_patterns:
                matches = list(handouts.glob(pattern))
                if matches and matches[0] != level_file:
                    has_better = True
                    break

            if size < 1500 and has_better:
                archive_file(level_file, "placeholder")
            elif size < 1500:
                print(f"  WARNING: {topic}/{level_file.name} is small ({size}b) but no better version")

def verify_tex_pdf_pairs():
    """Verify that PDFs have matching .tex source files."""
    print("\n=== TEX/PDF VERIFICATION ===")

    for topic_dir in sorted(TOPICS_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue

        handouts = topic_dir / "handouts"
        if not handouts.exists():
            continue

        pdfs = list(handouts.glob("*.pdf"))
        texs = {t.stem: t for t in handouts.glob("*.tex")}

        orphan_pdfs = []
        for pdf in pdfs:
            base = pdf.stem
            # Check for exact match or _compiled suffix
            if base in texs:
                continue
            if base.replace("_compiled", "") in texs:
                continue
            orphan_pdfs.append(pdf.name)

        if orphan_pdfs:
            print(f"  {topic_dir.name}: PDFs without .tex: {orphan_pdfs}")

def main():
    print("=" * 60)
    print("HANDOUTS CLEANUP")
    print(f"Archive location: {ARCHIVE_DIR}")
    print("=" * 60)

    cleanup_clustering()
    cleanup_generative_ai()
    cleanup_ml_foundations()
    cleanup_other_topics()
    verify_tex_pdf_pairs()

    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
