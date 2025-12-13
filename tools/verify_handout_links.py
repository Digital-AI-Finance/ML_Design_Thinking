"""Verify handout links in resources.md and generate correct links."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOPICS_DIR = ROOT / "topics"
RESOURCES_MD = ROOT / "content" / "resources.md"

GITHUB_BASE = "https://github.com/Digital-AI-Finance/ML_Design_Thinking/blob/main"

# Topic display names (same order as resources.md)
TOPICS = [
    ("ml_foundations", "ML Foundations"),
    ("supervised_learning", "Supervised Learning"),
    ("unsupervised_learning", "Unsupervised Learning"),
    ("clustering", "Clustering"),
    ("nlp_sentiment", "NLP & Sentiment"),
    ("classification", "Classification"),
    ("topic_modeling", "Topic Modeling"),
    ("generative_ai", "Generative AI"),
    ("neural_networks", "Neural Networks"),
    ("responsible_ai", "Responsible AI"),
    ("structured_output", "Structured Output"),
    ("validation_metrics", "Validation & Metrics"),
    ("ab_testing", "A/B Testing"),
    ("finance_applications", "Finance Applications"),
]

def find_handout_files(topic_dir: str) -> dict:
    """Find the best handout files for a topic."""
    handouts_path = TOPICS_DIR / topic_dir / "handouts"
    if not handouts_path.exists():
        return {"basic": None, "intermediate": None, "advanced": None}

    result = {"basic": None, "intermediate": None, "advanced": None}
    md_files = list(handouts_path.glob("*.md"))

    # Priority patterns for each level
    patterns = {
        "basic": ["handout_1_basic", "handout_1_level"],
        "intermediate": ["handout_2_intermediate", "handout_2_level"],
        "advanced": ["handout_3_advanced", "handout_3_level"],
    }

    for level, pats in patterns.items():
        for pat in pats:
            matches = [f for f in md_files if pat in f.name.lower()]
            if matches:
                # Pick the longest file (most content)
                best = max(matches, key=lambda f: f.stat().st_size)
                result[level] = best.name
                break

    return result

def generate_handout_table():
    """Generate the handout table for resources.md."""
    lines = []
    lines.append("| Topic | Basic | Intermediate | Advanced |")
    lines.append("|-------|-------|--------------|----------|")

    placeholder_topics = []

    for topic_dir, display_name in TOPICS:
        files = find_handout_files(topic_dir)

        cells = []
        for level in ["basic", "intermediate", "advanced"]:
            filename = files[level]
            if filename:
                size = (TOPICS_DIR / topic_dir / "handouts" / filename).stat().st_size
                if size < 500:
                    # Placeholder file
                    cells.append(f"[{level.title()}]({GITHUB_BASE}/topics/{topic_dir}/handouts/{filename}) *")
                    placeholder_topics.append((topic_dir, level))
                else:
                    cells.append(f"[{level.title()}]({GITHUB_BASE}/topics/{topic_dir}/handouts/{filename})")
            else:
                cells.append("-")

        lines.append(f"| {display_name} | {cells[0]} | {cells[1]} | {cells[2]} |")

    return "\n".join(lines), placeholder_topics

def verify_current_links():
    """Check if current links in resources.md are valid."""
    content = RESOURCES_MD.read_text(encoding="utf-8")

    # Find all GitHub links to handouts
    pattern = r'https://github\.com/Digital-AI-Finance/ML_Design_Thinking/blob/main/topics/([^/]+)/handouts/([^)]+)'
    matches = re.findall(pattern, content)

    print("Verifying current links...")
    broken = []
    valid = []

    for topic, filename in matches:
        filepath = TOPICS_DIR / topic / "handouts" / filename
        if filepath.exists():
            valid.append(f"  OK: {topic}/{filename}")
        else:
            broken.append(f"  BROKEN: {topic}/{filename}")

    print(f"\nValid links: {len(valid)}")
    print(f"Broken links: {len(broken)}")

    if broken:
        print("\nBroken links:")
        for b in broken:
            print(b)

    return broken

def update_resources_md(new_table: str):
    """Update resources.md with new handout table."""
    content = RESOURCES_MD.read_text(encoding="utf-8")

    # Find and replace the handout table
    # Table starts after the bullet points and ends before ## Python Dependencies
    pattern = r'(\| Topic \| Basic \| Intermediate \| Advanced \|.*?\| Finance Applications \|[^\n]+)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        new_content = content[:match.start()] + new_table + content[match.end():]
        RESOURCES_MD.write_text(new_content, encoding="utf-8")
        print("\nresources.md updated with verified links.")
        return True
    else:
        print("\nERROR: Could not find handout table in resources.md")
        return False

def main():
    print("=" * 60)
    print("HANDOUT LINK VERIFICATION")
    print("=" * 60)

    # Verify current links
    broken = verify_current_links()

    # Generate new table
    print("\nGenerating new handout table...")
    new_table, placeholders = generate_handout_table()

    print("\nNew table:")
    print(new_table)

    if placeholders:
        print(f"\nWARNING: {len(placeholders)} placeholder files marked with *")
        for topic, level in placeholders:
            print(f"  - {topic}: {level}")

    # Update if there were broken links
    if broken:
        print("\nUpdating resources.md to fix broken links...")
        update_resources_md(new_table)
    else:
        print("\nAll links valid. No update needed.")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
