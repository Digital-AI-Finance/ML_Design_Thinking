"""
Script to populate topic folders from archived week content.
Copies files from archive/weeks_original/ to topics/ structure.
"""

import shutil
import os
from pathlib import Path

BASE = Path(r"D:\Joerg\Research\slides\ML_Design_Thinking_16")
ARCHIVE = BASE / "archive" / "weeks_original"
TOPICS = BASE / "topics"

# Mapping: topic -> list of source weeks
TOPIC_MAPPING = {
    "ml_foundations": ["Week_00_Introduction_ML_AI", "Week_00a_ML_Foundations"],
    "supervised_learning": ["Week_00b_Supervised_Learning"],
    "unsupervised_learning": ["Week_00c_Unsupervised_Learning"],
    "clustering": ["Week_01", "Week_02"],
    "nlp_sentiment": ["Week_03"],
    "classification": ["Week_04"],
    "topic_modeling": ["Week_05"],
    "generative_ai": ["Week_06", "Week_00e_Generative_AI"],
    "neural_networks": ["Week_00d_Neural_Networks"],
    "responsible_ai": ["Week_07"],
    "structured_output": ["Week_08"],
    "validation_metrics": ["Week_09"],
    "ab_testing": ["Week_10"],
    "finance_applications": ["Week_00_Finance_Theory"],
}

def copy_folder_contents(src, dst, folder_name):
    """Copy contents of a subfolder (charts, scripts, handouts) if it exists."""
    src_folder = src / folder_name
    dst_folder = dst / folder_name
    if src_folder.exists():
        for item in src_folder.iterdir():
            dst_path = dst_folder / item.name
            if item.is_file():
                shutil.copy2(item, dst_path)
                print(f"  Copied: {item.name} -> {folder_name}/")
            elif item.is_dir() and folder_name != "archive":
                # Skip archive subfolders
                pass

def copy_slides(src_week, dst_topic, week_name):
    """Copy slide files (.tex, .pdf) to slides/ folder."""
    slides_dst = dst_topic / "slides"

    # Copy main .tex and .pdf files
    for item in src_week.iterdir():
        if item.is_file():
            if item.suffix in ['.tex', '.pdf']:
                # Rename to include source week for multi-week topics
                if 'main' in item.name.lower():
                    # Keep original name in slides folder
                    dst_path = slides_dst / item.name
                else:
                    dst_path = slides_dst / item.name
                shutil.copy2(item, dst_path)
                print(f"  Copied: {item.name} -> slides/")
            elif item.suffix == '.py' and 'compile' in item.name.lower():
                # Copy compile.py to topic root
                shutil.copy2(item, dst_topic / item.name)
                print(f"  Copied: {item.name} -> topic root")

def populate_topic(topic_name, source_weeks):
    """Populate a single topic folder from source weeks."""
    print(f"\n{'='*60}")
    print(f"Populating: {topic_name}")
    print(f"Sources: {', '.join(source_weeks)}")
    print('='*60)

    dst_topic = TOPICS / topic_name

    for week_name in source_weeks:
        src_week = ARCHIVE / week_name
        if not src_week.exists():
            print(f"  WARNING: {week_name} not found in archive!")
            continue

        print(f"\n  From {week_name}:")

        # Copy slides
        copy_slides(src_week, dst_topic, week_name)

        # Copy charts
        copy_folder_contents(src_week, dst_topic, "charts")

        # Copy scripts
        copy_folder_contents(src_week, dst_topic, "scripts")

        # Copy handouts
        copy_folder_contents(src_week, dst_topic, "handouts")

def main():
    print("Topic Population Script")
    print("="*60)
    print(f"Archive: {ARCHIVE}")
    print(f"Topics: {TOPICS}")

    for topic_name, source_weeks in TOPIC_MAPPING.items():
        populate_topic(topic_name, source_weeks)

    print("\n" + "="*60)
    print("DONE! All topics populated.")
    print("="*60)

if __name__ == "__main__":
    main()
