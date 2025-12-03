"""
Master Script: Generate ALL Charts for Week 0 Introduction to ML & AI
Runs all individual chart generation scripts
"""

import sys
import os
from pathlib import Path

# Change to scripts directory
os.chdir(Path(__file__).parent)

print("="*60)
print("GENERATING ALL CHARTS FOR WEEK 0")
print("="*60)

scripts = [
    'create_foundations_charts.py',
    'create_supervised_charts.py',
    'create_unsupervised_charts.py',
    'create_neural_networks_charts.py',
    'create_generative_ai_charts.py'
]

total_success = 0
total_failed = 0

for i, script in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] Running {script}...")
    print("-"*60)

    try:
        with open(script, encoding='utf-8') as f:
            exec(f.read())
        total_success += 1
    except Exception as e:
        print(f"[ERROR] Failed to run {script}: {e}")
        total_failed += 1

print("\n" + "="*60)
print("CHART GENERATION COMPLETE")
print("="*60)
print(f"Successfully generated: {total_success}/{len(scripts)} script sets")
print(f"Failed: {total_failed}/{len(scripts)} script sets")

# Count generated charts
charts_dir = Path('../charts')
if charts_dir.exists():
    pdf_count = len(list(charts_dir.glob('*.pdf')))
    png_count = len(list(charts_dir.glob('*.png')))
    print(f"\nTotal charts in charts/ directory:")
    print(f"  PDF files: {pdf_count}")
    print(f"  PNG files: {png_count}")

print("\n[OK] All done! Charts ready for LaTeX compilation.")