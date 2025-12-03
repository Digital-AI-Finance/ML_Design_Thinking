"""
Chart Generation for Week 0 Part 1: ML Foundations
WCAG AAA Compliant Color Palette
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# WCAG AAA Compliant Colors
COLORS = {
    'blue': '#1F77B4',
    'orange': '#FF7F0E',
    'green': '#2CA02C',
    'red': '#D62728',
    'purple': '#9467BD',
    'brown': '#8C564B',
    'pink': '#E377C2',
    'gray': '#7F7F7F',
    'olive': '#BCBD22',
    'cyan': '#17BECF'
}

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'


def create_ml_pipeline():
    """Create ML pipeline visualization"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    stages = [
        ('Data Collection', 1, COLORS['blue']),
        ('Data Preprocessing', 3.5, COLORS['orange']),
        ('Feature Engineering', 6, COLORS['green']),
        ('Model Training', 8.5, COLORS['red']),
        ('Model Evaluation', 11, COLORS['purple']),
    ]

    for i, (stage, x, color) in enumerate(stages):
        box = FancyBboxPatch((x, 2), 1.8, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black',
                             linewidth=2,
                             alpha=0.7)
        ax.add_patch(box)
        ax.text(x + 0.9, 3, stage,
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white',
                wrap=True)

        if i < len(stages) - 1:
            arrow = FancyArrowPatch((x + 1.8, 3), (stages[i+1][1], 3),
                                   arrowstyle='->',
                                   mutation_scale=30,
                                   linewidth=3,
                                   color='black')
            ax.add_patch(arrow)

    ax.text(7, 5.3, 'Machine Learning Pipeline',
            ha='center', fontsize=18, fontweight='bold')
    ax.text(7, 0.7, 'Iterative Process with Feedback Loops',
            ha='center', fontsize=12, style='italic', color=COLORS['gray'])

    plt.tight_layout()
    plt.savefig('../charts/ml_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/ml_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created ml_pipeline.pdf")


def create_data_splitting():
    """Create train/validation/test split visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))

    sizes = [0.6, 0.2, 0.2]
    labels = ['Training Set\n(60%)', 'Validation Set\n(20%)', 'Test Set\n(20%)']
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       colors=colors, autopct='%1.0f%%',
                                       startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')

    ax.set_title('Data Splitting Strategy', fontsize=18, fontweight='bold', pad=20)

    purposes = [
        'Model Training:\nLearn patterns',
        'Hyperparameter Tuning:\nModel selection',
        'Final Evaluation:\nUnbiased performance'
    ]

    for i, (purpose, color) in enumerate(zip(purposes, colors)):
        ax.text(0, -1.5 - i*0.25, purpose,
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))

    plt.tight_layout()
    plt.savefig('../charts/data_splitting.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/data_splitting.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created data_splitting.pdf")


if __name__ == '__main__':
    print("Generating ML Foundations Charts...")
    create_ml_pipeline()
    create_data_splitting()
    print("[OK] All foundations charts created successfully!")