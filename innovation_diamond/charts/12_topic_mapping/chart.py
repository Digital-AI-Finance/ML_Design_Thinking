"""Topic-Diamond Mapping - All 14 topics mapped to diamond stages"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

fig, ax = plt.subplots(figsize=(10, 6))

# Diamond stages with their topics
stages = [
    {'y': 5.5, 'stage': 'Challenge', 'topics': ['ML Foundations'], 'color': '#9467bd'},
    {'y': 4.5, 'stage': 'Exploration', 'topics': ['Unsupervised Learning'], 'color': '#3498db'},
    {'y': 3.5, 'stage': 'Discovery', 'topics': ['Supervised Learning', 'Neural Networks'], 'color': '#2ecc71'},
    {'y': 2.5, 'stage': 'Generation', 'topics': ['Generative AI', 'Topic Modeling'], 'color': '#2ecc71'},
    {'y': 1.5, 'stage': 'Peak', 'topics': ['NLP & Sentiment'], 'color': '#f1c40f'},
    {'y': 0.5, 'stage': 'Extraction', 'topics': ['Clustering'], 'color': '#e67e22'},
    {'y': -0.5, 'stage': 'Patterns', 'topics': ['Classification'], 'color': '#e67e22'},
    {'y': -1.5, 'stage': 'Insights', 'topics': ['Validation', 'A/B Testing'], 'color': '#e74c3c'},
    {'y': -2.5, 'stage': 'Strategy', 'topics': ['Responsible AI', 'Structured Output', 'Finance'], 'color': '#c0392b'},
]

# Draw simplified diamond outline
diamond_x = [0, 3, 5, 3, 0, -3, -5, -3, 0]
diamond_y = [5.5, 3.5, 1.5, -0.5, -2.5, -0.5, 1.5, 3.5, 5.5]
ax.plot(diamond_x, diamond_y, 'k-', linewidth=2, alpha=0.3)
ax.fill(diamond_x, diamond_y, color='gray', alpha=0.05)

# Plot stages and topics
for stage in stages:
    # Stage label on left
    ax.text(-6, stage['y'], stage['stage'], fontsize=9, fontweight='bold',
            color=stage['color'], ha='right', va='center')

    # Draw line
    ax.axhline(y=stage['y'], xmin=0.25, xmax=0.75, color=stage['color'], alpha=0.3, linewidth=1)

    # Topics on right
    topic_text = ', '.join(stage['topics'])
    ax.text(6, stage['y'], topic_text, fontsize=8, color='gray', ha='left', va='center')

    # Connecting dots
    ax.scatter([0], [stage['y']], s=80, c=[stage['color']], zorder=5, edgecolors='white', linewidth=1)

# Phase labels
ax.text(0, 6.2, 'DIVERGENT', fontsize=10, fontweight='bold', color='#9467bd', ha='center')
ax.text(0, -3.2, 'CONVERGENT', fontsize=10, fontweight='bold', color='#c0392b', ha='center')

ax.set_xlim(-8, 8)
ax.set_ylim(-4, 7)
ax.axis('off')
ax.set_title('14 Course Topics Mapped to Innovation Diamond', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("12_topic_mapping created")
