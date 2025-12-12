"""Innovation Diamond Overview - Main visualization showing 1->5000->5 journey"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch
import matplotlib.patheffects as path_effects
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

# Diamond stage colors
colors = {
    'challenge': '#9467bd',    # Purple
    'explore': '#3498db',      # Blue
    'generate': '#2ecc71',     # Green
    'peak': '#f1c40f',         # Yellow
    'filter': '#e67e22',       # Orange
    'refine': '#e74c3c',       # Red
    'strategy': '#c0392b',     # Dark Red
    'gray': '#7f7f7f'
}

np.random.seed(42)
fig, ax = plt.subplots(figsize=(10, 6))

# Define diamond stages
all_stages = [
    {'name': 'Challenge', 'width': 1.2, 'y': 5.5, 'count': 1, 'color': colors['challenge']},
    {'name': 'Exploration', 'width': 2.5, 'y': 4.5, 'count': 10, 'color': colors['explore']},
    {'name': 'Discovery', 'width': 4, 'y': 3.5, 'count': 100, 'color': colors['generate']},
    {'name': 'Generation', 'width': 6, 'y': 2.5, 'count': 1000, 'color': colors['generate']},
    {'name': 'Peak', 'width': 8, 'y': 1.5, 'count': 5000, 'color': colors['peak']},
    {'name': 'Extraction', 'width': 6, 'y': 0.5, 'count': 2000, 'color': colors['filter']},
    {'name': 'Patterns', 'width': 4, 'y': -0.5, 'count': 500, 'color': colors['filter']},
    {'name': 'Insights', 'width': 2.5, 'y': -1.5, 'count': 50, 'color': colors['refine']},
    {'name': 'Strategy', 'width': 1.2, 'y': -2.5, 'count': 5, 'color': colors['strategy']}
]

# Draw diamond shape
for i in range(len(all_stages) - 1):
    curr = all_stages[i]
    next_s = all_stages[i + 1]

    trapezoid = Polygon([
        (-curr['width']/2, curr['y']),
        (curr['width']/2, curr['y']),
        (next_s['width']/2, next_s['y']),
        (-next_s['width']/2, next_s['y'])
    ], facecolor=curr['color'], alpha=0.2, edgecolor=curr['color'], linewidth=1.5)
    ax.add_patch(trapezoid)

# Add stage labels
for stage in all_stages:
    ax.text(-5.5, stage['y'], stage['name'], fontsize=9, fontweight='bold',
            color=stage['color'], ha='right', va='center')

    count_text = f"{stage['count']:,}" if stage['count'] > 1 else "1"
    ax.text(5.5, stage['y'], count_text, fontsize=9, color=stage['color'],
            ha='left', va='center', style='italic')

# Phase labels
ax.text(0, 6.2, 'DIVERGENT', fontsize=11, fontweight='bold', color=colors['challenge'], ha='center')
ax.text(0, -3.2, 'CONVERGENT', fontsize=11, fontweight='bold', color=colors['strategy'], ha='center')

# Peak highlight
ax.annotate('5,000 IDEAS', xy=(0, 1.5), fontsize=12, fontweight='bold',
            color=colors['peak'], ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['peak'], alpha=0.9))

ax.set_xlim(-7, 7)
ax.set_ylim(-4, 7)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("01_diamond_overview created")
