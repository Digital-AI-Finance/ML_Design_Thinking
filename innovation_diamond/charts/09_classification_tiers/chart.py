"""Classification Tiers - Decision tree classification of ESG companies"""
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

# Create decision tree visualization
# Root node
root = mpatches.FancyBboxPatch((4, 5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#3498db', edgecolor='#2980b9', linewidth=2)
ax.add_patch(root)
ax.text(5, 5.4, 'ESG Score\n> 70?', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Level 2 nodes
node_l1 = mpatches.FancyBboxPatch((1.5, 3), 2, 0.8, boxstyle="round,pad=0.1",
                                   facecolor='#9b59b6', edgecolor='#8e44ad', linewidth=2)
ax.add_patch(node_l1)
ax.text(2.5, 3.4, 'Momentum\n> 0?', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

node_r1 = mpatches.FancyBboxPatch((6.5, 3), 2, 0.8, boxstyle="round,pad=0.1",
                                   facecolor='#9b59b6', edgecolor='#8e44ad', linewidth=2)
ax.add_patch(node_r1)
ax.text(7.5, 3.4, 'Risk\n< 0.3?', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Leaf nodes (Tiers)
colors_tier = {'Tier 1': '#2ecc71', 'Tier 2': '#f1c40f', 'Tier 3': '#e74c3c'}
tier_positions = [
    ('Tier 3', 0.3, 1, 'Laggards\n(Material Risks)'),
    ('Tier 2', 2.7, 1, 'Improvers\n(Positive Trend)'),
    ('Tier 2', 5.3, 1, 'Improvers\n(Moderate Risk)'),
    ('Tier 1', 7.7, 1, 'Leaders\n(Verified Impact)')
]

for tier, x, y, label in tier_positions:
    node = mpatches.FancyBboxPatch((x, y), 2, 0.8, boxstyle="round,pad=0.1",
                                    facecolor=colors_tier[tier], edgecolor='gray', linewidth=1)
    ax.add_patch(node)
    ax.text(x + 1, y + 0.4, f'{tier}\n{label}', ha='center', va='center',
            fontsize=8, color='white' if tier != 'Tier 2' else 'black', fontweight='bold')

# Draw edges
edges = [
    ((5, 5), (2.5, 3.8), 'No'),
    ((5, 5), (7.5, 3.8), 'Yes'),
    ((2.5, 3), (1.3, 1.8), 'No'),
    ((2.5, 3), (3.7, 1.8), 'Yes'),
    ((7.5, 3), (6.3, 1.8), 'No'),
    ((7.5, 3), (8.7, 1.8), 'Yes')
]

for (x1, y1), (x2, y2), label in edges:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mid_x + 0.2, mid_y, label, fontsize=8, color='gray')

ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0, 6.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Classification: ESG Sustainability Tiers', fontsize=12, fontweight='bold', color='#e67e22')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("09_classification_tiers created")
