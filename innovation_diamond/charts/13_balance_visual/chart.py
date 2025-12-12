"""Balance Visual - Divergent vs Convergent balance scale"""
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

# Draw balance scale
# Fulcrum
fulcrum = mpatches.RegularPolygon((5, 2), numVertices=3, radius=0.5,
                                   facecolor='#34495e', edgecolor='#2c3e50')
ax.add_patch(fulcrum)

# Beam (balanced)
ax.plot([1, 9], [4, 4], 'k-', linewidth=4, solid_capstyle='round')

# Left pan - Divergent
left_pan = mpatches.Circle((2, 3.5), 1.2, facecolor='#2ecc71', alpha=0.7, edgecolor='#27ae60', linewidth=2)
ax.add_patch(left_pan)
ax.text(2, 3.5, 'DIVERGENT', fontsize=9, fontweight='bold', ha='center', va='center', color='white')

# Right pan - Convergent
right_pan = mpatches.Circle((8, 3.5), 1.2, facecolor='#e74c3c', alpha=0.7, edgecolor='#c0392b', linewidth=2)
ax.add_patch(right_pan)
ax.text(8, 3.5, 'CONVERGENT', fontsize=9, fontweight='bold', ha='center', va='center', color='white')

# Support strings
ax.plot([2, 2], [4, 3.5 + 1.2], 'k-', linewidth=2)
ax.plot([8, 8], [4, 3.5 + 1.2], 'k-', linewidth=2)

# Labels below pans
divergent_items = ['Explore', 'Generate', 'Expand', 'Create']
convergent_items = ['Filter', 'Classify', 'Validate', 'Decide']

for i, item in enumerate(divergent_items):
    ax.text(2, 1.8 - i * 0.4, item, fontsize=8, ha='center', color='#2ecc71')

for i, item in enumerate(convergent_items):
    ax.text(8, 1.8 - i * 0.4, item, fontsize=8, ha='center', color='#e74c3c')

# Warning boxes
# Too much divergent
ax.text(0.5, 5.5, 'Too Much Divergent:', fontsize=9, fontweight='bold', color='#e67e22')
ax.text(0.5, 5.1, 'Analysis paralysis', fontsize=8, color='gray')
ax.text(0.5, 4.8, 'No action', fontsize=8, color='gray')

# Too much convergent
ax.text(9.5, 5.5, 'Too Much Convergent:', fontsize=9, fontweight='bold', color='#e67e22', ha='right')
ax.text(9.5, 5.1, 'Missed opportunities', fontsize=8, color='gray', ha='right')
ax.text(9.5, 4.8, 'Local maxima', fontsize=8, color='gray', ha='right')

# Central message
ax.text(5, 5.5, 'BALANCE', fontsize=14, fontweight='bold', ha='center', color='#2c3e50')
ax.text(5, 5.1, 'is the key to innovation', fontsize=10, ha='center', color='gray', style='italic')

ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("13_balance_visual created")
