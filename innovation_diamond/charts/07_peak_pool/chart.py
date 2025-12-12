"""Peak Pool Visualization - 5000 raw ideas at maximum expansion"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

np.random.seed(42)
fig, ax = plt.subplots(figsize=(10, 6))

# Generate 5000 points representing raw ideas
n_points = 5000

# Create multiple clusters representing different idea categories
clusters = [
    {'center': (0, 0), 'std': 2, 'n': 2000, 'color': '#f1c40f'},
    {'center': (-3, 2), 'std': 1.2, 'n': 800, 'color': '#2ecc71'},
    {'center': (3, 2), 'std': 1.2, 'n': 800, 'color': '#3498db'},
    {'center': (-3, -2), 'std': 1.2, 'n': 700, 'color': '#e74c3c'},
    {'center': (3, -2), 'std': 1.2, 'n': 700, 'color': '#9467bd'},
]

for cluster in clusters:
    x = np.random.normal(cluster['center'][0], cluster['std'], cluster['n'])
    y = np.random.normal(cluster['center'][1], cluster['std'], cluster['n'])
    ax.scatter(x, y, s=5, c=cluster['color'], alpha=0.4, edgecolors='none')

# Central highlight
ax.annotate('5,000\nIDEAS', xy=(0, 0), fontsize=14, fontweight='bold',
            ha='center', va='center', color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#f1c40f', linewidth=2))

# Category labels
labels = [('Company\nTargets', -3, 2), ('Weighting\nSchemes', 3, 2),
          ('Sector\nAllocations', -3, -2), ('Risk\nParameters', 3, -2)]
for label, lx, ly in labels:
    ax.text(lx, ly + 1.8, label, fontsize=8, ha='center', color='gray', fontweight='bold')

ax.set_xlim(-6, 6)
ax.set_ylim(-5, 5)
ax.set_xlabel('Idea Space Dimension 1', fontsize=10)
ax.set_ylabel('Idea Space Dimension 2', fontsize=10)
ax.set_title('Peak: Raw Ideas Pool (Maximum Expansion)', fontsize=12, fontweight='bold', color='#f1c40f')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("07_peak_pool created")
