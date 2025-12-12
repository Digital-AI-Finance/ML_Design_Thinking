"""Pitfalls Summary - Common pitfalls at each stage"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

fig, ax = plt.subplots(figsize=(10, 6))

# Pitfalls by stage
pitfalls = [
    {'stage': 'Challenge', 'pitfall': 'Too broad or narrow scope', 'y': 5, 'color': '#9467bd'},
    {'stage': 'Exploration', 'pitfall': 'Ignoring non-obvious dimensions', 'y': 4.3, 'color': '#3498db'},
    {'stage': 'Discovery', 'pitfall': 'Features without domain knowledge', 'y': 3.6, 'color': '#2ecc71'},
    {'stage': 'Generation', 'pitfall': 'Quantity over quality', 'y': 2.9, 'color': '#2ecc71'},
    {'stage': 'Peak', 'pitfall': 'Analysis paralysis', 'y': 2.2, 'color': '#f1c40f'},
    {'stage': 'Extraction', 'pitfall': 'Forcing non-existent clusters', 'y': 1.5, 'color': '#e67e22'},
    {'stage': 'Patterns', 'pitfall': 'Over-relying on historical data', 'y': 0.8, 'color': '#e67e22'},
    {'stage': 'Insights', 'pitfall': 'Overfitting / lookahead bias', 'y': 0.1, 'color': '#e74c3c'},
    {'stage': 'Strategy', 'pitfall': 'Black-box decisions', 'y': -0.6, 'color': '#c0392b'},
]

for p in pitfalls:
    # Stage box
    rect = mpatches.FancyBboxPatch((0.5, p['y'] - 0.25), 2, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=p['color'], alpha=0.8)
    ax.add_patch(rect)
    ax.text(1.5, p['y'], p['stage'], fontsize=8, fontweight='bold',
            ha='center', va='center', color='white')

    # Arrow
    ax.annotate('', xy=(3, p['y']), xytext=(2.5, p['y']),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Pitfall text
    ax.text(3.2, p['y'], p['pitfall'], fontsize=9, ha='left', va='center', color='#2c3e50')

    # Warning icon
    ax.text(9, p['y'], '!', fontsize=12, fontweight='bold', ha='center', va='center',
            color='#e74c3c', bbox=dict(boxstyle='circle,pad=0.2', facecolor='#ffeaa7', edgecolor='#e74c3c'))

ax.set_xlim(0, 10)
ax.set_ylim(-1.2, 5.8)
ax.axis('off')
ax.set_title('Common Pitfalls at Each Innovation Stage', fontsize=12, fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("14_pitfalls_summary created")
