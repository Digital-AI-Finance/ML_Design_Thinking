"""Final 5 Strategies - The converged innovation outcomes"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

fig, ax = plt.subplots(figsize=(10, 6))

# Five final strategies
strategies = [
    {'name': 'Climate Leaders\nFund', 'color': '#2ecc71', 'x': 1, 'desc': 'Carbon-negative companies'},
    {'name': 'Social Impact\nBlend', 'color': '#3498db', 'x': 3, 'desc': 'Labor & community focus'},
    {'name': 'Governance\nExcellence', 'color': '#9b59b6', 'x': 5, 'desc': 'Board & ethics priority'},
    {'name': 'ESG Momentum\nStrategy', 'color': '#f1c40f', 'x': 7, 'desc': 'Rising ESG scores'},
    {'name': 'Sustainable\nDividend Growth', 'color': '#e74c3c', 'x': 9, 'desc': 'Income + sustainability'}
]

for i, strat in enumerate(strategies):
    # Main strategy box
    rect = mpatches.FancyBboxPatch((strat['x'] - 0.8, 2), 1.6, 2.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor=strat['color'], alpha=0.8,
                                    edgecolor='gray', linewidth=2)
    ax.add_patch(rect)

    # Strategy number
    ax.text(strat['x'], 4.2, f'{i+1}', fontsize=14, fontweight='bold',
            ha='center', va='center', color='white',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor=strat['color'], edgecolor='white'))

    # Strategy name
    ax.text(strat['x'], 3.2, strat['name'], fontsize=9, fontweight='bold',
            ha='center', va='center', color='white')

    # Description
    ax.text(strat['x'], 2.3, strat['desc'], fontsize=7, ha='center', va='center',
            color='white', style='italic', wrap=True)

# Title and annotation
ax.set_title('Final Output: 5 Actionable ESG Strategies', fontsize=12, fontweight='bold', color='#c0392b')

# Arrow from 5000 to 5
ax.annotate('', xy=(5, 5.8), xytext=(5, 6.5),
           arrowprops=dict(arrowstyle='->', color='#c0392b', lw=3))
ax.text(5, 6.7, '5,000 ideas', fontsize=10, ha='center', va='bottom', color='gray')
ax.text(5, 5.6, '5 strategies', fontsize=11, ha='center', va='top', color='#c0392b', fontweight='bold')

ax.set_xlim(-0.5, 10.5)
ax.set_ylim(1, 7.5)
ax.axis('off')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("11_final_strategies created")
