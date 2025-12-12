"""Complete Journey - Full ESG case study visualization"""
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

# Journey stages as a flow
stages = [
    {'name': '1\nChallenge', 'x': 0.5, 'y': 3, 'color': '#9467bd', 'esg': 'ESG\nPortfolio?'},
    {'name': '10\nDimensions', 'x': 2, 'y': 4, 'color': '#3498db', 'esg': 'E/S/G\nMetrics'},
    {'name': '100\nFeatures', 'x': 3.5, 'y': 4.5, 'color': '#2ecc71', 'esg': 'NLP\nScores'},
    {'name': '1000\nIdeas', 'x': 5, 'y': 4.8, 'color': '#2ecc71', 'esg': 'LLM\nTheses'},
    {'name': '5000\nPeak', 'x': 6.5, 'y': 5, 'color': '#f1c40f', 'esg': 'All\nOptions'},
    {'name': '2000\nFiltered', 'x': 8, 'y': 4.5, 'color': '#e67e22', 'esg': '8\nClusters'},
    {'name': '500\nPatterns', 'x': 9, 'y': 3.5, 'color': '#e67e22', 'esg': '3\nTiers'},
    {'name': '50\nInsights', 'x': 9.5, 'y': 2.5, 'color': '#e74c3c', 'esg': 'CV\nTests'},
    {'name': '5\nStrategies', 'x': 9, 'y': 1.5, 'color': '#c0392b', 'esg': 'Final\nFunds'},
]

# Draw journey path
xs = [s['x'] for s in stages]
ys = [s['y'] for s in stages]
ax.plot(xs, ys, 'k-', linewidth=2, alpha=0.3, zorder=1)

# Draw stages
for i, s in enumerate(stages):
    circle = mpatches.Circle((s['x'], s['y']), 0.4, facecolor=s['color'], alpha=0.9,
                              edgecolor='white', linewidth=2, zorder=2)
    ax.add_patch(circle)

    # Stage count
    ax.text(s['x'], s['y'], s['name'], fontsize=7, ha='center', va='center',
            color='white', fontweight='bold', zorder=3)

    # ESG example below
    ax.text(s['x'], s['y'] - 0.7, s['esg'], fontsize=6, ha='center', va='top',
            color='gray', style='italic')

# Arrows showing direction
for i in range(len(stages) - 1):
    dx = stages[i+1]['x'] - stages[i]['x']
    dy = stages[i+1]['y'] - stages[i]['y']
    ax.annotate('', xy=(stages[i+1]['x'] - 0.3*dx/np.sqrt(dx**2+dy**2),
                        stages[i+1]['y'] - 0.3*dy/np.sqrt(dx**2+dy**2)),
               xytext=(stages[i]['x'] + 0.3*dx/np.sqrt(dx**2+dy**2),
                       stages[i]['y'] + 0.3*dy/np.sqrt(dx**2+dy**2)),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

# Phase labels
ax.text(3.5, 5.5, 'DIVERGENT PHASE', fontsize=10, fontweight='bold', color='#2ecc71', ha='center')
ax.text(8.5, 5.5, 'CONVERGENT PHASE', fontsize=10, fontweight='bold', color='#e74c3c', ha='center')

# Central annotation
ax.annotate('', xy=(7.5, 4.8), xytext=(6.5, 4.8),
           arrowprops=dict(arrowstyle='-[', color='#f1c40f', lw=2))

ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0.5, 6)
ax.axis('off')
ax.set_title('Complete ESG Innovation Journey: 1 Challenge to 5 Strategies',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("15_journey_complete created")
