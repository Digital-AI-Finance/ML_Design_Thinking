"""Convergent Process - Focusing from 5000 to 5"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

colors = {
    'peak': '#f1c40f',
    'filter': '#e67e22',
    'refine': '#e74c3c',
    'strategy': '#c0392b'
}

fig, ax = plt.subplots(figsize=(10, 6))

# Stages data
stages = ['Peak\n(5000)', 'Extraction\n(2000)', 'Patterns\n(500)', 'Insights\n(50)', 'Strategy\n(5)']
counts = [5000, 2000, 500, 50, 5]
stage_colors = [colors['peak'], colors['filter'], colors['filter'], colors['refine'], colors['strategy']]

x = np.arange(len(stages))

# Create converging visual (funnel)
for i, (stage, count, color) in enumerate(zip(stages, counts, stage_colors)):
    height = np.log10(count + 1) * 1.5
    width = 0.3 + (count / 5000) * 0.5

    ax.bar(i, height, width=width, color=color, alpha=0.7, edgecolor=color, linewidth=2)
    ax.text(i, height + 0.2, f'{count:,}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=color)

# Connecting arrows showing convergence
for i in range(len(stages) - 1):
    ax.annotate('', xy=(i + 0.6, 2), xytext=(i + 0.4, 2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel('Scale (log)', fontsize=10)
ax.set_title('Convergent Phase: Focusing to Strategy', fontsize=12, fontweight='bold', color=colors['strategy'])
ax.set_ylim(0, 7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ML technique annotations
techniques = ['NLP', 'Clustering', 'Classification', 'Validation', 'Responsible AI']
for i, tech in enumerate(techniques):
    ax.text(i, -0.8, tech, ha='center', va='top', fontsize=8, color='gray', style='italic')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("03_convergent_process created")
