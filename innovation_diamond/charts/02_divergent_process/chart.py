"""Divergent Process - Expansion from 1 to 5000"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

colors = {
    'challenge': '#9467bd',
    'explore': '#3498db',
    'generate': '#2ecc71',
    'peak': '#f1c40f'
}

fig, ax = plt.subplots(figsize=(10, 6))

# Stages data
stages = ['Challenge\n(1)', 'Exploration\n(10)', 'Discovery\n(100)', 'Generation\n(1000)', 'Peak\n(5000)']
counts = [1, 10, 100, 1000, 5000]
stage_colors = [colors['challenge'], colors['explore'], colors['generate'], colors['generate'], colors['peak']]

x = np.arange(len(stages))

# Create expanding visual
for i, (stage, count, color) in enumerate(zip(stages, counts, stage_colors)):
    # Bar height proportional to log scale for visibility
    height = np.log10(count + 1) * 1.5
    width = 0.3 + (count / 5000) * 0.5

    ax.bar(i, height, width=width, color=color, alpha=0.7, edgecolor=color, linewidth=2)

    # Add count label
    ax.text(i, height + 0.2, f'{count:,}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=color)

# Connecting arrows showing expansion
for i in range(len(stages) - 1):
    ax.annotate('', xy=(i + 0.6, 2), xytext=(i + 0.4, 2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel('Scale (log)', fontsize=10)
ax.set_title('Divergent Phase: Expanding Possibilities', fontsize=12, fontweight='bold', color=colors['challenge'])
ax.set_ylim(0, 7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add ML technique annotations
techniques = ['ML Foundations', 'Unsupervised', 'Supervised', 'Generative AI', 'NLP']
for i, tech in enumerate(techniques):
    ax.text(i, -0.8, tech, ha='center', va='top', fontsize=8, color='gray', style='italic')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("02_divergent_process created")
