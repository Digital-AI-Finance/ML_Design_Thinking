"""Efficient Frontier - Portfolio optimization with ESG constraint"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

np.random.seed(42)
fig, ax = plt.subplots(figsize=(10, 6))

# Generate efficient frontier curve
risk = np.linspace(0.05, 0.35, 100)
# Traditional frontier
return_trad = 0.02 + 0.5 * risk - 0.3 * risk**2
# ESG-constrained frontier (slightly lower)
return_esg = 0.018 + 0.48 * risk - 0.28 * risk**2

# Plot frontiers
ax.plot(risk, return_trad, 'b-', linewidth=2.5, label='Traditional Frontier', alpha=0.7)
ax.plot(risk, return_esg, 'g-', linewidth=2.5, label='ESG Frontier', alpha=0.7)

# Fill between to show ESG cost
ax.fill_between(risk, return_esg, return_trad, alpha=0.1, color='gray')

# Scatter random portfolios
n_portfolios = 200
port_risk = np.random.uniform(0.08, 0.32, n_portfolios)
port_return = 0.015 + 0.4 * port_risk - 0.25 * port_risk**2 + np.random.normal(0, 0.015, n_portfolios)
ax.scatter(port_risk, port_return, s=20, c='gray', alpha=0.3, label='Individual Portfolios')

# Mark optimal portfolios
ax.scatter([0.15], [return_esg[30]], s=200, c='#2ecc71', marker='*',
          edgecolors='white', linewidth=2, zorder=5, label='ESG Optimal')
ax.scatter([0.18], [return_trad[45]], s=200, c='#3498db', marker='*',
          edgecolors='white', linewidth=2, zorder=5, label='Traditional Optimal')

# Annotations
ax.annotate('ESG Optimal\n(Lower risk, sustainable)', xy=(0.15, return_esg[30]),
           xytext=(0.08, 0.12), fontsize=8,
           arrowprops=dict(arrowstyle='->', color='#2ecc71', alpha=0.7))
ax.annotate('Traditional Optimal\n(Higher return)', xy=(0.18, return_trad[45]),
           xytext=(0.22, 0.16), fontsize=8,
           arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.7))

# Add ESG cost annotation
ax.annotate('ESG "Cost"', xy=(0.25, (return_trad[65] + return_esg[65])/2),
           fontsize=9, ha='center', color='gray', style='italic')

ax.set_xlabel('Risk (Volatility)', fontsize=11)
ax.set_ylabel('Expected Return', fontsize=11)
ax.set_title('Efficient Frontier: Traditional vs ESG-Constrained', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(0.05, 0.35)
ax.set_ylim(0.02, 0.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Format axis as percentages
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("18_efficient_frontier created")
