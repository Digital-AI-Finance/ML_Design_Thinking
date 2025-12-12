"""ESG Dimensions Radar - 10 ESG dimensions for exploration stage"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

# ESG dimensions
categories = ['Carbon\nEmissions', 'Water\nUsage', 'Labor\nPractices', 'Board\nDiversity',
              'Supply Chain\nEthics', 'Community\nImpact', 'Waste\nManagement',
              'Energy\nEfficiency', 'Data\nPrivacy', 'Anti-\nCorruption']

# Simulated company scores (normalized 0-1)
np.random.seed(42)
company_a = np.random.uniform(0.5, 0.9, len(categories))
company_b = np.random.uniform(0.3, 0.7, len(categories))

# Number of dimensions
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

company_a = np.append(company_a, company_a[0])
company_b = np.append(company_b, company_b[0])

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))

# Plot data
ax.plot(angles, company_a, 'o-', linewidth=2, label='ESG Leader', color='#2ecc71')
ax.fill(angles, company_a, alpha=0.25, color='#2ecc71')

ax.plot(angles, company_b, 's-', linewidth=2, label='Industry Average', color='#e74c3c')
ax.fill(angles, company_b, alpha=0.15, color='#e74c3c')

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=8)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=8)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.set_title('ESG Performance: 10 Dimensions', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("04_esg_dimensions created")
