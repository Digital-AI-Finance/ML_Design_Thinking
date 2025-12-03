"""
Create radar chart comparing different prototype approaches across multiple dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define evaluation dimensions
dimensions = [
    'Accuracy',
    'Fairness', 
    'Robustness',
    'Novelty',
    'Usability',
    'Scalability',
    'Cost Efficiency',
    'Safety'
]

# Number of dimensions
N = len(dimensions)

# Create data for three different prototype approaches
# Scores from 0-100
prototypes = {
    'Baseline Model': [75, 60, 70, 30, 85, 90, 95, 80],
    'AI-Enhanced V1': [85, 75, 80, 70, 70, 75, 60, 85],
    'AI-Enhanced V2': [90, 85, 85, 90, 65, 70, 55, 90]
}

# Colors for each prototype
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
alphas = [0.25, 0.25, 0.25]

# Compute angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                subplot_kw=dict(projection='polar'))

# Left plot: All prototypes comparison
ax = ax1
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axis per variable and add labels
plt.xticks(angles[:-1], dimensions, size=14)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], 
          color="grey", size=8)
plt.ylim(0, 100)

# Plot data for each prototype
for idx, (name, values) in enumerate(prototypes.items()):
    values += values[:1]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[idx])
    ax.fill(angles, values, alpha=alphas[idx], color=colors[idx])

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13)
ax.set_title('Multi-Dimensional Prototype Comparison', size=16, fontweight='bold', pad=20)

# Right plot: Innovation trade-offs visualization
ax = ax2
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Focus on key trade-off dimensions
trade_off_dims = ['Accuracy', 'Novelty', 'Usability', 'Cost Efficiency']
trade_off_angles = [n / float(len(trade_off_dims)) * 2 * pi for n in range(len(trade_off_dims))]
trade_off_angles += trade_off_angles[:1]

plt.sca(ax)
plt.xticks(trade_off_angles[:-1], trade_off_dims, size=14)
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], 
          color="grey", size=8)
plt.ylim(0, 100)

# Extract trade-off values
for name, full_values in prototypes.items():
    indices = [dimensions.index(d) for d in trade_off_dims]
    values = [full_values[i] for i in indices]
    values += values[:1]
    
    idx = list(prototypes.keys()).index(name)
    ax.plot(trade_off_angles, values, 'o-', linewidth=2.5, 
           label=name, color=colors[idx], markersize=8)
    ax.fill(trade_off_angles, values, alpha=0.3, color=colors[idx])

ax.set_title('Key Innovation Trade-offs', size=16, fontweight='bold', pad=20)

# Add annotations for trade-offs
annotations = [
    (0.5, -0.15, "Higher accuracy often reduces novelty", ax1),
    (0.5, -0.20, "More features can hurt usability", ax1),
    (0.5, -0.15, "Innovation vs. Practicality", ax2),
    (0.5, -0.20, "Choose based on user priorities", ax2)
]

for x, y, text, axis in annotations:
    axis.text(x, y, text, transform=axis.transAxes,
             ha='center', fontsize=13, style='italic', color='#555')

# Overall title
fig.suptitle('Prototype Evaluation: No Single Winner', 
            fontsize=20, fontweight='bold', y=1.05)

# Add insights box
insights_text = (
    "Key Insights:\n"
    "• Baseline: High usability, low innovation\n"
    "• V1: Balanced approach, good fairness\n"
    "• V2: Maximum innovation, usability trade-off\n"
    "\n"
    "Decision: Match prototype to user segment needs"
)
fig.text(0.5, 0.02, insights_text, ha='center', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()

# Save the figure
plt.savefig('radar_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('radar_comparison.png', dpi=150, bbox_inches='tight')

print("Radar comparison chart created successfully!")
print(f"Compared {len(prototypes)} prototypes across {N} dimensions")