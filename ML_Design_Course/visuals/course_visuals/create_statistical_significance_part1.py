"""
Create statistical significance visualization Part 1 for Week 10
P-values and Confidence Intervals
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with 2 subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: P-value explanation
ax1.set_xlim(-4, 4)
ax1.set_ylim(0, 0.5)

# Draw normal distribution
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
ax1.plot(x, y, 'b-', linewidth=3)
ax1.fill_between(x, 0, y, where=(x < -1.96) | (x > 1.96), 
                 color='red', alpha=0.3, label='Rejection region (p < 0.05)')
ax1.fill_between(x, 0, y, where=(x >= -1.96) & (x <= 1.96),
                 color='green', alpha=0.3, label='Acceptance region')

ax1.axvline(x=2.5, color='orange', linewidth=4, label='Your result')
ax1.set_title('P-Value: Probability of Chance', fontsize=18, fontweight='bold')
ax1.set_xlabel('Difference from baseline', fontsize=14)
ax1.set_ylabel('Probability', fontsize=14)
ax1.legend(loc='upper right', fontsize=13)

# Add explanation
ax1.text(0, 0.45, 'If p < 0.05:\nResult is significant\n(Not due to chance)',
        ha='center', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))

# Add specific p-value annotation
ax1.annotate('p = 0.012\n(significant)', xy=(2.5, 0.01), xytext=(3.2, 0.15),
            fontsize=13, ha='center',
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white'))

# Right: Confidence intervals
ax2.set_xlim(0, 4)
ax2.set_ylim(45, 60)

# Sample data for confidence intervals
groups = ['Control', 'Test A', 'Test B']
means = [50, 55, 52]
ci_lower = [48, 53, 49]
ci_upper = [52, 57, 55]
colors_ci = ['blue', 'green', 'orange']

for i, (group, mean, lower, upper, color) in enumerate(zip(groups, means, ci_lower, ci_upper, colors_ci)):
    x_pos = i + 1
    
    # Draw confidence interval
    ax2.plot([x_pos, x_pos], [lower, upper], color=color, linewidth=4)
    ax2.plot([x_pos - 0.1, x_pos + 0.1], [lower, lower], color=color, linewidth=3)
    ax2.plot([x_pos - 0.1, x_pos + 0.1], [upper, upper], color=color, linewidth=3)
    
    # Draw mean point
    ax2.scatter(x_pos, mean, s=200, color=color, zorder=3, edgecolors='black', linewidth=2)
    
    # Label
    ax2.text(x_pos, 45.5, group, ha='center', fontsize=14, fontweight='bold')
    ax2.text(x_pos, mean + 0.8, f'{mean}%', ha='center', fontsize=13, fontweight='bold')
    
    # Add CI range label
    ax2.text(x_pos + 0.25, (lower + upper) / 2, f'95% CI', fontsize=11, style='italic')

ax2.set_title('95% Confidence Intervals', fontsize=18, fontweight='bold')
ax2.set_ylabel('Conversion Rate (%)', fontsize=14)
ax2.set_xlim(0.5, 3.5)
ax2.set_ylim(45, 60)
ax2.grid(True, alpha=0.3)

# Add interpretation boxes
ax2.text(1.75, 58, 'Overlapping CI = Not significant',
        ha='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffcccc', alpha=0.9))

ax2.text(2.75, 58, 'Separated CI = Significant',
        ha='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ccffcc', alpha=0.9))

# Draw comparison arrows
ax2.annotate('', xy=(1, 52), xytext=(3, 49),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2, alpha=0.5))
ax2.text(2, 50, 'Overlap', ha='center', fontsize=12, color='red', style='italic')

ax2.annotate('', xy=(1, 52), xytext=(2, 53),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2, alpha=0.5))
ax2.text(1.5, 52.5, 'No overlap', ha='center', fontsize=12, color='green', style='italic')

# Overall title
fig.suptitle('Statistical Significance: Understanding P-Values and Confidence', 
            fontsize=20, fontweight='bold', y=0.98)

plt.tight_layout()

# Save the figure
plt.savefig('statistical_significance_part1.pdf', dpi=300, bbox_inches='tight')
plt.savefig('statistical_significance_part1.png', dpi=150, bbox_inches='tight')

print("Statistical significance Part 1 visualization created successfully!")