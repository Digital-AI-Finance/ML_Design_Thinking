"""
Create statistical significance visualization for Week 10
Explains p-values, confidence intervals, and errors simply
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as patches

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with 4 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# Top left: P-value explanation
ax1.set_xlim(-4, 4)
ax1.set_ylim(0, 0.5)

# Draw normal distribution
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
ax1.plot(x, y, 'b-', linewidth=2)
ax1.fill_between(x, 0, y, where=(x < -1.96) | (x > 1.96), 
                 color='red', alpha=0.3, label='Rejection region (p < 0.05)')
ax1.fill_between(x, 0, y, where=(x >= -1.96) & (x <= 1.96),
                 color='green', alpha=0.3, label='Acceptance region')

ax1.axvline(x=2.5, color='orange', linewidth=3, label='Your result')
ax1.set_title('P-Value: Probability of Chance', fontsize=16, fontweight='bold')
ax1.set_xlabel('Difference from baseline', fontsize=13)
ax1.set_ylabel('Probability', fontsize=13)
ax1.legend(loc='upper right', fontsize=11)

# Add explanation
ax1.text(0, 0.45, 'If p < 0.05:\nResult is significant\n(Not due to chance)',
        ha='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

# Top right: Confidence intervals
ax2.set_xlim(0, 4)
ax2.set_ylim(0, 4)

# Sample data for confidence intervals
groups = ['Control', 'Test A', 'Test B']
means = [50, 55, 52]
ci_lower = [48, 53, 49]
ci_upper = [52, 57, 55]
colors_ci = ['blue', 'green', 'orange']

for i, (group, mean, lower, upper, color) in enumerate(zip(groups, means, ci_lower, ci_upper, colors_ci)):
    x_pos = i + 1
    
    # Draw confidence interval
    ax2.plot([x_pos, x_pos], [lower, upper], color=color, linewidth=3)
    ax2.plot([x_pos - 0.1, x_pos + 0.1], [lower, lower], color=color, linewidth=2)
    ax2.plot([x_pos - 0.1, x_pos + 0.1], [upper, upper], color=color, linewidth=2)
    
    # Draw mean point
    ax2.scatter(x_pos, mean, s=100, color=color, zorder=3)
    
    # Label
    ax2.text(x_pos, 47, group, ha='center', fontsize=12, fontweight='bold')
    ax2.text(x_pos, mean + 1, f'{mean}%', ha='center', fontsize=11)

ax2.set_title('95% Confidence Intervals', fontsize=16, fontweight='bold')
ax2.set_ylabel('Conversion Rate (%)', fontsize=13)
ax2.set_xlim(0.5, 3.5)
ax2.set_ylim(46, 60)
ax2.grid(True, alpha=0.3)

# Add interpretation
ax2.text(2, 58, 'Overlapping = Not significant\nSeparated = Significant',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.9))

# Bottom left: Sample size impact
ax3.set_xlim(0, 1000)
ax3.set_ylim(0, 10)

sample_sizes = [50, 100, 200, 500, 1000]
ci_widths = [8, 5.6, 4, 2.5, 1.8]

ax3.plot(sample_sizes, ci_widths, 'b-', linewidth=2, marker='o', markersize=8)
ax3.fill_between(sample_sizes, 0, ci_widths, alpha=0.3)

ax3.set_title('Sample Size Impact on Precision', fontsize=16, fontweight='bold')
ax3.set_xlabel('Sample Size', fontsize=13)
ax3.set_ylabel('Confidence Interval Width', fontsize=13)
ax3.grid(True, alpha=0.3)

# Add annotations
for size, width in zip([50, 1000], [8, 1.8]):
    if size == 50:
        ax3.annotate(f'Wide CI\nLess precise', xy=(size, width), 
                    xytext=(150, 7), fontsize=11,
                    arrowprops=dict(arrowstyle='->', color='red'))
    else:
        ax3.annotate(f'Narrow CI\nMore precise', xy=(size, width),
                    xytext=(700, 4), fontsize=11,
                    arrowprops=dict(arrowstyle='->', color='green'))

# Bottom right: Type I and Type II errors
ax4.axis('off')

# Create 2x2 table for errors
table_data = [
    ['', 'Reality: No Effect', 'Reality: Effect Exists'],
    ['Test: Significant', 'Type I Error\n(False Positive)\nα = 5%', 'Correct!\n(True Positive)\nPower = 80%'],
    ['Test: Not Significant', 'Correct!\n(True Negative)', 'Type II Error\n(False Negative)\nβ = 20%']
]

# Draw table
cell_colors = [
    ['lightgray', 'lightgray', 'lightgray'],
    ['lightgray', '#ffcccc', '#ccffcc'],
    ['lightgray', '#ccffcc', '#ffcccc']
]

table_x = 2
table_y = 6
cell_width = 2
cell_height = 1.5

for i, row in enumerate(table_data):
    for j, cell in enumerate(row):
        x = table_x + j * cell_width
        y = table_y - i * cell_height
        
        # Draw cell
        rect = patches.Rectangle((x, y), cell_width, cell_height,
                                facecolor=cell_colors[i][j],
                                edgecolor='black', linewidth=1)
        ax4.add_patch(rect)
        
        # Add text
        fontweight = 'bold' if i == 0 or j == 0 else 'normal'
        fontsize = 12 if i == 0 or j == 0 else 11
        ax4.text(x + cell_width/2, y + cell_height/2, cell,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight)

ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_title('Type I vs Type II Errors', fontsize=16, fontweight='bold',
             transform=ax4.transAxes, y=0.95, x=0.5)

# Add key insight
insight = 'Trade-off: Reducing Type I increases Type II'
ax4.text(5, 2, insight, ha='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))

# Overall title
fig.suptitle('Statistical Significance Made Simple', 
            fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('statistical_significance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('statistical_significance.png', dpi=150, bbox_inches='tight')

print("Statistical significance visualization created successfully!")