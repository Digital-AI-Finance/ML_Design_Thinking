"""
Create statistical significance visualization Part 2 for Week 10
Sample Size Impact and Type I/II Errors
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Sample size impact
ax1.set_xlim(0, 1100)
ax1.set_ylim(0, 10)

sample_sizes = [50, 100, 200, 500, 1000]
ci_widths = [8, 5.6, 4, 2.5, 1.8]

# Main line plot
ax1.plot(sample_sizes, ci_widths, 'b-', linewidth=3, marker='o', markersize=12, 
         markerfacecolor='lightblue', markeredgecolor='darkblue', markeredgewidth=2)
ax1.fill_between(sample_sizes, 0, ci_widths, alpha=0.3, color='lightblue')

ax1.set_title('Sample Size Impact on Precision', fontsize=18, fontweight='bold')
ax1.set_xlabel('Sample Size', fontsize=14)
ax1.set_ylabel('Confidence Interval Width (%)', fontsize=14)
ax1.grid(True, alpha=0.3)

# Add annotations with more detail
ax1.annotate('Wide CI\nLess precise\nMore uncertainty', xy=(50, 8), 
            xytext=(200, 8.5), fontsize=13,
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc'))

ax1.annotate('Narrow CI\nMore precise\nHigher confidence', xy=(1000, 1.8),
            xytext=(750, 4), fontsize=13,
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc'))

# Add power analysis info
ax1.text(550, 8, 'Power Analysis Rule:', fontsize=13, fontweight='bold')
ax1.text(550, 7.3, 'N > 30 per group (minimum)', fontsize=12)
ax1.text(550, 6.6, 'N > 100 per group (recommended)', fontsize=12)
ax1.text(550, 5.9, 'N > 500 per group (high confidence)', fontsize=12)

# Right: Type I and Type II errors
ax2.axis('off')

# Create confusion matrix style table
table_x = 1.5
table_y = 5
cell_width = 2.2
cell_height = 1.2

# Table data
table_data = [
    ['', 'Reality:\nNo Effect', 'Reality:\nEffect Exists'],
    ['Test Says:\nSignificant', 'Type I Error\n(False Positive)\nα = 5%', 'Correct!\n(True Positive)\nPower = 80%'],
    ['Test Says:\nNot Significant', 'Correct!\n(True Negative)\n95% confidence', 'Type II Error\n(False Negative)\nβ = 20%']
]

# Cell colors
cell_colors = [
    ['#e0e0e0', '#e0e0e0', '#e0e0e0'],
    ['#e0e0e0', '#ffcccc', '#ccffcc'],
    ['#e0e0e0', '#ccffcc', '#ffcccc']
]

# Draw table
for i, row in enumerate(table_data):
    for j, cell in enumerate(row):
        x = table_x + j * cell_width
        y = table_y - i * cell_height
        
        # Draw cell
        rect = patches.Rectangle((x, y), cell_width, cell_height,
                                facecolor=cell_colors[i][j],
                                edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        
        # Add text
        if i == 0 or j == 0:
            fontweight = 'bold'
            fontsize = 13
        else:
            fontweight = 'normal'
            fontsize = 12
            
        ax2.text(x + cell_width/2, y + cell_height/2, cell,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, multialignment='center')

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 7)
ax2.set_title('Type I vs Type II Errors: The Trade-off', fontsize=18, fontweight='bold',
             transform=ax2.transAxes, y=0.95, x=0.5)

# Add explanatory notes
notes = [
    ('Type I (α):', 'Saying there IS an effect when there is NOT', '#ff6666'),
    ('Type II (β):', 'Saying there is NO effect when there IS', '#ff9966'),
    ('Power (1-β):', 'Correctly detecting a real effect', '#66cc66')
]

y_start = 1.8
for i, (label, explanation, color) in enumerate(notes):
    y_pos = y_start - i * 0.5
    ax2.text(2, y_pos, label, fontsize=12, fontweight='bold')
    ax2.text(3.2, y_pos, explanation, fontsize=11, color=color)

# Trade-off explanation
ax2.text(5, 0.3, 'Key Insight: Reducing Type I errors increases Type II errors', 
        ha='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9))

# Overall title
fig.suptitle('Statistical Significance: Sample Size and Error Types', 
            fontsize=20, fontweight='bold', y=0.98)

plt.tight_layout()

# Save the figure
plt.savefig('statistical_significance_part2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('statistical_significance_part2.png', dpi=150, bbox_inches='tight')

print("Statistical significance Part 2 visualization created successfully!")