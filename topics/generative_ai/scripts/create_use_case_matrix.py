import matplotlib.pyplot as plt
import numpy as np

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Industries and use cases
industries = ['E-commerce', 'Gaming', 'Marketing', 'Architecture', 'Fashion',
              'Healthcare', 'Finance', 'Education', 'Media', 'Manufacturing']

use_cases = ['Product\nImages', 'Asset\nGeneration', 'Campaign\nCreation',
             'Concept\nDesign', 'Collection\nPlanning']

# Impact matrix (0-100 scale)
# Rows: Industries, Columns: Use cases
impact_matrix = np.array([
    [95, 30, 85, 40, 70],  # E-commerce
    [40, 98, 60, 80, 30],  # Gaming
    [80, 45, 95, 70, 50],  # Marketing
    [60, 70, 40, 98, 20],  # Architecture
    [90, 20, 75, 85, 98],  # Fashion
    [70, 15, 50, 65, 25],  # Healthcare
    [50, 10, 80, 40, 35],  # Finance
    [60, 40, 70, 55, 30],  # Education
    [85, 75, 90, 60, 40],  # Media
    [45, 60, 30, 75, 50],  # Manufacturing
])

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap
im = ax.imshow(impact_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set ticks and labels
ax.set_xticks(np.arange(len(use_cases)))
ax.set_yticks(np.arange(len(industries)))
ax.set_xticklabels(use_cases, fontsize=10)
ax.set_yticklabels(industries, fontsize=10)

# Rotate the tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

# Add text annotations
for i in range(len(industries)):
    for j in range(len(use_cases)):
        value = impact_matrix[i, j]
        color = 'white' if value > 60 else 'black'
        text = ax.text(j, i, f'{value}%', ha='center', va='center',
                      color=color, fontsize=9, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Impact Level (%)', rotation=270, labelpad=20, fontsize=11)

# Add title and labels
ax.set_title('GenAI Use Case Impact Matrix by Industry',
             fontsize=14, fontweight='bold', pad=20)

# Add ROI indicators on the side
roi_text = ['300%', '250%', '280%', '200%', '350%', '180%', '150%', '170%', '270%', '160%']
for i, roi in enumerate(roi_text):
    ax.text(5.5, i, f'ROI: {roi}', fontsize=9, color='#2c3e50', fontweight='bold')

# Add legend for impact levels
legend_elements = [
    plt.Rectangle((0,0),1,1, fc='#d73027', label='Low Impact (0-33%)'),
    plt.Rectangle((0,0),1,1, fc='#fee08b', label='Medium Impact (34-66%)'),
    plt.Rectangle((0,0),1,1, fc='#1a9850', label='High Impact (67-100%)')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

# Grid styling
ax.set_xticks(np.arange(len(use_cases))-.5, minor=True)
ax.set_yticks(np.arange(len(industries))-.5, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
ax.tick_params(which='minor', size=0)

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Save the figure
plt.tight_layout()
plt.savefig('../charts/use_case_matrix.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/use_case_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created use_case_matrix chart")