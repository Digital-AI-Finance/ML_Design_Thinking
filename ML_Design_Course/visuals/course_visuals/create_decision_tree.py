"""
Create decision tree visualization for Week 4
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
colors = {
    'root': '#9C27B0',
    'usability': '#2196F3',
    'performance': '#F44336',
    'feature': '#4CAF50',
    'integration': '#FF9800',
    'leaf': '#FFC107'
}

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define tree structure
# (x, y, text, color_key, is_leaf)
nodes = [
    # Root
    (7, 9, 'User Problem', 'root', False),
    
    # Level 1
    (3.5, 7, 'Technical Issue?', 'root', False),
    (10.5, 7, 'Experience Issue?', 'root', False),
    
    # Level 2 - Left branch
    (1.5, 5, 'Performance\nProblem?', 'performance', False),
    (5.5, 5, 'Feature\nMissing?', 'feature', False),
    
    # Level 2 - Right branch
    (8.5, 5, 'Usability\nIssue?', 'usability', False),
    (12.5, 5, 'Integration\nGap?', 'integration', False),
    
    # Level 3 - Leaves
    (0.5, 3, 'Speed\nOptimization\n(25%)', 'leaf', True),
    (2.5, 3, 'Resource\nUsage\n(15%)', 'leaf', True),
    (4.5, 3, 'Core Feature\nDevelopment\n(20%)', 'leaf', True),
    (6.5, 3, 'Enhancement\nRequest\n(10%)', 'leaf', True),
    (7.5, 3, 'Navigation\nRedesign\n(12%)', 'leaf', True),
    (9.5, 3, 'Onboarding\nImprovement\n(8%)', 'leaf', True),
    (11.5, 3, 'API\nDevelopment\n(7%)', 'leaf', True),
    (13.5, 3, 'Platform\nSupport\n(3%)', 'leaf', True),
]

# Draw connections first (so they appear behind nodes)
connections = [
    # From root
    ((7, 9), (3.5, 7)),
    ((7, 9), (10.5, 7)),
    
    # Left subtree
    ((3.5, 7), (1.5, 5)),
    ((3.5, 7), (5.5, 5)),
    ((1.5, 5), (0.5, 3)),
    ((1.5, 5), (2.5, 3)),
    ((5.5, 5), (4.5, 3)),
    ((5.5, 5), (6.5, 3)),
    
    # Right subtree
    ((10.5, 7), (8.5, 5)),
    ((10.5, 7), (12.5, 5)),
    ((8.5, 5), (7.5, 3)),
    ((8.5, 5), (9.5, 3)),
    ((12.5, 5), (11.5, 3)),
    ((12.5, 5), (13.5, 3)),
]

for (x1, y1), (x2, y2) in connections:
    ax.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.5, zorder=1)

# Draw nodes
for x, y, text, color_key, is_leaf in nodes:
    if is_leaf:
        # Leaf nodes - rectangular
        width = 1.2
        height = 0.8
        box = FancyBboxPatch((x - width/2, y - height/2),
                             width, height,
                             boxstyle="round,pad=0.05",
                             facecolor=colors[color_key], alpha=0.7,
                             edgecolor='black', linewidth=2, zorder=2)
        ax.add_patch(box)
        
        # Add text
        lines = text.split('\n')
        for i, line in enumerate(lines):
            y_offset = (len(lines) - 1) * 0.1 - i * 0.2
            fontsize = 12 if i < len(lines) - 1 else 13
            fontweight = 'normal' if i < len(lines) - 1 else 'bold'
            ax.text(x, y + y_offset, line, ha='center', va='center',
                   fontsize=fontsize, fontweight=fontweight, zorder=3)
    else:
        # Decision nodes - circular
        circle = plt.Circle((x, y), 0.6, facecolor=colors[color_key], 
                           alpha=0.7, edgecolor='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        # Add text
        lines = text.split('\n')
        for i, line in enumerate(lines):
            y_offset = (len(lines) - 1) * 0.1 - i * 0.2
            ax.text(x, y + y_offset, line, ha='center', va='center',
                   fontsize=13, fontweight='bold', color='white', zorder=3)

# Add decision labels on branches
branch_labels = [
    (5, 8, 'Yes', 'left'),
    (8.5, 8, 'No', 'right'),
    (2.5, 6, 'Yes', 'left'),
    (4, 6, 'No', 'right'),
    (9.5, 6, 'Yes', 'left'),
    (11.5, 6, 'No', 'right'),
]

for x, y, label, side in branch_labels:
    ax.text(x, y, label, fontsize=12, style='italic', color='gray',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Title
ax.text(7, 9.8, 'Problem Classification Tree', 
       fontsize=20, fontweight='bold', ha='center')

# Add statistics box
stats_text = (
    "Problem Distribution:\n"
    "• Technical: 60%\n"
    "• Experience: 40%\n"
    "• Avg. depth to solution: 3 steps\n"
    "• Coverage: 95% of issues"
)
ax.text(0.5, 1, stats_text, fontsize=13,
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))

# Add innovation insight
insight_text = (
    "Innovation Focus:\n"
    "Top 3 categories = 57% of problems\n"
    "Prioritize high-impact areas"
)
ax.text(13, 1, insight_text, fontsize=13, ha='right',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Add color legend
legend_y = 1.5
legend_items = [
    ('Decision Node', 'root'),
    ('Solution', 'leaf')
]

for i, (label, color_key) in enumerate(legend_items):
    x = 6.5 + i * 2
    circle = plt.Circle((x - 0.3, legend_y), 0.15, 
                       facecolor=colors[color_key], alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, legend_y, label, fontsize=12, va='center')

plt.tight_layout()

# Save the figure
plt.savefig('decision_tree.pdf', dpi=300, bbox_inches='tight')
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')

print("Decision tree visualization created successfully!")