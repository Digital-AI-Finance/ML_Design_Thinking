"""
Create journey roadmap visualization for Week 1
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors for each stage
stage_colors = {
    'Empathize': '#9C27B0',  # Purple
    'Define': '#2196F3',      # Blue
    'Ideate': '#4CAF50',      # Green
    'Prototype': '#FF9800',   # Orange
    'Test': '#F44336'         # Red
}

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')

# Week data
weeks = [
    (1, 'Foundation', 'neutral', 'Two Pipelines\nConverge'),
    (2, 'Empathize', 'Empathize', 'Hidden User\nSegments'),
    (3, 'Empathize', 'Empathize', 'Emotional\nContext'),
    (4, 'Define', 'Define', 'Problem\nPatterns'),
    (5, 'Define', 'Define', 'Hidden\nThemes'),
    (6, 'Ideate', 'Ideate', 'AI-Powered\nCreativity'),
    (7, 'Ideate', 'Ideate', 'Feature\nImpact'),
    (8, 'Prototype', 'Prototype', 'Structured\nGeneration'),
    (9, 'Prototype', 'Prototype', 'Smart\nValidation'),
    (10, 'Test', 'Test', 'Continuous\nEvolution')
]

# Draw the journey path
path_y = 3
for i in range(10):
    x = i + 0.5
    next_x = x + 1
    
    # Draw connecting line
    if i < 9:
        ax.plot([x + 0.3, next_x - 0.3], [path_y, path_y], 
               'gray', linewidth=2, alpha=0.3, zorder=1)

# Draw week nodes
for week_num, stage_name, stage_key, topic in weeks:
    x = week_num - 0.5
    
    # Determine color
    if stage_key == 'neutral':
        color = '#757575'
    else:
        color = stage_colors[stage_key]
    
    # Draw circle for week
    circle = Circle((x, path_y), 0.3, facecolor=color, 
                   edgecolor='black', linewidth=2, zorder=2)
    ax.add_patch(circle)
    
    # Add week number
    ax.text(x, path_y, str(week_num), ha='center', va='center',
           fontsize=15, fontweight='bold', color='white')
    
    # Add stage label above
    if week_num in [1, 2, 4, 6, 8, 10]:  # Key transition points
        ax.text(x, path_y + 0.7, stage_name, ha='center', va='center',
               fontsize=14, fontweight='bold', color=color)
    
    # Add topic below
    ax.text(x, path_y - 0.7, topic, ha='center', va='center',
           fontsize=13, style='italic')

# Add stage brackets
stage_ranges = [
    (2, 3, 'Empathize', stage_colors['Empathize']),
    (4, 5, 'Define', stage_colors['Define']),
    (6, 7, 'Ideate', stage_colors['Ideate']),
    (8, 9, 'Prototype', stage_colors['Prototype']),
    (10, 10, 'Test', stage_colors['Test'])
]

for start, end, stage, color in stage_ranges:
    x_start = start - 0.8
    x_end = end - 0.2
    y = 1.5
    
    # Draw bracket
    ax.plot([x_start, x_start, x_end, x_end], 
           [y - 0.1, y, y, y - 0.1], 
           color=color, linewidth=2, alpha=0.7)
    
    # Add stage label
    ax.text((x_start + x_end) / 2, y + 0.2, stage.upper(),
           ha='center', fontsize=14, fontweight='bold', color=color)

# Add innovation unlocks
unlocks = [
    (2.5, 4.5, 'Discover hidden needs'),
    (4.5, 4.5, 'Identify right problems'),
    (6.5, 4.5, 'Generate novel solutions'),
    (8.5, 4.5, 'Build smart concepts'),
    (10, 4.5, 'Evolve continuously')
]

for x, y, text in unlocks:
    ax.text(x, y, text, ha='center', fontsize=13,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Title
ax.text(5, 5.5, '10-Week Innovation Journey', 
       fontsize=20, fontweight='bold', ha='center')

# Add complexity indicator
ax.text(0.5, 0.5, 'Foundation', fontsize=13, style='italic', color='gray')
ax.text(5, 0.5, 'Progressive Complexity', fontsize=13, style='italic', color='gray')
ax.text(9.5, 0.5, 'Mastery', fontsize=13, style='italic', color='gray')

# Add arrow showing progression
ax.arrow(0.5, 0.3, 9, 0, head_width=0.1, head_length=0.2,
        fc='gray', ec='gray', alpha=0.3)

plt.tight_layout()

# Save the figure
plt.savefig('journey_roadmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('journey_roadmap.png', dpi=150, bbox_inches='tight')

print("Journey roadmap created successfully!")