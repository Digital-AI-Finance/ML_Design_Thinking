"""
Create a detailed linear breakdown of innovation stages showing the complete
process from challenge to strategy with ML techniques at each step.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patheffects as path_effects

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define colors
colors = {
    'challenge': '#9467bd',    # Purple
    'explore': '#3498db',      # Blue  
    'generate': '#2ecc71',     # Green
    'peak': '#f1c40f',         # Yellow
    'filter': '#e67e22',       # Orange
    'refine': '#e74c3c',       # Red
    'strategy': '#c0392b',     # Dark Red
    'gray': '#7f7f7f',
    'light_gray': '#bdc3c7'
}

np.random.seed(42)

# Define all stages in linear progression
stages = [
    {'name': 'Challenge', 'count': 1, 'color': colors['challenge'], 
     'ml': 'Problem\nDefinition', 'icon': '?', 'phase': 'start'},
    {'name': 'Context', 'count': 10, 'color': colors['explore'], 
     'ml': 'Data\nCollection', 'icon': 'i', 'phase': 'diverge'},
    {'name': 'Features', 'count': 100, 'color': colors['explore'], 
     'ml': 'Feature\nEngineering', 'icon': '*', 'phase': 'diverge'},
    {'name': 'Ideas', 'count': 1000, 'color': colors['generate'], 
     'ml': 'Idea\nGeneration', 'icon': '+', 'phase': 'diverge'},
    {'name': 'Pool', 'count': 5000, 'color': colors['peak'], 
     'ml': 'Complete\nDataset', 'icon': '=', 'phase': 'peak'},
    {'name': 'Filtered', 'count': 2000, 'color': colors['filter'], 
     'ml': 'Initial\nFiltering', 'icon': '/', 'phase': 'converge'},
    {'name': 'Patterns', 'count': 500, 'color': colors['filter'], 
     'ml': 'Pattern\nDiscovery', 'icon': '#', 'phase': 'converge'},
    {'name': 'Insights', 'count': 50, 'color': colors['refine'], 
     'ml': 'Deep\nAnalysis', 'icon': '!', 'phase': 'converge'},
    {'name': 'Strategy', 'count': 5, 'color': colors['strategy'], 
     'ml': 'Final\nOptimization', 'icon': '*', 'phase': 'end'}
]

# Calculate positions
x_positions = np.linspace(1, 13, len(stages))
y_base = 5

# Draw main process flow line
ax.plot(x_positions, [y_base]*len(stages), color=colors['light_gray'], 
       linewidth=3, alpha=0.5, zorder=1)

# Draw phase backgrounds
phase_regions = [
    {'start': 0, 'end': 4.5, 'color': colors['explore'], 'label': 'DIVERGENT PHASE', 'y': 8.5},
    {'start': 4.5, 'end': 5.5, 'color': colors['peak'], 'label': 'PEAK', 'y': 8.5},
    {'start': 5.5, 'end': 14, 'color': colors['refine'], 'label': 'CONVERGENT PHASE', 'y': 8.5}
]

for region in phase_regions:
    rect = FancyBboxPatch((region['start'], 0.5), region['end']-region['start'], 8,
                          boxstyle="round,pad=0.1",
                          facecolor=region['color'], alpha=0.05,
                          edgecolor='none')
    ax.add_patch(rect)
    ax.text((region['start'] + region['end'])/2, region['y'], region['label'],
           fontsize=11, fontweight='bold', color=region['color'], 
           ha='center', va='center', alpha=0.7)

# Draw stages
for i, (stage, x) in enumerate(zip(stages, x_positions)):
    # Main circle for stage
    if stage['phase'] == 'peak':
        radius = 0.6
    else:
        radius = 0.4
    
    circle = Circle((x, y_base), radius, facecolor=stage['color'], 
                   edgecolor='white', linewidth=3, zorder=5, alpha=0.9)
    ax.add_patch(circle)
    
    # Stage icon
    ax.text(x, y_base, stage['icon'], fontsize=16, fontweight='bold',
           color='white', ha='center', va='center', zorder=6)
    
    # Stage name above
    ax.text(x, y_base + 1, stage['name'], fontsize=10, fontweight='bold',
           color=stage['color'], ha='center', va='bottom')
    
    # Count below
    if stage['count'] == 1:
        count_text = '1'
    else:
        count_text = f"{stage['count']:,}"
    ax.text(x, y_base - 0.8, count_text, fontsize=9, 
           color=stage['color'], ha='center', va='top', fontweight='bold')
    
    # ML technique box below
    ml_box = FancyBboxPatch((x - 0.5, y_base - 2.5), 1, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor='white', alpha=0.9,
                            edgecolor=stage['color'], linewidth=1)
    ax.add_patch(ml_box)
    ax.text(x, y_base - 2.1, stage['ml'], fontsize=7, 
           color=stage['color'], ha='center', va='center')
    
    # Connecting arrow to next stage
    if i < len(stages) - 1:
        next_x = x_positions[i + 1]
        arrow = FancyArrowPatch((x + radius, y_base), 
                              (next_x - radius, y_base),
                              arrowstyle='->', mutation_scale=20,
                              color=stage['color'], alpha=0.5, linewidth=2,
                              zorder=2)
        ax.add_patch(arrow)
        
        # Multiplication factor
        factor = stages[i+1]['count'] / stage['count']
        if factor >= 2:
            factor_text = f"×{factor:.0f}"
        elif factor < 1:
            factor_text = f"÷{1/factor:.0f}"
        else:
            factor_text = f"×{factor:.1f}"
        
        mid_x = (x + next_x) / 2
        factor_circle = Circle((mid_x, y_base), 0.2, facecolor='white',
                              edgecolor=stage['color'], linewidth=1.5, zorder=4)
        ax.add_patch(factor_circle)
        ax.text(mid_x, y_base, factor_text, fontsize=7, fontweight='bold',
               color=stage['color'], ha='center', va='center', zorder=5)

# Add data flow visualization
n_flow_lines = 30
for i in range(n_flow_lines):
    flow_points = []
    y_offset = np.random.normal(0, 0.3)
    
    for j, x in enumerate(x_positions):
        if j < 4:  # Expansion
            y = y_base + y_offset * (j + 1) * 0.2
        elif j == 4:  # Peak
            y = y_base + y_offset * 0.8
        else:  # Convergence
            y = y_base + y_offset * (9 - j) * 0.2
        flow_points.append((x, y))
    
    # Plot flow line
    flow_x = [p[0] for p in flow_points]
    flow_y = [p[1] for p in flow_points]
    ax.plot(flow_x, flow_y, color=colors['gray'], alpha=0.05, linewidth=0.5, zorder=0)

# Add key metrics boxes
metrics = [
    {'x': 3, 'y': 3, 'title': 'Expansion Rate', 'value': '50x/stage', 'color': colors['generate']},
    {'x': 7, 'y': 3, 'title': 'Peak Ideas', 'value': '5000', 'color': colors['peak']},
    {'x': 11, 'y': 3, 'title': 'Reduction Rate', 'value': '90%', 'color': colors['refine']}
]

for metric in metrics:
    bbox = FancyBboxPatch((metric['x'] - 0.7, metric['y'] - 0.3), 1.4, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor=metric['color'], alpha=0.1,
                          edgecolor=metric['color'], linewidth=1.5)
    ax.add_patch(bbox)
    ax.text(metric['x'], metric['y'] + 0.1, metric['title'], fontsize=8,
           color=metric['color'], ha='center', va='center', fontweight='bold')
    ax.text(metric['x'], metric['y'] - 0.15, metric['value'], fontsize=9,
           color=metric['color'], ha='center', va='center', fontweight='bold')

# Add process descriptions
process_desc = [
    {'x': 2, 'y': 7.5, 'text': 'Explore\n&\nExpand', 'color': colors['explore']},
    {'x': 7, 'y': 7.5, 'text': 'Maximum\nCreativity', 'color': colors['peak']},
    {'x': 11, 'y': 7.5, 'text': 'Focus\n&\nRefine', 'color': colors['refine']}
]

for desc in process_desc:
    ax.text(desc['x'], desc['y'], desc['text'], fontsize=9,
           color=desc['color'], ha='center', va='center', 
           fontweight='bold', alpha=0.6)

# Add title and labels
ax.set_title('Innovation Pipeline: Stage-by-Stage ML Transformation', 
            fontsize=16, fontweight='bold', pad=20)

# Set axis limits and remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

# Add timeline indicator
ax.text(7, 0.5, 'TIME & EFFORT →', fontsize=10, fontweight='bold',
       color=colors['gray'], ha='center', va='center', alpha=0.7)

# Add bottom message
fig.text(0.5, 0.05, 
         'Each stage applies specific ML techniques to transform and refine innovation ideas',
         ha='center', fontsize=11, color=colors['gray'], fontweight='bold')
fig.text(0.5, 0.02,
         'From singular focus to strategic clarity through systematic exploration and refinement',
         ha='center', fontsize=9, color=colors['gray'], style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('innovation_stages_detail.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('innovation_stages_detail.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Innovation stages detail chart created successfully!")