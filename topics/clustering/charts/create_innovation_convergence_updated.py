"""
Create an updated Innovation Convergence visualization showing the filtering phase
where ML refines thousands of ideas into actionable strategies.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch, Circle, Wedge
import matplotlib.patheffects as path_effects

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define colors for convergence phase
colors = {
    'peak': '#f1c40f',         # Yellow
    'filter': '#e67e22',       # Orange
    'analyze': '#e74c3c',      # Light Red
    'refine': '#c0392b',       # Red
    'optimize': '#8e44ad',     # Purple
    'strategy': '#9b59b6',     # Light Purple
    'gray': '#7f7f7f'
}

np.random.seed(42)

# Define convergence stages
stages = [
    {'name': 'Raw Ideas Pool', 'width': 10, 'y': 8.5, 'count': 5000, 
     'color': colors['peak'], 'ml': 'Complete Dataset', 'filter': ''},
    {'name': 'Initial Filtering', 'width': 8, 'y': 7, 'count': 2000, 
     'color': colors['filter'], 'ml': 'Feasibility Check', 'filter': '60% filtered'},
    {'name': 'Feature Extraction', 'width': 6.5, 'y': 5.5, 'count': 1000,
     'color': colors['filter'], 'ml': 'Attribute Analysis', 'filter': '50% filtered'},
    {'name': 'Pattern Discovery', 'width': 5, 'y': 4, 'count': 500, 
     'color': colors['analyze'], 'ml': 'Clustering', 'filter': '50% filtered'},
    {'name': 'Theme Identification', 'width': 3.5, 'y': 2.5, 'count': 100,
     'color': colors['analyze'], 'ml': 'Topic Modeling', 'filter': '80% filtered'},
    {'name': 'Refined Insights', 'width': 2.5, 'y': 1, 'count': 50, 
     'color': colors['refine'], 'ml': 'Deep Analysis', 'filter': '50% filtered'},
    {'name': 'Strategic Options', 'width': 1.5, 'y': -0.5, 'count': 5, 
     'color': colors['strategy'], 'ml': 'Optimization', 'filter': '90% filtered'}
]

# Draw the converging funnel shape with filters
for i in range(len(stages) - 1):
    curr = stages[i]
    next = stages[i + 1]
    
    # Create converging trapezoid
    trapezoid = Polygon([
        (-curr['width']/2, curr['y']),
        (curr['width']/2, curr['y']),
        (next['width']/2, next['y']),
        (-next['width']/2, next['y'])
    ], 
    facecolor=curr['color'], alpha=0.12, edgecolor='none')
    ax.add_patch(trapezoid)
    
    # Draw converging borders
    ax.plot([-curr['width']/2, -next['width']/2], [curr['y'], next['y']], 
           color=curr['color'], linewidth=2.5, alpha=0.7)
    ax.plot([curr['width']/2, next['width']/2], [curr['y'], next['y']], 
           color=curr['color'], linewidth=2.5, alpha=0.7)
    
    # Add filter visualization between stages
    if i > 0:
        filter_y = (curr['y'] + next['y']) / 2
        # Draw filter mesh lines
        for x in np.linspace(-curr['width']/2.5, curr['width']/2.5, 7):
            ax.plot([x, x], [filter_y - 0.15, filter_y + 0.15],
                   color=curr['color'], alpha=0.3, linewidth=1, linestyle='--')

# Draw horizontal lines for each stage
for stage in stages:
    ax.plot([-stage['width']/2, stage['width']/2], [stage['y'], stage['y']], 
           color=stage['color'], linewidth=1.5, alpha=0.5)

# Add flowing particles with filtering effect
total_particles = 3000
all_x = []
all_y = []
all_colors = []
all_sizes = []

for i, stage in enumerate(stages):
    # Decreasing particles through filtering
    n_particles = int(total_particles * (stage['count'] / 5000))
    
    # Create converging pattern
    if i == 0:
        # Full spread at top
        x = np.random.uniform(-stage['width']/2 + 0.2, stage['width']/2 - 0.2, n_particles)
    else:
        # Increasingly concentrated
        concentration = 1 - (i / len(stages))
        x = np.random.normal(0, stage['width']/3 * concentration, n_particles)
        x = np.clip(x, -stage['width']/2 + 0.1, stage['width']/2 - 0.1)
    
    y = np.random.normal(stage['y'], 0.15, n_particles)
    
    # Particles get larger as they're refined (quality over quantity)
    size = 2 + i * 2
    alpha = 0.3 + i * 0.05
    
    all_x.extend(x)
    all_y.extend(y)
    all_colors.extend([stage['color']] * n_particles)
    all_sizes.extend([size] * n_particles)

# Plot all particles
ax.scatter(all_x, all_y, c=all_colors, s=all_sizes, alpha=0.4, edgecolors='none')

# Add filtering criteria boxes
filter_criteria = [
    {'y': 6.25, 'text': 'Feasibility\n& Resources', 'color': colors['filter']},
    {'y': 4.75, 'text': 'Market\nAlignment', 'color': colors['filter']},
    {'y': 3.25, 'text': 'Innovation\nPotential', 'color': colors['analyze']},
    {'y': 1.75, 'text': 'Strategic\nFit', 'color': colors['refine']},
    {'y': 0.25, 'text': 'Impact\n& ROI', 'color': colors['strategy']}
]

for fc in filter_criteria:
    # Filter indicator box
    bbox = FancyBboxPatch((-8.5, fc['y'] - 0.3), 2, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor='white', alpha=0.9,
                          edgecolor=fc['color'], linewidth=1.5)
    ax.add_patch(bbox)
    ax.text(-7.5, fc['y'], fc['text'],
           fontsize=8, ha='center', va='center',
           color=fc['color'], fontweight='bold')

# Add stage labels and metrics
for i, stage in enumerate(stages):
    # Stage name and count on right
    if stage['count'] >= 1000:
        count_label = f"{stage['count']:,}"
    else:
        count_label = str(stage['count'])
    
    text = ax.text(7, stage['y'], f"{stage['name']}\n{count_label} items", 
                  fontsize=10, fontweight='bold', color=stage['color'],
                  ha='center', va='center',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # ML technique label
    if stage['ml']:
        text2 = ax.text(9, stage['y'] - 0.5, f"{stage['ml']}", 
                       fontsize=8, color=stage['color'], style='italic',
                       ha='center', va='center')
    
    # Filter percentage
    if stage['filter']:
        circle = Circle((0, (stage['y'] + stages[i-1]['y'])/2), 0.35, 
                       facecolor='white', edgecolor=stage['color'], linewidth=2)
        ax.add_patch(circle)
        ax.text(0, (stage['y'] + stages[i-1]['y'])/2, stage['filter'],
               fontsize=7, fontweight='bold', color=stage['color'],
               ha='center', va='center')

# Add convergence arrows
for i in range(3):
    y_pos = 7 - i*2.5
    # Converging arrows
    arrow1 = patches.FancyArrowPatch((2, y_pos), (0.5, y_pos - 0.7),
                                    arrowstyle='->', mutation_scale=15,
                                    color=colors['refine'], alpha=0.3, linewidth=1.5)
    ax.add_patch(arrow1)
    arrow2 = patches.FancyArrowPatch((-2, y_pos), (-0.5, y_pos - 0.7),
                                    arrowstyle='->', mutation_scale=15,
                                    color=colors['refine'], alpha=0.3, linewidth=1.5)
    ax.add_patch(arrow2)

# Add title
ax.set_title('The Innovation Convergence: ML-Powered Strategic Focus', 
            fontsize=16, fontweight='bold', pad=20)
ax.text(0, 9.5, 'From Abundance to Action Through Intelligent Filtering', 
       fontsize=12, style='italic', color=colors['gray'], ha='center')

# Add quality vs quantity indicator
ax.text(-4.5, 9, 'QUANTITY', fontsize=10, fontweight='bold',
       color=colors['peak'], ha='center')
ax.text(4.5, 9, 'QUALITY', fontsize=10, fontweight='bold',
       color=colors['strategy'], ha='center')

# Add side process indicators
ax.text(-9.5, 8, '↓', fontsize=20, fontweight='bold', color=colors['filter'])
ax.text(-9.5, 5, '↓', fontsize=20, fontweight='bold', color=colors['analyze'])
ax.text(-9.5, 2, '↓', fontsize=20, fontweight='bold', color=colors['refine'])

# Add outcome metrics
outcomes = [
    {'x': 8.5, 'y': -1.5, 'text': 'Final 5:\n• Clear strategy\n• High impact\n• Feasible\n• Innovative', 
     'color': colors['strategy']}
]

for outcome in outcomes:
    bbox = FancyBboxPatch((outcome['x'] - 1.5, outcome['y'] - 0.6), 3, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=outcome['color'], alpha=0.1,
                          edgecolor=outcome['color'], linewidth=1.5)
    ax.add_patch(bbox)
    ax.text(outcome['x'], outcome['y'], outcome['text'],
           fontsize=8, ha='center', va='center',
           color=outcome['color'], fontweight='bold')

# Set axis limits and remove axes
ax.set_xlim(-10, 10)
ax.set_ylim(-3, 10)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

# Add bottom message
fig.text(0.5, 0.05, 
         'Machine Learning systematically filters and refines ideas using multi-criteria optimization',
         ha='center', fontsize=11, color=colors['refine'], fontweight='bold')
fig.text(0.5, 0.02,
         '5000 → 2000 (feasible) → 1000 (valuable) → 500 (patterns) → 100 (themes) → 50 (insights) → 5 (strategies)',
         ha='center', fontsize=9, color=colors['gray'], style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('innovation_convergence_updated.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('innovation_convergence_updated.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Innovation convergence (updated) chart created successfully!")