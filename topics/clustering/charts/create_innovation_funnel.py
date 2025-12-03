import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f'
}

np.random.seed(42)

# Define funnel stages
stages = [
    {'name': 'Raw Ideas', 'width': 10, 'y': 8, 'count': 5000, 'color': colors['mlgray']},
    {'name': 'Feature Extraction', 'width': 7, 'y': 6, 'count': 2000, 'color': colors['mlblue']},
    {'name': 'Pattern Discovery', 'width': 4.5, 'y': 4, 'count': 500, 'color': colors['mlorange']},
    {'name': 'Refined Insights', 'width': 2.5, 'y': 2, 'count': 50, 'color': colors['mlgreen']},
    {'name': 'Innovation Strategy', 'width': 1.5, 'y': 0, 'count': 5, 'color': colors['mlpurple']}
]

# Draw the funnel shape
for i in range(len(stages) - 1):
    curr = stages[i]
    next = stages[i + 1]
    
    # Create trapezoid for each funnel segment
    trapezoid = Polygon([
        (-curr['width']/2, curr['y']),
        (curr['width']/2, curr['y']),
        (next['width']/2, next['y']),
        (-next['width']/2, next['y'])
    ], 
    facecolor=curr['color'], alpha=0.15, edgecolor='none')
    ax.add_patch(trapezoid)
    
    # Draw borders
    ax.plot([-curr['width']/2, -next['width']/2], [curr['y'], next['y']], 
           color=curr['color'], linewidth=2, alpha=0.8)
    ax.plot([curr['width']/2, next['width']/2], [curr['y'], next['y']], 
           color=curr['color'], linewidth=2, alpha=0.8)

# Draw horizontal lines for each stage
for stage in stages:
    ax.plot([-stage['width']/2, stage['width']/2], [stage['y'], stage['y']], 
           color=stage['color'], linewidth=2, alpha=0.8)

# Add data points flowing through the funnel
total_points = 1500
points_x = []
points_y = []
point_colors = []
point_sizes = []

for i, stage in enumerate(stages):
    # Number of points at this stage (proportional to count)
    n_points = int(total_points * (stage['count'] / stages[0]['count']))
    
    # Generate points within the funnel width at this level
    if i == 0:
        # Top stage - full scatter
        x = np.random.uniform(-stage['width']/2 + 0.2, stage['width']/2 - 0.2, n_points)
        y = np.random.uniform(stage['y'] - 0.3, stage['y'] + 0.3, n_points)
    else:
        # Other stages - concentrated flow
        x = np.random.normal(0, stage['width']/4, n_points)
        x = np.clip(x, -stage['width']/2 + 0.2, stage['width']/2 - 0.2)
        y = np.random.uniform(stage['y'] - 0.2, stage['y'] + 0.2, n_points)
    
    points_x.extend(x)
    points_y.extend(y)
    point_colors.extend([stage['color']] * n_points)
    point_sizes.extend([max(2, 15 - i*3)] * n_points)

# Plot data points
scatter = ax.scatter(points_x, points_y, c=point_colors, s=point_sizes, 
                    alpha=0.4, edgecolors='none')

# Add stage labels and counts
for i, stage in enumerate(stages):
    # Stage name on the left
    text = ax.text(-6, stage['y'], stage['name'], 
                  fontsize=12, fontweight='bold', color=stage['color'],
                  ha='right', va='center')
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Count on the right
    count_text = f"{stage['count']:,} ideas"
    text2 = ax.text(6, stage['y'], count_text,
                   fontsize=10, color=stage['color'], style='italic',
                   ha='left', va='center')
    text2.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Add arrows showing flow direction
for i in range(3):
    y_pos = 7 - i*2
    arrow = patches.FancyArrowPatch((0, y_pos), (0, y_pos - 1),
                                   arrowstyle='->', mutation_scale=30,
                                   color=colors['mlpurple'], alpha=0.5, linewidth=2)
    ax.add_patch(arrow)

# Add title and labels
ax.set_title('The Innovation Refinement Funnel', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(-7, 7)
ax.set_ylim(-1, 9)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

# Add annotations with insights
annotation_boxes = [
    {'x': -4.5, 'y': 8.5, 'text': 'Chaos:\nUnstructured\ndata overload', 'color': colors['mlgray']},
    {'x': 4.5, 'y': 3, 'text': 'ML Magic:\nAutomatic\npattern detection', 'color': colors['mlorange']},
    {'x': -4.5, 'y': 0.5, 'text': 'Clarity:\nActionable\ninsights', 'color': colors['mlpurple']}
]

for ann in annotation_boxes:
    bbox = FancyBboxPatch((ann['x'] - 1.2, ann['y'] - 0.4), 2.4, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor=ann['color'], alpha=0.1,
                          edgecolor=ann['color'], linewidth=1)
    ax.add_patch(bbox)
    ax.text(ann['x'], ann['y'], ann['text'],
           fontsize=9, ha='center', va='center',
           color=ann['color'], fontweight='bold')

# Add bottom message
fig.text(0.5, 0.05, 
         'Machine Learning progressively refines thousands of raw ideas into strategic innovation opportunities',
         ha='center', fontsize=11, color=colors['mlpurple'], fontweight='bold')

# Add feature count progression
fig.text(0.5, 0.02,
         'Features: 0 → 100s → Patterns → Clusters → Strategy',
         ha='center', fontsize=10, color=colors['mlgray'], style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('innovation_funnel.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('innovation_funnel.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Innovation funnel chart (redesigned) created successfully!")