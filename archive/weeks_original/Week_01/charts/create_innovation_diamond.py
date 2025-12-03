"""
Create the complete Innovation Diamond visualization showing both
divergent (expansion) and convergent (filtering) phases of innovation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch
import matplotlib.patheffects as path_effects

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define colors for different phases
colors = {
    'challenge': '#9467bd',    # Purple
    'explore': '#3498db',      # Blue
    'generate': '#2ecc71',     # Green
    'peak': '#f1c40f',         # Yellow
    'filter': '#e67e22',       # Orange
    'refine': '#e74c3c',       # Red
    'strategy': '#c0392b',     # Dark Red
    'gray': '#7f7f7f'
}

np.random.seed(42)

# Define all stages (expansion + convergence)
all_stages = [
    # Expansion phase (top to middle)
    {'name': 'Innovation Challenge', 'width': 1.5, 'y': 9, 'count': 1, 
     'color': colors['challenge'], 'phase': 'expand'},
    {'name': 'Context Exploration', 'width': 3, 'y': 7.5, 'count': 10, 
     'color': colors['explore'], 'phase': 'expand'},
    {'name': 'Feature Discovery', 'width': 5, 'y': 6, 'count': 100, 
     'color': colors['generate'], 'phase': 'expand'},
    {'name': 'Idea Generation', 'width': 7.5, 'y': 4.5, 'count': 1000, 
     'color': colors['generate'], 'phase': 'expand'},
    
    # Peak (widest point)
    {'name': 'Raw Ideas Pool', 'width': 10, 'y': 3, 'count': 5000, 
     'color': colors['peak'], 'phase': 'peak'},
    
    # Convergence phase (middle to bottom)
    {'name': 'Feature Extraction', 'width': 7.5, 'y': 1.5, 'count': 2000, 
     'color': colors['filter'], 'phase': 'converge'},
    {'name': 'Pattern Discovery', 'width': 5, 'y': 0, 'count': 500, 
     'color': colors['filter'], 'phase': 'converge'},
    {'name': 'Refined Insights', 'width': 3, 'y': -1.5, 'count': 50, 
     'color': colors['refine'], 'phase': 'converge'},
    {'name': 'Innovation Strategy', 'width': 1.5, 'y': -3, 'count': 5, 
     'color': colors['strategy'], 'phase': 'converge'}
]

# Draw the diamond shape
for i in range(len(all_stages) - 1):
    curr = all_stages[i]
    next = all_stages[i + 1]
    
    # Create trapezoid for each segment
    trapezoid = Polygon([
        (-curr['width']/2, curr['y']),
        (curr['width']/2, curr['y']),
        (next['width']/2, next['y']),
        (-next['width']/2, next['y'])
    ], 
    facecolor=curr['color'], alpha=0.15, edgecolor='none')
    ax.add_patch(trapezoid)
    
    # Draw borders with gradient effect
    ax.plot([-curr['width']/2, -next['width']/2], [curr['y'], next['y']], 
           color=curr['color'], linewidth=2.5, alpha=0.8)
    ax.plot([curr['width']/2, next['width']/2], [curr['y'], next['y']], 
           color=curr['color'], linewidth=2.5, alpha=0.8)

# Draw horizontal lines for each stage
for stage in all_stages:
    ax.plot([-stage['width']/2, stage['width']/2], [stage['y'], stage['y']], 
           color=stage['color'], linewidth=2, alpha=0.6)

# Add flowing data points
total_points = 2000
points_x = []
points_y = []
point_colors = []
point_sizes = []

for i, stage in enumerate(all_stages):
    # Number of points proportional to stage
    if stage['phase'] == 'expand':
        n_points = int(total_points * (stage['count'] / 5000) * 2)  # More points as we expand
    elif stage['phase'] == 'peak':
        n_points = int(total_points * 0.4)  # Maximum points at peak
    else:  # converge
        n_points = int(total_points * (stage['count'] / 5000))
    
    # Generate points with appropriate spread
    if stage['phase'] == 'expand':
        # Expanding pattern
        spread_factor = stage['width'] / all_stages[4]['width']
        x = np.random.normal(0, stage['width']/3 * spread_factor, n_points)
    elif stage['phase'] == 'peak':
        # Maximum spread at peak
        x = np.random.uniform(-stage['width']/2 + 0.3, stage['width']/2 - 0.3, n_points)
    else:  # converge
        # Converging pattern
        spread_factor = stage['width'] / all_stages[4]['width']
        x = np.random.normal(0, stage['width']/4 * spread_factor, n_points)
    
    x = np.clip(x, -stage['width']/2 + 0.1, stage['width']/2 - 0.1)
    y = np.random.uniform(stage['y'] - 0.2, stage['y'] + 0.2, n_points)
    
    points_x.extend(x)
    points_y.extend(y)
    point_colors.extend([stage['color']] * n_points)
    
    # Size varies by phase
    if stage['phase'] == 'expand':
        size = max(2, 12 - i*2)
    elif stage['phase'] == 'peak':
        size = 8
    else:
        size = max(2, 10 - (i-4)*2)
    point_sizes.extend([size] * n_points)

# Plot data points
scatter = ax.scatter(points_x, points_y, c=point_colors, s=point_sizes, 
                    alpha=0.4, edgecolors='none')

# Add stage labels and counts
for i, stage in enumerate(all_stages):
    # Stage name on the left
    if i == 0:
        label_text = f"{stage['name']}"
    else:
        label_text = stage['name']
    
    text = ax.text(-6.5, stage['y'], label_text, 
                  fontsize=11, fontweight='bold', color=stage['color'],
                  ha='right', va='center')
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Count on the right
    if stage['count'] == 1:
        count_text = "1 challenge"
    elif stage['count'] < 100:
        count_text = f"{stage['count']} {('insights' if stage['count'] == 50 else 'strategies' if stage['count'] == 5 else 'dimensions')}"
    else:
        count_text = f"{stage['count']:,} {'ideas' if stage['count'] >= 1000 else 'features' if stage['count'] >= 100 else 'patterns'}"
    
    text2 = ax.text(6.5, stage['y'], count_text,
                   fontsize=10, color=stage['color'], style='italic',
                   ha='left', va='center')
    text2.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Add phase indicators with arrows
# Expansion arrows
for i in range(2):
    y_pos = 8 - i*2
    arrow = patches.FancyArrowPatch((0.3, y_pos), (1.5, y_pos - 0.5),
                                   arrowstyle='->', mutation_scale=20,
                                   color=colors['explore'], alpha=0.4, linewidth=2)
    ax.add_patch(arrow)
    arrow2 = patches.FancyArrowPatch((-0.3, y_pos), (-1.5, y_pos - 0.5),
                                    arrowstyle='->', mutation_scale=20,
                                    color=colors['explore'], alpha=0.4, linewidth=2)
    ax.add_patch(arrow2)

# Convergence arrows
for i in range(2):
    y_pos = -0.5 - i*1.5
    arrow = patches.FancyArrowPatch((1.5, y_pos), (0.3, y_pos - 0.5),
                                   arrowstyle='->', mutation_scale=20,
                                   color=colors['refine'], alpha=0.4, linewidth=2)
    ax.add_patch(arrow)
    arrow2 = patches.FancyArrowPatch((-1.5, y_pos), (-0.3, y_pos - 0.5),
                                    arrowstyle='->', mutation_scale=20,
                                    color=colors['refine'], alpha=0.4, linewidth=2)
    ax.add_patch(arrow2)

# Add title and phase labels
ax.set_title('The Innovation Diamond: From Challenge to Strategy', 
            fontsize=16, fontweight='bold', pad=20)

# Add phase annotations
ax.text(0, 10, 'DIVERGENT PHASE', fontsize=12, fontweight='bold',
       color=colors['challenge'], ha='center', alpha=0.8)
ax.text(0, 9.5, 'Exploring the Possible', fontsize=10, style='italic',
       color=colors['gray'], ha='center')

ax.text(0, -3.7, 'CONVERGENT PHASE', fontsize=12, fontweight='bold',
       color=colors['strategy'], ha='center', alpha=0.8)
ax.text(0, -4.2, 'Finding the Optimal', fontsize=10, style='italic',
       color=colors['gray'], ha='center')

# Add central transition zone highlight
transition_box = FancyBboxPatch((-5.5, 2.5), 11, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['peak'], alpha=0.1,
                               edgecolor=colors['peak'], linewidth=2,
                               linestyle='--')
ax.add_patch(transition_box)
ax.text(0, 3, '5000 IDEAS', fontsize=14, fontweight='bold',
       color=colors['peak'], ha='center', va='center')

# Add ML technique annotations
ml_annotations = [
    {'x': -8, 'y': 7.5, 'text': 'ML:\nData\nMining', 'color': colors['explore']},
    {'x': 8, 'y': 6, 'text': 'ML:\nFeature\nEngineering', 'color': colors['generate']},
    {'x': -8, 'y': 4.5, 'text': 'ML:\nGenerative\nAlgorithms', 'color': colors['generate']},
    {'x': 8, 'y': 1.5, 'text': 'ML:\nClustering', 'color': colors['filter']},
    {'x': -8, 'y': 0, 'text': 'ML:\nPattern\nRecognition', 'color': colors['filter']},
    {'x': 8, 'y': -1.5, 'text': 'ML:\nOptimization', 'color': colors['refine']}
]

for ann in ml_annotations:
    ax.text(ann['x'], ann['y'], ann['text'],
           fontsize=8, ha='center', va='center',
           color=ann['color'], fontweight='bold', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

# Set axis limits and remove axes
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 11)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

# Add bottom message
fig.text(0.5, 0.03, 
         'Machine Learning enables both creative expansion and strategic focus in innovation',
         ha='center', fontsize=11, color=colors['gray'], fontweight='bold')
fig.text(0.5, 0.01,
         '1 Challenge → 10 Dimensions → 100 Features → 1000 Ideas → 5000 Pool → 2000 Filtered → 500 Patterns → 50 Insights → 5 Strategies',
         ha='center', fontsize=9, color=colors['gray'], style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('innovation_diamond.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('innovation_diamond.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Innovation diamond chart created successfully!")