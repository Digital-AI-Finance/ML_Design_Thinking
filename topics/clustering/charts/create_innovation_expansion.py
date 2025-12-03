"""
Create the Innovation Expansion visualization focusing on the divergent phase
where ML amplifies a single challenge into thousands of possibilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch, Circle
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define colors for expansion phase
colors = {
    'challenge': '#9467bd',    # Purple
    'explore': '#3498db',      # Blue  
    'discover': '#17a2b8',     # Cyan
    'generate': '#2ecc71',     # Green
    'amplify': '#27ae60',      # Dark Green
    'peak': '#f1c40f',         # Yellow
    'gray': '#7f7f7f'
}

np.random.seed(42)

# Define expansion stages
stages = [
    {'name': 'Single Challenge', 'width': 1.5, 'y': 8.5, 'count': 1, 
     'color': colors['challenge'], 'ml': 'Problem Definition'},
    {'name': 'Context Exploration', 'width': 3, 'y': 7, 'count': 10, 
     'color': colors['explore'], 'ml': 'Data Collection'},
    {'name': 'Dimension Analysis', 'width': 4.5, 'y': 5.5, 'count': 50,
     'color': colors['explore'], 'ml': 'Feature Engineering'},
    {'name': 'Feature Discovery', 'width': 6, 'y': 4, 'count': 100, 
     'color': colors['discover'], 'ml': 'Attribute Extraction'},
    {'name': 'Pattern Exploration', 'width': 7.5, 'y': 2.5, 'count': 500,
     'color': colors['generate'], 'ml': 'Correlation Analysis'},
    {'name': 'Idea Generation', 'width': 9, 'y': 1, 'count': 1000, 
     'color': colors['amplify'], 'ml': 'Generative Models'},
    {'name': 'Possibility Universe', 'width': 10, 'y': -0.5, 'count': 5000, 
     'color': colors['peak'], 'ml': 'Full Exploration Space'}
]

# Draw the expanding funnel shape
for i in range(len(stages) - 1):
    curr = stages[i]
    next = stages[i + 1]
    
    # Create expanding trapezoid
    trapezoid = Polygon([
        (-curr['width']/2, curr['y']),
        (curr['width']/2, curr['y']),
        (next['width']/2, next['y']),
        (-next['width']/2, next['y'])
    ], 
    facecolor=curr['color'], alpha=0.12, edgecolor='none')
    ax.add_patch(trapezoid)
    
    # Draw expanding borders with gradient
    ax.plot([-curr['width']/2, -next['width']/2], [curr['y'], next['y']], 
           color=curr['color'], linewidth=2.5, alpha=0.7)
    ax.plot([curr['width']/2, next['width']/2], [curr['y'], next['y']], 
           color=curr['color'], linewidth=2.5, alpha=0.7)

# Draw horizontal lines for each stage
for stage in stages:
    ax.plot([-stage['width']/2, stage['width']/2], [stage['y'], stage['y']], 
           color=stage['color'], linewidth=1.5, alpha=0.5)

# Add expanding particle effect
total_particles = 2500
for i, stage in enumerate(stages):
    # Exponentially increasing particles
    n_particles = int(50 * (2 ** i))
    n_particles = min(n_particles, int(total_particles * (stage['count'] / 5000)))
    
    # Create spreading pattern
    if i == 0:
        # Single point at top
        x = [0]
        y = [stage['y']]
    else:
        # Expanding cloud pattern
        angle = np.random.uniform(0, 2*np.pi, n_particles)
        radius = np.random.uniform(0, stage['width']/2 - 0.1, n_particles)
        # Bias towards spreading outward
        radius = radius ** 0.7 * (stage['width']/2 - 0.1)
        x = radius * np.cos(angle)
        y = np.random.normal(stage['y'], 0.15, n_particles)
    
    # Size and alpha based on stage
    size = max(1, 15 - i*2)
    alpha = max(0.2, 0.6 - i*0.08)
    
    ax.scatter(x, y, c=stage['color'], s=size, alpha=alpha, edgecolors='none')

# Add expansion arrows showing amplification
for i in range(len(stages) - 1):
    curr = stages[i]
    next = stages[i + 1]
    y_mid = (curr['y'] + next['y']) / 2
    
    # Multiple arrows showing expansion
    for x_pos in [-0.5, 0.5]:
        arrow = patches.FancyArrowPatch(
            (x_pos * curr['width']/4, y_mid + 0.3),
            (x_pos * next['width']/3, y_mid - 0.3),
            arrowstyle='->', mutation_scale=15,
            color=curr['color'], alpha=0.3, linewidth=1.5
        )
        ax.add_patch(arrow)

# Add stage labels and ML techniques
for i, stage in enumerate(stages):
    # Stage name and count on left
    if stage['count'] == 1:
        count_label = "1"
        desc = "challenge"
    elif stage['count'] < 100:
        count_label = str(stage['count'])
        desc = "dimensions"
    else:
        count_label = f"{stage['count']:,}"
        desc = "ideas" if stage['count'] >= 1000 else "features"
    
    # Main label
    text = ax.text(-7, stage['y'], f"{stage['name']}\n{count_label} {desc}", 
                  fontsize=10, fontweight='bold', color=stage['color'],
                  ha='center', va='center',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # ML technique on right
    text2 = ax.text(7, stage['y'], f"ML: {stage['ml']}", 
                   fontsize=9, color=stage['color'], style='italic',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=stage['color'], 
                            alpha=0.1, edgecolor=stage['color']))

# Add amplification factors
for i in range(1, len(stages)):
    prev = stages[i-1]
    curr = stages[i]
    factor = curr['count'] / prev['count']
    
    y_pos = (prev['y'] + curr['y']) / 2
    if factor >= 2:
        factor_text = f"{factor:.0f}×"
    else:
        factor_text = f"{factor:.1f}×"
    
    # Amplification factor in circles
    circle = Circle((0, y_pos), 0.3, facecolor='white', 
                   edgecolor=prev['color'], linewidth=2, alpha=0.9)
    ax.add_patch(circle)
    ax.text(0, y_pos, factor_text, fontsize=9, fontweight='bold',
           color=prev['color'], ha='center', va='center')

# Add title and subtitle
ax.set_title('The Innovation Expansion: ML-Powered Divergent Thinking', 
            fontsize=16, fontweight='bold', pad=20)
ax.text(0, 9.5, 'From One Challenge to Thousands of Possibilities', 
       fontsize=12, style='italic', color=colors['gray'], ha='center')

# Add phase indicators
ax.text(-4, 9, 'FOCUS', fontsize=11, fontweight='bold',
       color=colors['challenge'], ha='center', rotation=0)
ax.text(4, 9, 'EXPLORE', fontsize=11, fontweight='bold',
       color=colors['explore'], ha='center', rotation=0)
ax.text(-4.5, -1.5, 'AMPLIFY', fontsize=11, fontweight='bold',
       color=colors['amplify'], ha='center', rotation=0)
ax.text(4.5, -1.5, 'GENERATE', fontsize=11, fontweight='bold',
       color=colors['peak'], ha='center', rotation=0)

# Add side annotations with process descriptions
process_annotations = [
    {'x': -9, 'y': 6, 'text': '▲\nDivergent\nThinking', 'color': colors['explore']},
    {'x': -9, 'y': 3, 'text': '▲\nCreative\nExpansion', 'color': colors['generate']},
    {'x': 9, 'y': 6, 'text': '◆\nSystematic\nExploration', 'color': colors['explore']},
    {'x': 9, 'y': 3, 'text': '◆\nCombinatorial\nGeneration', 'color': colors['generate']}
]

for ann in process_annotations:
    ax.text(ann['x'], ann['y'], ann['text'],
           fontsize=8, ha='center', va='center',
           color=ann['color'], fontweight='bold', alpha=0.7)

# Add expansion visualization lines
n_lines = 15
for i in range(n_lines):
    start_y = 8.5
    end_y = -0.5
    start_x = 0
    end_x = (i - n_lines/2) / 2
    
    line_alpha = 0.1
    line = plt.Line2D([start_x, end_x], [start_y, end_y],
                     color=colors['gray'], alpha=line_alpha, linewidth=0.5)
    ax.add_line(line)

# Set axis limits and remove axes
ax.set_xlim(-10, 10)
ax.set_ylim(-2, 10)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

# Add bottom message
fig.text(0.5, 0.05, 
         'Machine Learning amplifies innovation through systematic exploration and creative combination',
         ha='center', fontsize=11, color=colors['amplify'], fontweight='bold')
fig.text(0.5, 0.02,
         '1 → 10× → 5× → 2× → 5× → 2× → 5× = 5000 possibilities from a single seed',
         ha='center', fontsize=9, color=colors['gray'], style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('innovation_expansion.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('innovation_expansion.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Innovation expansion chart created successfully!")