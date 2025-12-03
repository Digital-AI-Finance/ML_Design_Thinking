"""
Create a 3D perspective view of the innovation flow showing depth and movement
through the complete diamond process.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with 3D subplot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Define colors
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

# Define stages with 3D coordinates
stages_3d = [
    # Expansion phase
    {'name': 'Challenge', 'radius': 0.5, 'z': 9, 'count': 1, 'color': colors['challenge']},
    {'name': 'Context', 'radius': 1.5, 'z': 7.5, 'count': 10, 'color': colors['explore']},
    {'name': 'Features', 'radius': 2.5, 'z': 6, 'count': 100, 'color': colors['explore']},
    {'name': 'Ideas', 'radius': 3.5, 'z': 4.5, 'count': 1000, 'color': colors['generate']},
    # Peak
    {'name': 'Pool', 'radius': 5, 'z': 3, 'count': 5000, 'color': colors['peak']},
    # Convergence phase
    {'name': 'Filtered', 'radius': 3.5, 'z': 1.5, 'count': 2000, 'color': colors['filter']},
    {'name': 'Patterns', 'radius': 2.5, 'z': 0, 'count': 500, 'color': colors['filter']},
    {'name': 'Insights', 'radius': 1.5, 'z': -1.5, 'count': 50, 'color': colors['refine']},
    {'name': 'Strategy', 'radius': 0.5, 'z': -3, 'count': 5, 'color': colors['strategy']}
]

# Create circular rings at each level
theta = np.linspace(0, 2*np.pi, 100)

for i, stage in enumerate(stages_3d):
    # Draw circular ring
    x_ring = stage['radius'] * np.cos(theta)
    y_ring = stage['radius'] * np.sin(theta)
    z_ring = np.full_like(x_ring, stage['z'])
    
    ax.plot(x_ring, y_ring, z_ring, color=stage['color'], linewidth=2, alpha=0.7)
    
    # Connect to next stage
    if i < len(stages_3d) - 1:
        next_stage = stages_3d[i + 1]
        # Draw connecting lines at 8 points
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x_points = [stage['radius'] * np.cos(angle), 
                       next_stage['radius'] * np.cos(angle)]
            y_points = [stage['radius'] * np.sin(angle),
                       next_stage['radius'] * np.sin(angle)]
            z_points = [stage['z'], next_stage['z']]
            ax.plot(x_points, y_points, z_points, 
                   color=stage['color'], alpha=0.3, linewidth=1)

# Add flowing particles through the system
n_particles = 2000
particle_paths = []

for i in range(n_particles):
    # Generate particle path through all stages
    path_x = []
    path_y = []
    path_z = []
    path_colors = []
    
    # Random angle for this particle
    base_angle = np.random.uniform(0, 2*np.pi)
    
    for j, stage in enumerate(stages_3d):
        # Add some randomness to radius and angle
        angle_variation = np.random.normal(0, 0.2)
        radius_variation = np.random.uniform(0.8, 1.0)
        
        if j < 4:  # Expansion
            # Spreading out
            radius = stage['radius'] * radius_variation
            angle = base_angle + angle_variation * (j + 1) * 0.1
        elif j == 4:  # Peak
            radius = stage['radius'] * radius_variation
            angle = base_angle + angle_variation
        else:  # Convergence
            # Coming together
            radius = stage['radius'] * radius_variation
            angle = base_angle * (1 - (j - 4) * 0.15) + angle_variation * 0.1
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = stage['z'] + np.random.normal(0, 0.1)
        
        path_x.append(x)
        path_y.append(y)
        path_z.append(z)
        path_colors.append(stage['color'])
    
    # Plot particle path (sample only some for performance)
    if i % 50 == 0:  # Plot every 50th particle
        for j in range(len(path_x) - 1):
            ax.plot([path_x[j], path_x[j+1]], 
                   [path_y[j], path_y[j+1]], 
                   [path_z[j], path_z[j+1]],
                   color=path_colors[j], alpha=0.1, linewidth=0.5)

# Add data points at each level
for stage in stages_3d:
    n_points = min(200, stage['count'])
    
    # Generate points in circular distribution
    angles = np.random.uniform(0, 2*np.pi, n_points)
    radii = np.random.uniform(0, stage['radius'], n_points)
    # Bias towards edge for better visibility
    radii = radii ** 0.5 * stage['radius']
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = np.full(n_points, stage['z']) + np.random.normal(0, 0.1, n_points)
    
    ax.scatter(x, y, z, c=stage['color'], s=10, alpha=0.6, edgecolors='none')

# Add stage labels
for stage in stages_3d:
    # Place label outside the ring
    label_radius = stage['radius'] + 1.5
    ax.text(label_radius, 0, stage['z'], 
           f"{stage['name']}\n{stage['count']:,}",
           fontsize=9, fontweight='bold', color=stage['color'],
           ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

# Add phase labels
ax.text(0, 7, 10, 'DIVERGENT', fontsize=12, fontweight='bold',
       color=colors['explore'], ha='center')
ax.text(0, 7, -4, 'CONVERGENT', fontsize=12, fontweight='bold',
       color=colors['refine'], ha='center')
ax.text(0, 7, 3, 'TRANSITION', fontsize=12, fontweight='bold',
       color=colors['peak'], ha='center')

# Add ML technique annotations
ml_annotations = [
    {'x': -6, 'y': 0, 'z': 7, 'text': 'Data Mining', 'color': colors['explore']},
    {'x': -6, 'y': 0, 'z': 4, 'text': 'Generation', 'color': colors['generate']},
    {'x': -6, 'y': 0, 'z': 0, 'text': 'Clustering', 'color': colors['filter']},
    {'x': -6, 'y': 0, 'z': -3, 'text': 'Optimization', 'color': colors['strategy']}
]

for ann in ml_annotations:
    ax.text(ann['x'], ann['y'], ann['z'], ann['text'],
           fontsize=8, color=ann['color'], fontweight='bold', alpha=0.7)

# Set labels and title
ax.set_title('Innovation Flow: 3D Perspective', fontsize=16, fontweight='bold', pad=20)

# Set axis properties
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_zlim(-4, 10)

# Hide axis for cleaner look
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Remove panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove grid
ax.grid(False)

# Set viewing angle for best perspective
ax.view_init(elev=20, azim=45)

# Add text annotations on 2D overlay
fig.text(0.5, 0.95, 'The Complete Innovation Journey in 3D', 
        ha='center', fontsize=14, fontweight='bold', color=colors['gray'])
fig.text(0.5, 0.05, 
         'Ideas flow through expanding and contracting stages, filtered by ML at each level',
         ha='center', fontsize=11, color=colors['gray'], fontweight='bold')
fig.text(0.5, 0.02,
         'Height represents process stage • Width represents idea diversity • Color represents transformation phase',
         ha='center', fontsize=9, color=colors['gray'], style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('innovation_flow_3d.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('innovation_flow_3d.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Innovation flow 3D chart created successfully!")