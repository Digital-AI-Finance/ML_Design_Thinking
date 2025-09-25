import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, FancyArrowPatch
import matplotlib.lines as mlines

# Set random seed for reproducibility
np.random.seed(42)

# Define color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f'
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Title
ax.text(0.5, 0.95, 'Week 2 Preview: Advanced Clustering', 
        fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes,
        color=colors['mlpurple'])

# Week 1 recap box
recap_box = FancyBboxPatch((0.05, 0.7), 0.4, 0.2,
                           boxstyle="round,pad=0.01",
                           linewidth=2, edgecolor=colors['mlgray'],
                           facecolor='lightgray', alpha=0.3)
ax.add_patch(recap_box)
ax.text(0.25, 0.85, 'Week 1: Basics', fontsize=14, fontweight='bold', ha='center',
        transform=ax.transAxes, color=colors['mlgray'])
ax.text(0.25, 0.78, '• K-means clustering', fontsize=10, ha='center',
        transform=ax.transAxes)
ax.text(0.25, 0.73, '• Finding optimal K', fontsize=10, ha='center',
        transform=ax.transAxes)

# Week 2 content box
content_box = FancyBboxPatch((0.55, 0.7), 0.4, 0.2,
                             boxstyle="round,pad=0.01",
                             linewidth=2, edgecolor=colors['mlgreen'],
                             facecolor=colors['mlgreen'], alpha=0.1)
ax.add_patch(content_box)
ax.text(0.75, 0.85, 'Week 2: Advanced', fontsize=14, fontweight='bold', ha='center',
        transform=ax.transAxes, color=colors['mlgreen'])
ax.text(0.75, 0.78, '• Dynamic clustering', fontsize=10, ha='center',
        transform=ax.transAxes)
ax.text(0.75, 0.73, '• Real-time analysis', fontsize=10, ha='center',
        transform=ax.transAxes)

# Arrow from Week 1 to Week 2
arrow = FancyArrowPatch((0.45, 0.8), (0.55, 0.8),
                       connectionstyle="arc3,rad=.2",
                       arrowstyle='->', mutation_scale=20,
                       color=colors['mlpurple'], linewidth=2,
                       transform=ax.transAxes)
ax.add_patch(arrow)

# Topics grid
topics = [
    {'name': 'DBSCAN Deep Dive', 'x': 0.2, 'y': 0.5, 'color': colors['mlblue']},
    {'name': 'GMM Models', 'x': 0.5, 'y': 0.5, 'color': colors['mlorange']},
    {'name': 'Online Clustering', 'x': 0.8, 'y': 0.5, 'color': colors['mlred']},
    {'name': 'Feature Engineering', 'x': 0.2, 'y': 0.3, 'color': colors['mlgreen']},
    {'name': 'Validation Methods', 'x': 0.5, 'y': 0.3, 'color': colors['mlpurple']},
    {'name': 'Real Applications', 'x': 0.8, 'y': 0.3, 'color': colors['mlbrown']}
]

for topic in topics:
    circle = Circle((topic['x'], topic['y']), 0.08, 
                   color=topic['color'], alpha=0.3, transform=ax.transAxes)
    ax.add_patch(circle)
    ax.text(topic['x'], topic['y'], topic['name'], fontsize=10,
            ha='center', va='center', transform=ax.transAxes,
            color=topic['color'], fontweight='bold')

# Add motivational text
ax.text(0.5, 0.15, 'Building on Week 1 Foundation', 
        fontsize=16, ha='center', transform=ax.transAxes,
        color=colors['mlpurple'], fontweight='bold')
ax.text(0.5, 0.08, 'From Basic Clustering to Advanced Pattern Recognition', 
        fontsize=12, ha='center', transform=ax.transAxes,
        color=colors['mlgray'], style='italic')

# Remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()

# Save the figure
plt.savefig('charts/week2_preview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/week2_preview.png', dpi=150, bbox_inches='tight')
print("Week 2 preview chart created successfully!")
plt.close()