import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import matplotlib.patches as patches

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
    'mlpink': '#e377c2'
}

# Generate different patterns of innovation data
np.random.seed(42)

# Group 1: Traditional Incremental Innovations (tight cluster)
n1 = 150
center1 = [2, 7]
X1 = np.random.randn(n1, 2) * 0.5 + center1

# Group 2: Disruptive Innovations (scattered but related)
n2 = 100
center2 = [8, 8]
X2 = np.random.randn(n2, 2) * 1.2 + center2

# Group 3: Platform Innovations (interconnected)
n3 = 120
center3 = [5, 3]
X3 = np.random.randn(n3, 2) * 0.8 + center3

# Group 4: Service Innovations (elongated)
n4 = 80
X4_base = np.random.randn(n4, 2)
X4_base[:, 0] *= 2.5
X4_base[:, 1] *= 0.6
# Rotate 30 degrees
theta = np.radians(30)
rotation = np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])
X4 = X4_base @ rotation + [9, 3]

# Group 5: Emerging Tech (small tight cluster)
n5 = 40
center5 = [3, 1]
X5 = np.random.randn(n5, 2) * 0.3 + center5

# Add some outliers (radical innovations)
n_outliers = 15
X_outliers = np.random.uniform(low=0, high=12, size=(n_outliers, 2))

# Plot all points with subtle coloring (as if unlabeled)
all_X = np.vstack([X1, X2, X3, X4, X5, X_outliers])

# Shuffle to mix groups
indices = np.random.permutation(len(all_X))
all_X_shuffled = all_X[indices]

# Create subtle color gradient based on position
colors_array = []
for point in all_X_shuffled:
    # Color based on position for subtle hint
    r = point[0] / 12
    b = point[1] / 10
    g = 0.5
    colors_array.append((r*0.3 + 0.4, g, b*0.3 + 0.4))

# Plot points
scatter = ax.scatter(all_X_shuffled[:, 0], all_X_shuffled[:, 1], 
                    c=colors_array, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add subtle visual hints with dotted circles (not obvious boundaries)
circle1 = plt.Circle(center1, 1.5, fill=False, edgecolor='gray', linestyle='--', alpha=0.3, linewidth=1)
circle2 = plt.Circle(center2, 2.5, fill=False, edgecolor='gray', linestyle='--', alpha=0.3, linewidth=1)
circle3 = plt.Circle(center3, 2, fill=False, edgecolor='gray', linestyle='--', alpha=0.3, linewidth=1)
circle5 = plt.Circle(center5, 1, fill=False, edgecolor='gray', linestyle='--', alpha=0.3, linewidth=1)

ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle5)

# Add ellipse for elongated group
from matplotlib.patches import Ellipse
ellipse = Ellipse([9, 3], width=6, height=2, angle=30, 
                  fill=False, edgecolor='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.add_patch(ellipse)

# Add some question marks to encourage thinking
ax.text(2, 8.5, '?', fontsize=20, color='gray', alpha=0.5, fontweight='bold')
ax.text(8, 9.5, '?', fontsize=20, color='gray', alpha=0.5, fontweight='bold')
ax.text(5, 4.5, '?', fontsize=20, color='gray', alpha=0.5, fontweight='bold')
ax.text(10, 4, '?', fontsize=20, color='gray', alpha=0.5, fontweight='bold')
ax.text(3, 2, '?', fontsize=20, color='gray', alpha=0.5, fontweight='bold')

# Add axis labels
ax.set_xlabel('Innovation Complexity Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Market Disruption Potential', fontsize=12, fontweight='bold')

# Add title
ax.set_title('Innovation Landscape: Can You Spot the Patterns?', fontsize=14, fontweight='bold')

# Add subtle grid
ax.grid(True, alpha=0.2)

# Set axis limits
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.5, 10.5)

# Add side annotations as hints
ax.text(13, 9, 'Hint 1:\nLook for\ndensity', fontsize=9, color='gray', alpha=0.7)
ax.text(13, 6, 'Hint 2:\nNotice\nshapes', fontsize=9, color='gray', alpha=0.7)
ax.text(13, 3, 'Hint 3:\nFind\noutliers', fontsize=9, color='gray', alpha=0.7)

# Add legend explaining axes
ax.text(-1.5, 5, 'Low\nDisruption', fontsize=9, color='gray', rotation=90, va='center')
ax.text(-1.5, 9, 'High\nDisruption', fontsize=9, color='gray', rotation=90, va='center')
ax.text(1, -1, 'Simple', fontsize=9, color='gray', ha='center')
ax.text(11, -1, 'Complex', fontsize=9, color='gray', ha='center')

# Add note at bottom
fig.text(0.5, 0.02, 'Exercise: How many distinct innovation groups do you see? Are some innovations isolated outliers?', 
         ha='center', fontsize=11, color=colors['mlpurple'], style='italic')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('discovery_patterns.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('discovery_patterns.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Discovery patterns chart created successfully!")