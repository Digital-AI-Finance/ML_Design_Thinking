import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd'
}

# 1. Volume concentration in high dimensions
ax1 = plt.subplot(2, 3, 1)
dimensions = np.arange(1, 101)
# Proportion of volume in outer shell
outer_shell_prop = 1 - (0.99)**dimensions
ax1.plot(dimensions, outer_shell_prop, color=colors['mlred'], linewidth=3)
ax1.fill_between(dimensions, outer_shell_prop, alpha=0.3, color=colors['mlred'])
ax1.set_xlabel('Number of Dimensions', fontsize=11)
ax1.set_ylabel('Proportion of Volume\nin Outer 1% Shell', fontsize=11)
ax1.set_title('Volume Concentrates at Boundaries', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.99, color='black', linestyle='--', alpha=0.5)
ax1.text(50, 0.92, '99% of volume\nat boundary!', fontsize=10, color=colors['mlred'], fontweight='bold')

# 2. Distance concentration
ax2 = plt.subplot(2, 3, 2)
np.random.seed(42)
dims_to_test = [2, 10, 50, 100]
for d in dims_to_test:
    # Generate random points
    points = np.random.randn(1000, d)
    # Calculate pairwise distances
    distances = []
    for i in range(100):
        for j in range(i+1, 100):
            distances.append(np.linalg.norm(points[i] - points[j]))
    
    # Plot histogram
    ax2.hist(distances, bins=30, alpha=0.5, label=f'{d}D', density=True)

ax2.set_xlabel('Distance Between Points', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('All Points Become Equally Distant', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Visualization of 2D vs 3D vs higher
ax3 = plt.subplot(2, 3, 3, projection='3d')
# Create sphere points
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot sphere with most volume at surface
ax3.plot_surface(x*0.99, y*0.99, z*0.99, alpha=0.2, color=colors['mlblue'])
ax3.plot_surface(x, y, z, alpha=0.8, color=colors['mlred'])
ax3.set_title('In High Dimensions:\nEverything is on the Surface', fontsize=12, fontweight='bold')
ax3.set_box_aspect([1,1,1])
ax3.axis('off')

# 4. Nearest neighbor becomes meaningless
ax4 = plt.subplot(2, 3, 4)
dims = np.array([2, 5, 10, 20, 50, 100])
# Ratio of nearest to farthest neighbor distance
ratio = 1 - 1/np.sqrt(dims)
ax4.plot(dims, ratio, 'o-', color=colors['mlpurple'], linewidth=3, markersize=10)
ax4.set_xlabel('Number of Dimensions', fontsize=11)
ax4.set_ylabel('Nearest/Farthest Distance Ratio', fontsize=11)
ax4.set_title('Nearest Neighbors Aren\'t "Near"', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1])
ax4.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
ax4.text(50, 0.85, 'All points equally far!', fontsize=10, color=colors['mlred'], fontweight='bold')

# 5. Sample size requirements
ax5 = plt.subplot(2, 3, 5)
dimensions = np.arange(1, 21)
# Samples needed for 0.1 resolution
samples_needed = 10**dimensions
ax5.semilogy(dimensions, samples_needed, color=colors['mlorange'], linewidth=3, marker='o')
ax5.set_xlabel('Number of Dimensions', fontsize=11)
ax5.set_ylabel('Samples Needed (log scale)', fontsize=11)
ax5.set_title('Data Requirements Explode', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=1e6, color='red', linestyle='--', alpha=0.5)
ax5.text(10, 1e7, '1 million', fontsize=10, color=colors['mlred'])
ax5.text(15, 1e16, 'More than atoms\nin universe!', fontsize=10, color=colors['mlred'], fontweight='bold')

# 6. Clustering becomes harder
ax6 = plt.subplot(2, 3, 6)
# Show how clustering quality degrades
dims = [2, 5, 10, 20, 50, 100]
silhouette_scores = [0.85, 0.72, 0.55, 0.35, 0.15, 0.05]
ax6.plot(dims, silhouette_scores, 'o-', color=colors['mlgreen'], linewidth=3, markersize=10)
ax6.fill_between(dims, silhouette_scores, alpha=0.3, color=colors['mlgreen'])
ax6.set_xlabel('Number of Dimensions', fontsize=11)
ax6.set_ylabel('Clustering Quality (Silhouette)', fontsize=11)
ax6.set_title('Clustering Gets Harder', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 1])
ax6.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
ax6.text(50, 0.55, 'Poor quality', fontsize=10, color=colors['mlorange'])
ax6.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
ax6.text(20, 0.75, 'Good quality', fontsize=10, color=colors['mlgreen'])

# Main title
fig.suptitle('The Curse of Dimensionality: Why High-Dimensional Spaces Are Strange', 
             fontsize=16, fontweight='bold', y=1.02)

# Add bottom text
fig.text(0.5, -0.02, 'Innovation data has 100+ dimensions - that\'s why we need specialized ML algorithms!', 
         ha='center', fontsize=12, color=colors['mlpurple'], fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('curse_of_dimensionality.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('curse_of_dimensionality.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Curse of dimensionality chart created successfully!")