import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

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
    'mlbrown': '#8c564b',
    'mlgray': '#7f7f7f'
}

# Generate diverse innovation clusters
np.random.seed(42)

# Create 5 distinct innovation patterns
patterns = []
pattern_labels = []
pattern_names = [
    'Digital Platform\nInnovations',
    'Sustainable Tech\nSolutions',
    'AI/ML\nApplications',
    'Customer Experience\nEnhancements',
    'Process\nOptimizations'
]

# Pattern 1: Digital Platforms (dense cluster)
center1 = [2, 7]
X1 = np.random.randn(200, 2) * 0.6 + center1
patterns.append(X1)
pattern_labels.extend([0] * 200)

# Pattern 2: Sustainable Tech (elongated)
X2_base = np.random.randn(150, 2)
X2_base[:, 0] *= 2.5
X2_base[:, 1] *= 0.7
theta = np.radians(-30)
rotation = np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])
X2 = X2_base @ rotation + [8, 6]
patterns.append(X2)
pattern_labels.extend([1] * 150)

# Pattern 3: AI/ML Applications (two sub-clusters)
X3a = np.random.randn(80, 2) * 0.5 + [5, 2]
X3b = np.random.randn(70, 2) * 0.5 + [6.5, 3]
X3 = np.vstack([X3a, X3b])
patterns.append(X3)
pattern_labels.extend([2] * 150)

# Pattern 4: Customer Experience (ring shape)
angles = np.random.uniform(0, 2*np.pi, 120)
radius = np.random.normal(2, 0.3, 120)
X4 = np.column_stack([radius * np.cos(angles) + 2, 
                      radius * np.sin(angles) + 3])
patterns.append(X4)
pattern_labels.extend([3] * 120)

# Pattern 5: Process Optimizations (scattered)
X5 = np.random.randn(100, 2) * 1.2 + [8, 2]
patterns.append(X5)
pattern_labels.extend([4] * 100)

# Combine all patterns
X_all = np.vstack(patterns)
y_all = np.array(pattern_labels)

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_all)

# Plot each cluster with different colors and styles
for i in range(5):
    mask = kmeans_labels == i
    ax.scatter(X_all[mask, 0], X_all[mask, 1], 
              c=list(colors.values())[i], 
              alpha=0.6, s=40, 
              label=pattern_names[i],
              edgecolors='white', linewidth=0.5)

# Add cluster centers
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], 
          c='black', s=300, marker='*', 
          edgecolors='white', linewidth=2,
          label='Innovation Hubs', zorder=10)

# Add pattern boundaries (visual hints)
for i, center in enumerate(centers):
    # Calculate cluster radius based on points
    mask = kmeans_labels == i
    cluster_points = X_all[mask]
    radius = np.percentile(np.linalg.norm(cluster_points - center, axis=1), 80)
    
    circle = Circle(center, radius, 
                   fill=False, 
                   edgecolor=list(colors.values())[i], 
                   linewidth=2, alpha=0.3, linestyle='--')
    ax.add_patch(circle)

# Add annotations for key insights
ax.annotate('High Density:\nMature Market', 
           xy=center1, xytext=(center1[0]-2, center1[1]+1.5),
           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
           fontsize=9, color='gray', style='italic')

ax.annotate('Bridged Clusters:\nConverging Tech', 
           xy=[5.75, 2.5], xytext=(5.75, 0.5),
           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
           fontsize=9, color='gray', style='italic')

ax.annotate('Dispersed:\nEmerging Field', 
           xy=[8, 2], xytext=(10, 1),
           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
           fontsize=9, color='gray', style='italic')

# Add title and labels
ax.set_xlabel('Innovation Complexity Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Market Impact Potential', fontsize=12, fontweight='bold')
ax.set_title('Innovation Patterns Discovered Through Clustering', fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.2)
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 9)

# Add legend
legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                  title='Innovation Categories', title_fontsize=11)
legend.get_title().set_fontweight('bold')

# Add insight boxes
insight_text1 = "Each cluster represents\na distinct innovation\nopportunity space"
bbox1 = FancyBboxPatch((9, 6.5), 2, 1.5, 
                       boxstyle="round,pad=0.1", 
                       facecolor=colors['mlpurple'], alpha=0.1,
                       edgecolor=colors['mlpurple'], linewidth=1)
ax.add_patch(bbox1)
ax.text(10, 7.25, insight_text1, fontsize=9, ha='center', va='center',
       color=colors['mlpurple'], fontweight='bold')

insight_text2 = "Distance between\nclusters indicates\ninnovation gaps"
bbox2 = FancyBboxPatch((9, 4.5), 2, 1.5, 
                       boxstyle="round,pad=0.1", 
                       facecolor=colors['mlorange'], alpha=0.1,
                       edgecolor=colors['mlorange'], linewidth=1)
ax.add_patch(bbox2)
ax.text(10, 5.25, insight_text2, fontsize=9, ha='center', va='center',
       color=colors['mlorange'], fontweight='bold')

# Add bottom message
fig.text(0.5, 0.02, 
         'Machine Learning reveals natural groupings in innovation data, enabling targeted strategy development',
         ha='center', fontsize=11, color=colors['mlpurple'], fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('innovation_patterns_visual.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('innovation_patterns_visual.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Innovation patterns visual chart created successfully!")