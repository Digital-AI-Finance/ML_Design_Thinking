"""
Create k-means clustering animation showing user segments emerging
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import seaborn as sns

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Generate synthetic user data (5 natural clusters)
n_samples = 500
n_clusters = 5

# Create user segments with different characteristics
X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters,
                      n_features=2, cluster_std=0.6,
                      random_state=42)

# Scale and shift for better visualization
X[:, 0] = X[:, 0] * 1.5 + 5
X[:, 1] = X[:, 1] * 1.2 + 5

# Define colors for clusters
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
segment_names = ['Power Users', 'Beginners', 'Social Users', 'Price-Sensitive', 'Quality-Focused']

# Create figure with 4 subplots showing progression
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Step 1: Raw data
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=30)
ax.set_title('Step 1: 500 Users - No Segments Visible', fontsize=16, fontweight='bold')
ax.set_xlabel('Engagement Score')
ax.set_ylabel('Feature Usage')
ax.text(0.5, 0.95, '1000 individual needs?', transform=ax.transAxes,
        ha='center', fontsize=14, style='italic', color='red')

# Step 2: Initial centers
ax = axes[1]
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1, max_iter=1)
kmeans.fit(X)
ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3, s=30)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
          c='red', s=200, marker='X', edgecolors='black', linewidth=2)
ax.set_title('Step 2: Algorithm Starts - Random Centers', fontsize=16, fontweight='bold')
ax.set_xlabel('Engagement Score')
ax.set_ylabel('Feature Usage')
ax.text(0.5, 0.95, 'K-means picks initial centers', transform=ax.transAxes,
        ha='center', fontsize=14, style='italic', color='blue')

# Step 3: Partial convergence
ax = axes[2]
kmeans_partial = KMeans(n_clusters=n_clusters, random_state=42, n_init=1, max_iter=3)
kmeans_partial.fit(X)
labels_partial = kmeans_partial.labels_

for i in range(n_clusters):
    mask = labels_partial == i
    ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=30)
ax.scatter(kmeans_partial.cluster_centers_[:, 0], kmeans_partial.cluster_centers_[:, 1],
          c='black', s=200, marker='X', edgecolors='white', linewidth=2)
ax.set_title('Step 3: Clusters Forming - Patterns Emerge', fontsize=16, fontweight='bold')
ax.set_xlabel('Engagement Score')
ax.set_ylabel('Feature Usage')
ax.text(0.5, 0.95, 'Users group by similarity', transform=ax.transAxes,
        ha='center', fontsize=14, style='italic', color='green')

# Step 4: Final clusters
ax = axes[3]
kmeans_final = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_final.fit(X)
labels_final = kmeans_final.labels_

for i in range(n_clusters):
    mask = labels_final == i
    ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.8, s=40,
              label=f'{segment_names[i]} ({np.sum(mask)} users)')
    
    # Add cluster center
    ax.scatter(kmeans_final.cluster_centers_[i, 0], 
              kmeans_final.cluster_centers_[i, 1],
              c='black', s=200, marker='X', edgecolors='white', linewidth=2)
    
    # Add segment label near center
    ax.annotate(segment_names[i], 
               xy=(kmeans_final.cluster_centers_[i, 0], 
                   kmeans_final.cluster_centers_[i, 1]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))

ax.set_title('Step 4: 5 Distinct User Segments Discovered!', fontsize=16, fontweight='bold')
ax.set_xlabel('Engagement Score')
ax.set_ylabel('Feature Usage')
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

# Overall title
fig.suptitle('K-Means Clustering: From Chaos to Clarity', fontsize=20, fontweight='bold', y=1.02)

# Add insights box
insights_text = (
    "Innovation Opportunities:\n"
    "• Power Users → Advanced features\n"
    "• Beginners → Better onboarding\n"
    "• Social Users → Community tools\n"
    "• Price-Sensitive → Freemium model\n"
    "• Quality-Focused → Premium tier"
)
fig.text(0.5, -0.02, insights_text, ha='center', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()

# Save the figure
plt.savefig('clustering_animation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('clustering_animation.png', dpi=150, bbox_inches='tight')

print("Clustering animation visualization created successfully!")
print(f"Total samples: {n_samples}")
print(f"Clusters found: {n_clusters}")