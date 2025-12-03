import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure showing K-means steps
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.ravel()

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd'
}

# Generate sample data
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2,
                      cluster_std=1.0, random_state=42)

# Step 1: Initial data
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=30)
ax.set_title('Step 1: Raw Innovation Data\n(300 ideas, no labels)', fontsize=11, fontweight='bold')
ax.set_xlabel('Innovation Dimension 1')
ax.set_ylabel('Innovation Dimension 2')
ax.text(0.5, -0.15, '❓ How many groups?', transform=ax.transAxes, 
        ha='center', fontsize=9, color=colors['mlpurple'])

# Step 2: Random initialization
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=30)
# Random initial centers
initial_centers = X[np.random.choice(X.shape[0], 3, replace=False)]
ax.scatter(initial_centers[:, 0], initial_centers[:, 1], 
          c=[colors['mlred'], colors['mlgreen'], colors['mlblue']], 
          s=200, marker='*', edgecolors='black', linewidth=2)
ax.set_title('Step 2: Initialize Centers\n(k=3 random starting points)', fontsize=11, fontweight='bold')
ax.set_xlabel('Innovation Dimension 1')
ax.set_ylabel('Innovation Dimension 2')
ax.text(0.5, -0.15, '🎯 Start with k=3 centers', transform=ax.transAxes,
        ha='center', fontsize=9, color=colors['mlpurple'])

# Step 3: First assignment
ax = axes[2]
kmeans_1 = KMeans(n_clusters=3, init=initial_centers, n_init=1, max_iter=1, random_state=42)
labels_1 = kmeans_1.fit_predict(X)
for i in range(3):
    mask = labels_1 == i
    ax.scatter(X[mask, 0], X[mask, 1], 
              c=[colors['mlred'], colors['mlgreen'], colors['mlblue']][i],
              alpha=0.6, s=30)
ax.scatter(kmeans_1.cluster_centers_[:, 0], kmeans_1.cluster_centers_[:, 1],
          c='black', s=200, marker='*', edgecolors='white', linewidth=2)
ax.set_title('Step 3: Assign to Nearest Center\n(Each point finds its cluster)', fontsize=11, fontweight='bold')
ax.set_xlabel('Innovation Dimension 1')
ax.set_ylabel('Innovation Dimension 2')
ax.text(0.5, -0.15, '📊 Distance-based assignment', transform=ax.transAxes,
        ha='center', fontsize=9, color=colors['mlpurple'])

# Step 4: Update centers
ax = axes[3]
kmeans_2 = KMeans(n_clusters=3, init=initial_centers, n_init=1, max_iter=2, random_state=42)
labels_2 = kmeans_2.fit_predict(X)
for i in range(3):
    mask = labels_2 == i
    ax.scatter(X[mask, 0], X[mask, 1],
              c=[colors['mlred'], colors['mlgreen'], colors['mlblue']][i],
              alpha=0.6, s=30)
# Show old centers as hollow
ax.scatter(kmeans_1.cluster_centers_[:, 0], kmeans_1.cluster_centers_[:, 1],
          c='gray', s=100, marker='*', alpha=0.3)
# Show new centers
ax.scatter(kmeans_2.cluster_centers_[:, 0], kmeans_2.cluster_centers_[:, 1],
          c='black', s=200, marker='*', edgecolors='white', linewidth=2)
# Draw arrows showing movement
for old, new in zip(kmeans_1.cluster_centers_, kmeans_2.cluster_centers_):
    ax.annotate('', xy=new, xytext=old,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.5))
ax.set_title('Step 4: Update Centers\n(Move to cluster mean)', fontsize=11, fontweight='bold')
ax.set_xlabel('Innovation Dimension 1')
ax.set_ylabel('Innovation Dimension 2')
ax.text(0.5, -0.15, '📍 Centers adjust position', transform=ax.transAxes,
        ha='center', fontsize=9, color=colors['mlpurple'])

# Step 5: Iterate (show iteration 5)
ax = axes[4]
kmeans_5 = KMeans(n_clusters=3, init=initial_centers, n_init=1, max_iter=5, random_state=42)
labels_5 = kmeans_5.fit_predict(X)
for i in range(3):
    mask = labels_5 == i
    ax.scatter(X[mask, 0], X[mask, 1],
              c=[colors['mlred'], colors['mlgreen'], colors['mlblue']][i],
              alpha=0.6, s=30)
ax.scatter(kmeans_5.cluster_centers_[:, 0], kmeans_5.cluster_centers_[:, 1],
          c='black', s=200, marker='*', edgecolors='white', linewidth=2)
ax.set_title('Step 5: Repeat Until Stable\n(After 5 iterations)', fontsize=11, fontweight='bold')
ax.set_xlabel('Innovation Dimension 1')
ax.set_ylabel('Innovation Dimension 2')
ax.text(0.5, -0.15, '🔄 Keep refining', transform=ax.transAxes,
        ha='center', fontsize=9, color=colors['mlpurple'])

# Step 6: Final convergence
ax = axes[5]
kmeans_final = KMeans(n_clusters=3, random_state=42)
labels_final = kmeans_final.fit_predict(X)
for i in range(3):
    mask = labels_final == i
    ax.scatter(X[mask, 0], X[mask, 1],
              c=[colors['mlred'], colors['mlgreen'], colors['mlblue']][i],
              alpha=0.6, s=30, label=f'Cluster {i+1}')
ax.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
          c='black', s=200, marker='*', edgecolors='white', linewidth=2)
# Draw decision boundaries (Voronoi regions)
from scipy.spatial import Voronoi
vor = Voronoi(kmeans_final.cluster_centers_)
# Add circles to show cluster boundaries
for center in kmeans_final.cluster_centers_:
    circle = plt.Circle(center, 3.5, fill=False, edgecolor='black', 
                       linestyle='--', alpha=0.3, linewidth=1)
    ax.add_patch(circle)
ax.set_title('Step 6: Convergence!\n(Stable clusters found)', fontsize=11, fontweight='bold')
ax.set_xlabel('Innovation Dimension 1')
ax.set_ylabel('Innovation Dimension 2')
ax.legend(loc='upper right', fontsize=8)
ax.text(0.5, -0.15, '✓ Algorithm complete!', transform=ax.transAxes,
        ha='center', fontsize=9, color=colors['mlgreen'], fontweight='bold')

# Main title
fig.suptitle('K-Means Algorithm: Step-by-Step Innovation Clustering', 
            fontsize=16, fontweight='bold')

# Add bottom annotation
fig.text(0.5, 0.02, 
         'Each iteration refines the innovation groups until they stabilize (typically 5-10 iterations)',
         ha='center', fontsize=11, color=colors['mlpurple'], fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('kmeans_animation.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('kmeans_animation.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("K-means animation chart created successfully!")