import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.patches as patches

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left subplot: Overwhelming data
ax1 = axes[0]

# Generate random points to represent reviews
n_reviews = 10000
x = np.random.randn(n_reviews) * 3
y = np.random.randn(n_reviews) * 2

# Plot as tiny dots
ax1.scatter(x, y, alpha=0.1, s=1, c='gray')
ax1.set_title('10,000 Product Reviews', fontsize=16, fontweight='bold')
ax1.set_xlabel('Feature 1 (e.g., length)', fontsize=12)
ax1.set_ylabel('Feature 2 (e.g., sentiment)', fontsize=12)

# Add text overlay
ax1.text(0, 0, '?', fontsize=200, ha='center', va='center', 
         alpha=0.3, color='red', fontweight='bold')
ax1.text(0, -3.5, 'Where do we even start?', fontsize=14, 
         ha='center', fontweight='bold', color='darkred')

# Add statistics
stats_text = "Time to read all: 83 hours\nCost of manual analysis: $5,000\nInsights found manually: 2-3"
ax1.text(-4.5, 3, stats_text, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Right subplot: After clustering
ax2 = axes[1]

# Generate clustered data
X, y_true = make_blobs(n_samples=1000, centers=5, n_features=2, 
                      cluster_std=0.6, random_state=42)

# Apply K-means
kmeans = KMeans(n_clusters=5, random_state=42)
y_pred = kmeans.fit_predict(X)

# Plot clusters
colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd']
cluster_names = ['Happy Users\n(35%)', 'Frustrated\n(20%)', 
                'Feature Requests\n(15%)', 'Bug Reports\n(10%)', 
                'Gift Buyers\n(20%)']

for i in range(5):
    mask = y_pred == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
               alpha=0.6, s=20, label=cluster_names[i])

# Plot centroids
ax2.scatter(kmeans.cluster_centers_[:, 0], 
           kmeans.cluster_centers_[:, 1], 
           c='black', s=200, alpha=0.8, marker='*', 
           edgecolors='white', linewidth=2)

ax2.set_title('After K-means Clustering', fontsize=16, fontweight='bold')
ax2.set_xlabel('Feature 1 (normalized)', fontsize=12)
ax2.set_ylabel('Feature 2 (normalized)', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)

# Add statistics
stats_text2 = "Time with ML: 4 hours\nCost: $50\nSegments found: 5\nInsights: Dozens"
ax2.text(-3, 5, stats_text2, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Main title
plt.suptitle('From Chaos to Clarity with K-means Clustering', 
            fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('clustering_demo.pdf', dpi=300, bbox_inches='tight')
plt.savefig('clustering_demo.png', dpi=150, bbox_inches='tight')

print("Chart saved: clustering_demo.pdf")
print("\nClustering Statistics:")
print(f"Total reviews processed: {n_reviews}")
print(f"Clusters discovered: 5")
print(f"Time reduction: 95.2% (83 hours to 4 hours)")
print(f"Cost reduction: 99% ($5,000 to $50)")
print(f"Scale increase: 100x (100 to 10,000 reviews)")