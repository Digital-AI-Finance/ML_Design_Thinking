import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Set random seed
np.random.seed(42)

# Define colors
colors_palette = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd'
}

# Create figure with 4 subplots for different applications
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Customer Segmentation (Top Left)
ax1 = axes[0, 0]
X1, y1 = make_blobs(n_samples=300, centers=3, n_features=2, 
                    cluster_std=0.8, random_state=42)
kmeans1 = KMeans(n_clusters=3, random_state=42)
labels1 = kmeans1.fit_predict(X1)

for i in range(3):
    mask = labels1 == i
    ax1.scatter(X1[mask, 0], X1[mask, 1], alpha=0.6, s=50)
ax1.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1],
           marker='*', s=300, c='red', edgecolors='black', linewidth=2)
ax1.set_title('Customer Segmentation', fontsize=14, fontweight='bold', 
              color=colors_palette['mlblue'])
ax1.set_xlabel('Purchase Frequency')
ax1.set_ylabel('Average Order Value')

# 2. Document Clustering (Top Right)
ax2 = axes[0, 1]
# Simulate document clusters
np.random.seed(43)
n_docs = 400
doc_clusters = []
centers = [(2, 3), (5, 7), (8, 2), (3, 8)]
for center in centers:
    cluster = np.random.normal(center, 0.8, (n_docs//4, 2))
    doc_clusters.append(cluster)
X2 = np.vstack(doc_clusters)
kmeans2 = KMeans(n_clusters=4, random_state=42)
labels2 = kmeans2.fit_predict(X2)

for i in range(4):
    mask = labels2 == i
    ax2.scatter(X2[mask, 0], X2[mask, 1], alpha=0.6, s=30)
ax2.set_title('Document Categorization', fontsize=14, fontweight='bold',
              color=colors_palette['mlorange'])
ax2.set_xlabel('Topic 1 Score')
ax2.set_ylabel('Topic 2 Score')

# 3. Image Segmentation (Bottom Left)
ax3 = axes[1, 0]
# Create synthetic image-like data
x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)
X_grid, Y_grid = np.meshgrid(x, y)
Z = np.sin(X_grid) * np.cos(Y_grid) + np.random.normal(0, 0.1, X_grid.shape)
im = ax3.contourf(X_grid, Y_grid, Z, levels=5, cmap='viridis', alpha=0.7)
ax3.set_title('Image Segmentation', fontsize=14, fontweight='bold',
              color=colors_palette['mlgreen'])
ax3.set_xlabel('Pixel X')
ax3.set_ylabel('Pixel Y')

# 4. Anomaly Detection (Bottom Right)
ax4 = axes[1, 1]
# Normal data
normal_data = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 300)
# Anomalies
anomalies = np.random.uniform(0, 10, (20, 2))
ax4.scatter(normal_data[:, 0], normal_data[:, 1], alpha=0.6, s=30,
           color=colors_palette['mlblue'], label='Normal')
ax4.scatter(anomalies[:, 0], anomalies[:, 1], alpha=0.8, s=50,
           color=colors_palette['mlred'], marker='^', label='Anomaly')
ax4.set_title('Anomaly Detection', fontsize=14, fontweight='bold',
              color=colors_palette['mlred'])
ax4.set_xlabel('Feature 1')
ax4.set_ylabel('Feature 2')
ax4.legend()

plt.suptitle('Real-World Clustering Applications', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

# Save the figure
plt.savefig('charts/clustering_examples.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/clustering_examples.png', dpi=150, bbox_inches='tight')
print("Clustering examples chart created successfully!")
plt.close()