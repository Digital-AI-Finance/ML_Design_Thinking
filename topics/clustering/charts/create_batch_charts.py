"""
Batch creation of essential charts for Week 1 slides
Creates multiple charts to restore visual content
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

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

np.random.seed(42)

# Chart 1: Innovation Discovery
print("Creating innovation_discovery.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))
# Create scattered innovation points
n_points = 1000
X_discovery = np.random.randn(n_points, 2) * 3
# Color by distance from origin (as proxy for innovation potential)
distances = np.linalg.norm(X_discovery, axis=1)
scatter = ax.scatter(X_discovery[:, 0], X_discovery[:, 1], 
                    c=distances, cmap='viridis', s=20, alpha=0.6)
ax.set_title('Innovation Discovery: Where Do We Start?', fontsize=14, fontweight='bold')
ax.set_xlabel('Technology Readiness')
ax.set_ylabel('Market Opportunity')
plt.colorbar(scatter, label='Innovation Potential Score')
# Add question marks to suggest exploration
for _ in range(5):
    x, y = np.random.uniform(-8, 8, 2)
    ax.text(x, y, '?', fontsize=20, color='red', alpha=0.3)
plt.savefig('innovation_discovery.pdf', dpi=300, bbox_inches='tight')
plt.savefig('innovation_discovery.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Innovation Feature Complexity
print("Creating innovation_feature_complexity.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))
# Show high-dimensional data projected to 2D
n_features = 50
n_samples = 500
X_complex = np.random.randn(n_samples, n_features)
# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_complex)
# Create clusters in high-D space
kmeans_hd = KMeans(n_clusters=4, random_state=42)
labels = kmeans_hd.fit_predict(X_complex)
# Plot reduced data with cluster colors
for i in range(4):
    mask = labels == i
    ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
              c=list(colors.values())[i], alpha=0.6, s=30,
              label=f'Pattern {i+1}')
ax.set_title(f'Innovation Complexity: {n_features} Features Reduced to 2D View', 
            fontsize=14, fontweight='bold')
ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.legend()
# Add annotation
ax.text(0.95, 0.95, f'Original: {n_features}D\nVisualized: 2D\nInformation preserved: {sum(pca.explained_variance_ratio_):.1%}',
       transform=ax.transAxes, ha='right', va='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.savefig('innovation_feature_complexity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('innovation_feature_complexity.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Current Reality Visual
print("Creating current_reality_visual.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Left: Traditional approach
ax = axes[0]
X_trad = np.random.randn(50, 2) * 2
ax.scatter(X_trad[:, 0], X_trad[:, 1], c='gray', s=50, alpha=0.5)
ax.set_title('Traditional Approach: Manual Analysis', fontsize=12, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.text(0.5, 0.05, '50 ideas analyzed\nWeeks of work\nSubjective grouping',
       transform=ax.transAxes, ha='center', fontsize=10, color=colors['mlred'])
# Right: ML approach
ax = axes[1]
X_ml, _ = make_blobs(n_samples=5000, centers=5, n_features=2, random_state=42)
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_ml)
for i in range(5):
    mask = labels == i
    ax.scatter(X_ml[mask, 0], X_ml[mask, 1],
              c=list(colors.values())[i], alpha=0.3, s=10)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
          c='black', s=200, marker='*', edgecolors='white', linewidth=2)
ax.set_title('ML Approach: Automated Discovery', fontsize=12, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.text(0.5, 0.05, '5000 ideas analyzed\nMinutes of processing\nData-driven patterns',
       transform=ax.transAxes, ha='center', fontsize=10, color=colors['mlgreen'])
fig.suptitle('The Current Reality: Scale Challenge in Innovation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('current_reality_visual.pdf', dpi=300, bbox_inches='tight')
plt.savefig('current_reality_visual.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: DBSCAN Shapes
print("Creating dbscan_shapes.pdf...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
datasets = [
    make_moons(n_samples=300, noise=0.05, random_state=42),
    make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42),
    make_blobs(n_samples=300, centers=3, random_state=42)
]
titles = ['Non-linear Patterns', 'Nested Structures', 'Traditional Clusters']
for idx, (X, y) in enumerate(datasets):
    ax = axes[idx]
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.3 if idx < 2 else 1.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    # Plot
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            # Noise points
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c='gray', s=20, alpha=0.3, label='Outliers')
        else:
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.7, label=f'Cluster {k+1}')
    ax.set_title(titles[idx], fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if idx == 0:
        ax.legend(loc='best', fontsize=8)
fig.suptitle('DBSCAN: Finding Patterns of Any Shape', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dbscan_shapes.pdf', dpi=300, bbox_inches='tight')
plt.savefig('dbscan_shapes.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Chaos to Clarity
print("Creating chaos_to_clarity.pdf...")
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
# Generate base data
X_base = np.random.randn(1000, 2) * 3
# Stage 1: Chaos
ax = axes[0]
ax.scatter(X_base[:, 0], X_base[:, 1], c='gray', alpha=0.2, s=5)
ax.set_title('Stage 1: Data Chaos', fontsize=12, fontweight='bold')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
# Stage 2: Processing
ax = axes[1]
# Show data with emerging patterns (use color gradient)
distances = np.linalg.norm(X_base, axis=1)
ax.scatter(X_base[:, 0], X_base[:, 1], c=distances, cmap='coolwarm', alpha=0.4, s=10)
ax.set_title('Stage 2: ML Processing', fontsize=12, fontweight='bold')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
# Add processing arrows
for _ in range(5):
    start = np.random.randn(2) * 4
    end = np.random.randn(2) * 2
    ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle='->', color='purple', alpha=0.3, lw=1))
# Stage 3: Clarity
ax = axes[2]
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_base)
for i in range(5):
    mask = labels == i
    ax.scatter(X_base[mask, 0], X_base[mask, 1],
              c=list(colors.values())[i], alpha=0.6, s=10)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
          c='black', s=150, marker='*', edgecolors='white', linewidth=2)
ax.set_title('Stage 3: Pattern Clarity', fontsize=12, fontweight='bold')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
fig.suptitle('From Chaos to Clarity: The ML Journey', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('chaos_to_clarity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('chaos_to_clarity.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 6: Dendrogram Example
print("Creating dendrogram_example.pdf...")
fig, ax = plt.subplots(figsize=(14, 8))
# Generate hierarchical data
X_hier, _ = make_blobs(n_samples=50, centers=4, n_features=2, 
                       cluster_std=0.5, random_state=42)
# Create linkage matrix
Z = linkage(X_hier, method='ward')
# Plot dendrogram
dendrogram(Z, ax=ax, color_threshold=10, above_threshold_color='gray')
ax.set_title('Hierarchical Clustering: Innovation Evolution Tree', fontsize=14, fontweight='bold')
ax.set_xlabel('Innovation ID', fontsize=12)
ax.set_ylabel('Distance (Dissimilarity)', fontsize=12)
# Add horizontal line for cut
ax.axhline(y=10, color=colors['mlred'], linestyle='--', linewidth=2)
ax.text(25, 11, 'Cut here for 4 clusters', ha='center', fontsize=10, 
       color=colors['mlred'], fontweight='bold')
plt.savefig('dendrogram_example.pdf', dpi=300, bbox_inches='tight')
plt.savefig('dendrogram_example.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll charts created successfully!")
print("Created files:")
print("- innovation_discovery.pdf/png")
print("- innovation_feature_complexity.pdf/png")
print("- current_reality_visual.pdf/png")
print("- dbscan_shapes.pdf/png")
print("- chaos_to_clarity.pdf/png")
print("- dendrogram_example.pdf/png")