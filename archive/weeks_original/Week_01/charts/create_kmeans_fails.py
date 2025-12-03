import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(14, 10))

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlpink': '#e377c2'
}

np.random.seed(42)

# 1. Crescent Moons
ax = axes[0, 0]
X_moons, y_moons = make_moons(n_samples=300, noise=0.05)
ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', s=30, alpha=0.7)
ax.set_title('Crescent Shapes\n(Technology Evolution Chains)', fontsize=11, fontweight='bold')
ax.set_xlabel('Innovation Stage')
ax.set_ylabel('Complexity')
# Add failed K-means attempt overlay
kmeans_moons = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans_moons.fit_predict(X_moons)
# Draw wrong boundaries
for center in kmeans_moons.cluster_centers_:
    circle = plt.Circle(center, 0.7, fill=False, edgecolor='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.add_patch(circle)
ax.text(0, -1.5, 'K-means tries circles!', color='red', fontsize=9, ha='center', fontweight='bold')

# 2. Nested Circles
ax = axes[0, 1]
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5)
ax.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='plasma', s=30, alpha=0.7)
ax.set_title('Nested Circles\n(Core vs Peripheral Innovation)', fontsize=11, fontweight='bold')
ax.set_xlabel('Market Position')
ax.set_ylabel('Innovation Impact')
# Add failed K-means
kmeans_circles = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans_circles.fit_predict(X_circles)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(0, -1.5, 'K-means splits wrong!', color='red', fontsize=9, ha='center', fontweight='bold')

# 3. Elongated Clusters
ax = axes[0, 2]
# Create elongated clusters
X_elongated = []
y_elongated = []
# Cluster 1: horizontal
X1 = np.random.randn(150, 2)
X1[:, 0] *= 3
X1[:, 1] *= 0.3
X_elongated.append(X1 + [0, 1])
y_elongated.extend([0] * 150)
# Cluster 2: vertical
X2 = np.random.randn(150, 2)
X2[:, 0] *= 0.3
X2[:, 1] *= 3
X_elongated.append(X2 + [0, -1])
y_elongated.extend([1] * 150)
X_elongated = np.vstack(X_elongated)
y_elongated = np.array(y_elongated)
ax.scatter(X_elongated[:, 0], X_elongated[:, 1], c=y_elongated, cmap='coolwarm', s=30, alpha=0.7)
ax.set_title('Chain Patterns\n(Innovation Pipelines)', fontsize=11, fontweight='bold')
ax.set_xlabel('Development Stage')
ax.set_ylabel('Market Segment')
# Add wrong K-means boundaries
kmeans_elongated = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans_elongated.fit_predict(X_elongated)
for center in kmeans_elongated.cluster_centers_:
    circle = plt.Circle(center, 2, fill=False, edgecolor='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.add_patch(circle)

# 4. Different Densities
ax = axes[1, 0]
# Create clusters with different densities
X_dense = np.random.randn(200, 2) * 0.3 + [-3, 0]
X_sparse = np.random.randn(50, 2) * 1.5 + [3, 0]
X_density = np.vstack([X_dense, X_sparse])
y_density = np.array([0]*200 + [1]*50)
ax.scatter(X_density[:, 0], X_density[:, 1], c=y_density, cmap='spring', s=30, alpha=0.7)
ax.set_title('Varying Densities\n(Mature vs Emerging Markets)', fontsize=11, fontweight='bold')
ax.set_xlabel('Market Size')
ax.set_ylabel('Innovation Rate')
# K-means fails
kmeans_density = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans_density.fit_predict(X_density)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(0, -3.5, 'Splits by distance,\nnot density!', color='red', fontsize=9, ha='center', fontweight='bold')

# 5. Clusters with Outliers
ax = axes[1, 1]
# Create main clusters with outliers
X_main, y_main = make_blobs(n_samples=250, centers=2, n_features=2, cluster_std=0.5, random_state=42)
# Add outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(30, 2))
X_with_outliers = np.vstack([X_main, X_outliers])
y_with_outliers = np.array(list(y_main) + [2]*30)
colors_outliers = ['blue' if y == 0 else 'green' if y == 1 else 'red' for y in y_with_outliers]
ax.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=colors_outliers, s=30, alpha=0.7)
ax.set_title('Clusters with Outliers\n(Disruptive Innovations)', fontsize=11, fontweight='bold')
ax.set_xlabel('Technology Readiness')
ax.set_ylabel('Market Fit')
# K-means assigns outliers
kmeans_outliers = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans_outliers.fit_predict(X_with_outliers)
for center in kmeans_outliers.cluster_centers_:
    circle = plt.Circle(center, 1.5, fill=False, edgecolor='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.add_patch(circle)
ax.text(0, -5, 'Forces outliers\ninto groups!', color='red', fontsize=9, ha='center', fontweight='bold')

# 6. Connected Components
ax = axes[1, 2]
# Create bridge-connected clusters
X_bridge1 = np.random.randn(100, 2) * 0.5 + [-2, 0]
X_bridge2 = np.random.randn(100, 2) * 0.5 + [2, 0]
# Bridge points
X_bridge_connect = np.random.randn(20, 2) * [1.5, 0.2] + [0, 0]
X_bridges = np.vstack([X_bridge1, X_bridge2, X_bridge_connect])
y_bridges = np.array([0]*120 + [1]*100)
ax.scatter(X_bridges[:, 0], X_bridges[:, 1], c='purple', s=30, alpha=0.7)
ax.set_title('Connected Ecosystems\n(Platform Innovations)', fontsize=11, fontweight='bold')
ax.set_xlabel('Ecosystem Size')
ax.set_ylabel('Integration Level')
# K-means splits the bridge
kmeans_bridges = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans_bridges.fit_predict(X_bridges)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(0, -2, 'Breaks connections!', color='red', fontsize=9, ha='center', fontweight='bold')

# Main title
fig.suptitle('Where K-Means Fails: Non-Spherical Innovation Patterns', 
             fontsize=16, fontweight='bold')

# Add main message
fig.text(0.5, 0.02, 'Real innovation patterns rarely form perfect circles - that\'s why we need advanced clustering methods!', 
         ha='center', fontsize=12, color=colors['mlpurple'], fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('kmeans_fails.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('kmeans_fails.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("K-means failure patterns chart created successfully!")