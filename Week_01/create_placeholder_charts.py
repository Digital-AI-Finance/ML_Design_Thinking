import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Set random seed
np.random.seed(42)

# Define colors
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

# Create too_few_clusters.pdf
fig, ax = plt.subplots(figsize=(6, 6))
X, y_true = make_blobs(n_samples=200, centers=4, n_features=2, cluster_std=0.8, random_state=42)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)
for i in range(2):
    mask = labels == i
    ax.scatter(X[mask, 0], X[mask, 1], alpha=0.6, s=50)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           marker='*', s=300, c='red', edgecolors='black', linewidth=2)
ax.set_title('Too Few Clusters (K=2)', fontsize=14, fontweight='bold', color=colors['mlred'])
ax.set_aspect('equal')
plt.savefig('charts/too_few_clusters.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Create just_right_clusters.pdf
fig, ax = plt.subplots(figsize=(6, 6))
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
for i in range(4):
    mask = labels == i
    ax.scatter(X[mask, 0], X[mask, 1], alpha=0.6, s=50)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           marker='*', s=300, c='red', edgecolors='black', linewidth=2)
ax.set_title('Just Right (K=4)', fontsize=14, fontweight='bold', color=colors['mlgreen'])
ax.set_aspect('equal')
plt.savefig('charts/just_right_clusters.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Create too_many_clusters.pdf
fig, ax = plt.subplots(figsize=(6, 6))
kmeans = KMeans(n_clusters=15, random_state=42)
labels = kmeans.fit_predict(X)
for i in range(15):
    mask = labels == i
    ax.scatter(X[mask, 0], X[mask, 1], alpha=0.6, s=50)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           marker='*', s=100, c='red', edgecolors='black', linewidth=1)
ax.set_title('Too Many Clusters (K=15)', fontsize=14, fontweight='bold', color=colors['mlorange'])
ax.set_aspect('equal')
plt.savefig('charts/too_many_clusters.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Placeholder charts created successfully!")