"""Clustering ESG Groups - K-means clustering of ESG strategies"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

np.random.seed(42)
fig, ax = plt.subplots(figsize=(10, 6))

# Generate sample ESG data points (2000 filtered ideas)
n_samples = 400  # Reduced for visibility

# Create 8 distinct clusters
cluster_data = [
    {'center': (0.8, 0.7), 'n': 50},   # Best-in-class
    {'center': (0.2, 0.8), 'n': 50},   # Exclusionary
    {'center': (0.9, 0.3), 'n': 50},   # Impact-first
    {'center': (0.5, 0.9), 'n': 50},   # ESG momentum
    {'center': (0.3, 0.3), 'n': 50},   # Dividend focus
    {'center': (0.7, 0.5), 'n': 50},   # Growth ESG
    {'center': (0.4, 0.6), 'n': 50},   # Balanced
    {'center': (0.6, 0.2), 'n': 50},   # Value ESG
]

X = []
for cluster in cluster_data:
    cx, cy = cluster['center']
    x = np.random.normal(cx, 0.08, cluster['n'])
    y = np.random.normal(cy, 0.08, cluster['n'])
    X.extend(zip(x, y))

X = np.array(X)
X = np.clip(X, 0, 1)

# Apply K-means
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Plot with cluster colors
colors = plt.cm.Set2(np.linspace(0, 1, 8))
for i in range(8):
    mask = labels == i
    ax.scatter(X[mask, 0], X[mask, 1], s=30, c=[colors[i]], alpha=0.6, label=f'Cluster {i+1}')

# Plot centroids
ax.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='X', edgecolors='white', linewidth=2, zorder=5)

# Label clusters
cluster_names = ['Best-in-Class', 'Exclusionary', 'Impact-First', 'ESG Momentum',
                 'Dividend Focus', 'Growth ESG', 'Balanced', 'Value ESG']
for i, (cx, cy) in enumerate(centers):
    ax.annotate(cluster_names[i], (cx, cy), xytext=(5, 5), textcoords='offset points',
               fontsize=7, fontweight='bold', color='black')

ax.set_xlabel('Environmental Score', fontsize=10)
ax.set_ylabel('Social Score', fontsize=10)
ax.set_title('K-Means Clustering: 8 ESG Investment Approaches', fontsize=12, fontweight='bold', color='#e67e22')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("08_clustering_esg created")
