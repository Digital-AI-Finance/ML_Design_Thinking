import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(14, 10))

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
X, y = make_blobs(n_samples=500, centers=4, n_features=2, 
                  cluster_std=0.7, center_box=(-10, 10), random_state=42)

# Test different numbers of clusters
cluster_numbers = [2, 3, 4, 5, 6, 7]

for idx, n_clusters in enumerate(cluster_numbers):
    ax = axes[idx // 3, idx % 3]
    
    # Fit the clusterer
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X)
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    y_lower = 10
    
    # Plot silhouette scores for each cluster
    for i in range(n_clusters):
        # Aggregate silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
    
    # Add vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)
    ax.text(silhouette_avg, ax.get_ylim()[1] * 0.95, f'Avg: {silhouette_avg:.2f}', 
            ha='center', fontsize=9, color='red', fontweight='bold')
    
    ax.set_xlabel('Silhouette Coefficient', fontsize=10)
    ax.set_ylabel('Cluster Label', fontsize=10)
    
    # Highlight the optimal case
    if n_clusters == 4:
        ax.set_title(f'k={n_clusters} (OPTIMAL: Score={silhouette_avg:.2f})', 
                    fontsize=11, fontweight='bold', color=colors['mlgreen'])
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(colors['mlgreen'])
            spine.set_linewidth(2)
    else:
        ax.set_title(f'k={n_clusters} (Score={silhouette_avg:.2f})', fontsize=11)
    
    # Set x limits
    ax.set_xlim([-0.2, 1])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    # Add interpretation guide for first subplot
    if idx == 0:
        ax.text(0.8, 0.9, 'Wide bars = good\nNarrow = poor', 
               transform=ax.transAxes, fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Main title
fig.suptitle('Silhouette Analysis: Cluster Quality Validation', 
            fontsize=16, fontweight='bold')

# Add interpretation guide at the bottom
fig.text(0.5, 0.01, 
         'Higher average silhouette score (red line) = better clustering. '
         'Wide, uniform bars indicate well-separated clusters.',
         ha='center', fontsize=11, color=colors['mlpurple'], fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('silhouette_score.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('silhouette_score.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Silhouette score chart created successfully!")