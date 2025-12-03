import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd'
}

# Generate sample data with clear clusters
np.random.seed(42)
X, y_true = make_blobs(n_samples=500, centers=4, n_features=2,
                      cluster_std=0.8, center_box=(-10, 10), random_state=42)

# Calculate inertia for different k values
k_range = range(1, 11)
inertias = []
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    if k > 1:  # Silhouette score requires at least 2 clusters
        silhouettes.append(silhouette_score(X, kmeans.labels_))

# Left plot: Elbow Method
ax1 = axes[0]
ax1.plot(k_range, inertias, 'o-', color=colors['mlblue'], linewidth=2, markersize=8)

# Mark the elbow point
elbow_k = 4
ax1.plot(elbow_k, inertias[elbow_k-1], 'o', color=colors['mlred'], 
         markersize=15, markeredgewidth=3, markeredgecolor='darkred')
ax1.annotate('Elbow Point\n(Optimal k=4)', 
            xy=(elbow_k, inertias[elbow_k-1]), 
            xytext=(elbow_k+1, inertias[elbow_k-1]+500),
            arrowprops=dict(arrowstyle='->', color=colors['mlred'], lw=2),
            fontsize=10, fontweight='bold', color=colors['mlred'])

# Add shaded region for good range
ax1.axvspan(3, 5, alpha=0.1, color=colors['mlgreen'])
ax1.text(4, max(inertias)*0.9, 'Good Range', 
         ha='center', fontsize=10, color=colors['mlgreen'], fontweight='bold')

ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12, fontweight='bold')
ax1.set_title('The Elbow Method: Finding Optimal k', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_range)

# Add interpretation text
ax1.text(0.98, 0.95, 'Look for the "elbow"\nwhere improvement slows', 
         transform=ax1.transAxes, fontsize=9, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right plot: Visual comparison of different k values
ax2 = axes[1]

# Create 4 small subplots within ax2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

k_values = [2, 3, 4, 6]
positions = [(0.1, 0.55, 0.35, 0.35),  # top-left
             (0.55, 0.55, 0.35, 0.35), # top-right
             (0.1, 0.1, 0.35, 0.35),   # bottom-left
             (0.55, 0.1, 0.35, 0.35)]  # bottom-right

for k, pos in zip(k_values, positions):
    ax_inset = inset_axes(ax2, width="100%", height="100%",
                          bbox_to_anchor=pos, bbox_transform=ax2.transAxes)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    for i in range(k):
        mask = labels == i
        ax_inset.scatter(X[mask, 0], X[mask, 1], s=10, alpha=0.6)
    
    ax_inset.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    c='black', s=50, marker='*', edgecolors='white', linewidth=1)
    
    if k == 4:
        ax_inset.set_title(f'k={k} (OPTIMAL)', fontsize=9, fontweight='bold', color=colors['mlgreen'])
        # Add border around optimal
        for spine in ax_inset.spines.values():
            spine.set_edgecolor(colors['mlgreen'])
            spine.set_linewidth(2)
    else:
        ax_inset.set_title(f'k={k}', fontsize=9)
    
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

ax2.set_title('Visual Comparison: Different k Values', fontsize=13, fontweight='bold')
ax2.axis('off')

# Add annotations for each subplot
ax2.text(0.275, 0.48, 'Too few\n(underfitting)', transform=ax2.transAxes, 
         ha='center', fontsize=8, color='gray')
ax2.text(0.725, 0.48, 'Still merged', transform=ax2.transAxes, 
         ha='center', fontsize=8, color='gray')
ax2.text(0.275, 0.03, 'Natural groups!', transform=ax2.transAxes, 
         ha='center', fontsize=8, color=colors['mlgreen'], fontweight='bold')
ax2.text(0.725, 0.03, 'Too many\n(overfitting)', transform=ax2.transAxes, 
         ha='center', fontsize=8, color='gray')

# Main title
fig.suptitle('Choosing the Right Number of Innovation Clusters', fontsize=16, fontweight='bold')

# Add bottom annotation
fig.text(0.5, 0.01, 'The elbow method helps find the sweet spot between too few and too many clusters',
         ha='center', fontsize=11, color=colors['mlpurple'], fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('elbow_method.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Elbow method chart created successfully!")