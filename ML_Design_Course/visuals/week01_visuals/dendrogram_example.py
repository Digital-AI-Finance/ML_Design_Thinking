import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Generate sample data - fewer points for clearer dendrogram
n_samples = 30
X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                  cluster_std=0.5, random_state=42)

# Create user labels
user_labels = [f'User_{i+1}' for i in range(n_samples)]

# Perform hierarchical clustering
linkage_matrix = linkage(X, method='ward')

# Create the dendrogram
fig, ax = plt.subplots(figsize=(12, 6))

# Plot dendrogram
dendrogram(linkage_matrix,
          labels=user_labels,
          leaf_rotation=90,
          leaf_font_size=10,
          color_threshold=10,
          above_threshold_color='gray')

# Add horizontal line to show cut
cut_height = 10
ax.axhline(y=cut_height, c='red', linestyle='--', linewidth=2, 
          label=f'Cut at height {cut_height} → 3 clusters')

# Styling
ax.set_xlabel('User ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance (Dissimilarity)', fontsize=12, fontweight='bold')
ax.set_title('Hierarchical Clustering Dendrogram: Building User Family Tree', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper right', fontsize=11)

# Add annotations
ax.text(5, 2, 'Best friends\n(most similar)', fontsize=9, ha='center',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.text(15, 5, 'Friend groups\nforming', fontsize=9, ha='center',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax.text(25, 12, 'Communities\nmerging', fontsize=9, ha='center',
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig('dendrogram_example.pdf', dpi=300, bbox_inches='tight')
plt.savefig('dendrogram_example.png', dpi=150, bbox_inches='tight')
plt.close()

print("Dendrogram visualization created!")