import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Generate sample data
X, y_true = make_blobs(n_samples=1000, centers=5, n_features=10, 
                      cluster_std=1.0, random_state=42)

# Calculate inertia for different K values
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Normalize inertias for better visualization
inertias_norm = np.array(inertias)
inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min())

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot elbow curve
ax.plot(K_range, inertias_norm, 'b-', linewidth=2.5, marker='o', markersize=8)

# Mark the elbow point (K=5)
elbow_k = 5
elbow_idx = elbow_k - 2  # Adjust for 0-indexing and starting at K=2
ax.scatter(elbow_k, inertias_norm[elbow_idx], color='red', s=200, zorder=5)

# Add elbow annotation
ax.annotate('Optimal K=5\n(The "Elbow")', 
            xy=(elbow_k, inertias_norm[elbow_idx]),
            xytext=(elbow_k + 1.5, inertias_norm[elbow_idx] + 0.1),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold', color='red')

# Add shaded regions
ax.axvspan(2, 4, alpha=0.2, color='orange', label='Too few clusters')
ax.axvspan(4, 6, alpha=0.2, color='green', label='Optimal range')
ax.axvspan(6, 10, alpha=0.2, color='red', label='Diminishing returns')

ax.set_xlabel('Number of Clusters (K)', fontsize=13, fontweight='bold')
ax.set_ylabel('Within-Cluster Sum of Squares (normalized)', fontsize=13, fontweight='bold')
ax.set_title('The Elbow Method: Finding Optimal K', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# Add text explanations
ax.text(3, 0.8, 'Loses detail', fontsize=10, ha='center', style='italic')
ax.text(5, 0.15, 'Sweet spot!', fontsize=10, ha='center', style='italic', 
        fontweight='bold', color='green')
ax.text(8, 0.05, 'Over-segmenting', fontsize=10, ha='center', style='italic')

# Add secondary plot showing cluster sizes
ax2 = ax.twinx()
cluster_sizes = [500, 333, 250, 200, 166, 142, 125, 111, 100]  # Approximate sizes
ax2.bar(K_range, cluster_sizes, alpha=0.3, color='gray', width=0.3)
ax2.set_ylabel('Average Cluster Size', fontsize=12, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# Add interpretation guide
interpretation_text = (
    "How to read this chart:\n"
    "• The curve shows clustering quality\n"
    "• Look for the 'elbow' - where improvement slows\n"
    "• K=5 balances detail with interpretability"
)

fig.text(0.15, 0.15, interpretation_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('empathy_scale.pdf', dpi=300, bbox_inches='tight')
plt.savefig('empathy_scale.png', dpi=150, bbox_inches='tight')

print("Chart saved: empathy_scale.pdf")
print("\nElbow Method Results:")
print(f"Optimal K: 5")
print(f"Inertia at K=5: {inertias[3]:.2f}")
print(f"Average cluster size at K=5: 200 data points")
print(f"Improvement from K=4 to K=5: {(inertias[2]-inertias[3])/inertias[2]*100:.1f}%")
print(f"Improvement from K=5 to K=6: {(inertias[3]-inertias[4])/inertias[3]*100:.1f}%")