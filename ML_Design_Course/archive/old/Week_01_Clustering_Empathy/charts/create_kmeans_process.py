import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as patches

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with 3 subplots showing K-means steps
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Generate sample data
np.random.seed(42)
n_points = 150
X = np.vstack([
    np.random.randn(50, 2) + [2, 2],
    np.random.randn(50, 2) + [-2, 2],
    np.random.randn(50, 2) + [0, -2]
])

# Step 1: Initialize random centers
ax1 = axes[0]
ax1.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, alpha=0.6)

# Random initial centers
initial_centers = np.array([[1, 1], [-1, 0], [0.5, -1]])
colors = ['#d62728', '#2ca02c', '#1f77b4']

for i, center in enumerate(initial_centers):
    ax1.scatter(center[0], center[1], c=colors[i], s=300, 
               marker='*', edgecolors='black', linewidth=2)

ax1.set_title('Step 1: Initialize K Centers', fontsize=14, fontweight='bold')
ax1.set_xlabel('Dimension 1', fontsize=11)
ax1.set_ylabel('Dimension 2', fontsize=11)
ax1.text(0, -4, 'Pick K=3 random starting points', 
        fontsize=11, ha='center', style='italic')

# Step 2: Assign points to nearest center
ax2 = axes[1]

# Calculate distances and assign
distances = np.zeros((len(X), 3))
for i, center in enumerate(initial_centers):
    distances[:, i] = np.sqrt(np.sum((X - center)**2, axis=1))

assignments = np.argmin(distances, axis=1)

# Plot assigned points
for i in range(3):
    mask = assignments == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=30, alpha=0.6)
    ax2.scatter(initial_centers[i, 0], initial_centers[i, 1], 
               c=colors[i], s=300, marker='*', 
               edgecolors='black', linewidth=2)

# Draw some distance lines
for i in range(0, len(X), 20):
    nearest = assignments[i]
    ax2.plot([X[i, 0], initial_centers[nearest, 0]], 
            [X[i, 1], initial_centers[nearest, 1]], 
            'k--', alpha=0.2, linewidth=0.5)

ax2.set_title('Step 2: Assign to Nearest Center', fontsize=14, fontweight='bold')
ax2.set_xlabel('Dimension 1', fontsize=11)
ax2.set_ylabel('Dimension 2', fontsize=11)
ax2.text(0, -4, 'Each point joins closest centroid', 
        fontsize=11, ha='center', style='italic')

# Step 3: Update centers and converge
ax3 = axes[2]

# Use actual K-means for final result
kmeans = KMeans(n_clusters=3, random_state=42, n_init=1)
final_labels = kmeans.fit_predict(X)
final_centers = kmeans.cluster_centers_

# Plot final clusters
for i in range(3):
    mask = final_labels == i
    ax3.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=30, alpha=0.6)
    
    # Show movement of centers
    ax3.annotate('', xy=final_centers[i], xytext=initial_centers[i],
                arrowprops=dict(arrowstyle='->', lw=2, 
                              color=colors[i], alpha=0.5))
    
    # Plot final centers
    ax3.scatter(final_centers[i, 0], final_centers[i, 1], 
               c=colors[i], s=300, marker='*', 
               edgecolors='black', linewidth=2)

ax3.set_title('Step 3: Update & Converge', fontsize=14, fontweight='bold')
ax3.set_xlabel('Dimension 1', fontsize=11)
ax3.set_ylabel('Dimension 2', fontsize=11)
ax3.text(0, -4, 'Centers move to cluster means, repeat until stable', 
        fontsize=11, ha='center', style='italic')

# Main title
plt.suptitle('K-means Algorithm: 3 Simple Steps', 
            fontsize=16, fontweight='bold', y=1.05)

# Add iteration counter
fig.text(0.5, 0.02, 'Typically converges in 10-20 iterations', 
        ha='center', fontsize=11, style='italic')

plt.tight_layout()
plt.savefig('kmeans_process.pdf', dpi=300, bbox_inches='tight')
plt.savefig('kmeans_process.png', dpi=150, bbox_inches='tight')

print("Chart saved: kmeans_process.pdf")
print("\nK-means Process:")
print("Step 1: Random initialization")
print("Step 2: Assign points to nearest centroid")  
print("Step 3: Update centroids to cluster means")
print("Repeat steps 2-3 until convergence")
print(f"\nFinal inertia (sum of squared distances): {kmeans.inertia_:.2f}")
print(f"Iterations to converge: {kmeans.n_iter_}")