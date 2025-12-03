import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Generate initial random data
n_points = 1000
n_clusters = 5

# Create data that will naturally form clusters
centers = np.array([[2, 2], [8, 2], [5, 8], [2, 7], [8, 7]])
X = []
for center in centers:
    cluster_data = np.random.randn(n_points // n_clusters, 2) * 0.8 + center
    X.append(cluster_data)
X = np.vstack(X)
np.random.shuffle(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1, max_iter=1)
kmeans.fit(X)

# Get intermediate steps
steps = []
for i in range(15):
    kmeans_step = KMeans(n_clusters=n_clusters, random_state=42, n_init=1, max_iter=i+1)
    kmeans_step.fit(X)
    steps.append({
        'centers': kmeans_step.cluster_centers_.copy(),
        'labels': kmeans_step.labels_.copy()
    })

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# Plot 1: Initial chaos
ax1 = axes[0]
ax1.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3, s=20)
ax1.set_title('Initial State:\nRandom Data Points', fontsize=14, fontweight='bold')
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 11)
ax1.set_ylim(-1, 11)

# Plot 2: Convergence process (middle state)
ax2 = axes[1]
mid_step = steps[7]
for i in range(n_clusters):
    mask = mid_step['labels'] == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.4, s=20)
ax2.scatter(mid_step['centers'][:, 0], mid_step['centers'][:, 1], 
           c='black', marker='X', s=200, edgecolors='white', linewidth=2)
ax2.set_title('Convergence Process:\nFinding Natural Groups', fontsize=14, fontweight='bold')
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, 11)
ax2.set_ylim(-1, 11)

# Add arrows showing movement
for i in range(n_clusters):
    if i > 0:
        ax2.annotate('', xy=mid_step['centers'][i], 
                    xytext=steps[3]['centers'][i],
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))

# Plot 3: Final clusters
ax3 = axes[2]
final_step = steps[-1]
for i in range(n_clusters):
    mask = final_step['labels'] == i
    ax3.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=20, 
               label=f'Cluster {i+1}')
ax3.scatter(final_step['centers'][:, 0], final_step['centers'][:, 1], 
           c='black', marker='X', s=200, edgecolors='white', linewidth=2,
           label='Centroids')
ax3.set_title('Final State:\nMeaningful Clusters', fontsize=14, fontweight='bold')
ax3.set_xlabel('Feature 1', fontsize=12)
ax3.set_ylabel('Feature 2', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1, 11)
ax3.set_ylim(-1, 11)
ax3.legend(loc='upper right', fontsize=10)

# Add main title
fig.suptitle('The Convergence Flow: From Chaos to Clarity', 
            fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('convergence_flow.pdf', dpi=300, bbox_inches='tight')
plt.savefig('convergence_flow.png', dpi=150, bbox_inches='tight')
plt.close()

print("Convergence flow visualization created successfully!")