"""
Chart 2: The K-Means Dance (6-panel animation)
Shows K-means algorithm progression step-by-step
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
mlblue = '#1f77b4'
mlorange = '#ff7f0e'
mlgreen = '#2ca02c'
mlred = '#d62728'
mlpurple = '#9467bd'
mlgray = '#7f7f7f'

# Generate data with 3 natural clusters
np.random.seed(42)
cluster1 = np.random.randn(8, 2) * 0.5 + [1, 6]
cluster2 = np.random.randn(7, 2) * 0.6 + [6, 4]
cluster3 = np.random.randn(5, 2) * 0.4 + [8, 2]
X = np.vstack([cluster1, cluster2, cluster3])

# Initial random centers
centers_0 = np.array([[2, 7], [5, 5], [8, 3]])

# Manual K-means iterations
def assign_clusters(X, centers):
    """Assign each point to nearest center"""
    distances = np.sqrt(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)

def update_centers(X, assignments, k=3):
    """Calculate new centers as means of assigned points"""
    new_centers = np.zeros((k, 2))
    for i in range(k):
        if np.sum(assignments == i) > 0:
            new_centers[i] = X[assignments == i].mean(axis=0)
    return new_centers

def calculate_variance(X, centers, assignments):
    """Calculate total within-cluster variance"""
    variance = 0
    for i in range(len(centers)):
        cluster_points = X[assignments == i]
        if len(cluster_points) > 0:
            variance += np.sum((cluster_points - centers[i])**2)
    return variance

# Perform iterations
centers_list = [centers_0]
assignments_list = []
variance_list = []

for iteration in range(5):
    assignments = assign_clusters(X, centers_list[-1])
    assignments_list.append(assignments)
    variance = calculate_variance(X, centers_list[-1], assignments)
    variance_list.append(variance)

    new_centers = update_centers(X, assignments)
    centers_list.append(new_centers)

# Create 6-panel figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

colors = [mlred, mlblue, mlgreen]
panel_titles = [
    'STEP 0: Random Start',
    'STEP 1: Assign Points',
    'STEP 2: Centers Move',
    'STEP 3: Reassign',
    'STEP 4: Move Again',
    'STEP 5: CONVERGED!'
]

for idx, ax in enumerate(axes):
    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Feature X', fontsize=10, fontweight='bold')
    ax.set_ylabel('Feature Y', fontsize=10, fontweight='bold')

    if idx == 0:
        # Step 0: Random initialization
        ax.scatter(X[:, 0], X[:, 1], c=mlgray, s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        for i, center in enumerate(centers_0):
            ax.scatter(center[0], center[1], c=colors[i], s=300, marker='*',
                      edgecolors='black', linewidth=2, zorder=5)
            ax.add_patch(Circle(center, 0.3, fill=False, edgecolor=colors[i], linewidth=3))
        ax.set_title(panel_titles[0], fontsize=12, fontweight='bold', pad=10)
        ax.text(0.05, 0.05, 'Centers: Random', transform=ax.transAxes,
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    elif idx % 2 == 1:  # Assignment steps (1, 3, 5)
        iteration_num = idx // 2
        assignments = assignments_list[iteration_num]
        centers = centers_list[iteration_num]

        for i in range(3):
            cluster_points = X[assignments == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=colors[i], s=60, alpha=0.6, edgecolors='black', linewidth=0.5)

        for i, center in enumerate(centers):
            ax.scatter(center[0], center[1], c=colors[i], s=300, marker='*',
                      edgecolors='black', linewidth=2, zorder=5)
            ax.add_patch(Circle(center, 0.3, fill=False, edgecolor=colors[i], linewidth=3))

        ax.set_title(panel_titles[idx], fontsize=12, fontweight='bold', pad=10)

        # Show cluster counts
        counts_text = f"Red:{np.sum(assignments==0)} Blue:{np.sum(assignments==1)} Green:{np.sum(assignments==2)}"
        ax.text(0.05, 0.05, counts_text, transform=ax.transAxes,
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        if iteration_num < len(variance_list):
            variance_text = f'Variance: {variance_list[iteration_num]:.1f}'
            ax.text(0.05, 0.15, variance_text, transform=ax.transAxes,
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    else:  # Movement steps (2, 4)
        iteration_num = (idx - 1) // 2
        assignments = assignments_list[iteration_num]
        old_centers = centers_list[iteration_num]
        new_centers = centers_list[iteration_num + 1]

        for i in range(3):
            cluster_points = X[assignments == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=colors[i], s=60, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Draw arrows showing movement
        for i in range(3):
            ax.annotate('', xy=new_centers[i], xytext=old_centers[i],
                       arrowprops=dict(arrowstyle='->', lw=2, color=colors[i]))
            ax.scatter(new_centers[i][0], new_centers[i][1], c=colors[i], s=300, marker='*',
                      edgecolors='black', linewidth=2, zorder=5)

        ax.set_title(panel_titles[idx], fontsize=12, fontweight='bold', pad=10)

        movement = np.sqrt(np.sum((new_centers - old_centers)**2, axis=1)).sum()
        movement_text = f'Movement: {movement:.2f}'
        ax.text(0.05, 0.05, movement_text, transform=ax.transAxes,
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Add convergence indicator to last panel
axes[5].text(0.5, 0.95, 'Centers stopped moving!', transform=axes[5].transAxes,
            fontsize=11, fontweight='bold', ha='center', color=mlpurple,
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

plt.tight_layout()

# Save
plt.savefig('../charts/discovery_chart_2_kmeans.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/discovery_chart_2_kmeans.png', dpi=150, bbox_inches='tight')
print("Chart 2 (K-Means Dance) created successfully!")
print(f"Variance progression: {[f'{v:.1f}' for v in variance_list]}")

plt.show()
