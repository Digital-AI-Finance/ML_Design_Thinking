"""
Create all missing Part 2 Technical charts for Week 1 slides
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, Ellipse
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlpink': '#e377c2',
    'mlbrown': '#8c564b',
    'mlgray': '#7f7f7f'
}

np.random.seed(42)

# Chart 1: Distance Metrics Detailed
print("Creating distance_metrics_detailed.pdf...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Generate sample points
point1 = np.array([1, 1])
point2 = np.array([4, 5])

metrics = ['Euclidean', 'Manhattan', 'Cosine']
distances = {
    'Euclidean': np.linalg.norm(point2 - point1),
    'Manhattan': np.sum(np.abs(point2 - point1)),
    'Cosine': 1 - np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
}

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Plot points
    ax.scatter(*point1, s=200, c=colors['mlblue'], marker='o', label='Point A', zorder=5)
    ax.scatter(*point2, s=200, c=colors['mlred'], marker='s', label='Point B', zorder=5)
    
    # Show distance path
    if metric == 'Euclidean':
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
               'g--', linewidth=3, alpha=0.7, label='Direct path')
    elif metric == 'Manhattan':
        ax.plot([point1[0], point2[0]], [point1[1], point1[1]], 
               'g--', linewidth=2, alpha=0.7)
        ax.plot([point2[0], point2[0]], [point1[1], point2[1]], 
               'g--', linewidth=2, alpha=0.7, label='Grid path')
    else:  # Cosine
        # Show angle
        origin = [0, 0]
        ax.plot([origin[0], point1[0]*5], [origin[1], point1[1]*5], 
               'b-', linewidth=1, alpha=0.5)
        ax.plot([origin[0], point2[0]*1.5], [origin[1], point2[1]*1.5], 
               'r-', linewidth=1, alpha=0.5)
        # Add arc for angle
        angle1 = np.arctan2(point1[1], point1[0])
        angle2 = np.arctan2(point2[1], point2[0])
        angles = np.linspace(angle1, angle2, 20)
        arc_x = 2 * np.cos(angles)
        arc_y = 2 * np.sin(angles)
        ax.plot(arc_x, arc_y, 'g--', linewidth=2, alpha=0.7, label='Angle')
    
    ax.set_title(f'{metric} Distance\nd = {distances[metric]:.2f}', 
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    
    # Add explanation
    if metric == 'Euclidean':
        ax.text(0.5, 0.95, 'Straight line\n"as the crow flies"', 
               transform=ax.transAxes, ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    elif metric == 'Manhattan':
        ax.text(0.5, 0.95, 'City block\ndistance', 
               transform=ax.transAxes, ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.95, 'Angle between\nvectors', 
               transform=ax.transAxes, ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Distance Metrics: Different Ways to Measure Innovation Similarity', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distance_metrics_detailed.pdf', dpi=300, bbox_inches='tight')
plt.savefig('distance_metrics_detailed.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Cluster Quality
print("Creating cluster_quality.pdf...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Generate different quality clusters
qualities = [
    ('Excellent', 0.3, 3),  # std, n_clusters
    ('Good', 0.5, 3),
    ('Fair', 0.8, 3),
    ('Poor', 1.2, 3),
    ('Overlapping', 0.7, 3),
    ('Wrong K', 0.5, 5)  # Good data but wrong number of clusters
]

for idx, (quality, std, n_clusters) in enumerate(qualities):
    ax = axes[idx // 3, idx % 3]
    
    if quality == 'Overlapping':
        # Create overlapping clusters
        X1, _ = make_blobs(n_samples=100, centers=[[0, 0]], cluster_std=std, random_state=42)
        X2, _ = make_blobs(n_samples=100, centers=[[1, 0]], cluster_std=std, random_state=42)
        X3, _ = make_blobs(n_samples=100, centers=[[0.5, 1]], cluster_std=std, random_state=42)
        X = np.vstack([X1, X2, X3])
        true_n = 3
    else:
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=std, random_state=42)
        true_n = 3
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score
    sil_score = silhouette_score(X, labels)
    
    # Plot clusters
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.7)
    
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c='black', s=100, marker='*', edgecolors='white', linewidth=1)
    
    # Color code the quality
    if quality == 'Excellent':
        title_color = colors['mlgreen']
    elif quality in ['Good', 'Fair']:
        title_color = colors['mlorange']
    else:
        title_color = colors['mlred']
    
    ax.set_title(f'{quality} Quality\nSilhouette: {sil_score:.2f}', 
                fontsize=10, fontweight='bold', color=title_color)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add quality indicators
    if quality == 'Wrong K':
        ax.text(0.5, 0.05, 'Too many clusters!', 
               transform=ax.transAxes, ha='center', fontsize=8, color=colors['mlred'])

fig.suptitle('Cluster Quality: From Excellent to Poor', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_quality.pdf', dpi=300, bbox_inches='tight')
plt.savefig('cluster_quality.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Davies-Bouldin Index
print("Creating davies_bouldin.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Generate data
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

# Calculate DB index for different k
k_range = range(2, 11)
db_scores = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    db_scores.append(davies_bouldin_score(X, labels))
    silhouette_scores.append(silhouette_score(X, labels))

# Left plot: DB Index
ax = axes[0]
ax.plot(k_range, db_scores, 'o-', color=colors['mlblue'], linewidth=2, markersize=8)
ax.axvline(x=4, color=colors['mlgreen'], linestyle='--', linewidth=2, alpha=0.5)
ax.scatter([4], [db_scores[2]], color=colors['mlgreen'], s=150, zorder=5)
ax.set_xlabel('Number of Clusters (k)', fontsize=11)
ax.set_ylabel('Davies-Bouldin Index', fontsize=11)
ax.set_title('Davies-Bouldin Index: Lower is Better', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.text(4, max(db_scores)*0.9, 'Optimal k=4\n(minimum DB)', 
       ha='center', fontsize=9, color=colors['mlgreen'], fontweight='bold')

# Right plot: Comparison with Silhouette
ax = axes[1]
ax2 = ax.twinx()
line1 = ax.plot(k_range, db_scores, 'o-', color=colors['mlred'], 
               linewidth=2, markersize=8, label='DB Index (lower better)')
line2 = ax2.plot(k_range, silhouette_scores, 's-', color=colors['mlblue'], 
                linewidth=2, markersize=8, label='Silhouette (higher better)')
ax.set_xlabel('Number of Clusters (k)', fontsize=11)
ax.set_ylabel('Davies-Bouldin Index', color=colors['mlred'], fontsize=11)
ax2.set_ylabel('Silhouette Score', color=colors['mlblue'], fontsize=11)
ax.tick_params(axis='y', labelcolor=colors['mlred'])
ax2.tick_params(axis='y', labelcolor=colors['mlblue'])
ax.set_title('Multiple Validation Metrics Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='best')

fig.suptitle('Cluster Validation: Davies-Bouldin Index', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('davies_bouldin.pdf', dpi=300, bbox_inches='tight')
plt.savefig('davies_bouldin.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Algorithm Visual Examples
print("Creating algorithm_visual_examples.pdf...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Generate diverse data
X_blob, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
from sklearn.datasets import make_moons
X_moon, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

algorithms = [
    ('K-Means', KMeans(n_clusters=3, random_state=42), X_blob, 'Best for spherical clusters'),
    ('DBSCAN', DBSCAN(eps=0.3, min_samples=5), X_moon, 'Finds arbitrary shapes'),
    ('Hierarchical', AgglomerativeClustering(n_clusters=3), X_blob, 'Shows relationships'),
    ('Gaussian Mixture', GaussianMixture(n_components=3, random_state=42), X_blob, 'Soft boundaries')
]

for idx, (name, algo, data, desc) in enumerate(algorithms):
    ax = axes[idx // 2, idx % 2]
    
    # Fit and predict
    if name == 'Gaussian Mixture':
        labels = algo.fit_predict(data)
        # Show probability contours
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        Z = -algo.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=10, linewidths=0.5, alpha=0.3)
    else:
        labels = algo.fit_predict(data)
    
    # Plot clusters
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            # Noise points in DBSCAN
            mask = labels == k
            ax.scatter(data[mask, 0], data[mask, 1], c='gray', s=20, alpha=0.3)
        else:
            mask = labels == k
            ax.scatter(data[mask, 0], data[mask, 1], s=40, alpha=0.7,
                      label=f'Cluster {k+1}')
    
    # Add centers for K-means
    if name == 'K-Means':
        ax.scatter(algo.cluster_centers_[:, 0], algo.cluster_centers_[:, 1],
                  c='black', s=150, marker='*', edgecolors='white', linewidth=2)
    
    ax.set_title(f'{name}\n{desc}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Add algorithm characteristics
    if name == 'K-Means':
        ax.text(0.95, 0.95, 'Requires k\nFast\nSpherical', 
               transform=ax.transAxes, ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor=colors['mlblue'], alpha=0.1))
    elif name == 'DBSCAN':
        ax.text(0.95, 0.95, 'No k needed\nFinds outliers\nDensity-based', 
               transform=ax.transAxes, ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor=colors['mlorange'], alpha=0.1))
    elif name == 'Hierarchical':
        ax.text(0.95, 0.95, 'Dendrogram\nNo retraining\nMemory intensive', 
               transform=ax.transAxes, ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor=colors['mlgreen'], alpha=0.1))
    else:
        ax.text(0.95, 0.95, 'Probabilistic\nOverlapping\nFlexible shapes', 
               transform=ax.transAxes, ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor=colors['mlpurple'], alpha=0.1))

fig.suptitle('Clustering Algorithm Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('algorithm_visual_examples.pdf', dpi=300, bbox_inches='tight')
plt.savefig('algorithm_visual_examples.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: GMM Detailed
print("Creating gmm_detailed.pdf...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Generate data and fit GMM
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=[0.5, 0.8, 1.2], random_state=42)
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X)

# Left plot: Data with cluster assignments
ax = axes[0]
for i in range(3):
    mask = labels == i
    ax.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.6)
ax.set_title('GMM Cluster Assignments', fontsize=11, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Middle plot: Probability contours
ax = axes[1]
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100))
Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
contour = ax.contour(xx, yy, Z, levels=15, cmap='viridis', alpha=0.4)
ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, alpha=0.5, cmap='viridis')
ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=200, 
          marker='*', edgecolors='white', linewidth=2)
ax.set_title('Probability Density Contours', fontsize=11, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Right plot: Soft assignments (probabilities)
ax = axes[2]
probs = gmm.predict_proba(X)
# Show points colored by uncertainty (entropy)
uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
scatter = ax.scatter(X[:, 0], X[:, 1], c=uncertainty, cmap='RdYlGn_r', s=30, alpha=0.7)
plt.colorbar(scatter, ax=ax, label='Uncertainty')
ax.set_title('Assignment Uncertainty\n(Red = uncertain)', fontsize=11, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

fig.suptitle('Gaussian Mixture Model: Probabilistic Clustering', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gmm_detailed.pdf', dpi=300, bbox_inches='tight')
plt.savefig('gmm_detailed.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 6: Feature Importance
print("Creating feature_importance.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Generate data with varying feature importance
np.random.seed(42)
n_samples = 500
n_features = 10

# Create features with different importance levels
important_features = np.random.randn(n_samples, 3) * 2  # Important features
noise_features = np.random.randn(n_samples, 7) * 0.1    # Noise features
X = np.hstack([important_features, noise_features])

# Calculate feature variance (proxy for importance)
feature_variance = np.var(X, axis=0)
feature_names = [f'F{i+1}' for i in range(n_features)]

# Left plot: Feature importance bar chart
ax = axes[0]
colors_list = [colors['mlgreen'] if v > 1 else colors['mlgray'] for v in feature_variance]
bars = ax.bar(feature_names, feature_variance, color=colors_list, alpha=0.7)
ax.set_xlabel('Features', fontsize=11)
ax.set_ylabel('Importance Score (Variance)', fontsize=11)
ax.set_title('Feature Importance for Clustering', fontsize=12, fontweight='bold')
ax.axhline(y=1, color=colors['mlred'], linestyle='--', linewidth=1, alpha=0.5)
ax.text(5, 1.1, 'Importance threshold', ha='center', fontsize=9, color=colors['mlred'])

# Right plot: Cumulative importance
ax = axes[1]
sorted_importance = sorted(feature_variance, reverse=True)
cumulative = np.cumsum(sorted_importance) / np.sum(sorted_importance) * 100
ax.plot(range(1, len(cumulative)+1), cumulative, 'o-', color=colors['mlblue'], 
       linewidth=2, markersize=8)
ax.axhline(y=80, color=colors['mlorange'], linestyle='--', linewidth=2, alpha=0.5)
ax.axvline(x=3, color=colors['mlorange'], linestyle='--', linewidth=2, alpha=0.5)
ax.fill_between(range(1, 4), 0, 100, alpha=0.2, color=colors['mlgreen'])
ax.set_xlabel('Number of Features', fontsize=11)
ax.set_ylabel('Cumulative Importance (%)', fontsize=11)
ax.set_title('Feature Selection: 80% with 3 Features', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.text(3.5, 40, 'Top 3 features\ncapture 80%\nof variance', fontsize=9, 
       color=colors['mlgreen'], fontweight='bold')

fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 7: Preprocessing Pipeline
print("Creating preprocessing_pipeline.pdf...")
fig, ax = plt.subplots(figsize=(14, 8))

# Define pipeline stages
stages = [
    {'name': 'Raw Data', 'y': 7, 'color': colors['mlgray'], 
     'desc': '• Missing values\n• Different scales\n• Outliers'},
    {'name': 'Cleaning', 'y': 5.5, 'color': colors['mlblue'],
     'desc': '• Fill missing\n• Remove duplicates\n• Fix errors'},
    {'name': 'Scaling', 'y': 4, 'color': colors['mlorange'],
     'desc': '• Standardization\n• Normalization\n• Min-Max scaling'},
    {'name': 'Feature Engineering', 'y': 2.5, 'color': colors['mlgreen'],
     'desc': '• Create new features\n• Combine features\n• Transform features'},
    {'name': 'Ready for ML', 'y': 1, 'color': colors['mlpurple'],
     'desc': '• Clean data\n• Scaled features\n• Optimized for clustering'}
]

# Draw pipeline boxes
for i, stage in enumerate(stages):
    # Main box
    rect = Rectangle((2, stage['y']-0.4), 4, 0.8, 
                    facecolor=stage['color'], alpha=0.3, 
                    edgecolor=stage['color'], linewidth=2)
    ax.add_patch(rect)
    
    # Stage name
    ax.text(4, stage['y'], stage['name'], fontsize=12, fontweight='bold',
           ha='center', va='center')
    
    # Description
    ax.text(7, stage['y'], stage['desc'], fontsize=9, va='center')
    
    # Arrow to next stage
    if i < len(stages) - 1:
        ax.annotate('', xy=(4, stages[i+1]['y'] + 0.4), 
                   xytext=(4, stage['y'] - 0.4),
                   arrowprops=dict(arrowstyle='->', lw=3, 
                                 color=stage['color'], alpha=0.7))

# Add example transformations on the right
examples_x = 10
ax.text(examples_x, 7, 'Example:', fontsize=10, fontweight='bold')
ax.text(examples_x, 6.5, '[NaN, 100, 0.5, 1000]', fontsize=8, family='monospace')
ax.text(examples_x, 5.5, '↓', fontsize=12)
ax.text(examples_x, 5, '[50, 100, 0.5, 1000]', fontsize=8, family='monospace')
ax.text(examples_x, 4.5, '↓', fontsize=12)
ax.text(examples_x, 4, '[-0.5, 0.8, -1.2, 1.5]', fontsize=8, family='monospace')
ax.text(examples_x, 3.5, '↓', fontsize=12)
ax.text(examples_x, 3, '[PC1, PC2, PC3]', fontsize=8, family='monospace')
ax.text(examples_x, 2.5, '↓', fontsize=12)
ax.text(examples_x, 2, 'Ready!', fontsize=10, fontweight='bold', color=colors['mlgreen'])

ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.set_title('Data Preprocessing Pipeline for Clustering', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('preprocessing_pipeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig('preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 8: Common Mistakes
print("Creating common_mistakes.pdf...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

mistakes = [
    ('No Scaling', 'Features have\ndifferent scales'),
    ('Wrong K', 'Too many/few\nclusters'),
    ('Ignoring Outliers', 'Outliers skew\ncenters'),
    ('No Validation', 'No quality\nmetrics used'),
    ('Wrong Algorithm', 'K-means for\nnon-spherical'),
    ('Overfitting', 'Too complex\nfor data')
]

for idx, (mistake, description) in enumerate(mistakes):
    ax = axes[idx // 3, idx % 3]
    
    # Generate problematic scenario
    if mistake == 'No Scaling':
        X1 = np.random.randn(50, 1) * 0.5
        X2 = np.random.randn(50, 1) * 100 + 500
        X = np.hstack([X1, X2])
        ax.scatter(X[:, 0], X[:, 1], c='red', alpha=0.5)
        ax.set_xlabel('Feature 1 (small scale)')
        ax.set_ylabel('Feature 2 (large scale)')
    
    elif mistake == 'Wrong K':
        X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=42)
        kmeans = KMeans(n_clusters=7, random_state=42)
        labels = kmeans.fit_predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.5)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    elif mistake == 'Ignoring Outliers':
        X, _ = make_blobs(n_samples=90, centers=2, cluster_std=0.5, random_state=42)
        outliers = np.array([[10, 10], [-10, -10], [10, -10]])
        X = np.vstack([X, outliers])
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)
        ax.scatter(X[:-3, 0], X[:-3, 1], c=labels[:-3], cmap='tab10', alpha=0.5)
        ax.scatter(X[-3:, 0], X[-3:, 1], c='red', s=100, marker='x', linewidth=3)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    else:
        # Generic bad clustering visualization
        X = np.random.randn(100, 2)
        ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
        ax.text(0.5, 0.5, '✗', transform=ax.transAxes, 
               fontsize=40, ha='center', va='center', color='red', alpha=0.5)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    ax.set_title(f'{mistake}\n{description}', fontsize=10, 
                fontweight='bold', color=colors['mlred'])

fig.suptitle('Common Clustering Mistakes to Avoid', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('common_mistakes.pdf', dpi=300, bbox_inches='tight')
plt.savefig('common_mistakes.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 9: Parameter Tuning Guide
print("Creating parameter_tuning_guide.pdf...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# K-means parameters
ax = axes[0, 0]
k_values = [2, 3, 4, 5, 6, 7, 8]
inertias = [1000, 600, 400, 350, 340, 335, 330]
ax.plot(k_values, inertias, 'o-', color=colors['mlblue'], linewidth=2)
ax.fill_between([3, 5], 0, 1100, alpha=0.2, color=colors['mlgreen'])
ax.set_xlabel('k (number of clusters)')
ax.set_ylabel('Inertia')
ax.set_title('K-Means: Choosing k', fontsize=11, fontweight='bold')
ax.text(4, 800, 'Sweet spot', ha='center', color=colors['mlgreen'], fontweight='bold')
ax.grid(True, alpha=0.3)

# DBSCAN parameters
ax = axes[0, 1]
eps_values = np.linspace(0.1, 2, 20)
n_clusters = [1, 1, 2, 3, 4, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ax.plot(eps_values, n_clusters, 'o-', color=colors['mlorange'], linewidth=2)
ax.fill_between([0.5, 1.0], 0, 5, alpha=0.2, color=colors['mlgreen'])
ax.set_xlabel('eps (neighborhood size)')
ax.set_ylabel('Number of clusters found')
ax.set_title('DBSCAN: Choosing eps', fontsize=11, fontweight='bold')
ax.text(0.75, 4, 'Good range', ha='center', color=colors['mlgreen'], fontweight='bold')
ax.grid(True, alpha=0.3)

# Hierarchical parameters
ax = axes[1, 0]
linkages = ['single', 'complete', 'average', 'ward']
scores = [0.4, 0.6, 0.65, 0.8]
colors_bar = [colors['mlred'], colors['mlorange'], colors['mlorange'], colors['mlgreen']]
ax.bar(linkages, scores, color=colors_bar, alpha=0.7)
ax.set_ylabel('Silhouette Score')
ax.set_title('Hierarchical: Choosing Linkage', fontsize=11, fontweight='bold')
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
ax.text(2, 0.72, 'Good threshold', fontsize=9, color='gray')

# GMM parameters
ax = axes[1, 1]
n_components = range(1, 8)
bic_scores = [1000, 700, 500, 480, 490, 510, 540]
aic_scores = [900, 650, 480, 470, 485, 505, 535]
ax.plot(n_components, bic_scores, 'o-', label='BIC', color=colors['mlpurple'], linewidth=2)
ax.plot(n_components, aic_scores, 's-', label='AIC', color=colors['mlpink'], linewidth=2)
ax.fill_between([3, 5], 400, 1100, alpha=0.2, color=colors['mlgreen'])
ax.set_xlabel('Number of components')
ax.set_ylabel('Information Criterion')
ax.set_title('GMM: Choosing Components', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Parameter Tuning Guide for Clustering Algorithms', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('parameter_tuning_guide.pdf', dpi=300, bbox_inches='tight')
plt.savefig('parameter_tuning_guide.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll Part 2 Technical charts created successfully!")
print("Created files:")
print("- distance_metrics_detailed.pdf/png")
print("- cluster_quality.pdf/png")
print("- davies_bouldin.pdf/png")
print("- algorithm_visual_examples.pdf/png")
print("- gmm_detailed.pdf/png")
print("- feature_importance.pdf/png")
print("- preprocessing_pipeline.pdf/png")
print("- common_mistakes.pdf/png")
print("- parameter_tuning_guide.pdf/png")