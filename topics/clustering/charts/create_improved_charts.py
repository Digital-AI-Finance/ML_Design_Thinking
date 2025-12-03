"""
Create improved charts for Week 1 slides:
1. Fix Common Mistakes chart (last 3 panels)
2. Split Parameter Tuning into 2 separate charts
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
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

# Chart 1: Improved Common Mistakes
print("Creating improved common_mistakes.pdf...")
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
    
    if mistake == 'No Scaling':
        # Show unscaled features problem
        X1 = np.random.randn(50, 1) * 0.5
        X2 = np.random.randn(50, 1) * 100 + 500
        X = np.hstack([X1, X2])
        ax.scatter(X[:, 0], X[:, 1], c='red', alpha=0.5)
        ax.set_xlabel('Feature 1 (small scale)')
        ax.set_ylabel('Feature 2 (large scale)')
    
    elif mistake == 'Wrong K':
        # Too many clusters for the data
        X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=42)
        kmeans = KMeans(n_clusters=7, random_state=42)
        labels = kmeans.fit_predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.5)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    elif mistake == 'Ignoring Outliers':
        # Outliers affecting clustering
        X, _ = make_blobs(n_samples=90, centers=2, cluster_std=0.5, random_state=42)
        outliers = np.array([[10, 10], [-10, -10], [10, -10]])
        X = np.vstack([X, outliers])
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)
        ax.scatter(X[:-3, 0], X[:-3, 1], c=labels[:-3], cmap='tab10', alpha=0.5)
        ax.scatter(X[-3:, 0], X[-3:, 1], c='red', s=100, marker='x', linewidth=3)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    elif mistake == 'No Validation':
        # Show clustering without validation metrics
        X, _ = make_blobs(n_samples=100, centers=4, cluster_std=1.5, random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        
        # Plot clusters
        for i in range(3):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        # Add warning text
        ax.text(0.5, 0.05, f'Silhouette: {silhouette_avg:.2f} (Poor!)', 
                transform=ax.transAxes, ha='center', fontsize=9, 
                color='red', fontweight='bold')
    
    elif mistake == 'Wrong Algorithm':
        # K-means on non-spherical data
        X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Show how K-means fails
        for i in range(2):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5)
        
        # Draw the wrong decision boundary
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.text(0.5, 0.95, 'K-means splits linearly!', 
                transform=ax.transAxes, ha='center', fontsize=9, 
                color='red', fontweight='bold')
    
    else:  # Overfitting
        # Show overfitting with too many clusters
        X, y_true = make_blobs(n_samples=50, centers=2, cluster_std=0.8, random_state=42)
        # Use too many clusters for small data
        kmeans = KMeans(n_clusters=10, random_state=42)
        labels = kmeans.fit_predict(X)
        
        for i in range(10):
            mask = labels == i
            if np.sum(mask) > 0:
                ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5, s=30)
        
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   c='black', s=50, marker='x', linewidth=2)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.text(0.5, 0.05, '10 clusters for 50 points!', 
                transform=ax.transAxes, ha='center', fontsize=9, 
                color='red', fontweight='bold')
    
    ax.set_title(f'{mistake}\n{description}', fontsize=10, 
                fontweight='bold', color=colors['mlred'])

fig.suptitle('Common Clustering Mistakes to Avoid', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('common_mistakes.pdf', dpi=300, bbox_inches='tight')
plt.savefig('common_mistakes.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Parameter Tuning Part 1 (K-Means and DBSCAN)
print("Creating parameter_tuning_part1.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-means parameters
ax = axes[0]
k_values = [2, 3, 4, 5, 6, 7, 8]
inertias = [1000, 600, 400, 350, 340, 335, 330]
ax.plot(k_values, inertias, 'o-', color=colors['mlblue'], linewidth=3, markersize=10)
ax.fill_between([3, 5], 0, 1100, alpha=0.2, color=colors['mlgreen'])
ax.set_xlabel('k (number of clusters)', fontsize=12)
ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
ax.set_title('K-Means: Choosing the Right k', fontsize=13, fontweight='bold')
ax.text(4, 800, 'Sweet spot\n(Elbow)', ha='center', fontsize=11, 
        color=colors['mlgreen'], fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(k_values)

# Add annotations
ax.annotate('Rapid decrease', xy=(3, 600), xytext=(2, 800),
            arrowprops=dict(arrowstyle='->', color=colors['mlred'], lw=2),
            fontsize=10, color=colors['mlred'])
ax.annotate('Diminishing returns', xy=(6, 340), xytext=(7, 500),
            arrowprops=dict(arrowstyle='->', color=colors['mlorange'], lw=2),
            fontsize=10, color=colors['mlorange'])

# DBSCAN parameters
ax = axes[1]
eps_values = np.linspace(0.1, 2, 20)
n_clusters = [1, 1, 2, 3, 4, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ax.plot(eps_values, n_clusters, 'o-', color=colors['mlorange'], linewidth=3, markersize=8)
ax.fill_between([0.5, 1.0], 0, 5, alpha=0.2, color=colors['mlgreen'])
ax.set_xlabel('eps (neighborhood size)', fontsize=12)
ax.set_ylabel('Number of clusters found', fontsize=12)
ax.set_title('DBSCAN: Finding the Right eps', fontsize=13, fontweight='bold')
ax.text(0.75, 4, 'Optimal\nrange', ha='center', fontsize=11,
        color=colors['mlgreen'], fontweight='bold')
ax.grid(True, alpha=0.3)

# Add guidance text
ax.text(0.3, 3.5, 'Too small:\nMany clusters', fontsize=9, color=colors['mlred'])
ax.text(1.5, 1.5, 'Too large:\nEverything merges', fontsize=9, color=colors['mlred'])

fig.suptitle('Parameter Tuning Guidelines - Part 1: Distance-Based Methods', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('parameter_tuning_part1.pdf', dpi=300, bbox_inches='tight')
plt.savefig('parameter_tuning_part1.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Parameter Tuning Part 2 (Hierarchical and GMM)
print("Creating parameter_tuning_part2.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Hierarchical parameters
ax = axes[0]
linkages = ['single', 'complete', 'average', 'ward']
scores = [0.4, 0.6, 0.65, 0.8]
colors_bar = [colors['mlred'], colors['mlorange'], colors['mlorange'], colors['mlgreen']]
bars = ax.bar(linkages, scores, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Silhouette Score', fontsize=12)
ax.set_title('Hierarchical: Best Linkage Method', fontsize=13, fontweight='bold')
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=2)
ax.text(1.5, 0.72, 'Good threshold', fontsize=10, color='gray')
ax.set_ylim(0, 1)

# Add descriptions
descriptions = ['Sensitive\nto noise', 'Can create\nchains', 'Balanced\napproach', 'Best for\ncompact']
for i, (link, desc) in enumerate(zip(linkages, descriptions)):
    ax.text(i, 0.1, desc, ha='center', fontsize=8, color='gray')

# GMM parameters
ax = axes[1]
n_components = range(1, 8)
bic_scores = [1000, 700, 500, 480, 490, 510, 540]
aic_scores = [900, 650, 480, 470, 485, 505, 535]
ax.plot(n_components, bic_scores, 'o-', label='BIC', color=colors['mlpurple'], 
        linewidth=3, markersize=10)
ax.plot(n_components, aic_scores, 's-', label='AIC', color=colors['mlpink'], 
        linewidth=3, markersize=10)
ax.fill_between([3, 5], 400, 1100, alpha=0.2, color=colors['mlgreen'])
ax.set_xlabel('Number of components', fontsize=12)
ax.set_ylabel('Information Criterion Score', fontsize=12)
ax.set_title('GMM: Optimal Number of Components', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# Mark optimal
ax.scatter([4], [480], color=colors['mlgreen'], s=200, zorder=5)
ax.annotate('Optimal: 4 components\n(minimum score)', xy=(4, 480), xytext=(5.5, 600),
            arrowprops=dict(arrowstyle='->', color=colors['mlgreen'], lw=2),
            fontsize=10, color=colors['mlgreen'], fontweight='bold')

fig.suptitle('Parameter Tuning Guidelines - Part 2: Advanced Methods', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('parameter_tuning_part2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('parameter_tuning_part2.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll improved charts created successfully!")
print("Created files:")
print("- common_mistakes.pdf/png (improved)")
print("- parameter_tuning_part1.pdf/png")
print("- parameter_tuning_part2.pdf/png")