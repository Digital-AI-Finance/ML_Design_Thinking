"""
Chart Generation for Week 0 Part 3: Unsupervised Learning
WCAG AAA Compliant Color Palette
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_circles
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import umap

# WCAG AAA Compliant Colors
COLORS = {
    'blue': '#1F77B4',
    'orange': '#FF7F0E',
    'green': '#2CA02C',
    'red': '#D62728',
    'purple': '#9467BD',
    'brown': '#8C564B',
    'pink': '#E377C2',
    'gray': '#7F7F7F',
    'olive': '#BCBD22',
    'cyan': '#17BECF'
}

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'


def create_kmeans_clustering():
    """Create K-means clustering process visualization"""
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, centers=3, n_features=2,
                     cluster_std=0.6, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    steps = [
        ('Initial Data', None),
        ('Random Initialization (k=3)', 'random'),
        ('After 5 Iterations', 5),
        ('Converged Solution', 20)
    ]

    for ax, (title, n_iter) in zip(axes.flat, steps):
        if n_iter is None:
            ax.scatter(X[:, 0], X[:, 1], c=COLORS['gray'], s=80, alpha=0.6, edgecolors='black')
        else:
            kmeans = KMeans(n_clusters=3, init='random' if n_iter == 'random' else 'k-means++',
                          max_iter=1 if n_iter == 'random' else n_iter, n_init=1, random_state=42)
            kmeans.fit(X)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]
            for i in range(3):
                mask = labels == i
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=80, alpha=0.6,
                          edgecolors='black', label=f'Cluster {i}')

            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.8,
                      edgecolors='black', marker='X', linewidths=3, label='Centroids')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('K-Means Clustering Process', fontsize=16, fontweight='bold', y=1.00)

    formula_text = r'Minimize: $\sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$'
    fig.text(0.5, 0.02, formula_text, ha='center', fontsize=13, style='italic')

    plt.tight_layout()
    plt.savefig('../charts/kmeans_clustering.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/kmeans_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created kmeans_clustering.pdf")


def create_hierarchical_clustering():
    """Create hierarchical clustering dendrogram"""
    np.random.seed(42)
    X, y = make_blobs(n_samples=50, centers=4, n_features=2,
                     cluster_std=0.6, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    linkage_matrix = linkage(X, method='ward')
    dendrogram(linkage_matrix, ax=ax1, color_threshold=10, above_threshold_color='gray')
    ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance', fontsize=12, fontweight='bold')
    ax1.axhline(y=10, c='red', linestyle='--', linewidth=2, label='Cut Threshold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    agg = AgglomerativeClustering(n_clusters=4)
    labels = agg.fit_predict(X)

    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red']]
    for i in range(4):
        mask = labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=100, alpha=0.6,
                   edgecolors='black', label=f'Cluster {i}')

    ax2.set_title('Resulting Clusters (k=4)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../charts/hierarchical_clustering.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/hierarchical_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created hierarchical_clustering.pdf")


def create_dbscan_clustering():
    """Create DBSCAN density-based clustering"""
    np.random.seed(42)
    X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    ax1.scatter(X[kmeans_labels==0, 0], X[kmeans_labels==0, 1],
               c=COLORS['blue'], s=60, alpha=0.6, edgecolors='black', label='Cluster 0')
    ax1.scatter(X[kmeans_labels==1, 0], X[kmeans_labels==1, 1],
               c=COLORS['orange'], s=60, alpha=0.6, edgecolors='black', label='Cluster 1')
    ax1.set_title('K-Means (Fails on Non-Convex)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    unique_labels = set(dbscan_labels)
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]

    for i, label in enumerate(unique_labels):
        if label == -1:
            col = COLORS['gray']
            label_name = 'Noise'
        else:
            col = colors[label % len(colors)]
            label_name = f'Cluster {label}'

        mask = dbscan_labels == label
        ax2.scatter(X[mask, 0], X[mask, 1], c=col, s=60, alpha=0.6,
                   edgecolors='black', label=label_name)

    ax2.set_title('DBSCAN (Handles Non-Convex)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Density-Based Clustering vs K-Means', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../charts/dbscan_clustering.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/dbscan_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created dbscan_clustering.pdf")


def create_pca_analysis():
    """Create PCA visualization"""
    np.random.seed(42)
    n_samples = 200
    theta = np.linspace(0, 2 * np.pi, n_samples)
    X = np.column_stack([
        2 * np.cos(theta) + np.random.normal(0, 0.3, n_samples),
        np.sin(theta) + np.random.normal(0, 0.3, n_samples)
    ])

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(X[:, 0], X[:, 1], c=COLORS['blue'], s=50, alpha=0.5, edgecolors='black')

    mean = np.mean(X, axis=0)
    components = pca.components_
    explained_var = pca.explained_variance_

    for i, (comp, var) in enumerate(zip(components, explained_var)):
        color = COLORS['red'] if i == 0 else COLORS['orange']
        label = f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)'

        ax1.arrow(mean[0], mean[1], comp[0]*np.sqrt(var)*2, comp[1]*np.sqrt(var)*2,
                 head_width=0.15, head_length=0.2, fc=color, ec=color, linewidth=3,
                 label=label, alpha=0.8)

    ax1.set_xlabel('Original Feature 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Original Feature 2', fontsize=12, fontweight='bold')
    ax1.set_title('Original Data with Principal Components', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=COLORS['green'], s=50, alpha=0.5, edgecolors='black')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
    ax2.set_title('Transformed Data in PC Space', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.suptitle('Principal Component Analysis (PCA)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../charts/pca_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created pca_analysis.pdf")


def create_autoencoder_architecture():
    """Create autoencoder network structure"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    layer_configs = [
        ('Input\n(784)', 2, 8, COLORS['blue']),
        ('Hidden 1\n(256)', 4.5, 6, COLORS['orange']),
        ('Latent\n(32)', 7, 4, COLORS['red']),
        ('Hidden 2\n(256)', 9.5, 6, COLORS['orange']),
        ('Output\n(784)', 12, 8, COLORS['green'])
    ]

    for name, x, n_neurons, color in layer_configs:
        y_positions = np.linspace(2, 8, min(n_neurons, 8))

        for y in y_positions:
            circle = plt.Circle((x, y), 0.25, color=color, alpha=0.6, edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)

        ax.text(x, 9.2, name, ha='center', va='top', fontsize=11, fontweight='bold')

    ax.text(3.25, 1, 'Encoder', ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['orange'], alpha=0.3))

    ax.text(10.75, 1, 'Decoder', ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['green'], alpha=0.3))

    ax.plot([3.25, 3.25], [1.5, 9.5], 'k--', linewidth=2, alpha=0.3)
    ax.plot([10.75, 10.75], [1.5, 9.5], 'k--', linewidth=2, alpha=0.3)

    ax.text(7, 9.7, 'Autoencoder Architecture', ha='center', fontsize=16, fontweight='bold')

    loss_text = r'Loss: $L = ||x - \hat{x}||^2$ (Reconstruction Error)'
    ax.text(7, 0.5, loss_text, ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.savefig('../charts/autoencoder_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/autoencoder_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created autoencoder_architecture.pdf")


def create_tsne_umap_comparison():
    """Create t-SNE vs UMAP comparison"""
    np.random.seed(42)
    n_samples = 300
    X, y = make_blobs(n_samples=n_samples, centers=5, n_features=50,
                     cluster_std=2.0, random_state=42)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    X_umap = reducer.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red'], COLORS['purple']]

    for i in range(5):
        mask = y == i
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[i], s=50,
                   alpha=0.6, edgecolors='black', label=f'Cluster {i}')

    ax1.set_title('t-SNE Projection', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for i in range(5):
        mask = y == i
        ax2.scatter(X_umap[mask, 0], X_umap[mask, 1], c=colors[i], s=50,
                   alpha=0.6, edgecolors='black', label=f'Cluster {i}')

    ax2.set_title('UMAP Projection', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Dimensionality Reduction: t-SNE vs UMAP (50D → 2D)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../charts/tsne_umap_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/tsne_umap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created tsne_umap_comparison.pdf")


if __name__ == '__main__':
    print("Generating Unsupervised Learning Charts...")
    create_kmeans_clustering()
    create_hierarchical_clustering()
    create_dbscan_clustering()
    create_pca_analysis()
    create_autoencoder_architecture()
    create_tsne_umap_comparison()
    print("[OK] All unsupervised learning charts created successfully!")