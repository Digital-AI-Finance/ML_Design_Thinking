"""
Week 2: Advanced Clustering Techniques
Shows DBSCAN, Hierarchical, and Gaussian Mixture Models
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f',
    'mlyellow': '#bcbd22',
    'mlcyan': '#17becf'
}

def generate_complex_data():
    """Generate data with different structures for clustering comparison"""
    # Dataset 1: Well-separated blobs (good for K-means)
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, n_features=2,
                                  cluster_std=0.5, random_state=42)
    
    # Dataset 2: Non-convex shapes (good for DBSCAN)
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # Dataset 3: Nested circles (challenging for most)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, 
                                        factor=0.5, random_state=42)
    
    # Dataset 4: Elongated clusters (good for GMM)
    np.random.seed(42)
    X_elongated = np.vstack([
        np.random.randn(100, 2) * [2, 0.5] + [2, 2],
        np.random.randn(100, 2) * [0.5, 2] + [6, 2],
        np.random.randn(100, 2) * [1, 1] + [4, 6]
    ])
    y_elongated = np.array([0]*100 + [1]*100 + [2]*100)
    
    return (X_blobs, y_blobs), (X_moons, y_moons), (X_circles, y_circles), (X_elongated, y_elongated)

def create_clustering_comparison():
    """Compare different clustering algorithms on various datasets"""
    datasets = generate_complex_data()
    dataset_names = ['Well-Separated', 'Moons (Non-convex)', 'Circles (Nested)', 'Elongated']
    
    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    
    algorithms = [
        ('Ground Truth', None),
        ('K-Means', KMeans(n_clusters=3, random_state=42)),
        ('DBSCAN', DBSCAN(eps=0.3, min_samples=5)),
        ('Hierarchical', AgglomerativeClustering(n_clusters=3)),
        ('Gaussian Mixture', GaussianMixture(n_components=3, random_state=42))
    ]
    
    for row, ((X, y_true), name) in enumerate(zip(datasets, dataset_names)):
        # Standardize data
        X_scaled = StandardScaler().fit_transform(X)
        
        for col, (algo_name, algorithm) in enumerate(algorithms):
            ax = axes[row, col]
            
            if col == 0:
                # Ground truth
                scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                    c=y_true, cmap='tab10', s=20, alpha=0.7)
                ax.set_ylabel(name, fontsize=10, fontweight='bold')
            else:
                # Apply clustering
                if algo_name == 'DBSCAN':
                    # Adjust eps for each dataset
                    if row == 0:  # blobs
                        algorithm.eps = 0.5
                    elif row == 1:  # moons
                        algorithm.eps = 0.2
                    elif row == 2:  # circles
                        algorithm.eps = 0.2
                    else:  # elongated
                        algorithm.eps = 0.4
                
                if algo_name == 'Gaussian Mixture':
                    labels = algorithm.fit_predict(X_scaled)
                else:
                    labels = algorithm.fit_predict(X_scaled)
                
                # Handle noise points in DBSCAN (label -1)
                unique_labels = np.unique(labels)
                colors_map = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for k, col_map in zip(unique_labels, colors_map):
                    if k == -1:
                        # Noise points in black
                        col_map = [0, 0, 0, 1]
                        marker = 'x'
                    else:
                        marker = 'o'
                    
                    class_member_mask = (labels == k)
                    xy = X_scaled[class_member_mask]
                    ax.scatter(xy[:, 0], xy[:, 1], s=20, c=[col_map],
                             marker=marker, alpha=0.7)
            
            if row == 0:
                ax.set_title(algo_name, fontsize=11, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Clustering Algorithm Comparison: Different Methods for Different Data',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_dbscan_visualization():
    """Detailed DBSCAN visualization showing core, border, and noise points"""
    # Generate data with noise
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2,
                     cluster_std=0.4, random_state=42)
    # Add noise points
    noise = np.random.uniform(-6, 6, (50, 2))
    X = np.vstack([X, noise])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Different eps values
    eps_values = [0.3, 0.5, 0.8]
    
    for ax, eps in zip(axes, eps_values):
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        # Get core samples mask
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        
        # Number of clusters (ignoring noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Plot
        unique_labels = set(labels)
        colors_list = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors_list):
            if k == -1:
                # Black for noise
                col = [0, 0, 0, 1]
            
            class_member_mask = (labels == k)
            
            # Core points (large)
            xy = X[class_member_mask & core_samples_mask]
            ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker='o',
                      s=50, alpha=0.8, edgecolors='black', linewidth=0.5,
                      label='Core' if k == 0 else '')
            
            # Border points (small)
            xy = X[class_member_mask & ~core_samples_mask]
            ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker='o',
                      s=20, alpha=0.6, edgecolors='black', linewidth=0.5,
                      label='Border' if k == 0 else '')
        
        # Add eps circle around a sample point
        sample_point = X[0]
        circle = plt.Circle((sample_point[0], sample_point[1]), eps,
                           color='red', fill=False, linestyle='--', linewidth=2)
        ax.add_patch(circle)
        ax.scatter(sample_point[0], sample_point[1], c='red', s=100,
                  marker='*', edgecolors='black', linewidth=1)
        
        ax.set_title(f'eps = {eps}\nClusters: {n_clusters}, Noise: {n_noise}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('DBSCAN: Density-Based Clustering with Different eps Values',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_hierarchical_visualization():
    """Create hierarchical clustering dendrogram and step-by-step merging"""
    # Generate small dataset for clarity
    X, y = make_blobs(n_samples=20, centers=3, n_features=2,
                     cluster_std=0.5, random_state=42)
    
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Dendrogram
    ax1 = plt.subplot(121)
    
    # Compute linkage
    Z = linkage(X, method='ward')
    
    # Create dendrogram
    dendrogram(Z, ax=ax1, color_threshold=0,
              above_threshold_color='black')
    ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Distance (Ward)', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add cut line for 3 clusters
    cut_height = Z[-3, 2] + 0.5
    ax1.axhline(y=cut_height, c='red', linestyle='--', linewidth=2,
               label=f'Cut for 3 clusters')
    ax1.legend(loc='upper right')
    
    # Right: Different number of clusters
    ax2 = plt.subplot(122)
    
    # Show clustering for different n_clusters
    n_clusters_list = [2, 3, 4, 5]
    colors_palette = sns.color_palette('husl', 5)
    
    for i, n_clusters in enumerate(n_clusters_list):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X)
        
        # Create subplot
        ax_sub = plt.subplot(2, 4, 5 + i)
        
        for k in range(n_clusters):
            mask = labels == k
            ax_sub.scatter(X[mask, 0], X[mask, 1], 
                          c=[colors_palette[k]], s=30, alpha=0.7,
                          edgecolors='black', linewidth=0.5)
        
        ax_sub.set_title(f'{n_clusters} Clusters', fontsize=10)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])
        ax_sub.grid(True, alpha=0.3)
    
    plt.suptitle('Hierarchical Clustering: Dendrogram and Cluster Formation',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_gmm_visualization():
    """Visualize Gaussian Mixture Models with ellipses"""
    # Generate elongated clusters
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(150, 2) * [2, 0.5] + [2, 2],
        np.random.randn(150, 2) * [0.5, 2] + [6, 2],
        np.random.randn(150, 2) * [1, 1] + [4, 6]
    ])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Different covariance types
    cov_types = ['full', 'tied', 'diag']
    
    for ax, cov_type in zip(axes, cov_types):
        # Fit GMM
        gmm = GaussianMixture(n_components=3, covariance_type=cov_type,
                             random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        
        # Plot points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10',
                            s=20, alpha=0.6)
        
        # Plot Gaussian ellipses
        for i in range(gmm.n_components):
            # Get mean and covariance
            mean = gmm.means_[i]
            
            if cov_type == 'full':
                cov = gmm.covariances_[i]
            elif cov_type == 'tied':
                cov = gmm.covariances_
            elif cov_type == 'diag':
                cov = np.diag(gmm.covariances_[i])
            elif cov_type == 'spherical':
                cov = np.eye(2) * gmm.covariances_[i]
            
            # Calculate ellipse parameters
            v, w = np.linalg.eigh(cov)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            angle = np.degrees(np.arctan2(w[0][1], w[0][0]))
            
            # Draw ellipse
            from matplotlib.patches import Ellipse
            ellipse = Ellipse(mean, v[0], v[1], angle=angle,
                             facecolor='none', edgecolor=plt.cm.tab10(i),
                             linewidth=2, linestyle='-', alpha=0.8)
            ax.add_patch(ellipse)
            
            # Mark center
            ax.scatter(mean[0], mean[1], c='black', s=100, marker='+',
                      linewidth=2)
        
        ax.set_title(f'GMM: {cov_type} covariance', fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
    
    plt.suptitle('Gaussian Mixture Models: Probabilistic Clustering with Different Covariances',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_method_selection_guide():
    """Create a visual guide for selecting clustering methods"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Clustering Method Selection Guide', 
           fontsize=18, fontweight='bold', ha='center')
    
    # Method boxes
    methods = [
        {
            'name': 'K-Means',
            'pos': (2, 7),
            'color': colors['mlblue'],
            'pros': ['Fast', 'Scalable', 'Simple'],
            'cons': ['Fixed K', 'Spherical', 'Sensitive'],
            'use_when': 'Well-separated,\nspherical clusters'
        },
        {
            'name': 'DBSCAN',
            'pos': (5, 7),
            'color': colors['mlorange'],
            'pros': ['No K needed', 'Any shape', 'Noise handling'],
            'cons': ['Parameters', 'Density', 'Memory'],
            'use_when': 'Arbitrary shapes,\nnoise present'
        },
        {
            'name': 'Hierarchical',
            'pos': (8, 7),
            'color': colors['mlgreen'],
            'pros': ['Dendrogram', 'No K upfront', 'Interpretable'],
            'cons': ['Slow', 'Memory', 'No undo'],
            'use_when': 'Need hierarchy,\nsmall datasets'
        },
        {
            'name': 'GMM',
            'pos': (3.5, 3.5),
            'color': colors['mlpurple'],
            'pros': ['Soft clustering', 'Flexible', 'Probabilistic'],
            'cons': ['Complex', 'Slow', 'Assumptions'],
            'use_when': 'Overlapping,\nelliptical clusters'
        },
        {
            'name': 'Mean Shift',
            'pos': (6.5, 3.5),
            'color': colors['mlred'],
            'pros': ['No K', 'Robust', 'Modes'],
            'cons': ['Very slow', 'Bandwidth', 'Memory'],
            'use_when': 'Mode seeking,\ncomputer vision'
        }
    ]
    
    for method in methods:
        # Method box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((method['pos'][0] - 1.2, method['pos'][1] - 1.2),
                            2.4, 2, boxstyle="round,pad=0.1",
                            facecolor=method['color'], alpha=0.2,
                            edgecolor=method['color'], linewidth=2)
        ax.add_patch(box)
        
        # Method name
        ax.text(method['pos'][0], method['pos'][1] + 0.6, method['name'],
               fontsize=12, fontweight='bold', ha='center',
               color=method['color'])
        
        # Pros
        ax.text(method['pos'][0], method['pos'][1] + 0.2, 'Pros:',
               fontsize=9, fontweight='bold', ha='center')
        pros_text = ' '.join(method['pros'])
        ax.text(method['pos'][0], method['pos'][1], pros_text,
               fontsize=8, ha='center', color=colors['mlgreen'])
        
        # Cons
        ax.text(method['pos'][0], method['pos'][1] - 0.3, 'Cons:',
               fontsize=9, fontweight='bold', ha='center')
        cons_text = ' '.join(method['cons'])
        ax.text(method['pos'][0], method['pos'][1] - 0.5, cons_text,
               fontsize=8, ha='center', color=colors['mlred'])
        
        # Use when
        ax.text(method['pos'][0], method['pos'][1] - 0.9, method['use_when'],
               fontsize=8, ha='center', style='italic',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Decision flow arrows
    from matplotlib.patches import FancyArrowPatch
    
    # Question at top
    ax.text(5, 1.5, 'Key Question: Do you know the number of clusters?',
           fontsize=11, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['mlyellow'], alpha=0.3))
    
    ax.text(2.5, 0.8, 'YES → K-Means, GMM', fontsize=10, ha='center',
           color=colors['mlblue'])
    ax.text(7.5, 0.8, 'NO → DBSCAN, Hierarchical, Mean Shift', fontsize=10, ha='center',
           color=colors['mlorange'])
    
    return fig

# Main execution
if __name__ == "__main__":
    # Create clustering comparison
    fig1 = create_clustering_comparison()
    plt.savefig('clustering_algorithm_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('clustering_algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("Created clustering_algorithm_comparison.pdf/png")
    
    # Create DBSCAN visualization
    fig2 = create_dbscan_visualization()
    plt.savefig('dbscan_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('dbscan_detailed.png', dpi=150, bbox_inches='tight')
    print("Created dbscan_detailed.pdf/png")
    
    # Create hierarchical visualization
    fig3 = create_hierarchical_visualization()
    plt.savefig('hierarchical_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('hierarchical_detailed.png', dpi=150, bbox_inches='tight')
    print("Created hierarchical_detailed.pdf/png")
    
    # Create GMM visualization
    fig4 = create_gmm_visualization()
    plt.savefig('gmm_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('gmm_detailed.png', dpi=150, bbox_inches='tight')
    print("Created gmm_detailed.pdf/png")
    
    # Create method selection guide
    fig5 = create_method_selection_guide()
    plt.savefig('clustering_selection_guide.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('clustering_selection_guide.png', dpi=150, bbox_inches='tight')
    print("Created clustering_selection_guide.pdf/png")
    
    plt.close('all')