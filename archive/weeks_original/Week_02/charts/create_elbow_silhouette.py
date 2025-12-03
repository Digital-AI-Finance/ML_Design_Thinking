"""
Week 2: Elbow Method and Silhouette Analysis for Optimal K
Shows both methods for determining the best number of clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

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

def generate_realistic_data(n_samples=2000):
    """Generate data with known cluster structure"""
    # Create 5 distinct clusters (known ground truth)
    centers = [
        [2, 2],    # Cluster 1
        [8, 2],    # Cluster 2
        [5, 8],    # Cluster 3
        [2, 7],    # Cluster 4
        [8, 8],    # Cluster 5
    ]
    
    X, y_true = make_blobs(n_samples=n_samples, centers=centers,
                          n_features=2, cluster_std=1.0,
                          random_state=42)
    return X, y_true

def create_elbow_analysis():
    """Create elbow method visualization"""
    X, _ = generate_realistic_data()
    
    # Calculate inertia for different K values
    K_range = range(2, 11)
    inertias = []
    silhouette_scores = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Elbow Method Plot
    ax1.plot(K_range, inertias, 'o-', linewidth=3, markersize=10, 
             color=colors['mlblue'], markerfacecolor=colors['mlorange'])
    
    # Mark the elbow (k=5)
    ax1.plot(5, inertias[3], 'o', markersize=20, color=colors['mlred'],
             markeredgewidth=3, markerfacecolor='none')
    ax1.annotate('Optimal K=5\n(Elbow Point)', xy=(5, inertias[3]),
                xytext=(5.5, inertias[3] + 5000),
                arrowprops=dict(arrowstyle='->', color=colors['mlred'], lw=2),
                fontsize=12, fontweight='bold', color=colors['mlred'])
    
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12, fontweight='bold')
    ax1.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(K_range)
    
    # Add trend lines to show elbow more clearly
    # Line before elbow
    ax1.plot([2, 5], [inertias[0], inertias[3]], '--', 
             alpha=0.5, color=colors['mlgray'], linewidth=2)
    # Line after elbow
    ax1.plot([5, 10], [inertias[3], inertias[-1]], '--', 
             alpha=0.5, color=colors['mlgray'], linewidth=2)
    
    # Silhouette Score Plot
    ax2.plot(K_range, silhouette_scores, 'o-', linewidth=3, markersize=10,
             color=colors['mlgreen'], markerfacecolor=colors['mlpurple'])
    
    # Mark the maximum (k=5)
    max_idx = np.argmax(silhouette_scores)
    ax2.plot(K_range[max_idx], silhouette_scores[max_idx], 'o', 
             markersize=20, color=colors['mlred'],
             markeredgewidth=3, markerfacecolor='none')
    ax2.annotate(f'Maximum at K={K_range[max_idx]}', 
                xy=(K_range[max_idx], silhouette_scores[max_idx]),
                xytext=(K_range[max_idx] + 0.5, silhouette_scores[max_idx] - 0.05),
                arrowprops=dict(arrowstyle='->', color=colors['mlred'], lw=2),
                fontsize=12, fontweight='bold', color=colors['mlred'])
    
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Silhouette Analysis for Optimal K', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(K_range)
    ax2.set_ylim([0.3, 0.7])
    
    # Add reference line at 0.5 (good clustering)
    ax2.axhline(y=0.5, color=colors['mlgray'], linestyle='--', 
                alpha=0.5, label='Good Clustering Threshold')
    ax2.legend(loc='lower right')
    
    plt.suptitle('Determining Optimal Number of Clusters: Two Methods Agree on K=5',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def create_silhouette_detailed():
    """Create detailed silhouette plot for different K values"""
    X, _ = generate_realistic_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    K_values = [2, 3, 4, 5, 6, 7]
    
    for idx, k in enumerate(K_values):
        ax = axes[idx]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        y_lower = 10
        for i in range(k):
            # Get silhouette scores for samples in cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            y_lower = y_upper + 10
        
        ax.set_title(f'K = {k}\nAvg Score: {silhouette_avg:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        
        # The vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color=colors['mlred'], linestyle="--", linewidth=2)
        
        # Set x-axis limits
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X) + (k + 1) * 10])
        
        # Remove y-axis labels
        ax.set_yticks([])
        
        # Highlight the optimal K=5
        if k == 5:
            ax.patch.set_facecolor('#ffffcc')
            ax.patch.set_alpha(0.3)
    
    plt.suptitle('Silhouette Analysis for K = 2 through 7\n(K=5 highlighted as optimal)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_comparison_visualization():
    """Create side-by-side comparison of different K values"""
    X, _ = generate_realistic_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    K_values = [2, 3, 4, 5, 6, 7]
    
    for idx, k in enumerate(K_values):
        ax = axes[idx]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)
        
        # Plot
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                           s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plot centers
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                  marker='*', edgecolors='black', linewidth=2)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, y_pred)
        
        ax.set_title(f'K = {k}\nInertia: {inertia:.0f}\nSilhouette: {silhouette:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Highlight the optimal K=5
        if k == 5:
            ax.patch.set_facecolor('#ccffcc')
            ax.patch.set_alpha(0.3)
            for spine in ax.spines.values():
                spine.set_edgecolor(colors['mlgreen'])
                spine.set_linewidth(3)
    
    plt.suptitle('Clustering Results for Different K Values\n(K=5 shows best natural grouping)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Create elbow and silhouette analysis
    fig1 = create_elbow_analysis()
    plt.savefig('elbow_silhouette_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('elbow_silhouette_analysis.png', dpi=150, bbox_inches='tight')
    print("Created elbow_silhouette_analysis.pdf/png")
    
    # Create detailed silhouette plots
    fig2 = create_silhouette_detailed()
    plt.savefig('silhouette_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('silhouette_detailed.png', dpi=150, bbox_inches='tight')
    print("Created silhouette_detailed.pdf/png")
    
    # Create comparison visualization
    fig3 = create_comparison_visualization()
    plt.savefig('clustering_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
    print("Created clustering_comparison.pdf/png")
    
    plt.close('all')