"""
Create missing charts for Week 1 compilation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlgray': '#808080'
}

def save_figure(name):
    """Save figure in both PDF and PNG formats"""
    plt.savefig(f'{name}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{name}.png', dpi=150, bbox_inches='tight', format='png')
    plt.close()

def create_dual_pipeline_overview():
    """Create dual pipeline overview chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Left: ML Pipeline
    ax1.set_title('Machine Learning Pipeline', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    steps = ['Data Collection', 'Preprocessing', 'Clustering', 'Validation', 'Insights']
    y_positions = np.linspace(8, 2, len(steps))

    for i, (step, y) in enumerate(zip(steps, y_positions)):
        rect = FancyBboxPatch((1, y-0.4), 8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['mlblue'],
                              alpha=0.3 + i*0.15,
                              edgecolor=colors['mlblue'],
                              linewidth=2)
        ax1.add_patch(rect)
        ax1.text(5, y, step, ha='center', va='center', fontsize=11, fontweight='bold')

        if i < len(steps) - 1:
            arrow = patches.FancyArrowPatch((5, y-0.5), (5, y_positions[i+1]+0.5),
                                          connectionstyle="arc3",
                                          arrowstyle='->',
                                          lw=2,
                                          color=colors['mlgray'])
            ax1.add_patch(arrow)

    # Right: Design Thinking Pipeline
    ax2.set_title('Design Thinking Pipeline', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    steps = ['Empathize', 'Define', 'Ideate', 'Prototype', 'Test']
    y_positions = np.linspace(8, 2, len(steps))

    for i, (step, y) in enumerate(zip(steps, y_positions)):
        rect = FancyBboxPatch((1, y-0.4), 8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['mlorange'],
                              alpha=0.3 + i*0.15,
                              edgecolor=colors['mlorange'],
                              linewidth=2)
        ax2.add_patch(rect)
        ax2.text(5, y, step, ha='center', va='center', fontsize=11, fontweight='bold')

        if i < len(steps) - 1:
            arrow = patches.FancyArrowPatch((5, y-0.5), (5, y_positions[i+1]+0.5),
                                          connectionstyle="arc3",
                                          arrowstyle='->',
                                          lw=2,
                                          color=colors['mlgray'])
            ax2.add_patch(arrow)

    plt.suptitle('Dual Pipeline Approach: ML + Design Thinking', fontsize=16, fontweight='bold', y=0.98)
    save_figure('dual_pipeline_overview')

def create_innovation_diamond_complete():
    """Create complete innovation diamond visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Diamond shape coordinates
    diamond_x = [0, 3, 6, 9, 12, 9, 6, 3, 0]
    diamond_y = [4, 6, 7, 6, 4, 2, 1, 2, 4]

    # Fill diamond
    ax.fill(diamond_x, diamond_y, color=colors['mlpurple'], alpha=0.2)
    ax.plot(diamond_x, diamond_y, color=colors['mlpurple'], linewidth=2)

    # Add stages
    stages = [
        (0, 4, '1\nChallenge', colors['mlpurple']),
        (3, 6, '10\nExplore', colors['mlblue']),
        (6, 7, '100\nGenerate', colors['mlgreen']),
        (9, 6, '5000\nPeak Ideas', colors['mlorange']),
        (12, 4, '500\nFilter', colors['mlred']),
        (9, 2, '50\nRefine', colors['mlpurple']),
        (6, 1, '5\nStrategy', colors['mlblue'])
    ]

    for x, y, label, color in stages:
        circle = Circle((x, y), 0.5, color=color, alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Add flow arrows
    flow_coords = [(0, 4, 3, 6), (3, 6, 6, 7), (6, 7, 9, 6),
                  (9, 6, 12, 4), (12, 4, 9, 2), (9, 2, 6, 1)]

    for x1, y1, x2, y2 in flow_coords:
        arrow = patches.FancyArrowPatch((x1+0.3, y1), (x2-0.3, y2),
                                      connectionstyle="arc3,rad=0.2",
                                      arrowstyle='->',
                                      lw=2,
                                      color=colors['mlgray'],
                                      alpha=0.6)
        ax.add_patch(arrow)

    # Labels
    ax.text(6, 8.5, 'Innovation Diamond', fontsize=16, fontweight='bold', ha='center')
    ax.text(6, 8, 'From Single Challenge to Strategic Solutions', fontsize=12, ha='center', style='italic')

    # Annotations
    ax.text(1.5, 5.5, 'Divergent\nThinking', fontsize=9, ha='center', color=colors['mlgray'])
    ax.text(10.5, 5.5, 'Convergent\nThinking', fontsize=9, ha='center', color=colors['mlgray'])

    ax.set_xlim(-1, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')

    save_figure('innovation_diamond_complete')

def create_kmeans_initialization():
    """Create K-means initialization visualization"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Random initialization
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=colors['mlgray'], alpha=0.6, s=50)
    random_centers = X[np.random.choice(len(X), 3, replace=False)]
    ax.scatter(random_centers[:, 0], random_centers[:, 1],
              c=colors['mlred'], s=200, marker='*', edgecolor='black', linewidth=2)
    ax.set_title('Random Initialization', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # K-means++ initialization
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c=colors['mlgray'], alpha=0.6, s=50)
    # Simulate k-means++ centers (simplified)
    kpp_centers = [X[0]]
    for _ in range(2):
        distances = np.min([np.sum((X - c)**2, axis=1) for c in kpp_centers], axis=0)
        probabilities = distances / distances.sum()
        cumulative = probabilities.cumsum()
        r = np.random.rand()
        kpp_centers.append(X[np.searchsorted(cumulative, r)])
    kpp_centers = np.array(kpp_centers)
    ax.scatter(kpp_centers[:, 0], kpp_centers[:, 1],
              c=colors['mlgreen'], s=200, marker='*', edgecolor='black', linewidth=2)
    ax.set_title('K-means++ Initialization', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')

    # Final clusters
    ax = axes[2]
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c=colors['mlpurple'], s=200, marker='*', edgecolor='black', linewidth=2)
    ax.set_title('Final Clusters', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')

    plt.suptitle('K-means Initialization Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('kmeans_initialization')

def create_kmeans_evolution():
    """Create K-means evolution/iteration visualization"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    # Manual K-means iterations
    centers = X[np.random.choice(len(X), 3, replace=False)]

    for iteration in range(6):
        ax = axes[iteration]

        # Assign points to nearest center
        distances = np.array([np.sum((X - c)**2, axis=1) for c in centers])
        labels = np.argmin(distances, axis=0)

        # Plot
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
        ax.scatter(centers[:, 0], centers[:, 1],
                  c='red', s=200, marker='*', edgecolor='black', linewidth=2)

        # Update centers
        new_centers = []
        for i in range(3):
            if np.sum(labels == i) > 0:
                new_centers.append(X[labels == i].mean(axis=0))
            else:
                new_centers.append(centers[i])

        # Draw movement arrows
        if iteration > 0:
            for old, new in zip(centers, new_centers):
                ax.annotate('', xy=new, xytext=old,
                           arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5))

        centers = np.array(new_centers)
        ax.set_title(f'Iteration {iteration + 1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1' if iteration >= 3 else '')
        ax.set_ylabel('Feature 2' if iteration % 3 == 0 else '')

    plt.suptitle('K-means Algorithm Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('kmeans_evolution')

def create_distance_metrics_comparison():
    """Create distance metrics comparison visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Sample points
    point1 = np.array([2, 2])
    point2 = np.array([5, 6])

    # Euclidean distance
    ax = axes[0]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.scatter(*point1, color=colors['mlblue'], s=100, zorder=3)
    ax.scatter(*point2, color=colors['mlred'], s=100, zorder=3)
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k--', alpha=0.5)
    ax.text(3.5, 4.5, f'd = {np.linalg.norm(point2 - point1):.2f}', fontsize=10)
    ax.set_title('Euclidean Distance', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)

    # Manhattan distance
    ax = axes[1]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.scatter(*point1, color=colors['mlblue'], s=100, zorder=3)
    ax.scatter(*point2, color=colors['mlred'], s=100, zorder=3)
    ax.plot([point1[0], point2[0], point2[0]], [point1[1], point1[1], point2[1]], 'k--', alpha=0.5)
    manhattan_dist = np.abs(point2[0] - point1[0]) + np.abs(point2[1] - point1[1])
    ax.text(3.5, 3, f'd = {manhattan_dist:.1f}', fontsize=10)
    ax.set_title('Manhattan Distance', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.grid(True, alpha=0.3)

    # Cosine similarity
    ax = axes[2]
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.arrow(0, 0, point1[0], point1[1], head_width=0.2, head_length=0.2,
            fc=colors['mlblue'], ec=colors['mlblue'])
    ax.arrow(0, 0, point2[0], point2[1], head_width=0.2, head_length=0.2,
            fc=colors['mlred'], ec=colors['mlred'])
    cosine_sim = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
    angle = np.arccos(cosine_sim) * 180 / np.pi
    ax.text(2, 5, f'θ = {angle:.1f}°\ncos(θ) = {cosine_sim:.3f}', fontsize=10)
    ax.set_title('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.grid(True, alpha=0.3)
    ax.scatter(0, 0, color='black', s=50, zorder=3)

    plt.suptitle('Distance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('distance_metrics_comparison')

def create_silhouette_analysis():
    """Create silhouette analysis visualization"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Clusters with silhouette scores
    ax = axes[0]
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    silhouette_vals = silhouette_samples(X, labels)

    scatter = ax.scatter(X[:, 0], X[:, 1], c=silhouette_vals, cmap='RdYlGn',
                        s=50, alpha=0.7, vmin=-0.2, vmax=1.0)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c='black', s=200, marker='*', edgecolor='white', linewidth=2)

    ax.set_title('Clusters with Silhouette Scores', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Silhouette Score', rotation=270, labelpad=20)

    # Right: Silhouette plot
    ax = axes[1]
    y_lower = 10

    for i in range(3):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()

        size_cluster = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster

        color = plt.cm.viridis(float(i) / 3)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
        y_lower = y_upper + 10

    ax.set_title('Silhouette Plot for Each Cluster', fontsize=12, fontweight='bold')
    ax.set_xlabel('Silhouette Score')
    ax.set_ylabel('Cluster')

    # Average silhouette score
    avg_score = silhouette_score(X, labels)
    ax.axvline(x=avg_score, color='red', linestyle='--', label=f'Average: {avg_score:.3f}')
    ax.legend()

    plt.suptitle('Silhouette Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('silhouette_analysis')

def create_dbscan_vs_kmeans():
    """Create DBSCAN vs K-means comparison"""
    np.random.seed(42)

    # Create non-globular clusters
    X1, _ = make_moons(n_samples=150, noise=0.05, random_state=42)
    X2, _ = make_blobs(n_samples=50, centers=1, n_features=2, random_state=42)
    X2 = X2 + [2, -0.5]
    X = np.vstack([X1, X2])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Original data
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=colors['mlgray'], alpha=0.6, s=50)
    ax.set_title('Original Data', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # K-means clustering
    ax = axes[1]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, s=50)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolor='black', linewidth=2)
    ax.set_title('K-means Clustering', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')

    # DBSCAN clustering
    ax = axes[2]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Plot DBSCAN results
    unique_labels = set(dbscan_labels)
    colors_list = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors_list):
        if k == -1:
            col = 'gray'
            marker = 'x'
        else:
            marker = 'o'

        class_member_mask = (dbscan_labels == k)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=marker, alpha=0.6,
                  label='Noise' if k == -1 else f'Cluster {k}')

    ax.set_title('DBSCAN Clustering', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('DBSCAN vs K-means: Non-Globular Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('dbscan_vs_kmeans')

# Create all missing charts
print("Creating missing charts...")

print("Creating dual_pipeline_overview.pdf...")
create_dual_pipeline_overview()

print("Creating innovation_diamond_complete.pdf...")
create_innovation_diamond_complete()

print("Creating kmeans_initialization.pdf...")
create_kmeans_initialization()

print("Creating kmeans_evolution.pdf...")
create_kmeans_evolution()

print("Creating distance_metrics_comparison.pdf...")
create_distance_metrics_comparison()

print("Creating silhouette_analysis.pdf...")
create_silhouette_analysis()

print("Creating dbscan_vs_kmeans.pdf...")
create_dbscan_vs_kmeans()

print("\nAll missing charts created successfully!")