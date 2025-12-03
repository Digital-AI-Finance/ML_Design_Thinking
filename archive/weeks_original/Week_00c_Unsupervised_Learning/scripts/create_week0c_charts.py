#!/usr/bin/env python3
"""
Chart generation script for Week 0c: Unsupervised Learning
Creates 15 visualization charts for the presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create charts directory
import os
chart_dir = os.path.join('..', 'charts')
os.makedirs(chart_dir, exist_ok=True)

def save_chart(fig, name):
    """Save chart in both PDF and PNG formats"""
    pdf_path = os.path.join(chart_dir, f'{name}.pdf')
    png_path = os.path.join(chart_dir, f'{name}.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name}")

def create_customer_data_sample():
    """Chart 1: Customer data sample visualization"""
    np.random.seed(42)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Generate sample customer data
    customers = pd.DataFrame({
        'Customer_ID': [f'C{i:04d}' for i in range(1, 11)],
        'Spending': np.random.uniform(200, 2000, 10),
        'Visits': np.random.randint(5, 50, 10),
        'Age': np.random.randint(25, 65, 10)
    })

    # Create table
    ax.axis('tight')
    ax.axis('off')

    table_data = customers.round(0).astype({'Spending': int, 'Visits': int, 'Age': int})
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Header styling
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')

    ax.set_title('Sample Customer Dataset (First 10 Records)', fontsize=14, fontweight='bold', pad=20)

    save_chart(fig, 'customer_data_sample')

def create_distance_calculation():
    """Chart 2: Distance calculation visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Customer points
    customer_a = np.array([500, 25])
    customer_b = np.array([520, 28])

    # Plot points
    ax1.scatter(*customer_a, color='red', s=100, label='Customer A', zorder=5)
    ax1.scatter(*customer_b, color='blue', s=100, label='Customer B', zorder=5)

    # Draw distance line
    ax1.plot([customer_a[0], customer_b[0]], [customer_a[1], customer_b[1]],
             'k--', linewidth=2, alpha=0.7, label='Distance = 20.3')

    ax1.set_xlabel('Spending ($)')
    ax1.set_ylabel('Visits per Month')
    ax1.set_title('Euclidean Distance Calculation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distance calculation breakdown
    ax2.axis('off')

    calculation_text = """
    Distance Calculation:

    Customer A: [500, 25]
    Customer B: [520, 28]

    d = √[(500-520)² + (25-28)²]
    d = √[(-20)² + (-3)²]
    d = √[400 + 9]
    d = √409 = 20.23
    """

    ax2.text(0.1, 0.5, calculation_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    save_chart(fig, 'distance_calculation')

def create_validation_problem():
    """Chart 3: Validation problem illustration"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Generate sample data
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=42)

    # Different clustering interpretations
    kmeans_3 = KMeans(n_clusters=3, random_state=42)
    kmeans_2 = KMeans(n_clusters=2, random_state=42)
    kmeans_5 = KMeans(n_clusters=5, random_state=42)

    labels_3 = kmeans_3.fit_predict(X)
    labels_2 = kmeans_2.fit_predict(X)
    labels_5 = kmeans_5.fit_predict(X)

    # Plot different clusterings
    ax1.scatter(X[:, 0], X[:, 1], c=labels_2, cmap='viridis', alpha=0.7)
    ax1.set_title('Geographic Segments (k=2)')

    ax2.scatter(X[:, 0], X[:, 1], c=labels_3, cmap='viridis', alpha=0.7)
    ax2.set_title('Spending-based Groups (k=3)')

    ax3.scatter(X[:, 0], X[:, 1], c=labels_5, cmap='viridis', alpha=0.7)
    ax3.set_title('Detailed Demographics (k=5)')

    # Question mark plot
    ax4.text(0.5, 0.5, '?', fontsize=100, ha='center', va='center',
             color='red', fontweight='bold')
    ax4.set_title('Which is "Correct"?')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    save_chart(fig, 'validation_problem')

def create_elbow_method():
    """Chart 4: Elbow method for k-selection"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=42)

    # Calculate WCSS for different k values
    k_range = range(1, 11)
    wcss = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Optimal k=4')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax.set_title('Elbow Method for Optimal k Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate the elbow
    ax.annotate('Elbow Point', xy=(4, wcss[3]), xytext=(6, wcss[3] + 50),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold')

    save_chart(fig, 'elbow_method')

def create_silhouette_analysis():
    """Chart 5: Silhouette analysis visualization"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    # Silhouette plot
    from sklearn.metrics import silhouette_samples
    silhouette_scores = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(3):
        cluster_silhouette_scores = silhouette_scores[labels == i]
        cluster_silhouette_scores.sort()

        size_cluster_i = cluster_silhouette_scores.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / 3)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_scores,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_xlabel('Silhouette Score')
    ax1.set_ylabel('Cluster Label')
    ax1.set_title('Silhouette Plot for Individual Samples')

    # Average silhouette score
    avg_score = silhouette_score(X, labels)
    ax1.axvline(x=avg_score, color="red", linestyle="--",
                label=f'Average Score: {avg_score:.3f}')
    ax1.legend()

    # Cluster visualization
    ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax2.set_title('Cluster Visualization')
    ax2.legend()

    plt.tight_layout()
    save_chart(fig, 'silhouette_analysis')

def create_kmeans_steps():
    """Chart 6: K-means algorithm steps"""
    np.random.seed(42)
    X = np.array([[2, 3], [3, 4], [8, 7], [9, 8], [1, 2], [7, 9]])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Step 1: Initial random centroids
    initial_centroids = np.array([[1, 1], [6, 6]])
    ax1.scatter(X[:, 0], X[:, 1], c='gray', s=100, alpha=0.7, label='Data Points')
    ax1.scatter(initial_centroids[:, 0], initial_centroids[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='Initial Centroids')
    ax1.set_title('Step 1: Random Initialization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Step 2: Assign to nearest centroid
    from scipy.spatial.distance import cdist
    distances = cdist(X, initial_centroids)
    labels = np.argmin(distances, axis=1)

    ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    ax2.scatter(initial_centroids[:, 0], initial_centroids[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    ax2.set_title('Step 2: Assign to Nearest')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Step 3: Update centroids
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(2)])

    ax3.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    ax3.scatter(new_centroids[:, 0], new_centroids[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='Updated Centroids')
    ax3.set_title('Step 3: Update Centroids')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Step 4: Repeat until convergence
    ax4.text(0.5, 0.5, 'Repeat Until\nConvergence', fontsize=16,
             ha='center', va='center', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Step 4: Iterate')

    plt.tight_layout()
    save_chart(fig, 'kmeans_steps')

def create_kmeans_example():
    """Chart 7: Worked example with coordinates"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Data points
    points = np.array([[2, 3], [3, 4], [8, 7], [9, 8]])
    point_labels = ['A', 'B', 'C', 'D']

    # Initial centroids
    centroids = np.array([[1, 1], [6, 6]])

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=150, alpha=0.7, label='Data Points')
    for i, label in enumerate(point_labels):
        ax.annotate(label, (points[i, 0], points[i, 1]), xytext=(5, 5),
                   textcoords='offset points', fontsize=12, fontweight='bold')

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x',
               s=200, linewidths=3, label='Initial Centroids')
    ax.annotate('μ₁', (centroids[0, 0], centroids[0, 1]), xytext=(5, 5),
               textcoords='offset points', fontsize=12, fontweight='bold', color='red')
    ax.annotate('μ₂', (centroids[1, 0], centroids[1, 1]), xytext=(5, 5),
               textcoords='offset points', fontsize=12, fontweight='bold', color='red')

    # Draw distance lines for point A
    ax.plot([points[0, 0], centroids[0, 0]], [points[0, 1], centroids[0, 1]],
            'g--', alpha=0.7, label='Distance to μ₁ = 2.24')
    ax.plot([points[0, 0], centroids[1, 0]], [points[0, 1], centroids[1, 1]],
            'orange', linestyle='--', alpha=0.7, label='Distance to μ₂ = 5.0')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('K-means Example: Distance Calculations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_chart(fig, 'kmeans_example')

def create_kmeans_success():
    """Chart 8: K-means success case"""
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True clusters
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax1.set_title('Ground Truth: Well-Separated Spherical Clusters')

    # K-means result
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', s=200, linewidths=3, color='red', label='Centroids')

    # Calculate silhouette score
    sil_score = silhouette_score(X, labels)
    ax2.set_title(f'K-means Result (Silhouette: {sil_score:.3f})')
    ax2.legend()

    plt.tight_layout()
    save_chart(fig, 'kmeans_success')

def create_kmeans_failure():
    """Chart 9: K-means failure on non-convex shapes"""
    np.random.seed(42)
    X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True clusters (moons)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax1.set_title('Ground Truth: Crescent-Shaped Clusters')

    # K-means result
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)

    ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', s=200, linewidths=3, color='red', label='Centroids')

    # Calculate silhouette score
    sil_score = silhouette_score(X, labels)
    ax2.set_title(f'K-means Failure (Silhouette: {sil_score:.3f})')
    ax2.legend()

    plt.tight_layout()
    save_chart(fig, 'kmeans_failure')

def create_crescent_data_table():
    """Chart 10: Crescent data table showing wrong assignments"""
    np.random.seed(42)
    X, y_true = make_moons(n_samples=20, noise=0.1, random_state=42)

    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Create DataFrame
    df = pd.DataFrame({
        'Point': [f'P{i+1:02d}' for i in range(20)],
        'X': X[:, 0].round(2),
        'Y': X[:, 1].round(2),
        'True_Cluster': y_true,
        'KMeans_Cluster': y_kmeans,
        'Correct': y_true == y_kmeans
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Create table with color coding
    table_data = df[['Point', 'X', 'Y', 'True_Cluster', 'KMeans_Cluster', 'Correct']]
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color incorrect assignments
    for i in range(len(df)):
        if not df.iloc[i]['Correct']:
            for j in range(len(table_data.columns)):
                table[(i+1, j)].set_facecolor('#FFB6C1')  # Light red

    # Header styling
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')

    ax.set_title('Crescent Data: K-means vs True Clustering\n(Red = Incorrect Assignment)',
                fontsize=14, fontweight='bold', pad=20)

    save_chart(fig, 'crescent_data_table')

def create_voronoi_boundaries():
    """Chart 11: Voronoi boundaries visualization"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Create mesh for decision boundaries
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

    # Plot data points and centroids
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200,
               linewidths=3, color='red', label='Centroids')

    ax.set_title('K-means Creates Voronoi Cell Boundaries\n(Always Convex Regions)')
    ax.legend()

    save_chart(fig, 'voronoi_boundaries')

def create_human_vs_kmeans():
    """Chart 12: Human vs K-means grouping comparison"""
    np.random.seed(42)
    X, _ = make_moons(n_samples=150, noise=0.15, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Human-like grouping (using DBSCAN as proxy)
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    human_labels = dbscan.fit_predict(X)

    ax1.scatter(X[:, 0], X[:, 1], c=human_labels, cmap='viridis', alpha=0.7)
    ax1.set_title('Human-like Grouping\n(Density-based, Natural Shapes)')

    # K-means grouping
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax2.set_title('K-means Grouping\n(Centroid-based, Artificial Split)')
    ax2.legend()

    plt.tight_layout()
    save_chart(fig, 'human_vs_kmeans')

def create_method_comparison():
    """Chart 13: Clustering method comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Generate different data types
    np.random.seed(42)

    # Spherical clusters
    X1, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)
    ax1.scatter(X1[:, 0], X1[:, 1], alpha=0.7)
    ax1.set_title('Spherical Clusters\nBest: K-means')

    # Moon shapes
    X2, _ = make_moons(n_samples=150, noise=0.1, random_state=42)
    ax2.scatter(X2[:, 0], X2[:, 1], alpha=0.7)
    ax2.set_title('Non-convex Shapes\nBest: DBSCAN')

    # Nested circles
    X3, _ = make_circles(n_samples=150, factor=0.6, noise=0.1, random_state=42)
    ax3.scatter(X3[:, 0], X3[:, 1], alpha=0.7)
    ax3.set_title('Nested Structures\nBest: Spectral')

    # Hierarchical structure
    X4 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50)
    X4 = np.vstack([X4, np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 25)])
    X4 = np.vstack([X4, np.random.multivariate_normal([0, 3], [[0.3, 0], [0, 0.3]], 25)])
    ax4.scatter(X4[:, 0], X4[:, 1], alpha=0.7)
    ax4.set_title('Hierarchical Structure\nBest: Agglomerative')

    plt.tight_layout()
    save_chart(fig, 'method_comparison')

def create_neighborhood_concept():
    """Chart 14: DBSCAN neighborhood concept"""
    np.random.seed(42)
    X = np.random.rand(30, 2) * 10

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot all points
    ax.scatter(X[:, 0], X[:, 1], c='lightblue', s=100, alpha=0.7, label='Data Points')

    # Select a core point example
    core_point = X[15]
    eps = 1.5

    # Draw neighborhood circle
    circle = plt.Circle(core_point, eps, fill=False, color='red', linewidth=3,
                       linestyle='--', label=f'ε-neighborhood (ε={eps})')
    ax.add_patch(circle)

    # Highlight core point
    ax.scatter(core_point[0], core_point[1], c='red', s=200, marker='*',
               label='Core Point', zorder=5)

    # Count neighbors
    distances = np.sqrt(((X - core_point)**2).sum(axis=1))
    neighbors = X[distances <= eps]

    # Highlight neighbors
    ax.scatter(neighbors[:, 0], neighbors[:, 1], c='green', s=150,
               alpha=0.8, label=f'Neighbors ({len(neighbors)} points)')

    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_title('DBSCAN Neighborhood Concept\n"Find Crowded Areas"')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_chart(fig, 'neighborhood_concept')

def create_epsilon_effect():
    """Chart 15: Effect of epsilon parameter"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    eps_values = [0.5, 1.0, 2.0, 4.0]
    titles = ['ε=0.5 (Too Small)', 'ε=1.0 (Good)', 'ε=2.0 (Large)', 'ε=4.0 (Too Large)']
    axes = [ax1, ax2, ax3, ax4]

    for i, (eps, title, ax) in enumerate(zip(eps_values, titles, axes)):
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)

        # Plot clusters
        unique_labels = set(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'black'  # Noise points

            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            ax.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.7, s=50)

        ax.set_title(f'{title}\n({len(set(labels)) - (1 if -1 in labels else 0)} clusters)')

    plt.tight_layout()
    save_chart(fig, 'epsilon_effect')

def create_dbscan_algorithm():
    """Chart 16: DBSCAN algorithm flowchart"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis('off')

    # Create flowchart text
    flowchart_text = """
    DBSCAN Algorithm Flowchart

    1. For each point p in dataset:
       └─ Count neighbors within ε distance

    2. Classify points:
       ├─ Core: ≥ MinPts neighbors
       ├─ Border: Within ε of core point
       └─ Noise: Neither core nor border

    3. Form clusters:
       ├─ Start with unvisited core point
       ├─ Add all density-reachable points
       └─ Repeat for remaining cores

    4. Output:
       └─ Clusters + noise points
    """

    ax.text(0.1, 0.5, flowchart_text, fontsize=14, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    ax.set_title('DBSCAN Algorithm Steps', fontsize=16, fontweight='bold', pad=20)

    save_chart(fig, 'dbscan_algorithm')

def create_dendrogram_example():
    """Chart 17: Dendrogram example"""
    np.random.seed(42)

    # Create simple dataset
    points = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])
    point_labels = ['A', 'B', 'C', 'D']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot points
    ax1.scatter(points[:, 0], points[:, 1], s=200, c='blue', alpha=0.7)
    for i, label in enumerate(point_labels):
        ax1.annotate(label, (points[i, 0], points[i, 1]), xytext=(10, 10),
                    textcoords='offset points', fontsize=14, fontweight='bold')

    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Data Points')
    ax1.grid(True, alpha=0.3)

    # Create dendrogram
    linkage_matrix = linkage(points, method='ward')
    dendrogram(linkage_matrix, labels=point_labels, ax=ax2)
    ax2.set_title('Hierarchical Clustering Dendrogram')
    ax2.set_ylabel('Distance')

    plt.tight_layout()
    save_chart(fig, 'dendrogram_example')

def create_dbscan_clusters():
    """Chart 18: DBSCAN cluster results"""
    np.random.seed(42)
    X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X)

    # Plot clusters
    unique_labels = set(labels)
    colors = ['blue', 'red', 'black']  # blue, red, black for noise

    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]

        if k == -1:
            # Noise points
            ax.scatter(xy[:, 0], xy[:, 1], c='black', marker='x', s=50, alpha=0.7, label='Noise')
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c=col, s=60, alpha=0.8, label=f'Cluster {k+1}')

    ax.set_title('DBSCAN Results: Crescent-Shaped Clusters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_chart(fig, 'dbscan_clusters')

def create_customer_dendrogram():
    """Chart 19: Customer dendrogram"""
    np.random.seed(42)

    # Generate customer-like data
    n_customers = 20
    customers = np.random.rand(n_customers, 5) * 100  # 5 features

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create dendrogram
    linkage_matrix = linkage(customers, method='ward')
    customer_labels = [f'C{i+1:02d}' for i in range(n_customers)]

    dendrogram(linkage_matrix, labels=customer_labels, ax=ax)
    ax.set_title('Customer Segmentation Dendrogram\n(Ward Linkage)')
    ax.set_ylabel('Distance')
    ax.set_xlabel('Customers')

    # Add colored horizontal line for optimal cut
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7,
               label='Optimal Cut (4 clusters)')
    ax.legend()

    plt.xticks(rotation=90)
    plt.tight_layout()
    save_chart(fig, 'customer_dendrogram')

def create_arbitrary_shapes():
    """Chart 20: Arbitrary shapes demo"""
    np.random.seed(42)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Spiral data
    t = np.linspace(0, 4*np.pi, 100)
    spiral1_x = t * np.cos(t) + np.random.normal(0, 0.3, 100)
    spiral1_y = t * np.sin(t) + np.random.normal(0, 0.3, 100)
    spiral2_x = t * np.cos(t + np.pi) + np.random.normal(0, 0.3, 100)
    spiral2_y = t * np.sin(t + np.pi) + np.random.normal(0, 0.3, 100)

    ax1.scatter(spiral1_x, spiral1_y, c='blue', alpha=0.7, label='Spiral 1')
    ax1.scatter(spiral2_x, spiral2_y, c='red', alpha=0.7, label='Spiral 2')
    ax1.set_title('Interleaved Spirals')
    ax1.legend()

    # Crescents
    X_crescents, y_crescents = make_moons(n_samples=200, noise=0.1, random_state=42)
    ax2.scatter(X_crescents[:, 0], X_crescents[:, 1], c=y_crescents, cmap='viridis', alpha=0.7)
    ax2.set_title('Crescent Shapes')

    # Nested circles
    X_circles, y_circles = make_circles(n_samples=200, factor=0.6, noise=0.1, random_state=42)
    ax3.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', alpha=0.7)
    ax3.set_title('Nested Circles')

    # Elongated clusters
    elongated = np.random.multivariate_normal([0, 0], [[4, 0], [0, 0.5]], 100)
    elongated2 = np.random.multivariate_normal([0, 5], [[4, 0], [0, 0.5]], 100)
    ax4.scatter(elongated[:, 0], elongated[:, 1], c='blue', alpha=0.7, label='Cluster 1')
    ax4.scatter(elongated2[:, 0], elongated2[:, 1], c='red', alpha=0.7, label='Cluster 2')
    ax4.set_title('Elongated Clusters')
    ax4.legend()

    plt.tight_layout()
    save_chart(fig, 'arbitrary_shapes')

def create_performance_table():
    """Chart 21: Performance comparison table"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Performance data
    data = {
        'Dataset': ['Spherical', 'Elongated', 'Crescent', 'Nested', 'Noisy'],
        'K-means_ARI': [0.92, 0.68, 0.23, 0.15, 0.84],
        'K-means_Silhouette': [0.75, 0.45, 0.12, -0.05, 0.65],
        'DBSCAN_ARI': [0.85, 0.89, 0.95, 0.88, 0.91],
        'DBSCAN_Silhouette': [0.68, 0.72, 0.82, 0.76, 0.79],
        'Hierarchical_ARI': [0.88, 0.82, 0.67, 0.78, 0.73],
        'Hierarchical_Silhouette': [0.71, 0.69, 0.58, 0.65, 0.61]
    }

    df = pd.DataFrame(data)
    table_display = df.round(2)

    table = ax.table(cellText=table_display.values,
                    colLabels=['Dataset', 'K-means\nARI', 'K-means\nSilhouette',
                              'DBSCAN\nARI', 'DBSCAN\nSilhouette',
                              'Hierarchical\nARI', 'Hierarchical\nSilhouette'],
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Color best performance in each row
    for i in range(1, len(df) + 1):
        # ARI columns (1, 3, 5)
        ari_values = [table_display.iloc[i-1, 1], table_display.iloc[i-1, 3], table_display.iloc[i-1, 5]]
        best_ari_idx = np.argmax(ari_values)
        table[(i, 1 + best_ari_idx * 2)].set_facecolor('#90EE90')  # Light green

        # Silhouette columns (2, 4, 6)
        sil_values = [table_display.iloc[i-1, 2], table_display.iloc[i-1, 4], table_display.iloc[i-1, 6]]
        best_sil_idx = np.argmax(sil_values)
        table[(i, 2 + best_sil_idx * 2)].set_facecolor('#90EE90')  # Light green

    # Header styling
    for i in range(7):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')

    ax.set_title('Algorithm Performance Comparison\n(Green = Best Performance)',
                fontsize=14, fontweight='bold', pad=20)

    save_chart(fig, 'performance_table')

def create_sklearn_pipeline():
    """Chart 22: Sklearn implementation pipeline"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')

    pipeline_text = """
    Scikit-learn Clustering Pipeline

    ┌─────────────────────────────────────────────────────────────┐
    │                    Data Preprocessing                        │
    │  • StandardScaler() for feature scaling                     │
    │  • PCA() for dimensionality reduction (optional)            │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    Algorithm Selection                       │
    │  • KMeans(n_clusters=k)                                     │
    │  • DBSCAN(eps=0.5, min_samples=5)                          │
    │  • AgglomerativeClustering(n_clusters=k)                   │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      Model Fitting                          │
    │  labels = model.fit_predict(X)                              │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      Evaluation                             │
    │  • silhouette_score(X, labels)                             │
    │  • adjusted_rand_score(true_labels, labels)                │
    └─────────────────────────────────────────────────────────────┘
    """

    ax.text(0.05, 0.5, pipeline_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    ax.set_title('Production Clustering Pipeline', fontsize=16, fontweight='bold')

    save_chart(fig, 'sklearn_pipeline')

def create_clustering_taxonomy():
    """Chart 23: Clustering taxonomy tree"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.axis('off')

    taxonomy_text = """
    Clustering Algorithm Taxonomy

    Clustering Algorithms
    ├── Centroid-Based
    │   ├── K-means (spherical clusters)
    │   ├── K-medoids (robust to outliers)
    │   └── Fuzzy C-means (soft assignments)
    │
    ├── Density-Based
    │   ├── DBSCAN (epsilon neighborhoods)
    │   ├── OPTICS (varying densities)
    │   └── Mean-shift (mode seeking)
    │
    ├── Hierarchical
    │   ├── Agglomerative (bottom-up)
    │   │   ├── Single linkage
    │   │   ├── Complete linkage
    │   │   ├── Average linkage
    │   │   └── Ward linkage
    │   └── Divisive (top-down)
    │
    ├── Graph-Based
    │   ├── Spectral clustering
    │   └── Community detection
    │
    └── Model-Based
        ├── Gaussian Mixture Models
        ├── Hidden Markov Models
        └── Deep clustering
    """

    ax.text(0.05, 0.5, taxonomy_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    ax.set_title('Complete Clustering Algorithm Taxonomy', fontsize=16, fontweight='bold')

    save_chart(fig, 'clustering_taxonomy')

def create_algorithm_selection():
    """Chart 24: Algorithm selection decision tree"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.axis('off')

    decision_tree = """
    Clustering Algorithm Selection Guide

    START: Do you know the number of clusters?
    ├── YES: Do clusters have spherical shape?
    │   ├── YES: Are clusters similar size?
    │   │   ├── YES → Use K-means (fast, reliable)
    │   │   └── NO → Use Hierarchical (handles size variation)
    │   └── NO: Is data noisy?
    │       ├── YES → Use DBSCAN (robust to noise)
    │       └── NO → Try Spectral clustering
    │
    └── NO: Is noise a major concern?
        ├── YES → Use DBSCAN (automatic noise detection)
        └── NO: Do you need hierarchy?
            ├── YES → Use Agglomerative clustering
            └── NO: Try multiple algorithms
                    ├── Start with K-means + elbow method
                    ├── Validate with DBSCAN
                    └── Compare results

    Dataset Size Considerations:
    • Small (<1,000): Any algorithm
    • Medium (1K-10K): All algorithms work
    • Large (>10K): Prefer K-means or Mini-batch variants
    """

    ax.text(0.05, 0.5, decision_tree, fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    ax.set_title('Algorithm Selection Decision Tree', fontsize=16, fontweight='bold')

    save_chart(fig, 'algorithm_selection')

def create_modern_applications():
    """Chart 25: Modern applications"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Anomaly detection visualization
    np.random.seed(42)
    normal_data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    anomalies = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]])

    ax1.scatter(normal_data[:, 0], normal_data[:, 1], c='blue', alpha=0.7, label='Normal')
    ax1.scatter(anomalies[:, 0], anomalies[:, 1], c='red', s=100, marker='x',
                linewidths=3, label='Anomalies')
    ax1.set_title('Anomaly Detection')
    ax1.legend()

    # Recommendation clustering
    users = np.random.rand(50, 2) * 10
    colors = ['red', 'blue', 'green', 'orange']
    labels = np.random.choice(4, 50)

    for i in range(4):
        mask = labels == i
        ax2.scatter(users[mask, 0], users[mask, 1], c=colors[i],
                   alpha=0.7, label=f'Cluster {i+1}')
    ax2.set_title('User Segmentation\n(Recommendation Systems)')
    ax2.legend()

    # Business intelligence
    ax3.axis('off')
    bi_text = """
    Business Intelligence Applications:

    • Customer Segmentation
      - Demographic groups
      - Behavioral patterns
      - Value-based segments

    • Market Research
      - Product categories
      - Competitor analysis
      - Trend identification

    • Operational Optimization
      - Resource allocation
      - Process improvement
      - Cost reduction
    """
    ax3.text(0.1, 0.5, bi_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax3.set_title('Business Intelligence')

    # Neural network preview
    ax4.axis('off')
    nn_text = """
    Neural Network Clustering:

    • Autoencoders
      - Learn data representations
      - Dimensionality reduction
      - Feature extraction

    • Self-Organizing Maps
      - Topological preservation
      - Visualization
      - Pattern recognition

    • Deep Embedded Clustering
      - End-to-end learning
      - Joint optimization
      - State-of-the-art results
    """
    ax4.text(0.1, 0.5, nn_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    ax4.set_title('Neural Network Approaches')

    plt.tight_layout()
    save_chart(fig, 'modern_applications')

def main():
    """Generate all charts for Week 0c"""
    print("Generating Week 0c charts...")

    # Create all charts
    create_customer_data_sample()           # Chart 1
    create_distance_calculation()           # Chart 2
    create_validation_problem()             # Chart 3
    create_elbow_method()                   # Chart 4
    create_silhouette_analysis()            # Chart 5
    create_kmeans_steps()                   # Chart 6
    create_kmeans_example()                 # Chart 7
    create_kmeans_success()                 # Chart 8
    create_kmeans_failure()                 # Chart 9
    create_crescent_data_table()            # Chart 10
    create_voronoi_boundaries()             # Chart 11
    create_human_vs_kmeans()                # Chart 12
    create_method_comparison()              # Chart 13
    create_neighborhood_concept()           # Chart 14
    create_epsilon_effect()                 # Chart 15

    # Additional charts needed for LaTeX references
    create_dbscan_algorithm()               # Chart 16
    create_dendrogram_example()             # Chart 17
    create_dbscan_clusters()                # Chart 18
    create_customer_dendrogram()            # Chart 19
    create_arbitrary_shapes()               # Chart 20
    create_performance_table()              # Chart 21
    create_sklearn_pipeline()               # Chart 22
    create_clustering_taxonomy()            # Chart 23
    create_algorithm_selection()            # Chart 24
    create_modern_applications()            # Chart 25

    print(f"\nCompleted! Generated 25 charts in {chart_dir}/")

if __name__ == "__main__":
    main()