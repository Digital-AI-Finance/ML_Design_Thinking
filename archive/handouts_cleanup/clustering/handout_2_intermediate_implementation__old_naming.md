# Week 2 Handout 2: Intermediate Implementation Guide
## Python Code for Advanced Clustering

**Target Audience:** Data scientists, ML engineers
**Prerequisites:** Python, pandas, scikit-learn, basic clustering knowledge from Week 1
**Time to Complete:** 1-2 hours

---

## Overview

This handout provides complete Python implementations of advanced clustering techniques covered in Week 2:
- DBSCAN for arbitrary shapes
- Hierarchical Clustering with dendrograms
- Gaussian Mixture Models for soft assignments
- Parameter tuning and validation
- Comparison framework

All code is production-ready with error handling and visualization.

---

## Setup

### Installation

```python
# Core libraries
pip install numpy pandas matplotlib seaborn scipy

# ML libraries
pip install scikit-learn

# Additional visualization
pip install plotly  # Optional for interactive dendrograms
```

### Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
```

---

## 1. DBSCAN Implementation

### Basic Usage

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate crescent-shaped data (K-means fails here)
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Scale features (IMPORTANT for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")
print(f"Labels: {np.unique(labels)}")  # -1 = noise
```

### Parameter Tuning: Finding Epsilon

**K-distance graph method:**

```python
def find_optimal_eps(X, k=5):
    """
    Find optimal epsilon using k-distance graph.

    Rule of thumb: epsilon = elbow point in k-distance plot
    k = min_samples parameter (usually 2 * n_features)
    """
    from sklearn.neighbors import NearestNeighbors

    # Fit k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Sort k-distances (distance to kth nearest neighbor)
    k_distances = np.sort(distances[:, k-1], axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.ylabel(f'{k}-NN Distance')
    plt.xlabel('Points sorted by distance')
    plt.title('K-distance Graph for Epsilon Selection')
    plt.axhline(y=np.percentile(k_distances, 90), color='r', linestyle='--',
                label='90th percentile (suggested epsilon)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Suggest epsilon
    suggested_eps = np.percentile(k_distances, 90)
    print(f"Suggested epsilon: {suggested_eps:.3f}")

    return suggested_eps

# Usage
eps = find_optimal_eps(X_scaled, k=5)
```

### Grid Search for Best Parameters

```python
def dbscan_grid_search(X, eps_range, min_samples_range):
    """
    Test multiple DBSCAN parameters, return best based on silhouette score.
    """
    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)

            # Skip if only noise or only one cluster
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue

            # Calculate metrics (excluding noise points)
            mask = labels != -1
            if mask.sum() > 0:
                score = silhouette_score(X[mask], labels[mask])
                n_noise = (labels == -1).sum()

                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': score
                })

    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('silhouette', ascending=False)

    print("Top 5 configurations:")
    print(df.head())

    return df

# Usage
eps_range = np.arange(0.1, 1.0, 0.1)
min_samples_range = range(3, 10)

results = dbscan_grid_search(X_scaled, eps_range, min_samples_range)
```

### Visualization

```python
def plot_dbscan_results(X, labels):
    """
    Visualize DBSCAN clustering results with noise highlighted.
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Colored by cluster
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in black
            color = 'k'
            marker = 'x'
            label_str = 'Noise'
        else:
            marker = 'o'
            label_str = f'Cluster {label}'

        mask = labels == label
        ax1.scatter(X[mask, 0], X[mask, 1], c=[color],
                   marker=marker, s=50, alpha=0.6, label=label_str)

    ax1.set_title('DBSCAN Clustering Results')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()

    # Plot 2: Core vs border vs noise
    # In DBSCAN: core points, border points, noise points
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.3, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Core points
    ax2.scatter(X[core_samples_mask, 0], X[core_samples_mask, 1],
               c='blue', marker='o', s=50, alpha=0.8, label='Core')

    # Border points
    border_mask = (labels != -1) & (~core_samples_mask)
    ax2.scatter(X[border_mask, 0], X[border_mask, 1],
               c='cyan', marker='o', s=50, alpha=0.6, label='Border')

    # Noise points
    noise_mask = labels == -1
    ax2.scatter(X[noise_mask, 0], X[noise_mask, 1],
               c='red', marker='x', s=50, alpha=0.8, label='Noise')

    ax2.set_title('Point Types in DBSCAN')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()

    plt.tight_layout()

# Usage
plot_dbscan_results(X_scaled, labels)
```

---

## 2. Hierarchical Clustering Implementation

### Basic Usage

```python
from sklearn.cluster import AgglomerativeClustering

# Apply hierarchical clustering
n_clusters = 3
agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = agg.fit_predict(X_scaled)

print(f"Cluster assignments: {labels}")
print(f"Cluster sizes: {np.bincount(labels)}")
```

### Creating Dendrograms

```python
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_dendrogram(X, method='ward', max_display=30):
    """
    Create dendrogram for hierarchical clustering.

    Parameters:
    - X: data matrix
    - method: linkage method ('single', 'complete', 'average', 'ward')
    - max_display: maximum number of points to display (for large datasets)
    """
    # Compute linkage matrix
    Z = linkage(X, method=method)

    # Plot
    plt.figure(figsize=(12, 6))
    dendrogram(Z,
              truncate_mode='lastp',  # Show only last p merged clusters
              p=max_display,          # Show last 30 merges
              leaf_rotation=90,
              leaf_font_size=10,
              show_contracted=True)   # Show height of contracted nodes

    plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)')
    plt.xlabel('Cluster size (or sample index)')
    plt.ylabel('Distance')
    plt.tight_layout()

# Usage
plot_dendrogram(X_scaled, method='ward')
```

### Finding Optimal Number of Clusters

**Method 1: Elbow in dendrogram distances**

```python
def find_optimal_clusters_hierarchical(X, max_clusters=10):
    """
    Find optimal number of clusters using elbow method on linkage distances.
    """
    Z = linkage(X, method='ward')

    # Last max_clusters merges
    last_merges = Z[-max_clusters:, 2]

    # Calculate acceleration (second derivative)
    acceleration = np.diff(last_merges, 2)

    # Optimal K = where acceleration is maximum
    k_optimal = acceleration.argmax() + 2  # +2 for indexing

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Linkage distances
    ax1.plot(range(1, max_clusters+1), last_merges[::-1], marker='o')
    ax1.axvline(x=k_optimal, color='r', linestyle='--', label=f'Optimal K = {k_optimal}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Distance')
    ax1.set_title('Linkage Distance by Number of Clusters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Acceleration
    ax2.plot(range(2, max_clusters), acceleration, marker='o', color='orange')
    ax2.axvline(x=k_optimal, color='r', linestyle='--', label=f'Optimal K = {k_optimal}')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Acceleration')
    ax2.set_title('Acceleration of Distance Increase')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    print(f"Suggested number of clusters: {k_optimal}")
    return k_optimal

# Usage
k_opt = find_optimal_clusters_hierarchical(X_scaled)
```

**Method 2: Silhouette scores for different cuts**

```python
def evaluate_hierarchical_cuts(X, linkage_method='ward', max_k=10):
    """
    Evaluate hierarchical clustering for different numbers of clusters.
    """
    silhouette_scores = []
    davies_bouldin_scores = []

    for k in range(2, max_k+1):
        agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = agg.fit_predict(X)

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        silhouette_scores.append(sil)
        davies_bouldin_scores.append(db)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette (higher is better)
    ax1.plot(range(2, max_k+1), silhouette_scores, marker='o', color='blue')
    best_k_sil = np.argmax(silhouette_scores) + 2
    ax1.axvline(x=best_k_sil, color='r', linestyle='--', label=f'Best K = {best_k_sil}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs Number of Clusters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Davies-Bouldin (lower is better)
    ax2.plot(range(2, max_k+1), davies_bouldin_scores, marker='o', color='orange')
    best_k_db = np.argmin(davies_bouldin_scores) + 2
    ax2.axvline(x=best_k_db, color='r', linestyle='--', label=f'Best K = {best_k_db}')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Davies-Bouldin Score')
    ax2.set_title('Davies-Bouldin Score vs Number of Clusters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return silhouette_scores, davies_bouldin_scores

# Usage
evaluate_hierarchical_cuts(X_scaled)
```

---

## 3. Gaussian Mixture Models Implementation

### Basic Usage

```python
from sklearn.mixture import GaussianMixture

# Fit GMM
n_components = 3
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# Hard assignments (like K-means)
hard_labels = gmm.predict(X_scaled)

# Soft assignments (probabilities)
soft_labels = gmm.predict_proba(X_scaled)

print(f"Hard labels shape: {hard_labels.shape}")
print(f"Soft labels shape: {soft_labels.shape}")  # (n_samples, n_components)
print(f"\nExample soft assignment (first point):")
print(f"Probabilities: {soft_labels[0]}")
print(f"Sum: {soft_labels[0].sum()}")  # Should be 1.0
```

### Finding Optimal Number of Components

**Method: BIC and AIC**

```python
def find_optimal_components_gmm(X, max_components=10):
    """
    Find optimal number of GMM components using BIC and AIC.

    Lower BIC/AIC = better model
    BIC = more conservative (penalizes complexity more)
    AIC = less conservative
    """
    bic_scores = []
    aic_scores = []

    for n in range(1, max_components+1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components+1), bic_scores, marker='o', label='BIC', color='blue')
    plt.plot(range(1, max_components+1), aic_scores, marker='s', label='AIC', color='orange')

    # Mark optimal
    optimal_bic = np.argmin(bic_scores) + 1
    optimal_aic = np.argmin(aic_scores) + 1

    plt.axvline(x=optimal_bic, color='blue', linestyle='--', alpha=0.5,
                label=f'Optimal (BIC) = {optimal_bic}')
    plt.axvline(x=optimal_aic, color='orange', linestyle='--', alpha=0.5,
                label=f'Optimal (AIC) = {optimal_aic}')

    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion')
    plt.title('GMM Model Selection (lower is better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"Optimal components (BIC): {optimal_bic}")
    print(f"Optimal components (AIC): {optimal_aic}")

    return optimal_bic, optimal_aic

# Usage
find_optimal_components_gmm(X_scaled)
```

### Visualizing Soft Assignments

```python
def plot_gmm_soft_assignments(X, gmm):
    """
    Visualize GMM with soft cluster boundaries.
    """
    # Get predictions
    labels = gmm.predict(X)
    proba = gmm.predict_proba(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Hard assignments (like K-means)
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    ax1.set_title('GMM Hard Assignments')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster')

    # Plot 2: Uncertainty (entropy of probabilities)
    from scipy.stats import entropy
    uncertainties = entropy(proba.T)  # Higher entropy = more uncertain

    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=uncertainties, cmap='YlOrRd', s=50, alpha=0.6)
    ax2.set_title('Assignment Uncertainty (darker = more uncertain)')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    plt.colorbar(scatter2, ax=ax2, label='Entropy')

    plt.tight_layout()

# Usage
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)
plot_gmm_soft_assignments(X_scaled, gmm)
```

---

## 4. Complete Comparison Framework

### Compare All Algorithms

```python
def compare_clustering_algorithms(X, n_clusters=3, eps=0.3, min_samples=5):
    """
    Compare K-means, DBSCAN, Hierarchical, and GMM on same data.
    """
    from sklearn.cluster import KMeans

    # Fit all algorithms
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X)

    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    hier_labels = hierarchical.fit_predict(X)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    algorithms = [
        ('K-Means', kmeans_labels),
        ('DBSCAN', dbscan_labels),
        ('Hierarchical (Ward)', hier_labels),
        ('GMM', gmm_labels)
    ]

    for ax, (name, labels) in zip(axes.flat, algorithms):
        # Handle DBSCAN noise
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'k'
                marker = 'x'
            else:
                marker = 'o'

            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], marker=marker, s=50, alpha=0.6)

        ax.set_title(name)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # Add metrics (excluding noise for DBSCAN)
        if name == 'DBSCAN':
            mask = labels != -1
            if mask.sum() > 0 and len(set(labels[mask])) > 1:
                sil = silhouette_score(X[mask], labels[mask])
                ax.text(0.02, 0.98, f'Silhouette: {sil:.3f}\nNoise points: {(labels==-1).sum()}',
                       transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            sil = silhouette_score(X, labels)
            ax.text(0.02, 0.98, f'Silhouette: {sil:.3f}',
                   transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

# Usage
compare_clustering_algorithms(X_scaled)
```

---

## 5. Production Pipeline

### Complete Workflow

```python
class AdvancedClusteringPipeline:
    """
    Production-ready clustering pipeline with algorithm selection.
    """

    def __init__(self, algorithm='auto', **kwargs):
        """
        Parameters:
        - algorithm: 'kmeans', 'dbscan', 'hierarchical', 'gmm', or 'auto'
        - **kwargs: algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.params = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None

    def fit(self, X):
        """Fit clustering model."""
        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # Auto-select algorithm if needed
        if self.algorithm == 'auto':
            self.algorithm = self._select_algorithm(X_scaled)
            print(f"Auto-selected algorithm: {self.algorithm}")

        # Fit appropriate model
        if self.algorithm == 'kmeans':
            from sklearn.cluster import KMeans
            self.model = KMeans(**self.params)
            self.labels_ = self.model.fit_predict(X_scaled)

        elif self.algorithm == 'dbscan':
            self.model = DBSCAN(**self.params)
            self.labels_ = self.model.fit_predict(X_scaled)

        elif self.algorithm == 'hierarchical':
            self.model = AgglomerativeClustering(**self.params)
            self.labels_ = self.model.fit_predict(X_scaled)

        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(**self.params)
            self.model.fit(X_scaled)
            self.labels_ = self.model.predict(X_scaled)

        return self

    def _select_algorithm(self, X):
        """Auto-select best algorithm based on data characteristics."""
        n_samples, n_features = X.shape

        # Large datasets: use K-means (fastest)
        if n_samples > 10000:
            return 'kmeans'

        # Check for outliers (use robust statistics)
        from scipy.stats import median_abs_deviation
        outlier_ratio = np.mean([np.sum(np.abs(X[:, i] - np.median(X[:, i])) > 3 * median_abs_deviation(X[:, i]))
                                 for i in range(n_features)]) / n_samples

        # If many outliers: use DBSCAN
        if outlier_ratio > 0.05:
            return 'dbscan'

        # Default: K-means
        return 'kmeans'

    def predict(self, X):
        """Predict cluster labels for new data."""
        X_scaled = self.scaler.transform(X)

        if self.algorithm == 'kmeans':
            return self.model.predict(X_scaled)
        elif self.algorithm == 'gmm':
            return self.model.predict(X_scaled)
        elif self.algorithm in ['dbscan', 'hierarchical']:
            # These don't support prediction on new data
            # Use nearest cluster center instead
            from sklearn.metrics import pairwise_distances_argmin
            if self.algorithm == 'hierarchical':
                # Compute cluster centers
                centers = np.array([X_scaled[self.labels_ == i].mean(axis=0)
                                   for i in range(self.labels_.max() + 1)])
                return pairwise_distances_argmin(X_scaled, centers)
            else:
                raise NotImplementedError(f"{self.algorithm} doesn't support prediction")

    def score(self, X):
        """Compute silhouette score."""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")

        # Handle DBSCAN noise
        if self.algorithm == 'dbscan':
            mask = self.labels_ != -1
            if mask.sum() == 0 or len(set(self.labels_[mask])) == 1:
                return -1
            return silhouette_score(self.scaler.transform(X)[mask], self.labels_[mask])

        return silhouette_score(self.scaler.transform(X), self.labels_)

# Usage
pipeline = AdvancedClusteringPipeline(algorithm='auto')
pipeline.fit(X)
print(f"Algorithm used: {pipeline.algorithm}")
print(f"Silhouette score: {pipeline.score(X):.3f}")
```

---

## Next Steps

- Handout 3: Mathematical foundations (advanced)
- Week 3: Apply clustering to NLP (sentiment clusters)
- Practice: Apply to your own dataset

---

**Status**: Implementation complete. Ready for production use or advanced study.
