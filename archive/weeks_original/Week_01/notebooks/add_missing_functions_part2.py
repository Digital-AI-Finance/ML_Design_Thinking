import json

# Load Part 2 notebook
with open(r'D:\Joerg\Research\slides\ML_Design_Thinking_16\Week_01\Week01_Part2_Technical_Design.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the Algorithm Demonstration Functions cell (should be at index 5)
functions_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'def demonstrate_kmeans_step_by_step' in source:
            functions_cell_idx = i
            break

if functions_cell_idx is None:
    print("Error: Could not find main functions cell")
    exit(1)

# Get existing source
existing_source = nb['cells'][functions_cell_idx]['source']
if isinstance(existing_source, str):
    existing_source = existing_source.split('\n')
existing_text = ''.join(existing_source)

# Add more algorithm and helper functions
additional_functions = '''

# ============================================================================
# ADDITIONAL ALGORITHM FUNCTIONS
# ============================================================================

def demonstrate_silhouette_analysis():
    """
    Detailed silhouette analysis for cluster validation.
    Shows how to interpret silhouette scores.
    """
    print("📊 Silhouette Analysis: Understanding Cluster Quality\\n")
    
    # Generate data with varying cluster quality
    X_good, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X_bad, _ = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for row, (X, quality) in enumerate([(X_good, 'Good'), (X_bad, 'Poor')]):
        # Standardize
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette scores
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(X_scaled, labels)
        silhouette_avg = silhouette_score(X_scaled, labels)
        
        # Plot clusters
        ax1 = axes[row, 0]
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
        ax1.set_title(f'{quality} Clustering\\nSilhouette: {silhouette_avg:.3f}', 
                     fontweight='bold')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        
        # Silhouette plot
        ax2 = axes[row, 1]
        y_lower = 10
        for i in range(3):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(i / 3)
            ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, 
                             cluster_silhouette_vals, 
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax2.set_xlabel('Silhouette Coefficient')
        ax2.set_ylabel('Cluster Label')
        ax2.set_title('Silhouette Plot', fontweight='bold')
        ax2.axvline(x=silhouette_avg, color='red', linestyle='--', 
                   label=f'Average: {silhouette_avg:.3f}')
        ax2.legend()
        
        # Distribution
        ax3 = axes[row, 2]
        ax3.hist(silhouette_vals, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Silhouette Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distribution', fontweight='bold')
    
    plt.suptitle('Silhouette Analysis: Good vs Poor Clustering', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n📚 Interpretation Guide:")
    print("• Score > 0.7: Strong clustering")
    print("• Score 0.5-0.7: Reasonable clustering")
    print("• Score 0.25-0.5: Weak clustering")
    print("• Score < 0.25: Poor/artificial clustering")

def demonstrate_clustering_stability():
    """
    Test clustering stability with different initializations.
    Shows importance of n_init parameter.
    """
    print("🔄 Clustering Stability Analysis\\n")
    
    # Generate data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Test different n_init values
    n_init_values = [1, 5, 10, 20]
    n_runs = 10
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, n_init in enumerate(n_init_values):
        ax = axes[idx]
        
        # Run clustering multiple times
        all_labels = []
        all_scores = []
        
        for run in range(n_runs):
            kmeans = KMeans(n_clusters=3, n_init=n_init, random_state=run)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            all_labels.append(labels)
            all_scores.append(score)
        
        # Plot all results overlaid
        for labels, alpha in zip(all_labels, np.linspace(0.2, 0.8, n_runs)):
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                      s=20, alpha=alpha)
        
        # Statistics
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        ax.set_title(f'n_init={n_init}\\nScore: {mean_score:.3f} ± {std_score:.3f}', 
                    fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    plt.suptitle('Clustering Stability: Impact of n_init Parameter', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n📊 Stability Results:")
    print("• n_init=1: Unstable, varies with initialization")
    print("• n_init=5: Better stability")
    print("• n_init=10: Good stability (default)")
    print("• n_init=20: Marginal improvement")
    print("\\n💡 Recommendation: Use n_init=10 for production")

def compare_distance_metrics():
    """
    Compare different distance metrics for clustering.
    Shows when to use each metric.
    """
    print("📏 Distance Metrics Comparison\\n")
    
    # Generate data with different characteristics
    X_normal, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    
    # Create elongated clusters
    transformation = [[0.5, -0.5], [-0.8, 0.8]]
    X_elongated = np.dot(X_normal, transformation)
    
    # Different metrics
    metrics = ['euclidean', 'manhattan', 'cosine']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for row, (X, data_type) in enumerate([(X_normal, 'Normal'), 
                                          (X_elongated, 'Elongated')]):
        X_scaled = StandardScaler().fit_transform(X)
        
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            
            # Apply clustering with different metrics
            if metric == 'cosine':
                # Use AgglomerativeClustering for cosine metric
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=3, 
                                                    metric=metric,
                                                    linkage='average')
                labels = clustering.fit_predict(X_scaled)
            else:
                # KMeans doesn't support custom metrics, use AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=3, 
                                                    metric=metric,
                                                    linkage='average')
                labels = clustering.fit_predict(X_scaled)
            
            # Plot
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, 
                               cmap='viridis', s=30, alpha=0.7)
            
            # Calculate score
            score = silhouette_score(X_scaled, labels, metric=metric)
            
            ax.set_title(f'{data_type} Data\\n{metric.capitalize()}\\nScore: {score:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
    
    plt.suptitle('Distance Metrics: Impact on Clustering Results', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n📚 When to Use Each Metric:")
    print("• Euclidean: General purpose, spherical clusters")
    print("• Manhattan: Grid-like data, robust to outliers")
    print("• Cosine: Text data, direction matters more than magnitude")
    print("• Correlation: Time series, pattern similarity")

def demonstrate_feature_scaling_impact():
    """
    Show the critical importance of feature scaling.
    Demonstrates what happens with and without scaling.
    """
    print("⚖️ Feature Scaling: Critical for Success\\n")
    
    # Create data with different scales
    np.random.seed(42)
    n_samples = 300
    
    # Feature 1: Scale 0-1
    feature1 = np.random.uniform(0, 1, n_samples)
    
    # Feature 2: Scale 0-1000
    feature2 = np.random.uniform(0, 1000, n_samples)
    
    # Create 3 true clusters
    true_labels = np.repeat([0, 1, 2], 100)
    feature1[true_labels == 0] += 0.5
    feature1[true_labels == 1] -= 0.3
    feature2[true_labels == 0] += 500
    feature2[true_labels == 1] -= 300
    
    X_unscaled = np.column_stack([feature1, feature2])
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Unscaled clustering
    kmeans_unscaled = KMeans(n_clusters=3, random_state=42)
    labels_unscaled = kmeans_unscaled.fit_predict(X_unscaled)
    
    # Scaled clustering
    kmeans_scaled = KMeans(n_clusters=3, random_state=42)
    labels_scaled = kmeans_scaled.fit_predict(X_scaled)
    
    # Plot unscaled
    axes[0, 0].scatter(X_unscaled[:, 0], X_unscaled[:, 1], 
                      c=true_labels, cmap='viridis', s=30, alpha=0.7)
    axes[0, 0].set_title('True Clusters\\n(Unscaled View)', fontweight='bold')
    axes[0, 0].set_xlabel('Feature 1 (0-1)')
    axes[0, 0].set_ylabel('Feature 2 (0-1000)')
    
    axes[0, 1].scatter(X_unscaled[:, 0], X_unscaled[:, 1], 
                      c=labels_unscaled, cmap='viridis', s=30, alpha=0.7)
    axes[0, 1].set_title('K-Means (No Scaling)\\n❌ Dominated by Feature 2', 
                        fontweight='bold', color='red')
    axes[0, 1].set_xlabel('Feature 1 (0-1)')
    axes[0, 1].set_ylabel('Feature 2 (0-1000)')
    
    # Feature importance (unscaled)
    axes[0, 2].bar(['Feature 1', 'Feature 2'], 
                   [np.std(X_unscaled[:, 0]), np.std(X_unscaled[:, 1])],
                   color=['blue', 'red'])
    axes[0, 2].set_title('Feature "Importance"\\n(Unscaled)', fontweight='bold')
    axes[0, 2].set_ylabel('Standard Deviation')
    
    # Plot scaled
    axes[1, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                      c=true_labels, cmap='viridis', s=30, alpha=0.7)
    axes[1, 0].set_title('True Clusters\\n(Scaled View)', fontweight='bold')
    axes[1, 0].set_xlabel('Feature 1 (scaled)')
    axes[1, 0].set_ylabel('Feature 2 (scaled)')
    
    axes[1, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                      c=labels_scaled, cmap='viridis', s=30, alpha=0.7)
    axes[1, 1].set_title('K-Means (With Scaling)\\n✅ Balanced Features', 
                        fontweight='bold', color='green')
    axes[1, 1].set_xlabel('Feature 1 (scaled)')
    axes[1, 1].set_ylabel('Feature 2 (scaled)')
    
    # Feature importance (scaled)
    axes[1, 2].bar(['Feature 1', 'Feature 2'], 
                   [np.std(X_scaled[:, 0]), np.std(X_scaled[:, 1])],
                   color=['blue', 'blue'])
    axes[1, 2].set_title('Feature Importance\\n(Scaled)', fontweight='bold')
    axes[1, 2].set_ylabel('Standard Deviation')
    axes[1, 2].set_ylim([0, 2])
    
    plt.suptitle('Feature Scaling: Essential for Correct Clustering', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Performance comparison
    score_unscaled = silhouette_score(X_unscaled, labels_unscaled)
    score_scaled = silhouette_score(X_scaled, labels_scaled)
    
    print(f"\\n📊 Performance Impact:")
    print(f"Without scaling: Silhouette = {score_unscaled:.3f}")
    print(f"With scaling: Silhouette = {score_scaled:.3f}")
    print(f"Improvement: {(score_scaled - score_unscaled) / abs(score_unscaled) * 100:.1f}%")
    print("\\n🚨 Always scale your features before clustering!")

def demonstrate_cluster_initialization():
    """
    Show different initialization methods for K-means.
    Compare random vs k-means++ initialization.
    """
    print("🎲 Cluster Initialization Methods\\n")
    
    # Generate challenging data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.2, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Different initialization methods
    init_methods = ['random', 'k-means++']
    n_runs = 5
    
    fig, axes = plt.subplots(2, n_runs, figsize=(16, 6))
    
    for row, init_method in enumerate(init_methods):
        scores = []
        inertias = []
        
        for col in range(n_runs):
            ax = axes[row, col]
            
            # Apply K-means with different initialization
            kmeans = KMeans(n_clusters=4, init=init_method, 
                          n_init=1, random_state=col)
            labels = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            score = silhouette_score(X_scaled, labels)
            scores.append(score)
            inertias.append(kmeans.inertia_)
            
            # Plot
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, 
                               cmap='viridis', s=20, alpha=0.7)
            ax.scatter(kmeans.cluster_centers_[:, 0] * np.std(X[:, 0]) + np.mean(X[:, 0]),
                      kmeans.cluster_centers_[:, 1] * np.std(X[:, 1]) + np.mean(X[:, 1]),
                      c='red', marker='*', s=200, edgecolors='black', linewidth=1.5)
            
            ax.set_title(f'Run {col+1}\\nScore: {score:.3f}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col == 0:
                ax.set_ylabel(f'{init_method}', fontsize=12, fontweight='bold')
        
        # Add statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{init_method:12} - Mean Score: {mean_score:.3f} ± {std_score:.3f}")
    
    plt.suptitle('K-Means Initialization: Random vs K-Means++', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n📚 Key Insights:")
    print("• Random: Varies significantly between runs")
    print("• K-means++: More consistent, better results")
    print("• K-means++ chooses centers far apart")
    print("• Default in sklearn is k-means++")

def demonstrate_incremental_clustering():
    """
    Show incremental/online clustering with MiniBatchKMeans.
    Useful for large datasets or streaming data.
    """
    print("📈 Incremental Clustering for Large Data\\n")
    
    # Simulate streaming data
    n_batches = 5
    batch_size = 200
    n_clusters = 3
    
    # Generate full dataset
    X_full, y_full = make_blobs(n_samples=n_batches * batch_size, 
                               centers=n_clusters, cluster_std=0.7, 
                               random_state=42)
    X_full_scaled = StandardScaler().fit_transform(X_full)
    
    # Compare regular KMeans vs MiniBatchKMeans
    from sklearn.cluster import MiniBatchKMeans
    
    fig, axes = plt.subplots(2, n_batches + 1, figsize=(16, 6))
    
    # Initialize models
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, 
                              batch_size=batch_size)
    
    # Process batches
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        X_batch = X_full_scaled[start_idx:end_idx]
        
        # MiniBatch update
        mbkmeans.partial_fit(X_batch)
        
        # Visualize current state
        ax_mb = axes[1, batch_idx]
        
        # Show data seen so far
        X_seen = X_full_scaled[:end_idx]
        labels_mb = mbkmeans.predict(X_seen)
        
        ax_mb.scatter(X_full[:end_idx, 0], X_full[:end_idx, 1], 
                     c=labels_mb, cmap='viridis', s=20, alpha=0.6)
        ax_mb.set_title(f'Batch {batch_idx+1}\\n({end_idx} points)', fontsize=10)
        ax_mb.set_xticks([])
        ax_mb.set_yticks([])
        
        if batch_idx == 0:
            ax_mb.set_ylabel('MiniBatch\\nK-Means', fontsize=11, fontweight='bold')
    
    # Final comparison
    # Regular K-means on full data
    labels_full = kmeans.fit_predict(X_full_scaled)
    
    ax_full = axes[0, n_batches]
    ax_full.scatter(X_full[:, 0], X_full[:, 1], c=labels_full, 
                   cmap='viridis', s=20, alpha=0.6)
    ax_full.set_title(f'Final\\nFull K-Means', fontsize=10, fontweight='bold')
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    
    # MiniBatch final
    labels_mb_final = mbkmeans.predict(X_full_scaled)
    ax_mb_final = axes[1, n_batches]
    ax_mb_final.scatter(X_full[:, 0], X_full[:, 1], c=labels_mb_final, 
                       cmap='viridis', s=20, alpha=0.6)
    ax_mb_final.set_title(f'Final\\nMiniBatch', fontsize=10, fontweight='bold')
    ax_mb_final.set_xticks([])
    ax_mb_final.set_yticks([])
    
    # Hide first row except last
    for i in range(n_batches):
        axes[0, i].axis('off')
    
    plt.suptitle('Incremental Clustering: Processing Data in Batches', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Performance comparison
    import time
    
    # Time regular K-means
    start = time.time()
    kmeans.fit(X_full_scaled)
    kmeans_time = time.time() - start
    
    # Time MiniBatchKMeans
    start = time.time()
    mbkmeans_new = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    mbkmeans_new.fit(X_full_scaled)
    mb_time = time.time() - start
    
    print(f"\\n⏱️ Performance Comparison:")
    print(f"Regular K-Means: {kmeans_time:.4f}s")
    print(f"MiniBatch K-Means: {mb_time:.4f}s")
    print(f"Speedup: {kmeans_time/mb_time:.2f}x")
    print(f"\\n💡 MiniBatch is ideal for large datasets or streaming data!")

# ============================================================================
# ADVANCED CLUSTERING TECHNIQUES
# ============================================================================

def demonstrate_spectral_clustering():
    """
    Show spectral clustering for non-convex shapes.
    Handles complex cluster boundaries.
    """
    print("🌈 Spectral Clustering: Beyond Simple Shapes\\n")
    
    from sklearn.cluster import SpectralClustering
    from sklearn.datasets import make_moons, make_circles
    
    # Generate non-convex datasets
    datasets = [
        ('Two Moons', make_moons(n_samples=300, noise=0.1, random_state=42)),
        ('Concentric Circles', make_circles(n_samples=300, noise=0.05, 
                                           factor=0.5, random_state=42))
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for row, (name, (X, y_true)) in enumerate(datasets):
        # K-Means (fails on these shapes)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        
        # Spectral Clustering (works well)
        spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                     n_neighbors=10, random_state=42)
        spectral_labels = spectral.fit_predict(X)
        
        # Plot true labels
        axes[row, 0].scatter(X[:, 0], X[:, 1], c=y_true, 
                           cmap='viridis', s=30, alpha=0.7)
        axes[row, 0].set_title(f'{name}\\nTrue Clusters', fontweight='bold')
        
        # Plot K-Means result
        axes[row, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, 
                           cmap='viridis', s=30, alpha=0.7)
        score_km = silhouette_score(X, kmeans_labels)
        axes[row, 1].set_title(f'K-Means\\nScore: {score_km:.3f}', fontweight='bold')
        
        # Plot Spectral result
        axes[row, 2].scatter(X[:, 0], X[:, 1], c=spectral_labels, 
                           cmap='viridis', s=30, alpha=0.7)
        score_sp = silhouette_score(X, spectral_labels)
        axes[row, 2].set_title(f'Spectral\\nScore: {score_sp:.3f}', fontweight='bold')
    
    plt.suptitle('Spectral Clustering: Handling Non-Convex Shapes', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n📚 When to Use Spectral Clustering:")
    print("• Non-convex cluster shapes")
    print("• Image segmentation")
    print("• Social network analysis")
    print("• When connectivity matters more than distance")

def demonstrate_mean_shift():
    """
    Show Mean Shift clustering - finds modes automatically.
    No need to specify number of clusters.
    """
    print("🎯 Mean Shift: Automatic Mode Finding\\n")
    
    from sklearn.cluster import MeanShift, estimate_bandwidth
    
    # Generate data with varying densities
    X1 = np.random.randn(100, 2) * 0.5 + [2, 2]
    X2 = np.random.randn(150, 2) * 0.7 + [-2, -1]
    X3 = np.random.randn(80, 2) * 0.3 + [1, -2]
    X = np.vstack([X1, X2, X3])
    
    # Estimate bandwidth
    bandwidth = estimate_bandwidth(X, quantile=0.2)
    
    # Apply Mean Shift
    ms = MeanShift(bandwidth=bandwidth)
    labels = ms.fit_predict(X)
    cluster_centers = ms.cluster_centers_
    n_clusters = len(cluster_centers)
    
    # Compare with K-means (needs K specified)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Mean Shift result
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    axes[0].scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                   c='red', marker='*', s=300, edgecolors='black', linewidth=2)
    axes[0].set_title(f'Mean Shift\\nFound {n_clusters} clusters automatically', 
                     fontweight='bold')
    
    # K-means with correct K
    kmeans_correct = KMeans(n_clusters=3, random_state=42)
    labels_correct = kmeans_correct.fit_predict(X)
    axes[1].scatter(X[:, 0], X[:, 1], c=labels_correct, 
                   cmap='viridis', s=30, alpha=0.7)
    axes[1].set_title('K-Means (K=3)\\nCorrect K', fontweight='bold')
    
    # K-means with wrong K
    kmeans_wrong = KMeans(n_clusters=5, random_state=42)
    labels_wrong = kmeans_wrong.fit_predict(X)
    axes[2].scatter(X[:, 0], X[:, 1], c=labels_wrong, 
                   cmap='viridis', s=30, alpha=0.7)
    axes[2].set_title('K-Means (K=5)\\nWrong K', fontweight='bold')
    
    for ax in axes:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    plt.suptitle('Mean Shift: Automatic Cluster Discovery', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\\n🎯 Mean Shift Results:")
    print(f"• Automatically found {n_clusters} clusters")
    print(f"• No need to specify K beforehand")
    print(f"• Bandwidth parameter controls granularity")
    print("• Works well for density-based clustering")

def demonstrate_affinity_propagation():
    """
    Show Affinity Propagation - finds exemplars automatically.
    Good for finding representative examples.
    """
    print("📍 Affinity Propagation: Finding Exemplars\\n")
    
    from sklearn.cluster import AffinityPropagation
    
    # Generate data
    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.5, random_state=42)
    
    # Apply Affinity Propagation
    af = AffinityPropagation(random_state=42, damping=0.9)
    labels = af.fit_predict(X)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters = len(cluster_centers_indices)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot clusters
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    for k, col in zip(range(n_clusters), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        ax1.plot(X[class_members, 0], X[class_members, 1], 'o', 
                markerfacecolor=col, markeredgecolor='k', markersize=8, alpha=0.6)
        ax1.plot(cluster_center[0], cluster_center[1], 'o', 
                markerfacecolor=col, markeredgecolor='red', markersize=15, 
                markeredgewidth=2)
        
        # Draw lines from exemplar to members
        for x in X[class_members]:
            ax1.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], 
                    col, alpha=0.2, linewidth=0.5)
    
    ax1.set_title(f'Affinity Propagation\\nFound {n_clusters} exemplars', 
                 fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    
    # Compare cluster sizes
    ax2 = axes[1]
    unique, counts = np.unique(labels, return_counts=True)
    ax2.bar(range(len(unique)), counts, color=colors)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Points')
    ax2.set_title('Cluster Size Distribution', fontweight='bold')
    
    plt.suptitle('Affinity Propagation: Automatic Exemplar Selection', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\\n📍 Key Features:")
    print(f"• Found {n_clusters} exemplars (actual data points)")
    print("• No need to specify number of clusters")
    print("• Each exemplar represents its cluster")
    print("• Useful for selecting representative samples")

# ============================================================================
# HELPER UTILITIES
# ============================================================================

def generate_complex_data(n_samples=500, pattern='spiral'):
    """
    Generate complex data patterns for testing algorithms.
    Includes spiral, swiss roll, and other challenging shapes.
    """
    if pattern == 'spiral':
        theta = np.sqrt(np.random.rand(n_samples)) * 4 * np.pi
        r = theta
        x = r * np.cos(theta) + np.random.randn(n_samples) * 0.5
        y = r * np.sin(theta) + np.random.randn(n_samples) * 0.5
        X = np.column_stack([x, y])
        
    elif pattern == 'swiss_roll':
        from sklearn.datasets import make_swiss_roll
        X, _ = make_swiss_roll(n_samples, noise=0.5, random_state=42)
        X = X[:, [0, 2]]  # Use 2D projection
        
    elif pattern == 'anisotropic':
        X, _ = make_blobs(n_samples=n_samples, centers=3, random_state=42)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
        
    else:
        X, _ = make_blobs(n_samples=n_samples, centers=4, random_state=42)
    
    return StandardScaler().fit_transform(X)

def evaluate_clustering_comprehensive(X, labels, algorithm_name='Algorithm'):
    """
    Comprehensive evaluation of clustering results.
    Returns multiple metrics and visualizations.
    """
    from sklearn.metrics import calinski_harabasz_score
    
    metrics = {
        'silhouette': silhouette_score(X, labels) if len(set(labels)) > 1 else -1,
        'davies_bouldin': davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.inf,
        'calinski_harabasz': calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else 0,
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'n_noise': list(labels).count(-1) if -1 in labels else 0
    }
    
    print(f"\\n📊 {algorithm_name} Evaluation:")
    print(f"Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}")
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.1f}")
    print(f"Number of Clusters: {metrics['n_clusters']}")
    if metrics['n_noise'] > 0:
        print(f"Noise Points: {metrics['n_noise']}")
    
    return metrics

def save_clustering_results(X, labels, algorithm_name, output_dir='./results'):
    """
    Save clustering results for later analysis.
    Creates visualizations and saves data.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    results_df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    results_df['Cluster'] = labels
    results_df['Algorithm'] = algorithm_name
    
    filename = f"{output_dir}/{algorithm_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(filename, index=False)
    
    print(f"✅ Results saved to {filename}")
    return results_df
'''

# Insert additional functions (convert existing source to list if it's a string)
if isinstance(existing_source, str):
    existing_source = existing_source.split('\n')
existing_source.extend(additional_functions.split('\n'))
nb['cells'][functions_cell_idx]['source'] = existing_source

# Save the updated notebook
with open(r'D:\Joerg\Research\slides\ML_Design_Thinking_16\Week_01\Week01_Part2_Technical_Design.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully added missing functions to Part 2")
print(f"Functions cell at index: {functions_cell_idx}")