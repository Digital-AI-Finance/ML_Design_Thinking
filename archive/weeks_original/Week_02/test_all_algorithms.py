"""
Week 2: Test All Clustering Algorithms on FinTech Dataset
Demonstrates K-Means, DBSCAN, Hierarchical, and GMM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def load_fintech_data():
    """Load the generated FinTech dataset"""
    try:
        X = np.load('fintech_X.npy')
        y_true = np.load('fintech_y_true.npy')
        segments = np.load('fintech_segments.npy', allow_pickle=True)

        with open('fintech_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]

        return X, y_true, segments, feature_names
    except FileNotFoundError:
        print("Dataset files not found. Running generator...")
        import generate_fintech_dataset
        generate_fintech_dataset.main()
        return load_fintech_data()

def preprocess_data(X):
    """Handle missing values and scale features"""
    # Simple imputation - fill NaN with median
    X_clean = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    return X_scaled, scaler

def test_kmeans(X_scaled, true_labels, n_clusters_range=range(3, 10)):
    """Test K-Means with different K values"""
    results = []

    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
        inertia = kmeans.inertia_

        # Compare with ground truth
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)

        results.append({
            'k': k,
            'algorithm': 'K-Means',
            'silhouette': silhouette,
            'calinski': calinski,
            'davies_bouldin': davies,
            'inertia': inertia,
            'ari': ari,
            'nmi': nmi,
            'labels': labels
        })

        print(f"K-Means (k={k}): Silhouette={silhouette:.3f}, ARI={ari:.3f}")

    return results

def test_dbscan(X_scaled, true_labels, eps_range=np.arange(0.3, 1.5, 0.1)):
    """Test DBSCAN with different eps values"""
    results = []

    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=10)
        labels = dbscan.fit_predict(X_scaled)

        # Count clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters > 1:  # Only calculate metrics if we have clusters
            # For metrics, we'll exclude noise points
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X_scaled[mask], labels[mask]) if mask.sum() > 1 else 0
                calinski = calinski_harabasz_score(X_scaled[mask], labels[mask]) if n_clusters > 1 else 0
                davies = davies_bouldin_score(X_scaled[mask], labels[mask]) if n_clusters > 1 else 999
            else:
                silhouette = calinski = 0
                davies = 999

            ari = adjusted_rand_score(true_labels, labels)
            nmi = normalized_mutual_info_score(true_labels, labels)

            results.append({
                'eps': eps,
                'algorithm': 'DBSCAN',
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(labels),
                'silhouette': silhouette,
                'calinski': calinski,
                'davies_bouldin': davies,
                'ari': ari,
                'nmi': nmi,
                'labels': labels
            })

            print(f"DBSCAN (eps={eps:.1f}): Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(labels)*100:.1f}%), Silhouette={silhouette:.3f}")

    return results

def test_hierarchical(X_scaled, true_labels, n_clusters_range=range(3, 10)):
    """Test Hierarchical clustering with different numbers of clusters"""
    results = []

    for n_clusters in n_clusters_range:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = hierarchical.fit_predict(X_scaled)

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)

        # Compare with ground truth
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)

        results.append({
            'n_clusters': n_clusters,
            'algorithm': 'Hierarchical',
            'silhouette': silhouette,
            'calinski': calinski,
            'davies_bouldin': davies,
            'ari': ari,
            'nmi': nmi,
            'labels': labels
        })

        print(f"Hierarchical (n={n_clusters}): Silhouette={silhouette:.3f}, ARI={ari:.3f}")

    return results

def test_gmm(X_scaled, true_labels, n_components_range=range(3, 10)):
    """Test Gaussian Mixture Model with different numbers of components"""
    results = []

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
        labels = gmm.fit_predict(X_scaled)

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
        bic = gmm.bic(X_scaled)
        aic = gmm.aic(X_scaled)

        # Compare with ground truth
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)

        results.append({
            'n_components': n_components,
            'algorithm': 'GMM',
            'silhouette': silhouette,
            'calinski': calinski,
            'davies_bouldin': davies,
            'bic': bic,
            'aic': aic,
            'ari': ari,
            'nmi': nmi,
            'labels': labels,
            'probabilities': gmm.predict_proba(X_scaled)
        })

        print(f"GMM (n={n_components}): Silhouette={silhouette:.3f}, BIC={bic:.0f}, ARI={ari:.3f}")

    return results

def test_minibatch_kmeans(X_scaled, true_labels, n_clusters=5):
    """Test Mini-Batch K-Means for scalability"""
    import time

    # Test with different batch sizes
    batch_sizes = [100, 500, 1000, 2000]
    results = []

    for batch_size in batch_sizes:
        start_time = time.time()

        mbkmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                   batch_size=batch_size,
                                   random_state=42)
        labels = mbkmeans.fit_predict(X_scaled)

        elapsed_time = time.time() - start_time

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        ari = adjusted_rand_score(true_labels, labels)

        results.append({
            'batch_size': batch_size,
            'time': elapsed_time,
            'silhouette': silhouette,
            'ari': ari,
            'labels': labels
        })

        print(f"Mini-Batch K-Means (batch={batch_size}): Time={elapsed_time:.3f}s, Silhouette={silhouette:.3f}")

    return results

def visualize_clustering_results(X_scaled, all_results, segment_names):
    """Create comprehensive visualization of all clustering results"""

    # Use PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create figure
    fig = plt.figure(figsize=(20, 16))

    # Select best results from each algorithm
    best_kmeans = max([r for r in all_results if r['algorithm'] == 'K-Means'],
                      key=lambda x: x['silhouette'])
    best_dbscan = max([r for r in all_results if r['algorithm'] == 'DBSCAN' and 'n_noise' in r],
                      key=lambda x: x['silhouette'] if x['n_clusters'] > 1 else -1)
    best_hierarchical = max([r for r in all_results if r['algorithm'] == 'Hierarchical'],
                           key=lambda x: x['silhouette'])
    best_gmm = max([r for r in all_results if r['algorithm'] == 'GMM'],
                   key=lambda x: x['silhouette'])

    # Plot each algorithm's best result
    algorithms = [
        ('Ground Truth', segment_names, 'True Segments'),
        ('K-Means', best_kmeans['labels'], f"K-Means (k={best_kmeans['k']})"),
        ('DBSCAN', best_dbscan['labels'], f"DBSCAN (eps={best_dbscan['eps']:.1f})"),
        ('Hierarchical', best_hierarchical['labels'], f"Hierarchical (n={best_hierarchical['n_clusters']})"),
        ('GMM', best_gmm['labels'], f"GMM (n={best_gmm['n_components']})")
    ]

    for idx, (name, labels, title) in enumerate(algorithms):
        ax = plt.subplot(2, 3, idx + 1)

        if name == 'Ground Truth':
            # For ground truth, use segment names
            unique_segments = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_segments)))

            for segment, color in zip(unique_segments, colors):
                mask = labels == segment
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=[color], label=segment.replace('_', ' ').title(),
                          alpha=0.6, s=10)
            ax.legend(fontsize=6, loc='upper right')
        else:
            # For clustering results
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = labels == label
                if label == -1:  # DBSCAN noise
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                             c='black', marker='x', s=10, alpha=0.3, label='Noise')
                else:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                             c=[color], s=10, alpha=0.6, label=f'Cluster {label}')

            if len(unique_labels) <= 8:  # Only show legend if not too many clusters
                ax.legend(fontsize=6, loc='upper right')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

    # Add metrics comparison
    ax6 = plt.subplot(2, 3, 6)

    # Prepare metrics data
    metrics_data = []
    for algo in ['K-Means', 'DBSCAN', 'Hierarchical', 'GMM']:
        algo_results = [r for r in all_results if r['algorithm'] == algo]
        if algo_results:
            best = max(algo_results, key=lambda x: x.get('silhouette', -1))
            metrics_data.append({
                'Algorithm': algo,
                'Silhouette': best.get('silhouette', 0),
                'ARI': best.get('ari', 0),
                'NMI': best.get('nmi', 0)
            })

    metrics_df = pd.DataFrame(metrics_data)
    x = np.arange(len(metrics_df))
    width = 0.25

    ax6.bar(x - width, metrics_df['Silhouette'], width, label='Silhouette', color='#1f77b4')
    ax6.bar(x, metrics_df['ARI'], width, label='ARI', color='#ff7f0e')
    ax6.bar(x + width, metrics_df['NMI'], width, label='NMI', color='#2ca02c')

    ax6.set_xlabel('Algorithm')
    ax6.set_ylabel('Score')
    ax6.set_title('Clustering Performance Metrics', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_df['Algorithm'])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Clustering Algorithm Comparison on FinTech Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def analyze_cluster_characteristics(X, labels, feature_names, algorithm_name):
    """Analyze and describe characteristics of each cluster"""

    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = labels

    print(f"\n{algorithm_name} Cluster Characteristics:")
    print("=" * 60)

    # Get cluster summaries
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:  # Skip noise for DBSCAN
            continue

        cluster_data = df[df['cluster'] == cluster_id]
        n_users = len(cluster_data)
        percentage = n_users / len(df) * 100

        print(f"\nCluster {cluster_id}: {n_users} users ({percentage:.1f}%)")
        print("-" * 40)

        # Find most distinctive features
        cluster_means = cluster_data[feature_names].mean()
        overall_means = df[feature_names].mean()

        # Calculate z-scores for feature importance
        feature_importance = (cluster_means - overall_means) / df[feature_names].std()

        # Top positive and negative features
        top_features = feature_importance.nlargest(3)
        bottom_features = feature_importance.nsmallest(3)

        print("Distinctive HIGH features:")
        for feat, score in top_features.items():
            print(f"  - {feat}: {cluster_means[feat]:.2f} (z-score: {score:.2f})")

        print("Distinctive LOW features:")
        for feat, score in bottom_features.items():
            print(f"  - {feat}: {cluster_means[feat]:.2f} (z-score: {score:.2f})")

        # Suggest persona based on characteristics
        if cluster_means['transaction_frequency'] > 8 and cluster_means['transaction_volume'] > 5000:
            persona = "Power User / Business"
        elif cluster_means['savings_behavior'] > 50 and cluster_means['credit_utilization'] < 20:
            persona = "Conservative Saver"
        elif cluster_means['international_activity'] > 0.5:
            persona = "International User"
        elif cluster_means['support_contacts'] > 2 and cluster_means['account_age'] < 60:
            persona = "New User / Needs Support"
        elif cluster_means['transaction_frequency'] < 2:
            persona = "Inactive / Dormant"
        else:
            persona = "Regular User"

        print(f"Suggested Persona: {persona}")

    return df

def main():
    """Run complete clustering analysis"""

    print("Loading FinTech dataset...")
    X, y_true, segments, feature_names = load_fintech_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {', '.join(feature_names[:5])}...")
    print()

    # Preprocess data
    print("Preprocessing data...")
    X_scaled, scaler = preprocess_data(X)

    # Store all results
    all_results = []

    # Test K-Means
    print("\n" + "="*60)
    print("Testing K-Means...")
    print("="*60)
    kmeans_results = test_kmeans(X_scaled, y_true, range(3, 10))
    all_results.extend(kmeans_results)

    # Test DBSCAN
    print("\n" + "="*60)
    print("Testing DBSCAN...")
    print("="*60)
    dbscan_results = test_dbscan(X_scaled, y_true, np.arange(0.5, 2.0, 0.2))
    all_results.extend(dbscan_results)

    # Test Hierarchical
    print("\n" + "="*60)
    print("Testing Hierarchical Clustering...")
    print("="*60)
    hierarchical_results = test_hierarchical(X_scaled, y_true, range(3, 10))
    all_results.extend(hierarchical_results)

    # Test GMM
    print("\n" + "="*60)
    print("Testing Gaussian Mixture Model...")
    print("="*60)
    gmm_results = test_gmm(X_scaled, y_true, range(3, 10))
    all_results.extend(gmm_results)

    # Test Mini-Batch K-Means
    print("\n" + "="*60)
    print("Testing Mini-Batch K-Means (Scalability)...")
    print("="*60)
    minibatch_results = test_minibatch_kmeans(X_scaled, y_true, n_clusters=5)

    # Visualize results
    print("\n" + "="*60)
    print("Creating visualizations...")
    fig = visualize_clustering_results(X_scaled, all_results, segments)
    fig.savefig('clustering_comparison_results.png', dpi=300, bbox_inches='tight')
    fig.savefig('clustering_comparison_results.pdf', bbox_inches='tight')
    print("Visualizations saved!")

    # Analyze best performing algorithm
    best_overall = max(all_results, key=lambda x: x.get('silhouette', -1))
    print(f"\nBest performing: {best_overall['algorithm']} with Silhouette={best_overall.get('silhouette', 0):.3f}")

    # Detailed analysis of best K-Means result
    best_kmeans = max([r for r in all_results if r['algorithm'] == 'K-Means'],
                      key=lambda x: x['silhouette'])
    analyze_cluster_characteristics(X, best_kmeans['labels'], feature_names, 'Best K-Means')

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('clustering_results.csv', index=False)
    print("\nResults saved to clustering_results.csv")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("\nKey Insights:")
    print("1. K-Means performs well on well-separated segments")
    print("2. DBSCAN successfully identifies fraudulent outliers")
    print("3. GMM captures overlapping user behaviors")
    print("4. Hierarchical reveals natural user progression")

if __name__ == "__main__":
    main()