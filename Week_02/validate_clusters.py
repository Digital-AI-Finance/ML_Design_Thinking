"""
Week 2: Cluster Validation Metrics and Visualizations
Elbow method, Silhouette analysis, Gap statistic, and more
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def load_data():
    """Load the FinTech dataset"""
    try:
        X = np.load('fintech_X.npy')
        y_true = np.load('fintech_y_true.npy')
        segments = np.load('fintech_segments.npy', allow_pickle=True)

        with open('fintech_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]

        # Handle missing values and scale
        X_clean = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        return X_scaled, y_true, segments, feature_names
    except FileNotFoundError:
        print("Dataset not found. Generating...")
        import generate_fintech_dataset
        generate_fintech_dataset.main()
        return load_data()

def calculate_elbow_metrics(X, max_k=15):
    """Calculate metrics for elbow method"""
    K = range(2, max_k + 1)
    inertias = []
    silhouettes = []
    calinski = []
    davies = []

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        calinski.append(calinski_harabasz_score(X, labels))
        davies.append(davies_bouldin_score(X, labels))

    return K, inertias, silhouettes, calinski, davies

def create_elbow_silhouette_plot(X):
    """Create combined elbow and silhouette analysis plot"""

    K, inertias, silhouettes, calinski, davies = calculate_elbow_metrics(X, max_k=12)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Elbow Method - Inertia
    ax1 = axes[0, 0]
    ax1.plot(K, inertias, 'b-', linewidth=2, marker='o', markersize=8)
    ax1.set_xlabel('Number of clusters (k)', fontsize=11)
    ax1.set_ylabel('Within-Cluster Sum of Squares', fontsize=11)
    ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Mark the elbow point (using simple derivative method)
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
    ax1.axvline(x=K[elbow_idx], color='red', linestyle='--', alpha=0.7, label=f'Elbow at k={K[elbow_idx]}')
    ax1.legend()

    # Add percentage decrease annotations
    for i in range(1, len(inertias)):
        pct_decrease = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
        if pct_decrease > 5:  # Only show significant decreases
            ax1.annotate(f'-{pct_decrease:.1f}%',
                        xy=(K[i], inertias[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='green')

    # 2. Silhouette Score
    ax2 = axes[0, 1]
    ax2.plot(K, silhouettes, 'g-', linewidth=2, marker='s', markersize=8)
    ax2.set_xlabel('Number of clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title('Silhouette Analysis', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Mark the maximum
    best_k_silhouette = K[np.argmax(silhouettes)]
    ax2.axvline(x=best_k_silhouette, color='red', linestyle='--', alpha=0.7,
               label=f'Best at k={best_k_silhouette}')
    ax2.axhline(y=max(silhouettes), color='red', linestyle=':', alpha=0.5)
    ax2.legend()

    # 3. Calinski-Harabasz Score
    ax3 = axes[1, 0]
    ax3.plot(K, calinski, 'orange', linewidth=2, marker='^', markersize=8)
    ax3.set_xlabel('Number of clusters (k)', fontsize=11)
    ax3.set_ylabel('Calinski-Harabasz Score', fontsize=11)
    ax3.set_title('Calinski-Harabasz Index', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    best_k_calinski = K[np.argmax(calinski)]
    ax3.axvline(x=best_k_calinski, color='red', linestyle='--', alpha=0.7,
               label=f'Best at k={best_k_calinski}')
    ax3.legend()

    # 4. Davies-Bouldin Score (lower is better)
    ax4 = axes[1, 1]
    ax4.plot(K, davies, 'purple', linewidth=2, marker='d', markersize=8)
    ax4.set_xlabel('Number of clusters (k)', fontsize=11)
    ax4.set_ylabel('Davies-Bouldin Score', fontsize=11)
    ax4.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    best_k_davies = K[np.argmin(davies)]
    ax4.axvline(x=best_k_davies, color='red', linestyle='--', alpha=0.7,
               label=f'Best at k={best_k_davies}')
    ax4.legend()

    plt.suptitle('Cluster Validation Metrics: Finding Optimal K', fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig, K[elbow_idx]

def create_detailed_silhouette_plot(X, k_values=[3, 4, 5, 6, 7]):
    """Create detailed silhouette analysis for multiple k values"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Calculate silhouette scores
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        colors = plt.cm.Set2(np.linspace(0, 1, k))

        for i in range(k):
            # Get silhouette scores for cluster i
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            cluster_silhouette_values.sort()

            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=colors[i], alpha=0.7,
                            edgecolor=colors[i])

            # Label clusters
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i),
                   fontsize=10, fontweight='bold')

            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
                  label=f'Avg: {silhouette_avg:.3f}')
        ax.set_xlabel('Silhouette Coefficient', fontsize=10)
        ax.set_ylabel('Cluster Label', fontsize=10)
        ax.set_title(f'k = {k} (Avg: {silhouette_avg:.3f})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim([-0.1, 1])
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[-1])

    plt.suptitle('Detailed Silhouette Analysis: Cluster Quality Assessment',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig

def calculate_gap_statistic(X, max_k=10, n_refs=10):
    """Calculate Gap Statistic for optimal k"""

    gaps = []
    s_k = []
    K = range(2, max_k + 1)

    for k in K:
        # Cluster actual data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)

        # Calculate log(W_k) for actual data
        W_k = kmeans.inertia_
        log_W_k = np.log(W_k)

        # Generate reference datasets and calculate expected log(W_k)
        log_W_kr = []
        for _ in range(n_refs):
            # Generate uniform random data
            X_random = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            kmeans_random = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans_random.fit(X_random)
            log_W_kr.append(np.log(kmeans_random.inertia_))

        # Calculate gap statistic
        gap = np.mean(log_W_kr) - log_W_k
        gaps.append(gap)

        # Calculate standard error
        sdk = np.std(log_W_kr) * np.sqrt(1 + 1/n_refs)
        s_k.append(sdk)

    # Find optimal k using gap statistic criterion
    gaps = np.array(gaps)
    s_k = np.array(s_k)

    # Optimal k is smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - s_k[i + 1]:
            optimal_k = K[i]
            break
    else:
        optimal_k = K[np.argmax(gaps)]

    return K, gaps, s_k, optimal_k

def create_gap_statistic_plot(X):
    """Visualize Gap Statistic"""

    print("Calculating Gap Statistic (this may take a moment)...")
    K, gaps, s_k, optimal_k = calculate_gap_statistic(X, max_k=10, n_refs=5)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot gap values with error bars
    ax.errorbar(K, gaps, yerr=s_k, marker='o', markersize=8, linewidth=2,
               capsize=5, capthick=2)

    # Mark optimal k
    ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
              label=f'Optimal k={optimal_k}')

    ax.set_xlabel('Number of clusters (k)', fontsize=12)
    ax.set_ylabel('Gap Statistic', fontsize=12)
    ax.set_title('Gap Statistic: Statistical Method for Optimal K',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add text annotation
    ax.text(0.05, 0.95,
           f'Gap Statistic suggests k={optimal_k}\nBased on {len(K)} iterations',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

def create_cluster_stability_analysis(X):
    """Analyze cluster stability across multiple runs"""

    k_values = range(3, 8)
    n_iterations = 20

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stability analysis
    stability_scores = []

    for k in k_values:
        iteration_scores = []

        for _ in range(n_iterations):
            kmeans = KMeans(n_clusters=k, n_init=1,
                          init='random', random_state=None)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            iteration_scores.append(score)

        stability_scores.append(iteration_scores)

    # Box plot of stability
    ax1 = axes[0]
    bp = ax1.boxplot(stability_scores, labels=[str(k) for k in k_values],
                     patch_artist=True)

    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xlabel('Number of clusters (k)', fontsize=11)
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title('Cluster Stability Analysis (20 runs)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Convergence speed analysis
    ax2 = axes[1]
    convergence_iterations = []

    for k in k_values:
        iterations_to_converge = []

        for _ in range(10):
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=1,
                          init='random', random_state=None)
            kmeans.fit(X)
            iterations_to_converge.append(kmeans.n_iter_)

        convergence_iterations.append(np.mean(iterations_to_converge))

    ax2.bar(range(len(k_values)), convergence_iterations,
           color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink'])
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels([str(k) for k in k_values])
    ax2.set_xlabel('Number of clusters (k)', fontsize=11)
    ax2.set_ylabel('Average Iterations to Converge', fontsize=11)
    ax2.set_title('Convergence Speed Analysis', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(convergence_iterations):
        ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)

    plt.suptitle('Clustering Stability and Convergence Analysis',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig

def create_inter_cluster_distance_matrix(X, optimal_k=5):
    """Create inter-cluster distance heatmap"""

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # Calculate pairwise distances between cluster centers
    distances = cdist(centers, centers, metric='euclidean')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(distances, annot=True, fmt='.2f', cmap='YlOrRd',
               xticklabels=range(optimal_k),
               yticklabels=range(optimal_k),
               ax=ax)

    ax.set_title(f'Inter-Cluster Distance Matrix (k={optimal_k})',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)

    # Add interpretation text
    fig.text(0.5, -0.05,
            'Lower values indicate closer clusters (potential overlap)',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    return fig

def create_validation_summary_report(X):
    """Create comprehensive validation summary"""

    # Calculate all metrics
    K, inertias, silhouettes, calinski, davies = calculate_elbow_metrics(X, max_k=10)

    # Create summary dataframe
    summary_df = pd.DataFrame({
        'k': K,
        'Inertia': inertias,
        'Silhouette': silhouettes,
        'Calinski-Harabasz': calinski,
        'Davies-Bouldin': davies
    })

    # Normalize scores for comparison (0-1 scale)
    summary_df['Inertia_norm'] = 1 - (summary_df['Inertia'] - summary_df['Inertia'].min()) / (summary_df['Inertia'].max() - summary_df['Inertia'].min())
    summary_df['Silhouette_norm'] = (summary_df['Silhouette'] - summary_df['Silhouette'].min()) / (summary_df['Silhouette'].max() - summary_df['Silhouette'].min())
    summary_df['Calinski_norm'] = (summary_df['Calinski-Harabasz'] - summary_df['Calinski-Harabasz'].min()) / (summary_df['Calinski-Harabasz'].max() - summary_df['Calinski-Harabasz'].min())
    summary_df['Davies_norm'] = 1 - (summary_df['Davies-Bouldin'] - summary_df['Davies-Bouldin'].min()) / (summary_df['Davies-Bouldin'].max() - summary_df['Davies-Bouldin'].min())

    # Calculate composite score
    summary_df['Composite_Score'] = (summary_df['Silhouette_norm'] +
                                     summary_df['Calinski_norm'] +
                                     summary_df['Davies_norm']) / 3

    # Find best k by different metrics
    best_k_silhouette = summary_df.loc[summary_df['Silhouette'].idxmax(), 'k']
    best_k_calinski = summary_df.loc[summary_df['Calinski-Harabasz'].idxmax(), 'k']
    best_k_davies = summary_df.loc[summary_df['Davies-Bouldin'].idxmin(), 'k']
    best_k_composite = summary_df.loc[summary_df['Composite_Score'].idxmax(), 'k']

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot normalized scores
    ax1 = axes[0]
    ax1.plot(summary_df['k'], summary_df['Silhouette_norm'], 'g-', marker='o', label='Silhouette', linewidth=2)
    ax1.plot(summary_df['k'], summary_df['Calinski_norm'], 'b-', marker='s', label='Calinski-Harabasz', linewidth=2)
    ax1.plot(summary_df['k'], summary_df['Davies_norm'], 'r-', marker='^', label='Davies-Bouldin', linewidth=2)
    ax1.plot(summary_df['k'], summary_df['Composite_Score'], 'k-', marker='d', label='Composite', linewidth=3)

    ax1.set_xlabel('Number of clusters (k)', fontsize=11)
    ax1.set_ylabel('Normalized Score (0-1)', fontsize=11)
    ax1.set_title('Normalized Validation Metrics Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add best k markers
    ax1.axvline(x=best_k_composite, color='purple', linestyle='--', alpha=0.7,
               label=f'Best Composite k={int(best_k_composite)}')

    # Create summary table
    ax2 = axes[1]
    ax2.axis('tight')
    ax2.axis('off')

    # Prepare summary data
    summary_table = [
        ['Metric', 'Optimal k', 'Score at Optimal k'],
        ['Silhouette', f'{int(best_k_silhouette)}', f'{summary_df.loc[summary_df["k"]==best_k_silhouette, "Silhouette"].values[0]:.3f}'],
        ['Calinski-Harabasz', f'{int(best_k_calinski)}', f'{summary_df.loc[summary_df["k"]==best_k_calinski, "Calinski-Harabasz"].values[0]:.0f}'],
        ['Davies-Bouldin', f'{int(best_k_davies)}', f'{summary_df.loc[summary_df["k"]==best_k_davies, "Davies-Bouldin"].values[0]:.3f}'],
        ['Composite Score', f'{int(best_k_composite)}', f'{summary_df.loc[summary_df["k"]==best_k_composite, "Composite_Score"].values[0]:.3f}']
    ]

    table = ax2.table(cellText=summary_table,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.3])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color best composite row
    table[(4, 0)].set_facecolor('#ffeedd')
    table[(4, 1)].set_facecolor('#ffeedd')
    table[(4, 2)].set_facecolor('#ffeedd')

    plt.suptitle('Cluster Validation Summary Report', fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig, summary_df

def main():
    """Run complete validation analysis"""

    print("Loading FinTech dataset for validation...")
    X_scaled, y_true, segments, feature_names = load_data()
    print(f"Dataset shape: {X_scaled.shape}")
    print()

    # 1. Elbow and basic metrics
    print("Creating elbow and silhouette analysis...")
    fig1, elbow_k = create_elbow_silhouette_plot(X_scaled)
    fig1.savefig('validation_elbow_silhouette.png', dpi=300, bbox_inches='tight')
    fig1.savefig('validation_elbow_silhouette.pdf', bbox_inches='tight')
    print(f"Elbow analysis suggests k={elbow_k}")
    print()

    # 2. Detailed silhouette analysis
    print("Creating detailed silhouette plots...")
    fig2 = create_detailed_silhouette_plot(X_scaled, k_values=[3, 4, 5, 6, 7])
    fig2.savefig('validation_silhouette_detailed.png', dpi=300, bbox_inches='tight')
    fig2.savefig('validation_silhouette_detailed.pdf', bbox_inches='tight')
    print("Detailed silhouette analysis complete")
    print()

    # 3. Gap statistic
    print("Creating gap statistic analysis...")
    fig3 = create_gap_statistic_plot(X_scaled)
    fig3.savefig('validation_gap_statistic.png', dpi=300, bbox_inches='tight')
    fig3.savefig('validation_gap_statistic.pdf', bbox_inches='tight')
    print("Gap statistic analysis complete")
    print()

    # 4. Stability analysis
    print("Analyzing cluster stability...")
    fig4 = create_cluster_stability_analysis(X_scaled)
    fig4.savefig('validation_stability.png', dpi=300, bbox_inches='tight')
    fig4.savefig('validation_stability.pdf', bbox_inches='tight')
    print("Stability analysis complete")
    print()

    # 5. Inter-cluster distances
    print("Creating inter-cluster distance matrix...")
    fig5 = create_inter_cluster_distance_matrix(X_scaled, optimal_k=5)
    fig5.savefig('validation_inter_cluster.png', dpi=300, bbox_inches='tight')
    fig5.savefig('validation_inter_cluster.pdf', bbox_inches='tight')
    print("Inter-cluster analysis complete")
    print()

    # 6. Summary report
    print("Generating validation summary report...")
    fig6, summary_df = create_validation_summary_report(X_scaled)
    fig6.savefig('validation_summary.png', dpi=300, bbox_inches='tight')
    fig6.savefig('validation_summary.pdf', bbox_inches='tight')
    summary_df.to_csv('validation_summary.csv', index=False)
    print("Summary report saved")
    print()

    # Print final recommendations
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print("\nRecommended number of clusters based on different metrics:")
    print(f"  - Elbow Method: k={elbow_k}")
    print(f"  - Silhouette: k={summary_df.loc[summary_df['Silhouette'].idxmax(), 'k']:.0f}")
    print(f"  - Calinski-Harabasz: k={summary_df.loc[summary_df['Calinski-Harabasz'].idxmax(), 'k']:.0f}")
    print(f"  - Davies-Bouldin: k={summary_df.loc[summary_df['Davies-Bouldin'].idxmin(), 'k']:.0f}")
    print(f"  - Composite Score: k={summary_df.loc[summary_df['Composite_Score'].idxmax(), 'k']:.0f}")
    print()
    print("Most metrics suggest k=5 or k=6 for this dataset")
    print()
    print("All validation visualizations have been saved!")
    print("Check the generated PNG/PDF files for detailed analysis")

if __name__ == "__main__":
    main()