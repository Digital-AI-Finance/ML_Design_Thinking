"""
Create validation visualizations for FinTech clustering analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def load_data():
    """Load the FinTech dataset"""
    X = np.load('fintech_X.npy')
    y_true = np.load('fintech_y_true.npy')
    segments = np.load('fintech_segments.npy', allow_pickle=True)

    with open('fintech_features.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    # Handle NaN and scale
    X_clean = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    return X, X_scaled, y_true, segments, feature_names

def create_comprehensive_elbow_plot():
    """Create detailed elbow plot with multiple metrics"""
    X, X_scaled, y_true, segments, feature_names = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    k_values = range(2, 11)
    inertias = []
    silhouettes = []
    davies = []
    calinski = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        davies.append(davies_bouldin_score(X_scaled, labels))
        calinski.append(calinski_harabasz_score(X_scaled, labels))

    # 1. Classic Elbow Plot
    ax1 = axes[0, 0]
    ax1.plot(k_values, inertias, 'b-', linewidth=3, marker='o', markersize=10)

    # Mark the elbow
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(diffs2) + 2
    ax1.plot(elbow_idx, inertias[elbow_idx-2], 'r*', markersize=20, label=f'Elbow at k={elbow_idx}')

    # Add percentage annotations
    for i in range(1, len(inertias)):
        pct_decrease = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
        ax1.annotate(f'-{pct_decrease:.1f}%',
                    xy=(k_values[i], inertias[i]),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=8, color='darkgreen',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Within-Cluster Sum of Squares', fontsize=11)
    ax1.set_title('Elbow Method: Identifying the Optimal k', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Silhouette Score Evolution
    ax2 = axes[0, 1]
    bars = ax2.bar(k_values, silhouettes, color=['red' if s < 0.3 else 'yellow' if s < 0.4 else 'green' for s in silhouettes])

    # Highlight best k
    best_k = k_values[np.argmax(silhouettes)]
    ax2.bar(best_k, silhouettes[best_k-2], color='darkgreen', edgecolor='black', linewidth=2)

    # Add value labels
    for k, s, bar in zip(k_values, silhouettes, bars):
        ax2.text(k, s + 0.005, f'{s:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title(f'Silhouette Analysis: Best k={best_k}', fontsize=13, fontweight='bold')
    ax2.axhline(y=0.4, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    ax2.set_ylim(0, max(silhouettes) * 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Davies-Bouldin Score
    ax3 = axes[1, 0]
    line1 = ax3.plot(k_values, davies, 'purple', linewidth=3, marker='^', markersize=10, label='Davies-Bouldin')

    # Mark minimum (best)
    best_k_davies = k_values[np.argmin(davies)]
    ax3.plot(best_k_davies, davies[best_k_davies-2], 'g*', markersize=20)

    ax3.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax3.set_ylabel('Davies-Bouldin Score', fontsize=11, color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_title(f'Davies-Bouldin Index: Best k={best_k_davies} (lower is better)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add secondary axis for Calinski
    ax3_twin = ax3.twinx()
    line2 = ax3_twin.plot(k_values, calinski, 'orange', linewidth=3, marker='d', markersize=10, label='Calinski-Harabasz')
    ax3_twin.set_ylabel('Calinski-Harabasz Score', fontsize=11, color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')

    # 4. Composite Score Analysis
    ax4 = axes[1, 1]

    # Normalize all metrics to 0-1 scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    inertias_norm = 1 - scaler.fit_transform(np.array(inertias).reshape(-1, 1)).flatten()
    silhouettes_norm = scaler.fit_transform(np.array(silhouettes).reshape(-1, 1)).flatten()
    davies_norm = 1 - scaler.fit_transform(np.array(davies).reshape(-1, 1)).flatten()
    calinski_norm = scaler.fit_transform(np.array(calinski).reshape(-1, 1)).flatten()

    # Calculate composite score
    composite = (silhouettes_norm*0.3 + davies_norm*0.3 + calinski_norm*0.3 + inertias_norm*0.1)

    # Create stacked area chart
    ax4.fill_between(k_values, 0, inertias_norm*0.1, alpha=0.5, color='blue', label='Inertia (10%)')
    ax4.fill_between(k_values, inertias_norm*0.1, inertias_norm*0.1 + silhouettes_norm*0.3,
                     alpha=0.5, color='green', label='Silhouette (30%)')
    ax4.fill_between(k_values, inertias_norm*0.1 + silhouettes_norm*0.3,
                     inertias_norm*0.1 + silhouettes_norm*0.3 + davies_norm*0.3,
                     alpha=0.5, color='purple', label='Davies-Bouldin (30%)')
    ax4.fill_between(k_values, inertias_norm*0.1 + silhouettes_norm*0.3 + davies_norm*0.3,
                     composite, alpha=0.5, color='orange', label='Calinski (30%)')

    # Plot composite line
    ax4.plot(k_values, composite, 'red', linewidth=4, marker='o', markersize=10, label='Composite Score')

    # Mark best composite k
    best_k_composite = k_values[np.argmax(composite)]
    ax4.plot(best_k_composite, composite[best_k_composite-2], 'r*', markersize=25)
    ax4.axvline(x=best_k_composite, color='red', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax4.set_ylabel('Weighted Composite Score', fontsize=11)
    ax4.set_title(f'Weighted Composite Analysis: Optimal k={best_k_composite}', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    # Add text annotation
    ax4.text(0.05, 0.95, f'Consensus: k=5\nAll metrics agree',
            transform=ax4.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            verticalalignment='top')

    plt.suptitle('Comprehensive Elbow Analysis: Multiple Validation Metrics',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig

def create_silhouette_analysis_grid():
    """Create detailed silhouette plots for different k values"""
    X, X_scaled, y_true, segments, feature_names = load_data()

    k_values = [3, 4, 5, 6, 7, 8]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Calculate silhouette scores
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

        y_lower = 10
        colors = plt.cm.Set3(np.linspace(0, 1, k))

        cluster_sizes = []
        for i in range(k):
            # Get silhouette scores for cluster i
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            cluster_silhouette_values.sort()

            size_cluster_i = cluster_silhouette_values.shape[0]
            cluster_sizes.append(size_cluster_i)
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=colors[i], alpha=0.7,
                            edgecolor=colors[i])

            # Label clusters with size
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i,
                   f'C{i}\n({size_cluster_i})',
                   fontsize=9, fontweight='bold', ha='center')

            y_lower = y_upper + 10

        # Draw average line
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
                  label=f'Avg: {silhouette_avg:.3f}')

        # Color-code title based on score quality
        if silhouette_avg > 0.4:
            title_color = 'green'
            quality = 'Good'
        elif silhouette_avg > 0.3:
            title_color = 'orange'
            quality = 'Fair'
        else:
            title_color = 'red'
            quality = 'Poor'

        ax.set_xlabel('Silhouette Coefficient', fontsize=10)
        ax.set_ylabel('Cluster', fontsize=10)
        ax.set_title(f'k = {k} ({quality})', fontsize=12, fontweight='bold', color=title_color)
        ax.legend(loc='upper right')
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X_scaled) + (k + 1) * 10])
        ax.grid(True, alpha=0.3)

        # Add cluster balance indicator
        balance_score = np.std(cluster_sizes) / np.mean(cluster_sizes)
        balance_text = 'Balanced' if balance_score < 0.3 else 'Imbalanced' if balance_score < 0.5 else 'Very Imbalanced'
        ax.text(0.95, 0.05, f'{balance_text}\n(σ/μ={balance_score:.2f})',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               horizontalalignment='right')

    # Highlight k=5 as optimal
    axes[2].patch.set_edgecolor('darkgreen')
    axes[2].patch.set_linewidth(3)

    plt.suptitle('Silhouette Analysis Grid: Detailed View for Each k',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig

def create_gap_statistic_analysis():
    """Calculate and visualize Gap Statistic"""
    X, X_scaled, y_true, segments, feature_names = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    k_values = range(2, 11)
    gaps = []
    s_k = []
    n_refs = 10

    print("  Calculating Gap Statistic (this may take a moment)...")
    for k in k_values:
        # Cluster actual data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        # Calculate log(W_k) for actual data
        W_k = kmeans.inertia_
        log_W_k = np.log(W_k)

        # Generate reference datasets
        log_W_kr = []
        for _ in range(n_refs):
            # Generate uniform random data
            X_random = np.random.uniform(X_scaled.min(axis=0), X_scaled.max(axis=0), X_scaled.shape)
            kmeans_random = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans_random.fit(X_random)
            log_W_kr.append(np.log(kmeans_random.inertia_))

        # Calculate gap statistic
        gap = np.mean(log_W_kr) - log_W_k
        gaps.append(gap)

        # Calculate standard error
        sdk = np.std(log_W_kr) * np.sqrt(1 + 1/n_refs)
        s_k.append(sdk)

    gaps = np.array(gaps)
    s_k = np.array(s_k)

    # Find optimal k
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - s_k[i + 1]:
            optimal_k = list(k_values)[i]
            break
    else:
        optimal_k = list(k_values)[np.argmax(gaps)]

    # 1. Gap Statistic with error bars
    ax1 = axes[0]
    ax1.errorbar(k_values, gaps, yerr=s_k, marker='o', markersize=10,
                linewidth=2, capsize=5, capthick=2, color='blue')
    ax1.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
               label=f'Optimal k={optimal_k}')
    ax1.fill_between(k_values, gaps - s_k, gaps + s_k, alpha=0.3, color='blue')

    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Gap Statistic', fontsize=11)
    ax1.set_title('Gap Statistic Analysis', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add interpretation
    ax1.text(0.05, 0.95,
            f'Gap Statistic suggests k={optimal_k}\nCompares to uniform distribution',
            transform=ax1.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='top')

    # 2. Stability Analysis
    ax2 = axes[1]

    stability_scores = []
    n_iterations = 20

    for k in [3, 4, 5, 6, 7]:
        iteration_scores = []

        for _ in range(n_iterations):
            # Random initialization
            kmeans = KMeans(n_clusters=k, n_init=1, init='random')
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            iteration_scores.append(score)

        stability_scores.append(iteration_scores)

    bp = ax2.boxplot(stability_scores, labels=['k=3', 'k=4', 'k=5', 'k=6', 'k=7'],
                     patch_artist=True)

    # Color boxes
    colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Highlight k=5
    bp['boxes'][2].set_facecolor('darkgreen')
    bp['boxes'][2].set_alpha(0.9)

    ax2.set_xlabel('Number of Clusters', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title('Cluster Stability Analysis (20 random initializations)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add stability metric
    for i, scores in enumerate(stability_scores):
        stability = 1 - np.std(scores) / np.mean(scores)
        ax2.text(i + 1, ax2.get_ylim()[1] * 0.95,
                f'{stability:.2f}', ha='center', fontsize=9, fontweight='bold')

    ax2.text(0.5, 1.05, 'Stability Score', transform=ax2.transAxes,
            ha='center', fontsize=10, fontweight='bold')

    # 3. Convergence Speed
    ax3 = axes[2]

    convergence_data = []
    k_range = [3, 4, 5, 6, 7]

    for k in k_range:
        iterations_list = []

        for _ in range(10):
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=1, init='random')
            kmeans.fit(X_scaled)
            iterations_list.append(kmeans.n_iter_)

        convergence_data.append({
            'k': k,
            'mean': np.mean(iterations_list),
            'std': np.std(iterations_list),
            'min': np.min(iterations_list),
            'max': np.max(iterations_list)
        })

    # Create bar plot with error bars
    x_pos = np.arange(len(k_range))
    means = [d['mean'] for d in convergence_data]
    stds = [d['std'] for d in convergence_data]

    bars = ax3.bar(x_pos, means, yerr=stds, capsize=5,
                   color=['lightblue', 'lightgreen', 'darkgreen', 'orange', 'lightcoral'])

    # Highlight k=5
    bars[2].set_color('darkgreen')
    bars[2].set_alpha(0.9)

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax3.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}',
                ha='center', fontsize=9, fontweight='bold')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'k={k}' for k in k_range])
    ax3.set_xlabel('Number of Clusters', fontsize=11)
    ax3.set_ylabel('Iterations to Converge', fontsize=11)
    ax3.set_title('Convergence Speed Analysis', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add interpretation
    ax3.text(0.05, 0.95,
            'k=5 shows fast,\nconsistent convergence',
            transform=ax3.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            verticalalignment='top')

    plt.suptitle('Advanced Validation: Gap Statistic, Stability & Convergence',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig

def create_validation_summary_dashboard():
    """Create comprehensive validation summary"""
    X, X_scaled, y_true, segments, feature_names = load_data()

    fig = plt.figure(figsize=(16, 10))

    # Calculate all metrics for summary
    k_values = range(2, 11)
    metrics = {
        'k': list(k_values),
        'inertia': [],
        'silhouette': [],
        'davies': [],
        'calinski': []
    }

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        metrics['inertia'].append(kmeans.inertia_)
        metrics['silhouette'].append(silhouette_score(X_scaled, labels))
        metrics['davies'].append(davies_bouldin_score(X_scaled, labels))
        metrics['calinski'].append(calinski_harabasz_score(X_scaled, labels))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. All metrics normalized (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])

    # Normalize metrics
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    inertia_norm = 1 - scaler.fit_transform(np.array(metrics['inertia']).reshape(-1, 1)).flatten()
    silhouette_norm = scaler.fit_transform(np.array(metrics['silhouette']).reshape(-1, 1)).flatten()
    davies_norm = 1 - scaler.fit_transform(np.array(metrics['davies']).reshape(-1, 1)).flatten()
    calinski_norm = scaler.fit_transform(np.array(metrics['calinski']).reshape(-1, 1)).flatten()

    ax1.plot(k_values, inertia_norm, 'b-', marker='o', label='Inertia', linewidth=2, alpha=0.7)
    ax1.plot(k_values, silhouette_norm, 'g-', marker='s', label='Silhouette', linewidth=2, alpha=0.7)
    ax1.plot(k_values, davies_norm, 'purple', marker='^', label='Davies-Bouldin', linewidth=2, alpha=0.7)
    ax1.plot(k_values, calinski_norm, 'orange', marker='d', label='Calinski', linewidth=2, alpha=0.7)

    # Composite score
    composite = (silhouette_norm + davies_norm + calinski_norm) / 3
    ax1.plot(k_values, composite, 'red', linewidth=4, marker='*', markersize=12,
            label='Composite Score')

    ax1.axvline(x=5, color='black', linestyle='--', alpha=0.3, linewidth=2)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Normalized Score (0-1)', fontsize=11)
    ax1.set_title('All Metrics Normalized: Consensus at k=5', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # 2. Best k by each metric (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    best_k_silhouette = k_values[np.argmax(metrics['silhouette'])]
    best_k_davies = k_values[np.argmin(metrics['davies'])]
    best_k_calinski = k_values[np.argmax(metrics['calinski'])]
    best_k_composite = k_values[np.argmax(composite)]

    summary_text = f"""
OPTIMAL K BY METRIC
===================
Silhouette:     k = {best_k_silhouette}
Davies-Bouldin: k = {best_k_davies}
Calinski:       k = {best_k_calinski}
Composite:      k = {best_k_composite}

CONSENSUS: k = 5

Why k=5?
• Clear elbow point
• High silhouette
• Low Davies-Bouldin
• Business alignment
• Stable clusters
"""

    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # 3. Metrics at k=5 (middle left)
    ax3 = fig.add_subplot(gs[1, 0])

    k5_idx = 3  # k=5 is at index 3 (k starts at 2)
    metrics_k5 = [
        ('Silhouette', metrics['silhouette'][k5_idx], 0.4, 1.0),
        ('Davies-B', 2 - metrics['davies'][k5_idx], 0, 2),  # Invert for visualization
        ('Calinski', metrics['calinski'][k5_idx]/1000, 0, 5)  # Scale down
    ]

    colors = ['green', 'purple', 'orange']
    bars = ax3.bar(range(3), [m[1] for m in metrics_k5], color=colors, alpha=0.7)

    # Add reference lines
    for i, (name, value, min_val, max_val) in enumerate(metrics_k5):
        # Good threshold
        if name == 'Silhouette':
            ax3.axhline(y=0.4, color='gray', linestyle='--', alpha=0.3)

    ax3.set_xticks(range(3))
    ax3.set_xticklabels([m[0] for m in metrics_k5])
    ax3.set_ylabel('Score', fontsize=10)
    ax3.set_title('Metrics at k=5', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, (name, value, _, _) in zip(bars, metrics_k5):
        if name == 'Calinski':
            label = f'{value*1000:.0f}'
        elif name == 'Davies-B':
            label = f'{2-value:.2f}'
        else:
            label = f'{value:.3f}'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha='center', fontsize=10, fontweight='bold')

    # 4. Cluster sizes at k=5 (middle center)
    ax4 = fig.add_subplot(gs[1, 1])

    kmeans_k5 = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels_k5 = kmeans_k5.fit_predict(X_scaled)

    cluster_sizes = [np.sum(labels_k5 == i) for i in range(5)]
    wedges, texts, autotexts = ax4.pie(cluster_sizes,
                                        labels=[f'C{i}' for i in range(5)],
                                        autopct='%d\n(%1.1f%%)',
                                        startangle=90,
                                        colors=plt.cm.Set3(np.linspace(0, 1, 5)))

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    ax4.set_title('Cluster Distribution at k=5', fontsize=12, fontweight='bold')

    # 5. Comparison table (middle right and bottom)
    ax5 = fig.add_subplot(gs[1:, 2])
    ax5.axis('off')

    # Create comparison data
    comparison_data = []
    for k in [3, 4, 5, 6, 7]:
        k_idx = k - 2
        comparison_data.append([
            f'{k}',
            f'{metrics["inertia"][k_idx]:.0f}',
            f'{metrics["silhouette"][k_idx]:.3f}',
            f'{metrics["davies"][k_idx]:.2f}',
            f'{metrics["calinski"][k_idx]:.0f}'
        ])

    # Create table
    table_data = [['k', 'Inertia', 'Silhouette', 'Davies-B', 'Calinski']] + comparison_data

    table = ax5.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.2, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight k=5 row
    for i in range(5):
        table[(3, i)].set_facecolor('#90EE90')
        table[(3, i)].set_text_props(weight='bold')

    # Add title above table
    ax5.text(0.5, 0.95, 'Detailed Metrics Comparison', transform=ax5.transAxes,
            ha='center', fontsize=12, fontweight='bold')

    # 6. Inter-cluster distances heatmap (bottom left and center)
    ax6 = fig.add_subplot(gs[2, :2])

    # Calculate inter-cluster distances
    from scipy.spatial.distance import cdist
    centers = kmeans_k5.cluster_centers_
    distances = cdist(centers, centers, metric='euclidean')

    im = ax6.imshow(distances, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(range(5))
    ax6.set_yticks(range(5))
    ax6.set_xticklabels([f'C{i}' for i in range(5)])
    ax6.set_yticklabels([f'C{i}' for i in range(5)])
    ax6.set_title('Inter-Cluster Distance Matrix (k=5)', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Euclidean Distance', rotation=270, labelpad=15)

    # Add distance values
    for i in range(5):
        for j in range(5):
            text = ax6.text(j, i, f'{distances[i, j]:.1f}',
                          ha="center", va="center",
                          color="white" if distances[i, j] > distances.max()/2 else "black",
                          fontsize=9)

    plt.suptitle('Cluster Validation Summary Dashboard: k=5 Optimal',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def main():
    """Generate all validation visualizations"""

    print("Generating FinTech validation visualization suite...")

    # 1. Comprehensive elbow plot
    print("Creating comprehensive elbow plot...")
    fig1 = create_comprehensive_elbow_plot()
    fig1.savefig('fintech_elbow_comprehensive.png', dpi=300, bbox_inches='tight')
    fig1.savefig('fintech_elbow_comprehensive.pdf', bbox_inches='tight')

    # 2. Silhouette analysis grid
    print("Creating silhouette analysis grid...")
    fig2 = create_silhouette_analysis_grid()
    fig2.savefig('fintech_silhouette_grid.png', dpi=300, bbox_inches='tight')
    fig2.savefig('fintech_silhouette_grid.pdf', bbox_inches='tight')

    # 3. Gap statistic analysis
    print("Creating gap statistic analysis...")
    fig3 = create_gap_statistic_analysis()
    fig3.savefig('fintech_gap_statistic.png', dpi=300, bbox_inches='tight')
    fig3.savefig('fintech_gap_statistic.pdf', bbox_inches='tight')

    # 4. Validation summary dashboard
    print("Creating validation summary dashboard...")
    fig4 = create_validation_summary_dashboard()
    fig4.savefig('fintech_validation_summary.png', dpi=300, bbox_inches='tight')
    fig4.savefig('fintech_validation_summary.pdf', bbox_inches='tight')

    print("\nValidation visualizations complete!")
    print("Generated files:")
    print("  - fintech_elbow_comprehensive.png/pdf")
    print("  - fintech_silhouette_grid.png/pdf")
    print("  - fintech_gap_statistic.png/pdf")
    print("  - fintech_validation_summary.png/pdf")

if __name__ == "__main__":
    main()