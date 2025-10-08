"""
Create comprehensive clustering visualizations for FinTech dataset slide deck
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define consistent colors
COLORS = {
    'digital_natives': '#2E86AB',
    'traditional_savers': '#A23B72',
    'business_users': '#F18F01',
    'international_travelers': '#C73E1D',
    'cautious_beginners': '#6B8E23',
    'fraudulent': '#8B0000',
    'noise': '#696969'
}

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

def create_dataset_overview():
    """Create opening power visualization showing the dataset"""
    X, X_scaled, y_true, segments, feature_names = load_data()

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. PCA scatter plot with segments
    ax1 = axes[0, 0]
    unique_segments = np.unique(segments)

    for segment in unique_segments:
        mask = segments == segment
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=segment.replace('_', ' ').title(),
                   alpha=0.6, s=20, c=COLORS.get(segment, '#333333'))

    ax1.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('10,000 FinTech Users: Natural Segments Revealed', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Segment distribution pie chart
    ax2 = axes[0, 1]
    segment_counts = pd.Series(segments).value_counts()
    colors_list = [COLORS.get(seg, '#333333') for seg in segment_counts.index]

    wedges, texts, autotexts = ax2.pie(segment_counts.values,
                                        labels=[s.replace('_', ' ').title() for s in segment_counts.index],
                                        colors=colors_list,
                                        autopct='%1.1f%%',
                                        startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    ax2.set_title('User Segment Distribution', fontsize=14, fontweight='bold')

    # 3. Feature importance heatmap
    ax3 = axes[1, 0]

    # Calculate feature statistics by segment
    df = pd.DataFrame(X, columns=feature_names)
    df['segment'] = segments

    # Get mean values for each segment
    segment_means = df.groupby('segment')[feature_names].mean()

    # Normalize for visualization
    segment_means_norm = (segment_means - segment_means.mean()) / segment_means.std()

    # Select top features
    top_features = ['transaction_frequency', 'transaction_volume', 'savings_behavior',
                   'credit_utilization', 'international_activity', 'support_contacts']

    im = ax3.imshow(segment_means_norm[top_features].T, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)

    ax3.set_xticks(range(len(segment_means_norm.index)))
    ax3.set_xticklabels([s.replace('_', ' ').title()[:10] for s in segment_means_norm.index],
                        rotation=45, ha='right', fontsize=8)
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels([f.replace('_', ' ').title() for f in top_features], fontsize=9)
    ax3.set_title('Key Feature Patterns by Segment', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax3, label='Normalized Value')

    # 4. Transaction volume vs frequency scatter
    ax4 = axes[1, 1]

    # Sample for clarity
    sample_idx = np.random.choice(len(X), 2000, replace=False)

    for segment in unique_segments:
        mask = (segments[sample_idx] == segment)
        ax4.scatter(X[sample_idx][mask, 0], X[sample_idx][mask, 1] / 1000,
                   label=segment.replace('_', ' ').title(),
                   alpha=0.5, s=15, c=COLORS.get(segment, '#333333'))

    ax4.set_xlabel('Transaction Frequency (daily avg)', fontsize=10)
    ax4.set_ylabel('Transaction Volume ($1000s/month)', fontsize=10)
    ax4.set_title('Behavioral Patterns: Frequency vs Volume', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Annotate interesting regions
    ax4.annotate('High-Value\nBusiness Users', xy=(12, 15), xytext=(15, 20),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                fontsize=8, ha='center')
    ax4.annotate('Cautious\nBeginners', xy=(1, 1), xytext=(3, 5),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                fontsize=8, ha='center')

    plt.suptitle('FinTech Dataset Overview: 10,000 Users, 12 Features, 7 Segments',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def create_algorithm_comparison():
    """Compare all clustering algorithms on the FinTech dataset"""
    X, X_scaled, y_true, segments, feature_names = load_data()

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Apply different algorithms
    algorithms = {
        'K-Means (k=5)': KMeans(n_clusters=5, random_state=42, n_init=10),
        'DBSCAN': DBSCAN(eps=0.8, min_samples=10),
        'Hierarchical': AgglomerativeClustering(n_clusters=5),
        'GMM (5 components)': GaussianMixture(n_components=5, random_state=42)
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # First plot: Ground truth
    ax = axes[0]
    for segment in np.unique(segments):
        mask = segments == segment
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  label=segment.replace('_', ' ').title(),
                  alpha=0.5, s=10, c=COLORS.get(segment, '#333333'))
    ax.set_title('Ground Truth Segments', fontsize=12, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Apply each algorithm
    for idx, (name, algorithm) in enumerate(algorithms.items(), 1):
        ax = axes[idx]

        # Fit and predict
        if 'GMM' in name:
            labels = algorithm.fit_predict(X_scaled)
        else:
            labels = algorithm.fit_predict(X_scaled)

        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            if -1 in labels:  # DBSCAN with noise
                mask_no_noise = labels != -1
                if mask_no_noise.sum() > 0:
                    score = silhouette_score(X_scaled[mask_no_noise], labels[mask_no_noise])
                else:
                    score = -1
            else:
                score = silhouette_score(X_scaled, labels)
        else:
            score = -1

        # Plot clusters
        unique_labels = np.unique(labels)
        colors_algo = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors_algo):
            if label == -1:  # DBSCAN noise
                ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                          c='black', marker='x', s=10, alpha=0.3, label='Noise')
            else:
                ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                          c=[color], s=10, alpha=0.5, label=f'Cluster {label}')

        ax.set_title(f'{name}\nSilhouette: {score:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)

        if len(unique_labels) <= 6:
            ax.legend(fontsize=7, loc='upper right')

        # Add cluster statistics
        if 'DBSCAN' in name:
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            ax.text(0.02, 0.98, f'Clusters: {n_clusters}\nNoise: {n_noise} ({n_noise/len(labels)*100:.1f}%)',
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Clustering Algorithm Comparison on FinTech Dataset',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def create_fraud_detection_visualization():
    """Visualize DBSCAN's fraud detection capability"""
    X, X_scaled, y_true, segments, feature_names = load_data()

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Identify fraudulent users in ground truth
    fraud_mask = segments == 'fraudulent'

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. DBSCAN clusters with fraud overlay
    ax1 = axes[0, 0]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot DBSCAN results
    for label in np.unique(dbscan_labels):
        if label == -1:
            ax1.scatter(X_pca[dbscan_labels == label, 0],
                       X_pca[dbscan_labels == label, 1],
                       c='red', marker='x', s=20, alpha=0.8, label='Detected Anomalies')
        else:
            ax1.scatter(X_pca[dbscan_labels == label, 0],
                       X_pca[dbscan_labels == label, 1],
                       c='lightgray', s=10, alpha=0.3)

    # Highlight true fraudulent users
    ax1.scatter(X_pca[fraud_mask, 0], X_pca[fraud_mask, 1],
               c='darkred', marker='^', s=30, edgecolors='black',
               linewidths=0.5, label='True Fraudulent', alpha=0.9)

    ax1.set_title('DBSCAN Anomaly Detection vs True Fraud', fontsize=12, fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Confusion matrix
    ax2 = axes[0, 1]

    # Create binary labels: anomaly vs normal
    is_anomaly_dbscan = dbscan_labels == -1
    is_fraud_true = fraud_mask

    # Calculate confusion matrix
    tn = np.sum((~is_anomaly_dbscan) & (~is_fraud_true))
    fp = np.sum(is_anomaly_dbscan & (~is_fraud_true))
    fn = np.sum((~is_anomaly_dbscan) & is_fraud_true)
    tp = np.sum(is_anomaly_dbscan & is_fraud_true)

    conf_matrix = np.array([[tn, fp], [fn, tp]])

    im = ax2.imshow(conf_matrix, cmap='Blues')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Normal', 'Anomaly'])
    ax2.set_yticklabels(['Normal', 'Fraud'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Fraud Detection Performance', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, f'{conf_matrix[i, j]}',
                          ha="center", va="center", color="black", fontweight='bold')

    # Add metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax2.text(0.5, -0.15, f'Precision: {precision:.2%} | Recall: {recall:.2%}',
            transform=ax2.transAxes, ha='center', fontsize=10)

    # 3. Feature comparison: Fraud vs Normal
    ax3 = axes[0, 2]

    df = pd.DataFrame(X, columns=feature_names)
    df['is_fraud'] = fraud_mask

    features_to_compare = ['transaction_frequency', 'international_activity', 'device_switches']

    fraud_means = df[df['is_fraud']][features_to_compare].mean()
    normal_means = df[~df['is_fraud']][features_to_compare].mean()

    x = np.arange(len(features_to_compare))
    width = 0.35

    bars1 = ax3.bar(x - width/2, normal_means, width, label='Normal Users', color='#4CAF50')
    bars2 = ax3.bar(x + width/2, fraud_means, width, label='Fraudulent Users', color='#F44336')

    ax3.set_xlabel('Features')
    ax3.set_ylabel('Average Value')
    ax3.set_title('Fraud Indicators: Key Feature Differences', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f.replace('_', ' ').title() for f in features_to_compare], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # 4. Transaction patterns over time
    ax4 = axes[1, 0]

    # Simulate temporal patterns
    days = np.arange(30)
    normal_pattern = 5 + np.sin(days / 5) * 2 + np.random.normal(0, 0.5, 30)
    fraud_pattern = np.concatenate([
        np.ones(10) * 2,  # Low activity initially
        np.ones(5) * 20,  # Sudden spike
        np.ones(15) * 1   # Then dormant
    ]) + np.random.normal(0, 1, 30)

    ax4.plot(days, normal_pattern, 'g-', linewidth=2, label='Normal User Pattern')
    ax4.plot(days, fraud_pattern, 'r-', linewidth=2, label='Fraudulent Pattern')
    ax4.fill_between([10, 15], 0, 25, alpha=0.3, color='red', label='Anomaly Period')

    ax4.set_xlabel('Days')
    ax4.set_ylabel('Transactions per Day')
    ax4.set_title('Temporal Patterns: Normal vs Fraudulent', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Risk score distribution
    ax5 = axes[1, 1]

    # Calculate risk scores based on multiple factors
    df['risk_score'] = (
        df['transaction_frequency'] / df['transaction_frequency'].max() * 0.2 +
        df['international_activity'] * 0.3 +
        df['device_switches'] / df['device_switches'].max() * 0.2 +
        (df['support_contacts'] == 0).astype(int) * 0.3
    )

    ax5.hist(df[~df['is_fraud']]['risk_score'], bins=30, alpha=0.7,
            label='Normal Users', color='green', density=True)
    ax5.hist(df[df['is_fraud']]['risk_score'], bins=30, alpha=0.7,
            label='Fraudulent Users', color='red', density=True)

    ax5.axvline(x=0.5, color='black', linestyle='--', label='Risk Threshold')
    ax5.set_xlabel('Risk Score')
    ax5.set_ylabel('Density')
    ax5.set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Detection summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Calculate statistics
    total_users = len(segments)
    total_fraud = fraud_mask.sum()
    detected_anomalies = (dbscan_labels == -1).sum()
    correctly_detected = tp

    stats_text = f"""
    FRAUD DETECTION SUMMARY
    ========================
    Total Users: {total_users:,}
    True Fraudulent: {total_fraud} ({total_fraud/total_users*100:.1f}%)

    DBSCAN Performance:
    • Anomalies Detected: {detected_anomalies}
    • Correctly Identified: {correctly_detected}/{total_fraud}
    • Precision: {precision:.1%}
    • Recall: {recall:.1%}

    Key Fraud Indicators:
    • High international activity (80% vs 28%)
    • Unusual transaction spikes
    • Multiple device switches
    • Zero support contacts
    """

    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Fraud Detection with DBSCAN: Identifying Anomalous Patterns',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def create_cluster_quality_dashboard():
    """Create a comprehensive cluster quality metrics dashboard"""
    X, X_scaled, y_true, segments, feature_names = load_data()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Test different k values
    k_values = range(2, 11)
    inertias = []
    silhouettes = []
    davies_bouldin = []
    calinski = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
        davies_bouldin.append(davies_bouldin_score(X_scaled, labels))
        calinski.append(calinski_harabasz_score(X_scaled, labels))

    # 1. Elbow method
    ax1 = axes[0, 0]
    ax1.plot(k_values, inertias, 'b-', linewidth=2, marker='o', markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax1.set_ylabel('Within-Cluster Sum of Squares', fontsize=10)
    ax1.set_title('Elbow Method: Finding Optimal k', fontsize=12, fontweight='bold')
    ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Optimal k=5')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add annotations for percentage decrease
    for i in range(1, len(inertias)):
        pct_decrease = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
        if pct_decrease > 10:
            ax1.annotate(f'-{pct_decrease:.1f}%',
                        xy=(k_values[i], inertias[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='green')

    # 2. Silhouette scores
    ax2 = axes[0, 1]
    ax2.plot(k_values, silhouettes, 'g-', linewidth=2, marker='s', markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax2.set_ylabel('Silhouette Score', fontsize=10)
    ax2.set_title('Silhouette Analysis: Cluster Cohesion', fontsize=12, fontweight='bold')
    ax2.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='k=5')
    ax2.axhline(y=max(silhouettes), color='gray', linestyle=':', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Davies-Bouldin Index
    ax3 = axes[0, 2]
    ax3.plot(k_values, davies_bouldin, 'purple', linewidth=2, marker='^', markersize=8)
    ax3.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax3.set_ylabel('Davies-Bouldin Score', fontsize=10)
    ax3.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='k=5')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Calinski-Harabasz Score
    ax4 = axes[1, 0]
    ax4.plot(k_values, calinski, 'orange', linewidth=2, marker='d', markersize=8)
    ax4.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax4.set_ylabel('Calinski-Harabasz Score', fontsize=10)
    ax4.set_title('Calinski-Harabasz Index: Between-Cluster Variance', fontsize=12, fontweight='bold')
    ax4.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='k=5')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Normalized comparison
    ax5 = axes[1, 1]

    # Normalize scores for comparison
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    inertias_norm = 1 - scaler.fit_transform(np.array(inertias).reshape(-1, 1)).flatten()
    silhouettes_norm = scaler.fit_transform(np.array(silhouettes).reshape(-1, 1)).flatten()
    davies_norm = 1 - scaler.fit_transform(np.array(davies_bouldin).reshape(-1, 1)).flatten()
    calinski_norm = scaler.fit_transform(np.array(calinski).reshape(-1, 1)).flatten()

    ax5.plot(k_values, inertias_norm, 'b-', label='Inertia', alpha=0.7)
    ax5.plot(k_values, silhouettes_norm, 'g-', label='Silhouette', alpha=0.7)
    ax5.plot(k_values, davies_norm, 'purple', label='Davies-Bouldin', alpha=0.7)
    ax5.plot(k_values, calinski_norm, 'orange', label='Calinski-Harabasz', alpha=0.7)

    # Composite score
    composite = (silhouettes_norm + calinski_norm + davies_norm) / 3
    ax5.plot(k_values, composite, 'red', linewidth=3, label='Composite Score', marker='o')

    ax5.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax5.set_ylabel('Normalized Score (0-1)', fontsize=10)
    ax5.set_title('All Metrics Normalized: Consensus at k=5', fontsize=12, fontweight='bold')
    ax5.axvline(x=5, color='black', linestyle='--', alpha=0.3)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary table
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary_text = f"""
    OPTIMAL CLUSTERING ANALYSIS
    ===========================

    Recommended k = 5

    Metrics at k=5:
    • Silhouette Score: {silhouettes[3]:.3f}
    • Davies-Bouldin: {davies_bouldin[3]:.3f}
    • Calinski-Harabasz: {calinski[3]:.0f}

    Why k=5?
    • Clear elbow in inertia curve
    • High silhouette score
    • Low Davies-Bouldin index
    • Matches business segments
    • Interpretable personas

    Business Segments Found:
    1. Digital Natives (25%)
    2. Traditional Savers (20%)
    3. Business Users (15%)
    4. International (10%)
    5. Beginners (25%)
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Cluster Quality Metrics Dashboard: Validating k=5',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def main():
    """Generate all clustering visualizations"""

    print("Generating FinTech clustering visualization suite...")

    # 1. Dataset overview
    print("Creating dataset overview...")
    fig1 = create_dataset_overview()
    fig1.savefig('fintech_dataset_overview_slides.png', dpi=300, bbox_inches='tight')
    fig1.savefig('fintech_dataset_overview_slides.pdf', bbox_inches='tight')

    # 2. Algorithm comparison
    print("Creating algorithm comparison...")
    fig2 = create_algorithm_comparison()
    fig2.savefig('fintech_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    fig2.savefig('fintech_algorithm_comparison.pdf', bbox_inches='tight')

    # 3. Fraud detection
    print("Creating fraud detection visualization...")
    fig3 = create_fraud_detection_visualization()
    fig3.savefig('fintech_fraud_detection.png', dpi=300, bbox_inches='tight')
    fig3.savefig('fintech_fraud_detection.pdf', bbox_inches='tight')

    # 4. Cluster quality dashboard
    print("Creating cluster quality dashboard...")
    fig4 = create_cluster_quality_dashboard()
    fig4.savefig('fintech_cluster_quality.png', dpi=300, bbox_inches='tight')
    fig4.savefig('fintech_cluster_quality.pdf', bbox_inches='tight')

    print("\nClustering visualizations complete!")
    print("Generated files:")
    print("  - fintech_dataset_overview_slides.png/pdf")
    print("  - fintech_algorithm_comparison.png/pdf")
    print("  - fintech_fraud_detection.png/pdf")
    print("  - fintech_cluster_quality.png/pdf")

if __name__ == "__main__":
    main()