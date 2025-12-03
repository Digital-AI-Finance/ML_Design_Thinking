"""
Week 2: FinTech User Behavior Dataset Generator
MSc-level clustering challenges with real-world financial patterns
Author: ML Design Thinking Course
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

np.random.seed(42)

class FinTechDataGenerator:
    """Generate realistic FinTech user behavior data with built-in clustering challenges"""

    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.feature_names = [
            'transaction_frequency',  # Daily avg transactions
            'transaction_volume',     # Monthly total USD
            'session_duration',       # Avg app usage minutes
            'payment_diversity',      # Different payment types (1-10)
            'international_activity', # Cross-border ratio (0-1)
            'savings_behavior',       # Deposit frequency score (0-100)
            'credit_utilization',     # Credit line usage % (0-100)
            'support_contacts',       # Monthly service interactions
            'device_switches',        # Multi-device score (0-10)
            'peak_hour_usage',        # Business hours ratio (0-1)
            'merchant_categories',    # Spending diversity (1-20)
            'account_age'            # Days since registration
        ]

    def generate_segment(self, n, segment_type):
        """Generate data for a specific user segment"""

        if segment_type == 'digital_natives':
            # Heavy users, tech-savvy, diverse usage
            data = {
                'transaction_frequency': np.random.gamma(5, 2, n),  # High frequency
                'transaction_volume': np.random.lognormal(7.5, 0.8, n),  # $1,000-$5,000
                'session_duration': np.random.gamma(4, 3, n) + 10,  # 15-30 min
                'payment_diversity': np.random.poisson(7, n) + 3,  # Many payment types
                'international_activity': np.random.beta(2, 5, n),  # Some international
                'savings_behavior': np.random.gamma(3, 10, n),  # Moderate saving
                'credit_utilization': np.random.beta(3, 2, n) * 100,  # Active credit use
                'support_contacts': np.random.poisson(0.5, n),  # Low support need
                'device_switches': np.random.poisson(3, n) + 2,  # Multiple devices
                'peak_hour_usage': np.random.beta(2, 2, n),  # Mixed hours
                'merchant_categories': np.random.poisson(8, n) + 5,  # Diverse spending
                'account_age': np.random.gamma(3, 100, n) + 180  # 6+ months
            }

        elif segment_type == 'traditional_savers':
            # High deposits, low transactions, conservative
            data = {
                'transaction_frequency': np.random.gamma(2, 1, n),  # Low frequency
                'transaction_volume': np.random.lognormal(8, 0.5, n),  # $2,000-$8,000
                'session_duration': np.random.gamma(2, 2, n) + 5,  # 5-15 min
                'payment_diversity': np.random.poisson(2, n) + 1,  # Few payment types
                'international_activity': np.random.beta(1, 10, n),  # Mostly domestic
                'savings_behavior': np.random.gamma(5, 15, n) + 40,  # High saving
                'credit_utilization': np.random.beta(1, 5, n) * 100,  # Low credit use
                'support_contacts': np.random.poisson(1, n),  # Occasional support
                'device_switches': np.random.poisson(1, n),  # Single device
                'peak_hour_usage': np.random.beta(5, 2, n),  # Business hours
                'merchant_categories': np.random.poisson(3, n) + 2,  # Limited categories
                'account_age': np.random.gamma(5, 150, n) + 365  # 1+ years
            }

        elif segment_type == 'business_users':
            # High volume, peak hours, B2B patterns
            data = {
                'transaction_frequency': np.random.gamma(8, 1.5, n),  # Very high frequency
                'transaction_volume': np.random.lognormal(9, 0.6, n),  # $5,000-$20,000
                'session_duration': np.random.gamma(5, 4, n) + 20,  # 20-40 min
                'payment_diversity': np.random.poisson(5, n) + 2,  # Business payment types
                'international_activity': np.random.beta(3, 3, n),  # Mixed international
                'savings_behavior': np.random.gamma(2, 8, n),  # Low personal saving
                'credit_utilization': np.random.beta(4, 2, n) * 100,  # High credit use
                'support_contacts': np.random.poisson(2, n),  # Regular support
                'device_switches': np.random.poisson(2, n) + 1,  # Desktop + mobile
                'peak_hour_usage': np.random.beta(8, 2, n),  # Strong business hours
                'merchant_categories': np.random.poisson(6, n) + 3,  # Business categories
                'account_age': np.random.gamma(4, 120, n) + 90  # 3+ months
            }

        elif segment_type == 'international_travelers':
            # Cross-border focus, irregular patterns
            data = {
                'transaction_frequency': np.random.gamma(4, 1.8, n),  # Moderate frequency
                'transaction_volume': np.random.lognormal(7.8, 1, n),  # $1,500-$6,000
                'session_duration': np.random.gamma(3, 3, n) + 8,  # 10-20 min
                'payment_diversity': np.random.poisson(6, n) + 2,  # Various currencies
                'international_activity': np.random.beta(5, 2, n),  # High international
                'savings_behavior': np.random.gamma(2, 12, n),  # Low saving
                'credit_utilization': np.random.beta(3, 3, n) * 100,  # Moderate credit
                'support_contacts': np.random.poisson(1.5, n),  # Some support (timezone)
                'device_switches': np.random.poisson(4, n) + 1,  # Multiple devices/locations
                'peak_hour_usage': np.random.beta(2, 3, n),  # Off-peak (timezones)
                'merchant_categories': np.random.poisson(10, n) + 5,  # Very diverse
                'account_age': np.random.gamma(3, 90, n) + 60  # 2+ months
            }

        elif segment_type == 'cautious_beginners':
            # New users, low activity, learning phase
            data = {
                'transaction_frequency': np.random.gamma(1.5, 0.8, n),  # Very low
                'transaction_volume': np.random.lognormal(6.5, 0.7, n),  # $300-$1,500
                'session_duration': np.random.gamma(2, 5, n) + 3,  # 5-15 min
                'payment_diversity': np.random.poisson(1, n) + 1,  # 1-2 types only
                'international_activity': np.random.beta(1, 20, n),  # Almost none
                'savings_behavior': np.random.gamma(1, 5, n),  # Testing waters
                'credit_utilization': np.random.beta(1, 10, n) * 100,  # Minimal credit
                'support_contacts': np.random.poisson(3, n),  # High support need
                'device_switches': np.maximum(np.random.poisson(0.5, n), 0),  # Single device
                'peak_hour_usage': np.random.beta(3, 3, n),  # Random times
                'merchant_categories': np.random.poisson(2, n) + 1,  # Few categories
                'account_age': np.random.gamma(2, 15, n)  # 0-2 months
            }

        elif segment_type == 'fraudulent':
            # Anomalous patterns for DBSCAN to detect
            data = {
                'transaction_frequency': np.concatenate([
                    np.random.gamma(15, 1, n//2),  # Sudden spike
                    np.random.gamma(1, 1, n//2)    # Then dormant
                ]),
                'transaction_volume': np.concatenate([
                    np.random.lognormal(10, 0.3, n//2),  # Very high volume
                    np.random.lognormal(5, 1, n//2)      # Small tests
                ]),
                'session_duration': np.random.exponential(2, n),  # Very short sessions
                'payment_diversity': np.random.randint(8, 10, n),  # Many types quickly
                'international_activity': np.random.beta(8, 2, n),  # Heavy international
                'savings_behavior': np.zeros(n),  # No savings
                'credit_utilization': np.random.choice([0, 95, 100], n),  # Extreme values
                'support_contacts': np.zeros(n),  # Avoid support
                'device_switches': np.random.poisson(8, n),  # Many devices
                'peak_hour_usage': np.random.uniform(0, 1, n),  # Random times
                'merchant_categories': np.random.randint(15, 20, n),  # Unusual variety
                'account_age': np.random.randint(0, 30, n)  # Very new accounts
            }

        else:  # noise
            # Random noise points
            data = {col: np.random.uniform(
                np.percentile(np.random.gamma(2, 2, 1000), 5),
                np.percentile(np.random.gamma(2, 2, 1000), 95),
                n
            ) for col in self.feature_names}

        # Add some realistic noise and constraints
        for col in data:
            data[col] = np.maximum(data[col], 0)  # No negative values
            if 'frequency' in col or 'diversity' in col or 'categories' in col:
                data[col] = np.round(data[col])
            if 'activity' in col or 'usage' in col or 'utilization' in col:
                data[col] = np.clip(data[col], 0, 100 if 'utilization' in col else 1)

        return pd.DataFrame(data)

    def add_temporal_correlation(self, df):
        """Add realistic temporal patterns"""
        # Older accounts tend to have higher transaction volumes
        df['transaction_volume'] *= (1 + 0.0005 * df['account_age'])

        # Support contacts decrease with account age
        df['support_contacts'] *= np.maximum(0.3, 1 - 0.001 * df['account_age'])

        # Credit utilization grows with trust
        df['credit_utilization'] *= np.minimum(1.5, 1 + 0.0003 * df['account_age'])

        return df

    def generate_full_dataset(self):
        """Generate complete dataset with all segments"""

        # Define segment sizes (must sum to n_samples)
        segments = {
            'digital_natives': int(0.25 * self.n_samples),      # 25%
            'traditional_savers': int(0.20 * self.n_samples),   # 20%
            'business_users': int(0.15 * self.n_samples),       # 15%
            'international_travelers': int(0.10 * self.n_samples), # 10%
            'cautious_beginners': int(0.25 * self.n_samples),   # 25%
            'fraudulent': int(0.03 * self.n_samples),           # 3%
            'noise': int(0.02 * self.n_samples)                 # 2%
        }

        # Generate each segment
        dfs = []
        labels = []

        for i, (segment, size) in enumerate(segments.items()):
            df_segment = self.generate_segment(size, segment)
            df_segment['true_segment'] = segment
            df_segment['true_label'] = i
            dfs.append(df_segment)
            labels.extend([i] * size)

        # Combine all segments
        df = pd.concat(dfs, ignore_index=True)

        # Shuffle the data
        shuffle_idx = np.random.permutation(len(df))
        df = df.iloc[shuffle_idx].reset_index(drop=True)

        # Add temporal correlations
        df = self.add_temporal_correlation(df)

        # Add some missing values (realistic)
        missing_prob = 0.02
        for col in ['international_activity', 'credit_utilization', 'device_switches']:
            missing_mask = np.random.random(len(df)) < missing_prob
            df.loc[missing_mask, col] = np.nan

        # Create feature matrix and labels
        X = df[self.feature_names].values
        y_true = df['true_label'].values
        segment_names = df['true_segment'].values

        return X, y_true, segment_names, df

    def get_feature_statistics(self, df):
        """Calculate feature statistics for interpretation"""
        stats = pd.DataFrame({
            'mean': df[self.feature_names].mean(),
            'std': df[self.feature_names].std(),
            'min': df[self.feature_names].min(),
            'max': df[self.feature_names].max(),
            'skewness': df[self.feature_names].skew(),
            'kurtosis': df[self.feature_names].kurtosis()
        })
        return stats

def visualize_dataset(X, y_true, segment_names, feature_names):
    """Create comprehensive visualization of the dataset"""

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. PCA visualization (2D)
    ax1 = plt.subplot(3, 3, 1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    # Handle NaN values before PCA
    X_clean = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_clean))

    unique_segments = np.unique(segment_names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_segments)))

    for segment, color in zip(unique_segments, colors):
        mask = segment_names == segment
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=segment.replace('_', ' ').title(),
                   alpha=0.6, s=10, c=[color])

    ax1.set_title('PCA Visualization of FinTech User Segments', fontsize=12, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # 2. Feature distributions by segment
    ax2 = plt.subplot(3, 3, 2)
    segment_counts = pd.Series(segment_names).value_counts()
    ax2.bar(range(len(segment_counts)), segment_counts.values)
    ax2.set_xticks(range(len(segment_counts)))
    ax2.set_xticklabels(segment_counts.index, rotation=45, ha='right', fontsize=8)
    ax2.set_title('Segment Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Users')

    # 3. Correlation heatmap
    ax3 = plt.subplot(3, 3, 3)
    correlation = pd.DataFrame(X, columns=feature_names).corr()
    im = ax3.imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(feature_names)))
    ax3.set_yticks(range(len(feature_names)))
    ax3.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
    ax3.set_yticklabels(feature_names, fontsize=7)
    ax3.set_title('Feature Correlations', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # 4-6. Key feature relationships
    for idx, (feat1, feat2) in enumerate([
        ('transaction_frequency', 'transaction_volume'),
        ('savings_behavior', 'credit_utilization'),
        ('account_age', 'support_contacts')
    ]):
        ax = plt.subplot(3, 3, 4 + idx)

        for segment, color in zip(unique_segments, colors):
            mask = segment_names == segment
            if feat1 in feature_names and feat2 in feature_names:
                i1 = feature_names.index(feat1)
                i2 = feature_names.index(feat2)
                ax.scatter(X[mask, i1], X[mask, i2],
                          alpha=0.5, s=5, c=[color], label=segment.replace('_', ' ').title())

        ax.set_xlabel(feat1.replace('_', ' ').title(), fontsize=9)
        ax.set_ylabel(feat2.replace('_', ' ').title(), fontsize=9)
        ax.set_title(f'{feat1} vs {feat2}'.replace('_', ' ').title(), fontsize=10)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    # 7-9. Feature statistics by segment
    for idx, feature in enumerate(['transaction_volume', 'savings_behavior', 'international_activity']):
        ax = plt.subplot(3, 3, 7 + idx)

        if feature in feature_names:
            feat_idx = feature_names.index(feature)

            data_by_segment = []
            labels_by_segment = []

            for segment in unique_segments:
                mask = segment_names == segment
                data_by_segment.append(X[mask, feat_idx])
                labels_by_segment.append(segment.replace('_', ' ').title())

            bp = ax.boxplot(data_by_segment, labels=labels_by_segment, patch_artist=True)

            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xticklabels(labels_by_segment, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=9)
            ax.set_title(f'{feature} by Segment'.replace('_', ' ').title(), fontsize=10)

    plt.suptitle('FinTech User Behavior Dataset Overview', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig

def save_dataset(X, y_true, segment_names, df, feature_names):
    """Save dataset in multiple formats"""

    # Save as CSV with all information
    df.to_csv('fintech_user_behavior_full.csv', index=False)

    # Save as NumPy arrays for direct sklearn usage
    np.save('fintech_X.npy', X)
    np.save('fintech_y_true.npy', y_true)
    np.save('fintech_segments.npy', segment_names)

    # Save feature names
    with open('fintech_features.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    # Save dataset description
    description = """
    FinTech User Behavior Dataset
    =============================

    Dataset for demonstrating advanced clustering techniques
    Designed for MSc-level machine learning education

    Size: 10,000 users
    Features: 12 behavioral and transactional metrics
    Segments: 7 (including fraudulent and noise)

    Segments:
    - Digital Natives (25%): Tech-savvy heavy users
    - Traditional Savers (20%): Conservative, high deposits
    - Business Users (15%): High volume, business hours
    - International Travelers (10%): Cross-border focus
    - Cautious Beginners (25%): New users, learning phase
    - Fraudulent (3%): Anomalous patterns for outlier detection
    - Noise (2%): Random patterns

    Features:
    1. transaction_frequency: Daily average transactions
    2. transaction_volume: Monthly total in USD
    3. session_duration: Average app usage in minutes
    4. payment_diversity: Number of different payment types (1-10)
    5. international_activity: Cross-border transaction ratio (0-1)
    6. savings_behavior: Deposit frequency score (0-100)
    7. credit_utilization: Credit line usage percentage (0-100)
    8. support_contacts: Monthly customer service interactions
    9. device_switches: Multi-device usage score (0-10)
    10. peak_hour_usage: Business hours activity ratio (0-1)
    11. merchant_categories: Spending category diversity (1-20)
    12. account_age: Days since registration

    Clustering Challenges:
    - Varying cluster densities
    - Overlapping segments (GMM suitable)
    - Outliers and anomalies (DBSCAN suitable)
    - Natural hierarchy (Hierarchical suitable)
    - Different cluster shapes
    - Missing values (2% in some features)
    - Temporal correlations
    - Skewed distributions

    Files:
    - fintech_user_behavior_full.csv: Complete dataset with labels
    - fintech_X.npy: Feature matrix (10000, 12)
    - fintech_y_true.npy: True segment labels
    - fintech_segments.npy: Segment names
    - fintech_features.txt: Feature names list
    """

    with open('fintech_dataset_description.txt', 'w') as f:
        f.write(description)

    print("Dataset saved successfully!")
    print(f"Shape: {X.shape}")
    print(f"Files created: CSV, NPY arrays, description")

def main():
    """Generate and save the complete FinTech dataset"""

    print("Generating FinTech User Behavior Dataset...")
    print("=" * 50)

    # Initialize generator
    generator = FinTechDataGenerator(n_samples=10000)

    # Generate dataset
    X, y_true, segment_names, df = generator.generate_full_dataset()

    print(f"Dataset generated: {X.shape[0]} users, {X.shape[1]} features")
    print(f"Segments: {len(np.unique(segment_names))} unique")
    print()

    # Get feature statistics
    stats = generator.get_feature_statistics(df)
    print("Feature Statistics:")
    print(stats.round(2))
    print()

    # Visualize dataset
    print("Creating visualizations...")
    fig = visualize_dataset(X, y_true, segment_names, generator.feature_names)
    fig.savefig('fintech_dataset_overview.png', dpi=300, bbox_inches='tight')
    fig.savefig('fintech_dataset_overview.pdf', bbox_inches='tight')
    print("Visualizations saved: PNG and PDF")

    # Save dataset
    print("\nSaving dataset files...")
    save_dataset(X, y_true, segment_names, df, generator.feature_names)

    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print("\nNext steps:")
    print("1. Run test_all_algorithms.py to apply clustering methods")
    print("2. Run validate_clusters.py for validation metrics")
    print("3. Run create_visualization_suite.py for all Week 2 charts")

if __name__ == "__main__":
    main()