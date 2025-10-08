"""
Create descriptive analysis visualizations for the FinTech dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def create_descriptive_statistics_dashboard():
    """Create comprehensive descriptive statistics visualization"""

    # Load dataset
    df = pd.read_csv('fintech_user_behavior_full.csv')

    # Select numeric columns
    numeric_cols = [col for col in df.columns if col not in ['true_segment', 'true_label']]

    fig = plt.figure(figsize=(20, 14))

    # 1. Distribution plots for all features
    for idx, col in enumerate(numeric_cols[:12]):
        ax = plt.subplot(4, 4, idx + 1)

        # Histogram with KDE
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)

        # Add KDE
        kde_data = np.linspace(data.min(), data.max(), 100)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_data, kde(kde_data), 'r-', linewidth=2, label='KDE')

        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')

        ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_ylabel('Density', fontsize=8)

        # Add skewness annotation
        skew = data.skew()
        ax.text(0.95, 0.95, f'Skew: {skew:.2f}', transform=ax.transAxes,
               fontsize=8, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    # 2. Correlation heatmap (bottom left)
    ax_corr = plt.subplot(4, 4, (13, 16))

    corr_matrix = df[numeric_cols[:12]].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax_corr)

    ax_corr.set_title('Feature Correlations', fontsize=12, fontweight='bold')

    plt.suptitle('FinTech Dataset: Descriptive Statistics Dashboard (10,000 Simulated Users)',
                fontsize=16, fontweight='bold')

    # Add note about simulated data
    plt.figtext(0.5, 0.01,
               'Note: This is SIMULATED data generated for educational purposes',
               ha='center', fontsize=10, style='italic', color='red')

    plt.tight_layout()
    return fig

def create_segment_statistics_comparison():
    """Create segment-wise statistical comparison"""

    df = pd.read_csv('fintech_user_behavior_full.csv')

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Key features to analyze
    features = [
        'transaction_frequency',
        'transaction_volume',
        'savings_behavior',
        'credit_utilization',
        'international_activity',
        'support_contacts'
    ]

    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]

        # Box plot by segment
        segment_data = []
        labels = []

        for segment in df['true_segment'].unique():
            if segment not in ['noise']:  # Exclude noise
                segment_data.append(df[df['true_segment'] == segment][feature].dropna())
                labels.append(segment.replace('_', ' ').title()[:15])

        bp = ax.boxplot(segment_data, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f'{feature.replace("_", " ").title()} by Segment',
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add mean values as text
        for i, data in enumerate(segment_data):
            mean_val = np.mean(data)
            ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.1f}',
                   ha='center', fontsize=8, fontweight='bold')

    plt.suptitle('Segment-wise Feature Distribution Analysis (Simulated FinTech Data)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_data_quality_report():
    """Create data quality and summary statistics report"""

    df = pd.read_csv('fintech_user_behavior_full.csv')

    fig = plt.figure(figsize=(16, 10))

    # 1. Missing data heatmap
    ax1 = plt.subplot(2, 3, 1)

    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) > 0:
        ax1.barh(range(len(missing_pct)), missing_pct.values, color='coral')
        ax1.set_yticks(range(len(missing_pct)))
        ax1.set_yticklabels(missing_pct.index)
        ax1.set_xlabel('Missing %')
        ax1.set_title('Missing Data Analysis', fontweight='bold')

        # Add values
        for i, v in enumerate(missing_pct.values):
            ax1.text(v + 0.01, i, f'{v:.2f}%', va='center', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No missing data in numeric features',
                ha='center', va='center', fontsize=12)
        ax1.set_title('Missing Data Analysis', fontweight='bold')

    # 2. Summary statistics table
    ax2 = plt.subplot(2, 3, (2, 3))
    ax2.axis('off')

    # Calculate summary stats - only for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    summary_stats = numeric_df.describe().T
    summary_stats['CV'] = (summary_stats['std'] / summary_stats['mean']).abs()  # Coefficient of variation

    # Select key columns
    table_data = summary_stats[['mean', 'std', 'min', 'max', 'CV']].round(2)

    # Create table
    table = ax2.table(cellText=table_data.values,
                     rowLabels=[col.replace('_', ' ').title()[:20] for col in table_data.index],
                     colLabels=['Mean', 'Std Dev', 'Min', 'Max', 'CV'],
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Summary Statistics Table', fontsize=12, fontweight='bold', pad=20)

    # 3. Segment distribution pie chart
    ax3 = plt.subplot(2, 3, 4)

    segment_counts = df['true_segment'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))

    wedges, texts, autotexts = ax3.pie(segment_counts.values,
                                        labels=[s.replace('_', ' ').title() for s in segment_counts.index],
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        startangle=90)

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax3.set_title('Segment Distribution', fontsize=11, fontweight='bold')

    # 4. Feature importance based on variance
    ax4 = plt.subplot(2, 3, 5)

    # Calculate normalized variance (CV) - only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_importance = (df[numeric_cols].std() / df[numeric_cols].mean()).abs()
    feature_importance = feature_importance.sort_values(ascending=True)[-10:]

    ax4.barh(range(len(feature_importance)), feature_importance.values, color='steelblue')
    ax4.set_yticks(range(len(feature_importance)))
    ax4.set_yticklabels([f.replace('_', ' ').title()[:20] for f in feature_importance.index], fontsize=9)
    ax4.set_xlabel('Coefficient of Variation')
    ax4.set_title('Feature Variability (Top 10)', fontsize=11, fontweight='bold')

    # 5. Sample size and data info
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')

    info_text = f"""
    DATASET INFORMATION
    ===================

    Total Samples: {len(df):,}
    Features: {len(df.columns) - 2}
    Segments: {df['true_segment'].nunique()}

    Data Type: SIMULATED
    Purpose: Educational

    Segment Breakdown:
    ------------------
    """

    for segment, count in segment_counts.items():
        info_text += f"\n    {segment.replace('_', ' ').title()}: {count:,} ({count/len(df)*100:.1f}%)"

    info_text += f"""

    Missing Data: {df.isnull().sum().sum()} values
    ({df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}% of total)

    Note: This dataset was synthetically
    generated to demonstrate clustering
    techniques for FinTech applications.
    """

    ax5.text(0.1, 0.9, info_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('FinTech Dataset Quality Report (Simulated Data)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def main():
    """Generate descriptive analysis visualizations"""

    print("Creating descriptive analysis visualizations...")

    # 1. Descriptive statistics dashboard
    print("  Creating descriptive statistics dashboard...")
    fig1 = create_descriptive_statistics_dashboard()
    fig1.savefig('fintech_descriptive_statistics.png', dpi=300, bbox_inches='tight')
    fig1.savefig('fintech_descriptive_statistics.pdf', bbox_inches='tight')

    # 2. Segment statistics comparison
    print("  Creating segment statistics comparison...")
    fig2 = create_segment_statistics_comparison()
    fig2.savefig('fintech_segment_statistics.png', dpi=300, bbox_inches='tight')
    fig2.savefig('fintech_segment_statistics.pdf', bbox_inches='tight')

    # 3. Data quality report
    print("  Creating data quality report...")
    fig3 = create_data_quality_report()
    fig3.savefig('fintech_data_quality.png', dpi=300, bbox_inches='tight')
    fig3.savefig('fintech_data_quality.pdf', bbox_inches='tight')

    print("\nDescriptive analysis complete!")
    print("Generated files:")
    print("  - fintech_descriptive_statistics.png/pdf")
    print("  - fintech_segment_statistics.png/pdf")
    print("  - fintech_data_quality.png/pdf")

if __name__ == "__main__":
    main()