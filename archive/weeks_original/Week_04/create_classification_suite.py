"""
Create main classification visualizations for Week 4 presentation.
Generates comprehensive charts demonstrating classification concepts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define consistent color palette
colors = {
    'failed': '#e74c3c',        # Red
    'struggling': '#f39c12',    # Orange
    'growing': '#3498db',       # Blue
    'breakthrough': '#27ae60',  # Green
    'tech': '#9b59b6',          # Purple
    'consumer': '#1abc9c',      # Turquoise
    'b2b': '#34495e',          # Dark gray
    'social': '#e67e22',        # Carrot
}

def load_data():
    """Load the innovation dataset."""
    df = pd.read_csv('innovation_products_full.csv')

    # Load numpy arrays
    X_train = np.load('innovation_X_train.npy')
    X_test = np.load('innovation_X_test.npy')
    y_train = np.load('innovation_y_train_binary.npy')
    y_test = np.load('innovation_y_test_binary.npy')
    y_train_multi = np.load('innovation_y_train_multi.npy')
    y_test_multi = np.load('innovation_y_test_multi.npy')
    feature_names = np.load('innovation_feature_names.npy', allow_pickle=True)

    return df, X_train, X_test, y_train, y_test, y_train_multi, y_test_multi, feature_names

def create_dataset_overview():
    """Create comprehensive dataset overview visualization."""

    df, _, _, _, _, _, _, _ = load_data()

    fig = plt.figure(figsize=(16, 10))

    # Title
    fig.suptitle('Product Innovation Dataset Overview\n10,000 Products Across 6 Segments',
                fontsize=16, fontweight='bold', y=0.98)

    # 1. Segment distribution (pie chart)
    ax1 = plt.subplot(2, 3, 1)
    segment_counts = df['segment'].value_counts()
    wedges, texts, autotexts = ax1.pie(segment_counts.values,
                                        labels=[s.replace('_', ' ').title() for s in segment_counts.index],
                                        colors=[colors.get(s.split('_')[0], '#95a5a6') for s in segment_counts.index],
                                        autopct='%1.1f%%',
                                        startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax1.set_title('Segment Distribution', fontweight='bold', fontsize=12)

    # 2. Success rate by segment
    ax2 = plt.subplot(2, 3, 2)
    success_by_segment = df.groupby('segment')['success'].mean().sort_values()
    bars = ax2.barh(range(len(success_by_segment)), success_by_segment.values)

    # Color bars
    for i, (segment, bar) in enumerate(zip(success_by_segment.index, bars)):
        bar.set_color(colors.get(segment.split('_')[0], '#95a5a6'))
        bar.set_alpha(0.7)

    ax2.set_yticks(range(len(success_by_segment)))
    ax2.set_yticklabels([s.replace('_', ' ').title() for s in success_by_segment.index])
    ax2.set_xlabel('Success Rate')
    ax2.set_xlim(0, 1)
    ax2.set_title('Success Rate by Segment', fontweight='bold', fontsize=12)

    # Add percentage labels
    for i, v in enumerate(success_by_segment.values):
        ax2.text(v + 0.01, i, f'{v:.1%}', va='center')

    # 3. Success level distribution
    ax3 = plt.subplot(2, 3, 3)
    success_levels = df['success_level'].value_counts()
    level_colors = [colors[level] for level in success_levels.index]
    bars = ax3.bar(range(len(success_levels)), success_levels.values, color=level_colors, alpha=0.8)

    ax3.set_xticks(range(len(success_levels)))
    ax3.set_xticklabels([l.title() for l in success_levels.index], rotation=45, ha='right')
    ax3.set_ylabel('Count')
    ax3.set_title('Success Level Distribution', fontweight='bold', fontsize=12)

    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom')

    # 4. Feature distributions (violin plot for key features)
    ax4 = plt.subplot(2, 3, 4)

    key_features = ['novelty_score', 'market_size', 'team_experience', 'timing_score']
    feature_data = []
    for feat in key_features:
        feature_data.append(df[feat].dropna())

    parts = ax4.violinplot(feature_data, positions=range(len(key_features)),
                           showmeans=True, showextrema=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.6)

    ax4.set_xticks(range(len(key_features)))
    ax4.set_xticklabels([f.replace('_', '\n').title() for f in key_features])
    ax4.set_ylabel('Score (0-10)')
    ax4.set_title('Key Feature Distributions', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 5. Funding vs Success
    ax5 = plt.subplot(2, 3, 5)

    # Bin funding into categories
    df['funding_category'] = pd.qcut(df['initial_funding'], q=5,
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    funding_success = df.groupby('funding_category')['success'].mean()

    bars = ax5.bar(range(len(funding_success)), funding_success.values,
                   color='#2ecc71', alpha=0.7)

    ax5.set_xticks(range(len(funding_success)))
    ax5.set_xticklabels(funding_success.index, rotation=45, ha='right')
    ax5.set_ylabel('Success Rate')
    ax5.set_ylim(0, 1)
    ax5.set_title('Success Rate by Funding Level', fontweight='bold', fontsize=12)

    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')

    # 6. Temporal trends
    ax6 = plt.subplot(2, 3, 6)

    yearly_success = df.groupby('launch_year')['success'].mean()
    ax6.plot(yearly_success.index, yearly_success.values,
            marker='o', linewidth=2, markersize=8, color='#e74c3c')

    ax6.set_xlabel('Launch Year')
    ax6.set_ylabel('Success Rate')
    ax6.set_ylim(0, 1)
    ax6.set_title('Success Rate Over Time', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(yearly_success.index, yearly_success.values, 1)
    p = np.poly1d(z)
    ax6.plot(yearly_success.index, p(yearly_success.index),
            "--", alpha=0.5, color='gray', label=f'Trend: {z[0]:.3f}x')
    ax6.legend()

    plt.tight_layout()

    # Add footer
    plt.figtext(0.5, 0.01, 'Note: SIMULATED data for educational purposes',
               ha='center', fontsize=10, style='italic', color='gray')

    return fig

def create_algorithm_comparison():
    """Create algorithm performance comparison visualization."""

    _, X_train, X_test, y_train, y_test, _, _, feature_names = load_data()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Classification Algorithm Comparison\nProduct Innovation Success Prediction',
                fontsize=16, fontweight='bold', y=0.98)

    # Train multiple classifiers
    classifiers = {
        'Logistic\nRegression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision\nTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random\nForest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient\nBoosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    }

    # Scale data for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. ROC Curves
    ax1 = plt.subplot(2, 3, 1)

    for name, clf in classifiers.items():
        print(f"Training {name.replace(chr(10), ' ')}...")

        if 'SVM' in name:
            clf.fit(X_train_scaled, y_train)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        ax1.plot(fpr, tpr, linewidth=2,
                label=f'{name.replace(chr(10), " ")} (AUC = {roc_auc:.3f})')

        results[name] = {'clf': clf, 'auc': roc_auc, 'proba': y_pred_proba}

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves', fontweight='bold', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Precision-Recall Curves
    ax2 = plt.subplot(2, 3, 2)

    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['proba'])
        ax2.plot(recall, precision, linewidth=2,
                label=name.replace(chr(10), ' '))

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves', fontweight='bold', fontsize=12)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy comparison
    ax3 = plt.subplot(2, 3, 3)

    accuracies = []
    names = []

    for name, clf in classifiers.items():
        if 'SVM' in name:
            y_pred = clf.predict(X_test_scaled)
        else:
            y_pred = clf.predict(X_test)

        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        names.append(name.replace(chr(10), ' '))

    bars = ax3.bar(range(len(accuracies)), accuracies, color='#3498db', alpha=0.7)

    # Color best performer
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_color('#27ae60')

    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0, 1)
    ax3.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    # 4. Feature Importance (from Random Forest)
    ax4 = plt.subplot(2, 3, 4)

    rf_clf = classifiers['Random\nForest']
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10

    ax4.barh(range(len(indices)), importances[indices], color='#9b59b6', alpha=0.7)
    ax4.set_yticks(range(len(indices)))
    ax4.set_yticklabels([feature_names[i].replace('_', ' ').title() for i in indices])
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Feature Importances\n(Random Forest)', fontweight='bold', fontsize=12)

    # 5. Confusion Matrix (best performer)
    ax5 = plt.subplot(2, 3, 5)

    best_clf = list(classifiers.values())[best_idx]
    if 'SVM' in list(classifiers.keys())[best_idx]:
        y_pred = best_clf.predict(X_test_scaled)
    else:
        y_pred = best_clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    im = ax5.imshow(cm, interpolation='nearest', cmap='Blues')
    ax5.set_xticks([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_xticklabels(['Failed', 'Success'])
    ax5.set_yticklabels(['Failed', 'Success'])
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    ax5.set_title(f'Confusion Matrix\n(Best: {names[best_idx]})', fontweight='bold', fontsize=12)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, f'{cm[i, j]:,}\n({cm[i, j]/cm.sum():.1%})',
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.colorbar(im, ax=ax5)

    # 6. AUC comparison
    ax6 = plt.subplot(2, 3, 6)

    aucs = [results[name]['auc'] for name in classifiers.keys()]
    bars = ax6.bar(range(len(aucs)), aucs, color='#e67e22', alpha=0.7)

    # Color best performer
    best_auc_idx = np.argmax(aucs)
    bars[best_auc_idx].set_color('#27ae60')

    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=45, ha='right')
    ax6.set_ylabel('AUC Score')
    ax6.set_ylim(0, 1)
    ax6.set_title('AUC Score Comparison', fontweight='bold', fontsize=12)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    # Add footer
    plt.figtext(0.5, 0.01, 'Note: SIMULATED data for educational purposes',
               ha='center', fontsize=10, style='italic', color='gray')

    return fig

def create_multiclass_analysis():
    """Create multi-class classification analysis."""

    df, X_train, X_test, _, _, y_train_multi, y_test_multi, _ = load_data()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Multi-Class Classification Analysis\nFailed / Struggling / Growing / Breakthrough',
                fontsize=16, fontweight='bold', y=0.98)

    # Class labels
    class_names = ['Failed', 'Struggling', 'Growing', 'Breakthrough']

    # Train Random Forest for multi-class
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train_multi)
    y_pred = rf.predict(X_test)

    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, (1, 4))

    cm = confusion_matrix(y_test_multi, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax1.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('True Class', fontsize=12)
    ax1.set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=14)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax1.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                          ha='center', va='center',
                          color='white' if cm_normalized[i, j] > 0.5 else 'black')

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Class-wise metrics
    ax2 = plt.subplot(2, 3, 3)

    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test_multi, y_pred)

    x = np.arange(len(class_names))
    width = 0.25

    bars1 = ax2.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax2.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.set_title('Class-wise Performance Metrics', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Class distribution in predictions vs actual
    ax3 = plt.subplot(2, 3, 5)

    actual_dist = np.bincount(y_test_multi, minlength=4)
    pred_dist = np.bincount(y_pred, minlength=4)

    x = np.arange(len(class_names))
    width = 0.35

    bars1 = ax3.bar(x - width/2, actual_dist, width, label='Actual', color='#34495e', alpha=0.7)
    bars2 = ax3.bar(x + width/2, pred_dist, width, label='Predicted', color='#9b59b6', alpha=0.7)

    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names)
    ax3.set_ylabel('Count')
    ax3.set_title('Class Distribution: Actual vs Predicted', fontweight='bold', fontsize=12)
    ax3.legend()

    # Add count labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    # 4. Feature importance for multi-class
    ax4 = plt.subplot(2, 3, 6)

    importances = rf.feature_importances_
    feature_names = np.load('innovation_feature_names.npy', allow_pickle=True)
    indices = np.argsort(importances)[-10:]

    ax4.barh(range(len(indices)), importances[indices], color='#e67e22', alpha=0.7)
    ax4.set_yticks(range(len(indices)))
    ax4.set_yticklabels([feature_names[i].replace('_', ' ').title() for i in indices], fontsize=9)
    ax4.set_xlabel('Importance')
    ax4.set_title('Top Features for Multi-Class', fontweight='bold', fontsize=12)

    plt.tight_layout()

    # Add footer
    plt.figtext(0.5, 0.01, 'Note: SIMULATED data for educational purposes',
               ha='center', fontsize=10, style='italic', color='gray')

    return fig

def create_success_prediction_dashboard():
    """Create a comprehensive success prediction dashboard."""

    df, _, _, _, _, _, _, _ = load_data()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Product Innovation Success Prediction Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    # 1. Success factors correlation
    ax1 = plt.subplot(2, 3, (1, 2))

    success_factors = ['novelty_score', 'market_size', 'timing_score',
                      'team_experience', 'domain_expertise', 'user_sentiment_score',
                      'innovation_intensity', 'market_readiness', 'team_strength']

    correlations = []
    for factor in success_factors:
        if factor in df.columns:
            corr = df[factor].corr(df['success'])
            correlations.append(corr)
        else:
            correlations.append(0)

    # Sort by absolute correlation
    sorted_idx = np.argsort(np.abs(correlations))[::-1]
    sorted_factors = [success_factors[i] for i in sorted_idx]
    sorted_corrs = [correlations[i] for i in sorted_idx]

    colors_bar = ['#27ae60' if c > 0 else '#e74c3c' for c in sorted_corrs]
    bars = ax1.barh(range(len(sorted_factors)), sorted_corrs, color=colors_bar, alpha=0.7)

    ax1.set_yticks(range(len(sorted_factors)))
    ax1.set_yticklabels([f.replace('_', ' ').title() for f in sorted_factors])
    ax1.set_xlabel('Correlation with Success')
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_title('Success Factor Correlations', fontweight='bold', fontsize=12)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='x')

    # Add correlation values
    for i, (bar, val) in enumerate(zip(bars, sorted_corrs)):
        ax1.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}',
                va='center', ha='left' if val > 0 else 'right', fontsize=9)

    # 2. Success probability distribution
    ax2 = plt.subplot(2, 3, 3)

    ax2.hist(df[df['success']==1]['success_probability'], bins=30,
            alpha=0.7, label='Successful', color='#27ae60', density=True)
    ax2.hist(df[df['success']==0]['success_probability'], bins=30,
            alpha=0.7, label='Failed', color='#e74c3c', density=True)

    ax2.set_xlabel('Success Probability Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Success Probability Distributions', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Innovation vs Market readiness scatter
    ax3 = plt.subplot(2, 3, 4)

    # Sample for visibility
    sample_idx = np.random.choice(len(df), 500, replace=False)
    sample_df = df.iloc[sample_idx]

    for success_level in ['failed', 'struggling', 'growing', 'breakthrough']:
        mask = sample_df['success_level'] == success_level
        ax3.scatter(sample_df.loc[mask, 'innovation_intensity'],
                   sample_df.loc[mask, 'market_readiness'],
                   label=success_level.title(),
                   color=colors[success_level],
                   alpha=0.6, s=50)

    ax3.set_xlabel('Innovation Intensity')
    ax3.set_ylabel('Market Readiness')
    ax3.set_title('Innovation vs Market Positioning', fontweight='bold', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Team strength impact
    ax4 = plt.subplot(2, 3, 5)

    # Bin team strength
    df['team_category'] = pd.qcut(df['team_strength'], q=4,
                                  labels=['Weak', 'Average', 'Strong', 'Elite'])
    team_success = df.groupby('team_category')['success'].agg(['mean', 'count'])

    bars = ax4.bar(range(len(team_success)), team_success['mean'],
                  color='#9b59b6', alpha=0.7)

    ax4.set_xticks(range(len(team_success)))
    ax4.set_xticklabels(team_success.index)
    ax4.set_ylabel('Success Rate')
    ax4.set_ylim(0, 1)
    ax4.set_title('Success Rate by Team Strength', fontweight='bold', fontsize=12)

    # Add labels
    for i, (bar, (idx, row)) in enumerate(zip(bars, team_success.iterrows())):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{row["mean"]:.1%}\n(n={int(row["count"]):,})',
                ha='center', va='bottom', fontsize=9)

    # 5. Development efficiency
    ax5 = plt.subplot(2, 3, 6)

    # Calculate development efficiency score
    df['dev_efficiency'] = df['iterations'] / df['development_time']

    # Bin by efficiency
    df['efficiency_category'] = pd.qcut(df['dev_efficiency'], q=4,
                                        labels=['Slow', 'Normal', 'Fast', 'Rapid'])

    eff_success = df.groupby('efficiency_category')['success'].mean()

    bars = ax5.bar(range(len(eff_success)), eff_success.values,
                  color='#1abc9c', alpha=0.7)

    ax5.set_xticks(range(len(eff_success)))
    ax5.set_xticklabels(eff_success.index)
    ax5.set_ylabel('Success Rate')
    ax5.set_ylim(0, 1)
    ax5.set_title('Success vs Development Speed', fontweight='bold', fontsize=12)

    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')

    plt.tight_layout()

    # Add footer
    plt.figtext(0.5, 0.01, 'Note: SIMULATED data for educational purposes',
               ha='center', fontsize=10, style='italic', color='gray')

    return fig

def main():
    """Generate all classification visualizations."""

    print("\nGenerating classification visualizations...")
    print("=" * 60)

    # 1. Dataset overview
    print("Creating dataset overview...")
    fig1 = create_dataset_overview()
    fig1.savefig('innovation_dataset_overview.png', dpi=300, bbox_inches='tight')
    fig1.savefig('innovation_dataset_overview.pdf', bbox_inches='tight')
    plt.close()

    # 2. Algorithm comparison
    print("Creating algorithm comparison...")
    fig2 = create_algorithm_comparison()
    fig2.savefig('innovation_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    fig2.savefig('innovation_algorithm_comparison.pdf', bbox_inches='tight')
    plt.close()

    # 3. Multi-class analysis
    print("Creating multi-class analysis...")
    fig3 = create_multiclass_analysis()
    fig3.savefig('innovation_multiclass_analysis.png', dpi=300, bbox_inches='tight')
    fig3.savefig('innovation_multiclass_analysis.pdf', bbox_inches='tight')
    plt.close()

    # 4. Success prediction dashboard
    print("Creating success prediction dashboard...")
    fig4 = create_success_prediction_dashboard()
    fig4.savefig('innovation_success_dashboard.png', dpi=300, bbox_inches='tight')
    fig4.savefig('innovation_success_dashboard.pdf', bbox_inches='tight')
    plt.close()

    print("\nVisualization generation complete!")
    print("Generated files:")
    print("  - innovation_dataset_overview.pdf")
    print("  - innovation_algorithm_comparison.pdf")
    print("  - innovation_multiclass_analysis.pdf")
    print("  - innovation_success_dashboard.pdf")
    print("=" * 60)

if __name__ == "__main__":
    main()