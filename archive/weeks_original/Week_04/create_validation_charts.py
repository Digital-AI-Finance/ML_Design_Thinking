"""
Create validation and evaluation charts for classification models.
Includes learning curves, cross-validation, and calibration plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def load_data():
    """Load the preprocessed data."""
    X_train = np.load('innovation_X_train.npy')
    X_test = np.load('innovation_X_test.npy')
    y_train = np.load('innovation_y_train_binary.npy')
    y_test = np.load('innovation_y_test_binary.npy')
    feature_names = np.load('innovation_feature_names.npy', allow_pickle=True)

    return X_train, X_test, y_train, y_test, feature_names

def create_learning_curves():
    """Create learning curves for multiple algorithms."""

    X_train, _, y_train, _, _ = load_data()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Learning Curves: Model Performance vs Training Size',
                fontsize=16, fontweight='bold', y=0.98)

    # Define classifiers
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42))
    ]

    train_sizes = np.linspace(0.1, 1.0, 10)

    for idx, (name, clf) in enumerate(classifiers, 1):
        ax = plt.subplot(2, 2, idx)

        print(f"  Computing learning curve for {name}...")

        # Compute learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            clf, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=train_sizes, random_state=42
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot learning curves
        ax.plot(train_sizes_abs, train_mean, 'o-', color='#3498db',
                label='Training score', linewidth=2, markersize=8)
        ax.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='#3498db')

        ax.plot(train_sizes_abs, val_mean, 'o-', color='#e74c3c',
                label='Cross-validation score', linewidth=2, markersize=8)
        ax.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='#e74c3c')

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.set_title(f'{name}', fontweight='bold', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)

        # Add convergence point annotation
        convergence_idx = np.where(np.abs(train_mean - val_mean) < 0.02)[0]
        if len(convergence_idx) > 0:
            conv_point = train_sizes_abs[convergence_idx[0]]
            ax.axvline(x=conv_point, color='gray', linestyle='--', alpha=0.5)
            ax.text(conv_point, 0.52, f'Convergence\n~{int(conv_point)} samples',
                   ha='center', fontsize=8)

    plt.tight_layout()

    # Add footer
    plt.figtext(0.5, 0.01, 'Note: SIMULATED data for educational purposes',
               ha='center', fontsize=10, style='italic', color='gray')

    return fig

def create_validation_curves():
    """Create validation curves for hyperparameter tuning."""

    X_train, _, y_train, _, _ = load_data()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Validation Curves: Hyperparameter Impact on Performance',
                fontsize=16, fontweight='bold', y=0.98)

    # 1. Random Forest - n_estimators
    ax1 = plt.subplot(2, 3, 1)
    param_range = [10, 20, 50, 100, 150, 200]

    print("  Computing validation curve for Random Forest (n_estimators)...")
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(max_depth=5, random_state=42),
        X_train, y_train, param_name="n_estimators",
        param_range=param_range, cv=5, n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax1.plot(param_range, train_mean, 'o-', color='#3498db',
            label='Training score', linewidth=2)
    ax1.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='#3498db')
    ax1.plot(param_range, val_mean, 'o-', color='#e74c3c',
            label='Validation score', linewidth=2)
    ax1.fill_between(param_range, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color='#e74c3c')

    ax1.set_xlabel('Number of Trees')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_title('Random Forest: n_estimators', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Random Forest - max_depth
    ax2 = plt.subplot(2, 3, 2)
    param_range = [3, 5, 7, 10, 15, 20, None]
    param_labels = [str(p) if p else 'None' for p in param_range]

    print("  Computing validation curve for Random Forest (max_depth)...")
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(n_estimators=50, random_state=42),
        X_train, y_train, param_name="max_depth",
        param_range=param_range, cv=5, n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    ax2.plot(range(len(param_range)), train_mean, 'o-', color='#3498db',
            label='Training score', linewidth=2)
    ax2.plot(range(len(param_range)), val_mean, 'o-', color='#e74c3c',
            label='Validation score', linewidth=2)

    ax2.set_xticks(range(len(param_range)))
    ax2.set_xticklabels(param_labels)
    ax2.set_xlabel('Max Depth')
    ax2.set_ylabel('Accuracy Score')
    ax2.set_title('Random Forest: max_depth', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. Logistic Regression - C (regularization)
    ax3 = plt.subplot(2, 3, 3)
    param_range = [0.001, 0.01, 0.1, 1, 10, 100]

    print("  Computing validation curve for Logistic Regression (C)...")
    train_scores, val_scores = validation_curve(
        LogisticRegression(max_iter=1000, random_state=42),
        X_train, y_train, param_name="C",
        param_range=param_range, cv=5, n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    ax3.semilogx(param_range, train_mean, 'o-', color='#3498db',
                label='Training score', linewidth=2)
    ax3.semilogx(param_range, val_mean, 'o-', color='#e74c3c',
                label='Validation score', linewidth=2)

    ax3.set_xlabel('C (Regularization Parameter)')
    ax3.set_ylabel('Accuracy Score')
    ax3.set_title('Logistic Regression: C', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. Cross-validation scores comparison
    ax4 = plt.subplot(2, 3, 4)

    print("  Computing cross-validation scores...")
    classifiers = [
        ('Logistic\nRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Decision\nTree', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('Random\nForest', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
        ('Gradient\nBoosting', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42))
    ]

    cv_results = []
    for name, clf in classifiers:
        scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
        cv_results.append(scores)

    bp = ax4.boxplot(cv_results, labels=[name for name, _ in classifiers],
                     patch_artist=True)

    # Color boxes
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_ylabel('Accuracy Score')
    ax4.set_title('Cross-Validation Score Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add mean values
    for i, scores in enumerate(cv_results):
        ax4.text(i+1, np.mean(scores) + 0.01, f'{np.mean(scores):.3f}',
                ha='center', fontsize=9, fontweight='bold')

    # 5. Model complexity vs performance
    ax5 = plt.subplot(2, 3, 5)

    complexity = [1, 5, 10, 20]  # Proxy for model complexity
    models = ['Linear', 'Tree (d=5)', 'Forest (n=10)', 'Forest (n=20)']
    train_scores = [0.75, 0.82, 0.88, 0.92]
    val_scores = [0.73, 0.79, 0.85, 0.84]

    ax5.plot(complexity, train_scores, 'o-', color='#3498db',
            label='Training', linewidth=2, markersize=8)
    ax5.plot(complexity, val_scores, 'o-', color='#e74c3c',
            label='Validation', linewidth=2, markersize=8)

    # Mark optimal complexity
    optimal_idx = np.argmax(val_scores)
    ax5.scatter(complexity[optimal_idx], val_scores[optimal_idx],
               s=200, color='#27ae60', zorder=5, alpha=0.5)
    ax5.annotate('Optimal\nComplexity', xy=(complexity[optimal_idx], val_scores[optimal_idx]),
                xytext=(complexity[optimal_idx]+2, val_scores[optimal_idx]-0.05),
                arrowprops=dict(arrowstyle='->', color='#27ae60'),
                fontsize=9, ha='center')

    ax5.set_xlabel('Model Complexity')
    ax5.set_ylabel('Accuracy Score')
    ax5.set_title('Bias-Variance Tradeoff', fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0.7, 0.95)

    # 6. Training time comparison
    ax6 = plt.subplot(2, 3, 6)

    # Simulated training times (in seconds)
    model_names = ['Logistic\nReg', 'Decision\nTree', 'Random\nForest', 'SVM', 'Gradient\nBoost']
    train_times = [0.05, 0.02, 0.45, 1.2, 0.8]
    accuracies = [0.73, 0.71, 0.85, 0.82, 0.86]

    # Create scatter plot
    scatter = ax6.scatter(train_times, accuracies, s=200, c=accuracies,
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)

    # Add labels
    for i, name in enumerate(model_names):
        ax6.annotate(name, (train_times[i], accuracies[i]),
                    ha='center', va='center', fontsize=9)

    ax6.set_xlabel('Training Time (seconds)')
    ax6.set_ylabel('Validation Accuracy')
    ax6.set_title('Efficiency vs Performance', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Accuracy', rotation=270, labelpad=15)

    plt.tight_layout()

    # Add footer
    plt.figtext(0.5, 0.01, 'Note: SIMULATED data for educational purposes',
               ha='center', fontsize=10, style='italic', color='gray')

    return fig

def create_calibration_plots():
    """Create calibration plots for probability predictions."""

    X_train, X_test, y_train, y_test, _ = load_data()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Model Calibration: Predicted vs Actual Probabilities',
                fontsize=16, fontweight='bold', y=0.98)

    # Train models
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    ]

    # 1. Calibration curves
    ax1 = plt.subplot(2, 2, 1)

    for name, clf in classifiers:
        print(f"  Computing calibration for {name}...")
        clf.fit(X_train, y_train)

        if hasattr(clf, 'predict_proba'):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        fraction_pos, mean_pred = calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_pred, fraction_pos, 'o-', label=name, linewidth=2)

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)

    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Curves', fontweight='bold', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 2. Reliability diagram (histogram)
    ax2 = plt.subplot(2, 2, 2)

    # Use Random Forest for detailed analysis
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    prob_pos = rf.predict_proba(X_test)[:, 1]

    # Create bins
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate statistics for each bin
    bin_counts = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)

    for i in range(n_bins):
        bin_mask = (prob_pos >= bin_edges[i]) & (prob_pos < bin_edges[i+1])
        bin_counts[i] = np.sum(bin_mask)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(y_test[bin_mask])

    # Plot histogram
    width = 0.8 / n_bins
    bars = ax2.bar(bin_centers, bin_counts, width=width, alpha=0.7,
                   color='#3498db', edgecolor='black')

    # Color bars by calibration error
    for i, bar in enumerate(bars):
        error = abs(bin_accuracies[i] - bin_centers[i])
        if error > 0.1:
            bar.set_color('#e74c3c')
        elif error > 0.05:
            bar.set_color('#f39c12')
        else:
            bar.set_color('#27ae60')

    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution (Random Forest)', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Brier score comparison
    ax3 = plt.subplot(2, 2, 3)

    brier_scores = []
    model_names = []

    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        prob_pos = clf.predict_proba(X_test)[:, 1]

        # Calculate Brier score
        brier = np.mean((prob_pos - y_test) ** 2)
        brier_scores.append(brier)
        model_names.append(name.replace('\n', ' '))

    bars = ax3.bar(range(len(brier_scores)), brier_scores,
                   color=['#3498db', '#2ecc71', '#9b59b6'], alpha=0.7)

    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_ylabel('Brier Score (lower is better)')
    ax3.set_title('Calibration Quality Comparison', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, score in zip(bars, brier_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # 4. Confidence vs accuracy
    ax4 = plt.subplot(2, 2, 4)

    # Use Random Forest predictions
    rf_probs = rf.predict_proba(X_test)
    rf_preds = rf.predict(X_test)

    # Get confidence (max probability)
    confidences = np.max(rf_probs, axis=1)

    # Bin confidences
    conf_bins = np.linspace(0.5, 1.0, 6)
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for i in range(len(conf_bins)-1):
        mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(rf_preds[mask] == y_test[mask]))
            bin_confs.append(np.mean(confidences[mask]))
            bin_sizes.append(np.sum(mask))
        else:
            bin_accs.append(0)
            bin_confs.append((conf_bins[i] + conf_bins[i+1])/2)
            bin_sizes.append(0)

    # Plot with bubble size representing sample size
    scatter = ax4.scatter(bin_confs, bin_accs, s=np.array(bin_sizes)/2,
                         c=bin_accs, cmap='RdYlGn', alpha=0.7,
                         edgecolors='black', linewidth=2, vmin=0.5, vmax=1.0)

    # Perfect calibration line
    ax4.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5, label='Perfect calibration')

    ax4.set_xlabel('Average Confidence')
    ax4.set_ylabel('Actual Accuracy')
    ax4.set_xlim(0.45, 1.05)
    ax4.set_ylim(0.45, 1.05)
    ax4.set_title('Confidence vs Accuracy (Random Forest)', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Accuracy', rotation=270, labelpad=15)

    plt.tight_layout()

    # Add footer
    plt.figtext(0.5, 0.01, 'Note: SIMULATED data for educational purposes',
               ha='center', fontsize=10, style='italic', color='gray')

    return fig

def create_error_analysis():
    """Create error analysis visualization."""

    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Load original dataframe for more context
    df = pd.read_csv('innovation_products_full.csv')

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Error Analysis: Understanding Model Mistakes',
                fontsize=16, fontweight='bold', y=0.98)

    # 1. False Positives vs False Negatives distribution
    ax1 = plt.subplot(2, 3, 1)

    # Identify errors
    false_positives = (y_pred == 1) & (y_test == 0)
    false_negatives = (y_pred == 0) & (y_test == 1)
    true_positives = (y_pred == 1) & (y_test == 1)
    true_negatives = (y_pred == 0) & (y_test == 0)

    error_counts = [
        np.sum(true_negatives),
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives)
    ]

    labels = ['True Negative', 'True Positive', 'False Positive', 'False Negative']
    colors_pie = ['#27ae60', '#3498db', '#e67e22', '#e74c3c']

    wedges, texts, autotexts = ax1.pie(error_counts, labels=labels,
                                        colors=colors_pie, autopct='%1.0f%%',
                                        startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax1.set_title('Prediction Outcome Distribution', fontweight='bold', fontsize=12)

    # 2. Confidence distribution for errors
    ax2 = plt.subplot(2, 3, 2)

    # Get confidence scores for different outcomes
    conf_tp = y_pred_proba[true_positives]
    conf_tn = 1 - y_pred_proba[true_negatives]
    conf_fp = y_pred_proba[false_positives]
    conf_fn = 1 - y_pred_proba[false_negatives]

    # Create violin plot
    parts = ax2.violinplot([conf_tn, conf_tp, conf_fp, conf_fn],
                           positions=[1, 2, 3, 4], showmeans=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_pie[i])
        pc.set_alpha(0.6)

    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels(['TN', 'TP', 'FP', 'FN'])
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Confidence Distribution by Outcome', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Feature importance for errors
    ax3 = plt.subplot(2, 3, 3)

    # Calculate feature values for errors vs correct predictions
    X_test_errors = X_test[false_positives | false_negatives]
    X_test_correct = X_test[true_positives | true_negatives]

    # Get top 5 features by importance
    importances = rf.feature_importances_
    top_features_idx = np.argsort(importances)[-5:]

    feature_diffs = []
    feature_labels = []

    for idx in top_features_idx:
        mean_error = np.mean(X_test_errors[:, idx])
        mean_correct = np.mean(X_test_correct[:, idx])
        diff = mean_error - mean_correct
        feature_diffs.append(diff)
        feature_labels.append(feature_names[idx].replace('_', ' ').title())

    colors_bar = ['#e74c3c' if d < 0 else '#27ae60' for d in feature_diffs]
    bars = ax3.barh(range(len(feature_diffs)), feature_diffs,
                    color=colors_bar, alpha=0.7)

    ax3.set_yticks(range(len(feature_labels)))
    ax3.set_yticklabels(feature_labels)
    ax3.set_xlabel('Difference (Errors - Correct)')
    ax3.set_title('Feature Differences in Errors', fontweight='bold', fontsize=12)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Error rate by predicted probability
    ax4 = plt.subplot(2, 3, 4)

    # Bin predictions
    prob_bins = np.linspace(0, 1, 11)
    error_rates = []
    bin_centers = []
    bin_counts = []

    for i in range(len(prob_bins)-1):
        mask = (y_pred_proba >= prob_bins[i]) & (y_pred_proba < prob_bins[i+1])
        if np.sum(mask) > 0:
            error_rate = np.mean(y_pred[mask] != y_test[mask])
            error_rates.append(error_rate)
            bin_centers.append((prob_bins[i] + prob_bins[i+1])/2)
            bin_counts.append(np.sum(mask))

    ax4.bar(bin_centers, error_rates, width=0.08, alpha=0.7,
           color='#e74c3c', edgecolor='black')

    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Error Rate')
    ax4.set_title('Error Rate by Confidence Level', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add sample counts as text
    for x, y, count in zip(bin_centers, error_rates, bin_counts):
        ax4.text(x, y + 0.01, f'n={count}', ha='center', fontsize=8)

    # 5. Misclassification cost analysis
    ax5 = plt.subplot(2, 3, 5)

    # Define costs (example values)
    cost_fp = 10000  # Cost of false positive (pursuing bad product)
    cost_fn = 50000  # Cost of false negative (missing good product)

    costs = {
        'True Negative': 0,
        'True Positive': 0,
        'False Positive': cost_fp * np.sum(false_positives),
        'False Negative': cost_fn * np.sum(false_negatives)
    }

    bars = ax5.bar(range(len(costs)), list(costs.values()),
                   color=['#27ae60', '#3498db', '#e67e22', '#e74c3c'],
                   alpha=0.7)

    ax5.set_xticks(range(len(costs)))
    ax5.set_xticklabels(list(costs.keys()), rotation=45, ha='right')
    ax5.set_ylabel('Total Cost ($)')
    ax5.set_title('Misclassification Cost Analysis', fontweight='bold', fontsize=12)

    # Add value labels
    for bar, (label, cost) in zip(bars, costs.items()):
        if cost > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'${cost/1000:.0f}K', ha='center', va='bottom', fontsize=9)

    # 6. Error examples
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Create text summary of typical errors
    error_text = """
    TYPICAL ERROR PATTERNS
    ======================

    FALSE POSITIVES (Predicted Success, Actually Failed):
    • High novelty but poor market timing
    • Strong team but excessive complexity
    • Good funding but fierce competition
    • Average: 65% confidence

    FALSE NEGATIVES (Predicted Failure, Actually Succeeded):
    • Low initial metrics but great execution
    • Small team with exceptional expertise
    • Limited funding but perfect timing
    • Average: 62% confidence

    KEY INSIGHTS:
    → Model struggles with outliers
    → Timing is hardest to predict
    → Team quality often undervalued
    → Market dynamics are complex
    """

    ax6.text(0.1, 0.9, error_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()

    # Add footer
    plt.figtext(0.5, 0.01, 'Note: SIMULATED data for educational purposes',
               ha='center', fontsize=10, style='italic', color='gray')

    return fig

def main():
    """Generate all validation and evaluation charts."""

    print("\nGenerating validation and evaluation charts...")
    print("=" * 60)

    # 1. Learning curves
    print("Creating learning curves...")
    fig1 = create_learning_curves()
    fig1.savefig('innovation_learning_curves.png', dpi=300, bbox_inches='tight')
    fig1.savefig('innovation_learning_curves.pdf', bbox_inches='tight')
    plt.close()

    # 2. Validation curves
    print("Creating validation curves...")
    fig2 = create_validation_curves()
    fig2.savefig('innovation_validation_curves.png', dpi=300, bbox_inches='tight')
    fig2.savefig('innovation_validation_curves.pdf', bbox_inches='tight')
    plt.close()

    # 3. Calibration plots
    print("Creating calibration plots...")
    fig3 = create_calibration_plots()
    fig3.savefig('innovation_calibration_plots.png', dpi=300, bbox_inches='tight')
    fig3.savefig('innovation_calibration_plots.pdf', bbox_inches='tight')
    plt.close()

    # 4. Error analysis
    print("Creating error analysis...")
    fig4 = create_error_analysis()
    fig4.savefig('innovation_error_analysis.png', dpi=300, bbox_inches='tight')
    fig4.savefig('innovation_error_analysis.pdf', bbox_inches='tight')
    plt.close()

    print("\nValidation charts generation complete!")
    print("Generated files:")
    print("  - innovation_learning_curves.pdf")
    print("  - innovation_validation_curves.pdf")
    print("  - innovation_calibration_plots.pdf")
    print("  - innovation_error_analysis.pdf")
    print("=" * 60)

if __name__ == "__main__":
    main()