"""
Create advanced metrics visualizations for Week 4 presentation.
Includes precision-recall curves, learning curves, and cross-validation comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.metrics import precision_recall_curve, auc, roc_curve
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
    'logreg': '#3498db',      # Blue
    'tree': '#2ecc71',        # Green
    'rf': '#9b59b6',          # Purple
    'svm': '#e74c3c',         # Red
    'gb': '#f39c12',          # Orange
    'train': '#34495e',       # Dark gray
    'val': '#e67e22',         # Carrot
}

def load_data():
    """Load the innovation dataset."""
    X_train = np.load('innovation_X_train.npy')
    X_test = np.load('innovation_X_test.npy')
    y_train = np.load('innovation_y_train_binary.npy')
    y_test = np.load('innovation_y_test_binary.npy')
    feature_names = np.load('innovation_feature_names.npy', allow_pickle=True)

    return X_train, X_test, y_train, y_test, feature_names

def create_precision_recall_curves():
    """Create precision-recall curves for multiple classifiers."""
    print("Creating precision-recall curves...")

    # Load data
    X_train, X_test, y_train, y_test, _ = load_data()

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define classifiers
    classifiers = {
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), colors['logreg']),
        'Random Forest': (RandomForestClassifier(n_estimators=50, random_state=42), colors['rf']),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=50, random_state=42), colors['gb']),
        'SVM': (SVC(probability=True, random_state=42), colors['svm'])
    }

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot precision-recall curves
    for name, (clf, color) in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_scores = clf.predict_proba(X_test_scaled)[:, 1]

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)
        ax1.plot(recall, precision, color=color, lw=2,
                label=f'{name} (AUC: {pr_auc:.3f})')

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC: {roc_auc:.3f})')

    # Baseline for PR curve
    baseline_pr = y_test.sum() / len(y_test)
    ax1.axhline(y=baseline_pr, color='gray', linestyle='--', label=f'Baseline: {baseline_pr:.3f}')

    # Diagonal for ROC
    ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    # Styling
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.suptitle('Model Performance: Precision-Recall vs ROC Analysis',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    plt.savefig('charts/precision_recall_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('charts/precision_recall_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: charts/precision_recall_curves.pdf")
    plt.close()

def create_learning_curves():
    """Create learning curves for different algorithms."""
    print("Creating learning curves...")

    # Load data
    X_train, _, y_train, _, _ = load_data()

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use subset for faster computation
    n_samples = min(3000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_subset = X_train_scaled[indices]
    y_subset = y_train[indices]

    # Define classifiers
    classifiers = [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=500), colors['logreg']),
        ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42), colors['tree']),
        ('Random Forest', RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42), colors['rf']),
        ('SVM', SVC(random_state=42), colors['svm'])
    ]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    train_sizes = np.linspace(0.1, 1.0, 10)

    for idx, (name, clf, color) in enumerate(classifiers):
        ax = axes[idx]

        # Calculate learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            clf, X_subset, y_subset, train_sizes=train_sizes,
            cv=5, n_jobs=-1, scoring='accuracy'
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot
        ax.plot(train_sizes_abs, train_mean, 'o-', color=colors['train'],
               label='Training score')
        ax.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color=colors['train'])

        ax.plot(train_sizes_abs, val_mean, 'o-', color=colors['val'],
               label='Cross-validation score')
        ax.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color=colors['val'])

        # Styling
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.05])

        # Add final scores
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_train - final_val
        ax.text(0.02, 0.98,
               f'Final Train: {final_train:.3f}\nFinal Val: {final_val:.3f}\nGap: {gap:.3f}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Learning Curves: Training vs Validation Performance',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig('charts/learning_curves_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('charts/learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: charts/learning_curves_comparison.pdf")
    plt.close()

def create_cross_validation_comparison():
    """Create cross-validation score comparison."""
    print("Creating cross-validation comparison...")

    # Load data
    X_train, _, y_train, _, _ = load_data()

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use subset for faster computation
    n_samples = min(2000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_subset = X_train_scaled[indices]
    y_subset = y_train[indices]

    # Define classifiers
    classifiers = {
        'Logistic\nRegression': LogisticRegression(random_state=42, max_iter=500),
        'Decision\nTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random\nForest': RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42),
        'SVM': SVC(random_state=42),
        'Gradient\nBoosting': GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    }

    # Perform cross-validation
    cv_results = []
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_subset, y_subset, cv=10, scoring='accuracy')
        cv_results.extend([(name, score) for score in scores])

    # Create DataFrame
    df = pd.DataFrame(cv_results, columns=['Classifier', 'Accuracy'])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    sns.boxplot(data=df, x='Classifier', y='Accuracy', ax=ax1, palette='Set2')
    ax1.set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Classifier', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.5, 1.0])

    # Violin plot
    sns.violinplot(data=df, x='Classifier', y='Accuracy', ax=ax2, palette='Set2')
    ax2.set_title('Score Distribution Density', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_xlabel('Classifier', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.5, 1.0])

    # Add mean lines
    for ax in [ax1, ax2]:
        means = df.groupby('Classifier')['Accuracy'].mean()
        for i, (name, mean_val) in enumerate(means.items()):
            ax.hlines(mean_val, i - 0.4, i + 0.4, colors='red',
                     linestyles='dashed', alpha=0.6)

    plt.suptitle('10-Fold Cross-Validation Performance Comparison',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    plt.savefig('charts/cross_validation_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('charts/cross_validation_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: charts/cross_validation_comparison.pdf")
    plt.close()

def create_hyperparameter_sensitivity():
    """Create hyperparameter sensitivity analysis."""
    print("Creating hyperparameter sensitivity analysis...")

    # Load data
    X_train, _, y_train, _, _ = load_data()

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use subset
    n_samples = min(1500, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_subset = X_train_scaled[indices]
    y_subset = y_train[indices]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Random Forest - n_estimators
    ax = axes[0, 0]
    param_range = [10, 20, 30, 50, 75, 100]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42, max_depth=5),
        X_subset, y_subset, param_name='n_estimators',
        param_range=param_range, cv=5, scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.plot(param_range, train_mean, 'o-', color=colors['train'], label='Training')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color=colors['train'])
    ax.plot(param_range, val_mean, 'o-', color=colors['val'], label='Validation')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color=colors['val'])
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Accuracy')
    ax.set_title('Random Forest: n_estimators', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Decision Tree - max_depth
    ax = axes[0, 1]
    param_range = [2, 3, 4, 5, 7, 10, 15]
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(random_state=42),
        X_subset, y_subset, param_name='max_depth',
        param_range=param_range, cv=5, scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.plot(param_range, train_mean, 'o-', color=colors['train'], label='Training')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color=colors['train'])
    ax.plot(param_range, val_mean, 'o-', color=colors['val'], label='Validation')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color=colors['val'])
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('Decision Tree: max_depth', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. SVM - C parameter
    ax = axes[1, 0]
    param_range = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    train_scores, val_scores = validation_curve(
        SVC(random_state=42),
        X_subset, y_subset, param_name='C',
        param_range=param_range, cv=5, scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.semilogx(param_range, train_mean, 'o-', color=colors['train'], label='Training')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color=colors['train'])
    ax.semilogx(param_range, val_mean, 'o-', color=colors['val'], label='Validation')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color=colors['val'])
    ax.set_xlabel('C (Regularization)')
    ax.set_ylabel('Accuracy')
    ax.set_title('SVM: C parameter', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Logistic Regression - C parameter
    ax = axes[1, 1]
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, val_scores = validation_curve(
        LogisticRegression(random_state=42, max_iter=500),
        X_subset, y_subset, param_name='C',
        param_range=param_range, cv=5, scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.semilogx(param_range, train_mean, 'o-', color=colors['train'], label='Training')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color=colors['train'])
    ax.semilogx(param_range, val_mean, 'o-', color=colors['val'], label='Validation')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color=colors['val'])
    ax.set_xlabel('C (Inverse Regularization)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Logistic Regression: C parameter', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hyperparameter Sensitivity Analysis',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig('charts/hyperparameter_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('charts/hyperparameter_sensitivity.png', dpi=150, bbox_inches='tight')
    print("Saved: charts/hyperparameter_sensitivity.pdf")
    plt.close()

if __name__ == "__main__":
    print("Generating advanced metrics visualizations...")
    create_precision_recall_curves()
    create_learning_curves()
    create_cross_validation_comparison()
    create_hyperparameter_sensitivity()
    print("All advanced metrics visualizations completed!")