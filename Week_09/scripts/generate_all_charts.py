import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Create charts directory if not exists
import os
os.makedirs('../charts', exist_ok=True)

def save_chart(name):
    """Save chart as both PDF and PNG"""
    plt.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {name}")

# Chart 1: Accuracy Trap
def chart_accuracy_trap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Data distribution
    labels = ['Not Spam\n(95%)', 'Spam\n(5%)']
    sizes = [95, 5]
    colors = ['#4CAF50', '#F44336']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax1.set_title('Email Dataset Distribution', fontsize=14, fontweight='bold')

    # Right: Model performance
    ax2.text(0.5, 0.8, '95% Accurate Model', ha='center', fontsize=16, fontweight='bold')
    ax2.text(0.5, 0.6, 'Predicts "Not Spam" for Everything', ha='center', fontsize=12)
    ax2.text(0.5, 0.4, 'Catches: 0 spam emails', ha='center', fontsize=11, color='red')
    ax2.text(0.5, 0.2, 'High accuracy, Zero value', ha='center', fontsize=11, style='italic')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    save_chart('accuracy_trap')

# Chart 2: Confusion Matrix Anatomy
def chart_confusion_matrix():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create matrix
    cm = np.array([[810, 90], [15, 85]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                square=True, linewidths=2, linecolor='black')

    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Negative', 'Positive'], fontsize=12)
    ax.set_yticklabels(['Negative', 'Positive'], fontsize=12, rotation=0)
    ax.set_title('Confusion Matrix: Four Numbers, Infinite Insights',
                 fontsize=16, fontweight='bold', pad=20)

    # Annotations
    ax.text(0.5, -0.15, 'TN=810\nCorrectly identified\nnegative',
            ha='center', transform=ax.transData, fontsize=10)
    ax.text(1.5, -0.15, 'FP=90\nFalse alarm\n(Type I error)',
            ha='center', transform=ax.transData, fontsize=10, color='red')
    ax.text(0.5, 1.15, 'FN=15\nMissed positive\n(Type II error)',
            ha='center', transform=ax.transData, fontsize=10, color='red')
    ax.text(1.5, 1.15, 'TP=85\nCorrectly identified\npositive',
            ha='center', transform=ax.transData, fontsize=10)

    plt.tight_layout()
    save_chart('confusion_matrix_anatomy')

# Chart 3: Precision-Recall Tradeoff
def chart_precision_recall_tradeoff():
    thresholds = np.linspace(0, 1, 100)
    precision = 0.5 + 0.5 * thresholds
    recall = 1 - 0.9 * thresholds

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(thresholds, precision, 'b-', linewidth=3, label='Precision')
    ax.plot(thresholds, recall, 'r-', linewidth=3, label='Recall')
    ax.fill_between(thresholds, precision, alpha=0.3)
    ax.fill_between(thresholds, recall, alpha=0.3)

    # Mark intersection
    intersect = 0.55
    ax.plot(intersect, 0.775, 'go', markersize=15, label='F1 optimal')
    ax.axvline(intersect, color='green', linestyle='--', alpha=0.5)

    ax.set_xlabel('Decision Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('The Precision-Recall Trade-Off:\nCannot Optimize Both Simultaneously',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_chart('precision_recall_tradeoff')

# Chart 4: F-Beta Family
def chart_f_beta_family():
    precision = 0.8
    recall = 0.9
    betas = [0.5, 1.0, 2.0]

    fig, ax = plt.subplots(figsize=(10, 7))

    scores = []
    for beta in betas:
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        scores.append(f_beta)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(['F0.5\n(Precision focus)', 'F1\n(Balanced)', 'F2\n(Recall focus)'],
                  scores, color=colors, width=0.6, edgecolor='black', linewidth=2)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('F-Beta Score', fontsize=14, fontweight='bold')
    ax.set_title('F-Beta Family: Choose Based on Error Costs\n(Precision=0.8, Recall=0.9)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)

    # Add use case annotations
    ax.text(0, 0.75, 'Use case:\nSpam filtering', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(1, 0.75, 'Use case:\nGeneral purpose', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(2, 0.75, 'Use case:\nDisease screening', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_chart('f_beta_family')

# Chart 5: ROC Curve Explained
def chart_roc_curve():
    # Generate ROC curves for 3 models
    fpr_random = np.linspace(0, 1, 100)
    tpr_random = fpr_random

    fpr_good = np.linspace(0, 1, 100)
    tpr_good = 1 - (1 - fpr_good)**2

    fpr_excellent = np.linspace(0, 1, 100)
    tpr_excellent = 1 - (1 - fpr_excellent)**4

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(fpr_random, tpr_random, 'k--', linewidth=2, label='Random (AUC=0.50)')
    ax.plot(fpr_good, tpr_good, 'b-', linewidth=3, label='Good Model (AUC=0.75)')
    ax.plot(fpr_excellent, tpr_excellent, 'r-', linewidth=3, label='Excellent Model (AUC=0.93)')

    # Mark key points
    ax.plot(0, 1, 'go', markersize=15, label='Perfect (0,1)')
    ax.plot(0.2, 0.85, 'ro', markersize=12)
    ax.text(0.22, 0.85, 'Point A:\nHigh threshold', fontsize=10)

    ax.plot(0.5, 0.95, 'ro', markersize=12)
    ax.text(0.52, 0.95, 'Point B:\nBalanced', fontsize=10)

    ax.plot(0.8, 0.98, 'ro', markersize=12)
    ax.text(0.55, 0.98, 'Point C: Low threshold', fontsize=10)

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR = Recall)', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curve: Threshold-Independent Evaluation',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_chart('roc_curve_explained')

# Chart 6: AUC Interpretation
def chart_auc_interpretation():
    aucs = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    labels = ['Random\n(Useless)', 'Fair', 'Good', 'Excellent', 'Outstanding\n(Suspicious?)', 'Perfect\n(Check leakage!)']
    colors = ['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50', '#2196F3']

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(aucs)), aucs, color=colors, edgecolor='black', linewidth=2)

    ax.set_yticks(range(len(aucs)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('AUC Score', fontsize=14, fontweight='bold')
    ax.set_title('AUC Interpretation: What Do These Numbers Mean?',
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, axis='x', alpha=0.3)

    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{auc:.2f}', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_chart('auc_interpretation')

# Chart 7: PR vs ROC
def chart_pr_vs_roc():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Balanced dataset ROC
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr)**3
    ax1.plot(fpr, tpr, 'b-', linewidth=3, label='Model')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.3)
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve\n(Balanced: 50% positive)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.text(0.5, 0.3, 'Looks great!\nAUC = 0.89', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Imbalanced dataset PR curve
    recall = np.linspace(0, 1, 100)
    precision = 0.5 * np.exp(-2 * recall) + 0.3
    ax2.plot(recall, precision, 'r-', linewidth=3, label='Model')
    ax2.axhline(0.05, color='k', linestyle='--', linewidth=2, label='Random (baseline)')
    ax2.fill_between(recall, precision, alpha=0.3)
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve\n(Imbalanced: 5% positive)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.6, 'Reality check:\nStruggles at\nhigh recall', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('ROC vs PR: Use PR for Imbalanced Data', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_chart('pr_vs_roc')

# Chart 8: Multi-Class Strategies
def chart_multi_class():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # One-vs-Rest
    categories = ['Cat', 'Dog', 'Bird', 'Fish']
    ax1.text(0.5, 0.9, 'One-vs-Rest Strategy', ha='center', fontsize=16, fontweight='bold')
    ax1.text(0.5, 0.75, 'Cat vs (Dog + Bird + Fish)', ha='center', fontsize=12)
    ax1.text(0.5, 0.65, 'Dog vs (Cat + Bird + Fish)', ha='center', fontsize=12)
    ax1.text(0.5, 0.55, 'Bird vs (Cat + Dog + Fish)', ha='center', fontsize=12)
    ax1.text(0.5, 0.45, 'Fish vs (Cat + Dog + Bird)', ha='center', fontsize=12)
    ax1.text(0.5, 0.3, 'Result: 4 binary classifiers', ha='center', fontsize=11, style='italic')
    ax1.text(0.5, 0.2, 'Pros: Fast, simple', ha='center', fontsize=10, color='green')
    ax1.text(0.5, 0.1, 'Cons: Class imbalance', ha='center', fontsize=10, color='red')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Averaging strategies
    strategies = ['Macro\nAverage', 'Micro\nAverage', 'Weighted\nAverage']
    descriptions = [
        'Equal weight\nper class',
        'Aggregate all\npredictions',
        'Weight by\nfrequency'
    ]
    y_positions = [0.7, 0.45, 0.2]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    ax2.text(0.5, 0.9, 'Averaging Strategies', ha='center', fontsize=16, fontweight='bold')
    for strat, desc, y, color in zip(strategies, descriptions, y_positions, colors):
        rect = FancyBboxPatch((0.15, y-0.08), 0.7, 0.15, boxstyle="round,pad=0.01",
                              edgecolor='black', facecolor=color, alpha=0.6, linewidth=2)
        ax2.add_patch(rect)
        ax2.text(0.5, y, strat, ha='center', fontsize=13, fontweight='bold')
        ax2.text(0.5, y-0.05, desc, ha='center', fontsize=10)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    save_chart('multi_class_strategies')

# Chart 9: Cross-Validation Strategies
def chart_cross_validation():
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    n_samples = 100
    n_splits = 5

    # K-Fold
    ax = axes[0]
    for i in range(n_splits):
        train_idx = [j for j in range(n_samples) if j % n_splits != i]
        test_idx = [j for j in range(n_samples) if j % n_splits == i]
        ax.barh(i, len(train_idx), left=0, height=0.8, color='skyblue', label='Train' if i == 0 else '')
        ax.barh(i, len(test_idx), left=len(train_idx), height=0.8, color='salmon', label='Test' if i == 0 else '')
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax.set_xlabel('Samples', fontsize=12)
    ax.set_title('K-Fold Cross-Validation (K=5)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, n_samples)

    # Stratified K-Fold
    ax = axes[1]
    for i in range(n_splits):
        # Simulate stratified split
        ax.barh(i, 80, left=0, height=0.8, color='skyblue')
        ax.barh(i, 16, left=80, height=0.8, color='salmon')
        # Add markers for class distribution
        for j in range(0, 80, 10):
            ax.plot(j, i, 'go', markersize=3)
            ax.plot(j+5, i, 'ro', markersize=3)
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax.set_xlabel('Samples', fontsize=12)
    ax.set_title('Stratified K-Fold (Preserves class distribution)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, n_samples)

    # Time Series Split
    ax = axes[2]
    splits = [(20, 20), (40, 20), (60, 20), (80, 20)]
    for i, (train_size, test_size) in enumerate(splits):
        ax.barh(i, train_size, left=0, height=0.8, color='skyblue')
        ax.barh(i, test_size, left=train_size, height=0.8, color='salmon')
        ax.axvline(train_size, color='black', linestyle='--', alpha=0.3)
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f'Split {i+1}' for i in range(len(splits))])
    ax.set_xlabel('Time (samples in temporal order)', fontsize=12)
    ax.set_title('Time Series Split (Respects temporal order)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, n_samples)
    ax.arrow(10, -0.7, 80, 0, head_width=0.2, head_length=3, fc='black', ec='black')
    ax.text(50, -0.9, 'Time flow →', ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    save_chart('cross_validation_strategies')

# Chart 10: Metric Correlation Heatmap
def chart_metric_correlation():
    # Simulate correlation matrix
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Specificity']
    corr_matrix = np.array([
        [1.00, 0.65, 0.70, 0.85, 0.75, 0.45],
        [0.65, 1.00, 0.25, 0.70, 0.60, 0.80],
        [0.70, 0.25, 1.00, 0.75, 0.65, 0.20],
        [0.85, 0.70, 0.75, 1.00, 0.80, 0.50],
        [0.75, 0.60, 0.65, 0.80, 1.00, 0.55],
        [0.45, 0.80, 0.20, 0.50, 0.55, 1.00]
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                xticklabels=metrics, yticklabels=metrics, square=True,
                cbar_kws={'label': 'Correlation'}, ax=ax, vmin=0, vmax=1)
    ax.set_title('Metric Correlation: Which Metrics Are Redundant?',
                 fontsize=16, fontweight='bold', pad=20)

    # Add note
    ax.text(3, -0.5, 'High correlation (>0.8) = redundant metrics',
            ha='center', fontsize=11, style='italic')
    ax.text(3, -0.8, 'Low correlation (<0.3) = complementary information',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    save_chart('metric_correlation_heatmap')

# Generate remaining charts 11-15 (simplified versions for length)
def chart_threshold_optimization():
    thresholds = np.linspace(0.1, 0.9, 50)
    cost = 100 * (thresholds - 0.3)**2 + 50

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(thresholds, cost, 'b-', linewidth=3)
    optimal_idx = np.argmin(cost)
    ax.plot(thresholds[optimal_idx], cost[optimal_idx], 'ro', markersize=15)
    ax.axvline(thresholds[optimal_idx], color='red', linestyle='--', alpha=0.5)
    ax.text(thresholds[optimal_idx], cost[optimal_idx]-10,
            f'Optimal: {thresholds[optimal_idx]:.2f}', ha='center', fontsize=12)

    ax.set_xlabel('Decision Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Expected Cost (thousands)', fontsize=14, fontweight='bold')
    ax.set_title('Threshold Optimization: Minimize Business Cost', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_chart('threshold_optimization')

def chart_model_comparison_dashboard():
    models = ['LogReg', 'RF', 'XGBoost', 'SVM', 'NN']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    data = np.array([
        [0.89, 0.85, 0.82, 0.83, 0.91],
        [0.92, 0.90, 0.88, 0.89, 0.95],
        [0.93, 0.91, 0.90, 0.90, 0.96],
        [0.90, 0.87, 0.85, 0.86, 0.93],
        [0.91, 0.88, 0.87, 0.87, 0.94]
    ])

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn', center=0.88,
                xticklabels=metrics, yticklabels=models, cbar_kws={'label': 'Score'},
                vmin=0.8, vmax=1.0, ax=ax)
    ax.set_title('Model Comparison Dashboard: 5 Models × 5 Metrics',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    save_chart('model_comparison_dashboard')

def chart_validation_pipeline():
    fig, ax = plt.subplots(figsize=(12, 8))

    stages = ['Data Split', 'Training', 'Cross-Validation', 'Threshold Opt', 'Final Test', 'Deploy']
    y_pos = np.linspace(0.9, 0.1, len(stages))

    for i, (stage, y) in enumerate(zip(stages, y_pos)):
        rect = FancyBboxPatch((0.1, y-0.05), 0.8, 0.08, boxstyle="round,pad=0.01",
                              edgecolor='black', facecolor=f'C{i}', alpha=0.7, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y, stage, ha='center', va='center', fontsize=14, fontweight='bold')
        if i < len(stages) - 1:
            ax.arrow(0.5, y-0.05, 0, -0.05, head_width=0.05, head_length=0.02,
                    fc='black', ec='black')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Production Validation Pipeline: End-to-End', fontsize=16, fontweight='bold')

    plt.tight_layout()
    save_chart('validation_pipeline')

def chart_business_metric_alignment():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ML metrics
    ml_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    ax1.barh(ml_metrics, [0.89, 0.87, 0.85, 0.86, 0.93], color='skyblue', edgecolor='black')
    ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('ML Metrics', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(True, axis='x', alpha=0.3)

    # Business metrics
    business_metrics = ['Revenue', 'Cost Savings', 'User Satisfaction', 'Time Saved', 'ROI']
    business_values = [85, 70, 90, 75, 95]
    ax2.barh(business_metrics, business_values, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Impact Score', fontsize=12, fontweight='bold')
    ax2.set_title('Business Impact', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Alignment: ML Metrics → Business KPIs', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_chart('business_metric_alignment')

def chart_validation_checklist():
    fig, ax = plt.subplots(figsize=(10, 12))

    categories = [
        ('Technical Validation', ['Multiple metrics calculated', 'Confusion matrix analyzed',
                                 'Threshold optimized', 'Cross-validation performed']),
        ('Statistical Testing', ['Confidence intervals computed', 'Statistical tests passed',
                                'Significance verified', 'Variance assessed']),
        ('Business Alignment', ['ROI projected', 'Cost-benefit done',
                               'Stakeholder approval', 'Deployment plan ready'])
    ]

    y_start = 0.9
    for cat_name, items in categories:
        ax.text(0.5, y_start, cat_name, ha='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        y_start -= 0.08
        for item in items:
            ax.text(0.15, y_start, f'☐ {item}', fontsize=11)
            y_start -= 0.06
        y_start -= 0.04

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Pre-Deployment Validation Checklist', fontsize=16, fontweight='bold')

    plt.tight_layout()
    save_chart('validation_checklist')

def chart_validation_pyramid():
    fig, ax = plt.subplots(figsize=(10, 8))

    levels = [
        ('Business Metrics\n(ROI, Cost, Impact)', 0.8, 'orange'),
        ('Multi-Metric\n(Precision, Recall, F1, AUC)', 0.5, 'lightblue'),
        ('Single Metric\n(Accuracy Only)', 0.2, 'lightcoral')
    ]

    for i, (label, width, color) in enumerate(levels):
        y = i * 0.3
        ax.barh(y, width, height=0.25, color=color, edgecolor='black', linewidth=2)
        ax.text(width/2, y, label, ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1)
    ax.axis('off')
    ax.set_title('The Validation Pyramid: Three Levels of Evaluation',
                 fontsize=16, fontweight='bold')

    # Add annotations
    ax.text(0.9, 0.8, 'Most Important', ha='center', fontsize=10, style='italic')
    ax.text(0.6, 0.5, 'Necessary', ha='center', fontsize=10, style='italic')
    ax.text(0.3, 0.2, 'Insufficient Alone', ha='center', fontsize=10, style='italic', color='red')

    plt.tight_layout()
    save_chart('validation_pyramid')

def chart_validation_depth_decision():
    """Decision tree for determining validation depth"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'When to Use Multi-Metric Validation: Decision Framework',
            ha='center', fontsize=16, fontweight='bold')

    # Root question
    root_box = FancyBboxPatch((5.5, 8.2), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(root_box)
    ax.text(7, 8.6, 'What are the stakes?', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Three main branches
    # Branch 1: High Stakes
    ax.arrow(6.5, 8.2, -2, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    high_box = FancyBboxPatch((2.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor='#F44336', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(high_box)
    ax.text(4, 6.2, 'HIGH STAKES', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    ax.text(4, 5.4, 'Production deployment\nMedical/Financial/Legal\nRegulatory compliance\nCost-asymmetric errors',
            ha='center', va='top', fontsize=9)

    ax.arrow(4, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_high = FancyBboxPatch((2.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                 facecolor='#4CAF50', edgecolor='black', linewidth=2)
    ax.add_patch(result_high)
    ax.text(4, 4.5, 'COMPREHENSIVE', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(4, 4.1, '10+ metrics\nCross-validation\nStatistical tests\nBusiness alignment',
            ha='center', va='center', fontsize=8)

    # Branch 2: Medium Stakes
    ax.arrow(7, 8.2, 0, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    med_box = FancyBboxPatch((5.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='#FF9800', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(med_box)
    ax.text(7, 6.2, 'MEDIUM STAKES', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    ax.text(7, 5.4, 'Business critical\nModerate volume\nSome automation\nReversible decisions',
            ha='center', va='top', fontsize=9)

    ax.arrow(7, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_med = FancyBboxPatch((5.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                facecolor='#FFEB3B', edgecolor='black', linewidth=2)
    ax.add_patch(result_med)
    ax.text(7, 4.5, 'MODERATE', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(7, 4.1, '3-5 key metrics\nBasic validation\nConfusion matrix\nStakeholder review',
            ha='center', va='center', fontsize=8)

    # Branch 3: Low Stakes
    ax.arrow(7.5, 8.2, 2, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    low_box = FancyBboxPatch((8.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='#9E9E9E', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(low_box)
    ax.text(10, 6.2, 'LOW STAKES', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    ax.text(10, 5.4, 'Prototyping\nExploratory analysis\nQuick iterations\nAcademic/research',
            ha='center', va='top', fontsize=9)

    ax.arrow(10, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_low = FancyBboxPatch((8.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                facecolor='#E0E0E0', edgecolor='black', linewidth=2)
    ax.add_patch(result_low)
    ax.text(10, 4.5, 'LIGHTWEIGHT', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(10, 4.1, 'Accuracy + 1-2 metrics\nSimple train/test split\nQuick assessment\nIterate rapidly',
            ha='center', va='center', fontsize=8)

    # Additional considerations box
    consider_box = FancyBboxPatch((0.5, 0.5), 13, 2.5, boxstyle="round,pad=0.1",
                                  facecolor='#F0F0F0', edgecolor='black', linewidth=2)
    ax.add_patch(consider_box)
    ax.text(7, 2.7, 'Additional Considerations', ha='center', va='center',
            fontsize=12, fontweight='bold')

    considerations_text = """
Volume: High volume (1000+/day) → More rigorous validation needed
Regulation: FDA, SEC, GDPR compliance → Mandatory comprehensive validation
Cost Asymmetry: If FN costs >> FP costs → Multi-metric essential (optimize recall vs precision)
Class Imbalance: Severe imbalance (>95:5) → Accuracy alone completely insufficient
Stakeholders: Multiple non-technical stakeholders → Business metric translation critical
Time Constraints: Tight deadlines → May need lightweight first, then iterate to comprehensive
    """
    ax.text(7, 1.5, considerations_text, ha='center', va='center', fontsize=8,
            family='monospace')

    # Bottom principle
    ax.text(7, 0.2, 'Principle: Match validation rigor to decision consequences - comprehensive for production, lightweight for exploration',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    save_chart('validation_depth_decision')

# Generate all charts
print("Generating all 16 charts for Week 9...")
print("=" * 50)

chart_accuracy_trap()
chart_confusion_matrix()
chart_precision_recall_tradeoff()
chart_f_beta_family()
chart_roc_curve()
chart_auc_interpretation()
chart_pr_vs_roc()
chart_multi_class()
chart_cross_validation()
chart_metric_correlation()
chart_threshold_optimization()
chart_model_comparison_dashboard()
chart_validation_pipeline()
chart_business_metric_alignment()
chart_validation_checklist()
chart_validation_pyramid()
chart_validation_depth_decision()

print("=" * 50)
print("All 16 charts generated successfully!")
print("Charts saved in Week_09/charts/ as both PDF and PNG")