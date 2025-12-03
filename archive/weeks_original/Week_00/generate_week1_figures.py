"""
Generate all figures for Week 1: Innovation Foundations
BSc ML for Design Thinking Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Standard settings
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (12, 5)
FIGSIZE_QUAD = (15, 10)
DPI = 300

# Create figures directory
import os
os.makedirs('figures', exist_ok=True)

# Color scheme
colors = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

def save_figure(fig, name):
    """Save figure in PDF format"""
    fig.savefig(f'figures/{name}.pdf', dpi=DPI, bbox_inches='tight')
    print(f"Saved: figures/{name}.pdf")
    plt.close()

# Figure 1: Innovation Failure Rate
def plot_innovation_failure_rate():
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    categories = ['New\nProducts', 'Startups', 'R&D\nProjects', 'Patents\nCommercialized']
    failure_rates = [95, 90, 85, 92]
    success_rates = [5, 10, 15, 8]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, failure_rates, width, label='Failure Rate', color=colors['red'], alpha=0.8)
    bars2 = ax.bar(x + width/2, success_rates, width, label='Success Rate', color=colors['green'], alpha=0.8)
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Innovation Failure Rates Across Domains', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height}%', ha='center', va='bottom', fontsize=10)
    
    save_figure(fig, 'innovation_failure_rate')

# Figure 2: Learning Paradigms Comparison
def plot_learning_paradigms():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    
    # Supervised Learning
    X_sup, y_sup = make_classification(n_samples=100, n_features=2,
                                       n_redundant=0, n_clusters_per_class=1,
                                       n_classes=2, random_state=42)
    axes[0].scatter(X_sup[:, 0], X_sup[:, 1], c=y_sup, cmap='viridis', 
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0].set_title('Supervised Learning\n(With Labels)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature 1', fontsize=12)
    axes[0].set_ylabel('Feature 2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Unsupervised Learning
    X_unsup, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    axes[1].scatter(X_unsup[:, 0], X_unsup[:, 1], alpha=0.7, s=50,
                   edgecolors='black', linewidth=0.5, color=colors['gray'])
    axes[1].set_title('Unsupervised Learning\n(No Labels)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature 1', fontsize=12)
    axes[1].set_ylabel('Feature 2', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Learning Paradigms in Machine Learning', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'learning_paradigms')

# Figure 3: Loss Functions Comparison
def plot_loss_functions():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    
    # Supervised loss over iterations
    iterations = np.arange(0, 100)
    loss_supervised = 10 * np.exp(-iterations/20) + np.random.normal(0, 0.1, 100)
    
    axes[0].plot(iterations, loss_supervised, color=colors['blue'], linewidth=2)
    axes[0].fill_between(iterations, loss_supervised - 0.5, loss_supervised + 0.5, 
                         alpha=0.2, color=colors['blue'])
    axes[0].set_xlabel('Iterations', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Supervised Learning Loss\n(MSE Decreasing)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 12)
    
    # Unsupervised objective (inertia)
    k_values = np.arange(1, 11)
    inertia = 1000 / k_values + np.random.normal(0, 20, 10)
    
    axes[1].plot(k_values, inertia, 'o-', color=colors['orange'], linewidth=2, markersize=8)
    axes[1].axvline(x=4, color=colors['red'], linestyle='--', label='Optimal k')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Inertia', fontsize=12)
    axes[1].set_title('Unsupervised Learning\n(Elbow Method)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'loss_functions_comparison')

# Figure 4: Design-ML Integration
def plot_design_ml_integration():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Design thinking stages
    stages = ['Empathize', 'Define', 'Ideate', 'Prototype', 'Test']
    ml_methods = ['Unsupervised\nClustering', 'Supervised\nClassification', 
                  'Generative AI\nCreation', 'Supervised\nPrediction', 
                  'Both Methods\nAnalysis']
    
    # Create circular flow
    theta = np.linspace(0, 2*np.pi, len(stages)+1)[:-1]
    r = 3
    
    for i, (stage, method, t) in enumerate(zip(stages, ml_methods, theta)):
        x, y = r * np.cos(t), r * np.sin(t)
        
        # Draw stage circle
        circle = plt.Circle((x, y), 0.8, color=plt.cm.Set3(i), alpha=0.6)
        ax.add_patch(circle)
        
        # Add labels
        ax.text(x, y+0.1, stage, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x, y-0.2, method, ha='center', va='center', fontsize=9, style='italic')
        
        # Draw arrows to next stage
        next_i = (i + 1) % len(stages)
        next_t = theta[next_i]
        next_x, next_y = r * np.cos(next_t), r * np.sin(next_t)
        
        # Calculate arrow position
        dx, dy = next_x - x, next_y - y
        length = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/length, dy/length
        
        ax.arrow(x + 0.8*dx, y + 0.8*dy, 
                (length-1.6)*dx, (length-1.6)*dy,
                head_width=0.15, head_length=0.1, fc='gray', ec='gray', alpha=0.7)
    
    # Add center text
    ax.text(0, 0, 'ML-Enhanced\nDesign\nThinking', ha='center', va='center', 
           fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='white', edgecolor='black'))
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Integration of ML Methods with Design Thinking Process', 
                fontsize=16, fontweight='bold', pad=20)
    
    save_figure(fig, 'design_ml_integration')

# Figure 5: Learning Objectives
def plot_learning_objectives():
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    categories = ['Theory\nUnderstanding', 'Practical\nSkills', 'Integration\nAbility', 
                 'Problem\nSolving', 'Innovation\nCapacity']
    current = [20, 15, 10, 25, 20]
    target = [90, 85, 95, 88, 92]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, current, width, label='Week 1 Start', color=colors['red'], alpha=0.7)
    bars2 = ax.bar(x + width/2, target, width, label='Week 1 Target', color=colors['green'], alpha=0.7)
    
    ax.set_ylabel('Competency Level (%)', fontsize=12)
    ax.set_title('Week 1 Learning Objectives', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    save_figure(fig, 'learning_objectives')

# Figure 6: Decision Boundary
def plot_decision_boundary():
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                              n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Scale for better visualization
    X[:, 0] = X[:, 0] * 2 + 5  # Innovation score (0-10 scale)
    X[:, 1] = X[:, 1] * 2 + 5  # Feasibility score (0-10 scale)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Create mesh
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                        edgecolor='black', s=50, alpha=0.8)
    
    ax.set_xlabel('Innovation Score', fontsize=12)
    ax.set_ylabel('Feasibility Score', fontsize=12)
    ax.set_title('Supervised Learning: Innovation Success Prediction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.6, label='Predicted Failure'),
                      Patch(facecolor='blue', alpha=0.6, label='Predicted Success')]
    ax.legend(handles=legend_elements, loc='upper left')
    
    save_figure(fig, 'decision_boundary')

# Figure 7: Clustering Results
def plot_clustering_results():
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # Generate data
    X, y_true = make_blobs(n_samples=300, centers=5, n_features=2,
                          cluster_std=0.8, random_state=42)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # Plot clusters
    unique_labels = set(labels)
    colors_list = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors_list):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], s=50, c=[col],
                  marker='o', edgecolor='black', alpha=0.6,
                  label=f'Pattern {k+1}')
    
    # Plot centers
    ax.scatter(centers[:, 0], centers[:, 1], s=300, c='red',
              marker='*', edgecolor='black', linewidth=2,
              label='Cluster Centers', zorder=10)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Unsupervised Learning: Innovation Pattern Discovery', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'clustering_results')

# Figure 8: Performance Comparison
def plot_performance_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['Human\nOnly', 'ML\nAssisted', 'Full\nIntegration']
    ideas_generated = [30, 300, 3000]
    success_rate = [5, 25, 35]
    time_hours = [40, 10, 4]
    
    # Ideas Generated
    bars1 = axes[0].bar(methods, ideas_generated, color=colors['blue'], alpha=0.7)
    axes[0].set_ylabel('Ideas Generated', fontsize=12)
    axes[0].set_title('Quantity', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, ideas_generated):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    str(val), ha='center', fontsize=10)
    
    # Success Rate
    bars2 = axes[1].bar(methods, success_rate, color=colors['green'], alpha=0.7)
    axes[1].set_ylabel('Success Rate (%)', fontsize=12)
    axes[1].set_title('Quality', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, success_rate):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val}%', ha='center', fontsize=10)
    
    # Time Investment
    bars3 = axes[2].bar(methods, time_hours, color=colors['orange'], alpha=0.7)
    axes[2].set_ylabel('Time (hours)', fontsize=12)
    axes[2].set_title('Speed', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, time_hours):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}h', ha='center', fontsize=10)
    
    plt.suptitle('Innovation Performance: Traditional vs ML-Enhanced', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'performance_comparison')

# Figure 9: Learning Curves
def plot_learning_curves():
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # Generate synthetic learning curve data
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores_mean = 0.6 + 0.35 * train_sizes
    train_scores_std = 0.1 - 0.08 * train_sizes
    val_scores_mean = 0.5 + 0.25 * train_sizes + 0.05 * np.sin(train_sizes * 10)
    val_scores_std = 0.15 - 0.1 * train_sizes
    
    # Plot
    ax.plot(train_sizes, train_scores_mean, 'o-', color=colors['blue'],
           label='Training score', linewidth=2, markersize=8)
    ax.fill_between(train_sizes,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.2, color=colors['blue'])
    
    ax.plot(train_sizes, val_scores_mean, 'o-', color=colors['red'],
           label='Validation score', linewidth=2, markersize=8)
    ax.fill_between(train_sizes,
                    val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std,
                    alpha=0.2, color=colors['red'])
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_title('Learning Curves: Model Performance vs Training Size', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)
    
    save_figure(fig, 'learning_curves')

# Figure 10: Case Study Results
def plot_case_study_results():
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_QUAD)
    
    # Before/After clustering
    np.random.seed(42)
    X = np.random.randn(1000, 2) * 2
    
    axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.3, s=20, color=colors['gray'])
    axes[0, 0].set_title('Before: 1000+ Unorganized Ideas', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Dimension 1')
    axes[0, 0].set_ylabel('Dimension 2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # After clustering
    kmeans = KMeans(n_clusters=8, random_state=42)
    labels = kmeans.fit_predict(X)
    
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.5, s=20)
    axes[0, 1].set_title('After: 8 Clear Categories', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Dimension 1')
    axes[0, 1].set_ylabel('Dimension 2')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Success probability distribution
    success_probs = np.random.beta(2, 5, 1000)
    axes[1, 0].hist(success_probs, bins=30, color=colors['green'], alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0.3, color='red', linestyle='--', label='Threshold')
    axes[1, 0].set_xlabel('Success Probability')
    axes[1, 0].set_ylabel('Number of Ideas')
    axes[1, 0].set_title('ML-Predicted Success Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # ROI comparison
    categories = ['Traditional', 'ML-Enhanced']
    roi = [150, 750]
    bars = axes[1, 1].bar(categories, roi, color=[colors['orange'], colors['green']], alpha=0.7)
    axes[1, 1].set_ylabel('ROI (%)')
    axes[1, 1].set_title('Return on Investment', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, roi):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                       f'{val}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Case Study: Product Innovation Pipeline Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'case_study_results')

# Figure 11: Integration Challenge
def plot_integration_challenge():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pipeline flow
    stages = [
        ('Input:\nIdeas', 0.5, 4, colors['gray']),
        ('Generate\n(Gen AI)', 2, 5, colors['purple']),
        ('Organize\n(Unsupervised)', 4, 5, colors['orange']),
        ('Predict\n(Supervised)', 6, 5, colors['blue']),
        ('Filter\n(Threshold)', 8, 5, colors['green']),
        ('Output:\nTop Ideas', 9.5, 4, colors['cyan'])
    ]
    
    # Draw boxes and connections
    for i, (label, x, y, color) in enumerate(stages):
        # Draw box
        rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Draw arrow to next stage
        if i < len(stages) - 1:
            next_x = stages[i+1][1]
            ax.arrow(x+0.4, y, next_x-x-0.8, 0, 
                    head_width=0.15, head_length=0.1, fc='gray', ec='gray')
    
    # Add data flow annotations
    annotations = [
        (1.25, 3.5, '100 ideas'),
        (3, 3.5, '1000 ideas'),
        (5, 3.5, '8 clusters'),
        (7, 3.5, 'probabilities'),
        (8.75, 3.5, 'top 20%')
    ]
    
    for x, y, text in annotations:
        ax.text(x, y, text, ha='center', fontsize=9, style='italic', color='gray')
    
    # Add feedback loop
    ax.annotate('', xy=(2, 4.5), xytext=(8, 4.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                             color='red', linestyle='--', alpha=0.5))
    ax.text(5, 3, 'Feedback Loop', ha='center', fontsize=9, 
           color='red', style='italic')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 6)
    ax.axis('off')
    ax.set_title('Integration Challenge: Complete ML Innovation Pipeline', 
                fontsize=16, fontweight='bold', pad=20)
    
    save_figure(fig, 'integration_challenge')

# Figure 12: Problem Evolution
def plot_problem_evolution():
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Problem evolution stages
    stages = [
        ('Week 1:\nGenerate\n5000 ideas', 2, 2, colors['green']),
        ('Problem:\nToo many!\nChaos!', 5, 2, colors['red']),
        ('Week 2:\nNeed\nOrganization', 8, 2, colors['blue'])
    ]
    
    for i, (text, x, y, color) in enumerate(stages):
        # Draw circle
        circle = plt.Circle((x, y), 0.8, color=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Draw arrow
        if i < len(stages) - 1:
            ax.arrow(x + 0.9, y, 1.3, 0, head_width=0.2, head_length=0.15,
                    fc='gray', ec='gray', alpha=0.7)
    
    # Add annotations
    ax.text(3.5, 1, 'Success creates\nnew challenge', ha='center', fontsize=9, style='italic')
    ax.text(6.5, 1, 'Complexity requires\nnew solution', ha='center', fontsize=9, style='italic')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Problem Evolution: How Success Creates New Challenges', 
                fontsize=16, fontweight='bold', pad=20)
    
    save_figure(fig, 'problem_evolution_w1')

# Figure 13: Chaos to Order Preview
def plot_chaos_to_order_preview():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    
    # Before - Chaos
    np.random.seed(42)
    X = np.random.randn(500, 2) * 3
    axes[0].scatter(X[:, 0], X[:, 1], alpha=0.3, s=30, color=colors['gray'])
    axes[0].set_title('Current State: 5000 Ideas in Chaos', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].grid(True, alpha=0.3)
    
    # After - Organized (preview)
    kmeans = KMeans(n_clusters=8, random_state=42)
    labels = kmeans.fit_predict(X)
    axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.6, s=30)
    axes[1].set_title('Week 2 Preview: Organized into Themes', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Next Week: K-Means Clustering Solution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'chaos_to_order_preview')

# Figure 14: Metrics Comparison
def plot_metrics_comparison():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    
    # Classification metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [0.82, 0.85, 0.78, 0.81]
    
    bars = axes[0].barh(metrics, values, color=colors['blue'], alpha=0.7)
    axes[0].set_xlabel('Score')
    axes[0].set_title('Classification Metrics', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        axes[0].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=10)
    
    # Clustering metrics
    metrics = ['Silhouette', 'Davies-Bouldin\n(inverted)', 'Calinski\n(normalized)']
    values = [0.72, 0.75, 0.68]  # Note: DB inverted for consistency
    
    bars = axes[1].barh(metrics, values, color=colors['orange'], alpha=0.7)
    axes[1].set_xlabel('Score')
    axes[1].set_title('Clustering Metrics', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=10)
    
    plt.suptitle('Model Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'metrics_comparison')

# Main execution
if __name__ == "__main__":
    print("Generating Week 1 figures...")
    print("-" * 50)
    
    # Generate all figures
    plot_innovation_failure_rate()
    plot_learning_paradigms()
    plot_loss_functions()
    plot_design_ml_integration()
    plot_learning_objectives()
    plot_decision_boundary()
    plot_clustering_results()
    plot_performance_comparison()
    plot_learning_curves()
    plot_case_study_results()
    plot_integration_challenge()
    plot_problem_evolution()
    plot_chaos_to_order_preview()
    plot_metrics_comparison()
    
    print("-" * 50)
    print("All figures generated successfully!")
    print(f"Total figures created: 14")
    print("Location: figures/ directory")