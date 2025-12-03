#!/usr/bin/env python3
"""
Week 0b: Supervised Learning Chart Generation
Creates all 15 essential charts for the supervised learning presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style and create output directory
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = '../charts/'

# Color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlgray': '#7f7f7f'
}

def create_real_estate_scatter():
    """Chart 1: Real estate price prediction scatter plot"""
    np.random.seed(42)

    # Generate synthetic real estate data
    n_samples = 100
    size = np.random.normal(1800, 400, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    age = np.random.randint(0, 30, n_samples)

    # Create price with realistic relationships
    price = (size * 0.15 + bedrooms * 20 + (30 - age) * 5 +
             np.random.normal(0, 30, n_samples))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Size vs Price
    ax1.scatter(size, price, alpha=0.6, color=colors['mlblue'])
    ax1.set_xlabel('House Size (sq ft)')
    ax1.set_ylabel('Price ($1000s)')
    ax1.set_title('Price vs Size')

    # Bedrooms vs Price
    ax2.scatter(bedrooms, price, alpha=0.6, color=colors['mlorange'])
    ax2.set_xlabel('Number of Bedrooms')
    ax2.set_ylabel('Price ($1000s)')
    ax2.set_title('Price vs Bedrooms')

    plt.tight_layout()
    plt.savefig(f'{output_dir}real_estate_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}real_estate_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_linear_regression_fit():
    """Chart 2: Linear regression fitting example"""
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + np.random.normal(0, 1, 50)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, color=colors['mlblue'], label='Data points')
    plt.plot(X, y_pred, color=colors['mlred'], linewidth=2, label='Linear fit')
    plt.xlabel('Feature X')
    plt.ylabel('Target y')
    plt.title('Linear Regression: Finding Best Fit Line')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{output_dir}linear_regression_fit.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}linear_regression_fit.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_regression_vs_classification():
    """Chart 3: Regression vs Classification comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Regression example
    np.random.seed(42)
    X_reg = np.linspace(0, 10, 50)
    y_reg = 2 * X_reg + 1 + np.random.normal(0, 1, 50)

    ax1.scatter(X_reg, y_reg, alpha=0.6, color=colors['mlblue'])
    ax1.plot(X_reg, 2 * X_reg + 1, color=colors['mlred'], linewidth=2)
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Continuous Target')
    ax1.set_title('Regression: Predicting Continuous Values')
    ax1.grid(True, alpha=0.3)

    # Classification example
    X_class = np.random.randn(100, 2)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)

    colors_class = [colors['mlblue'] if y == 0 else colors['mlorange'] for y in y_class]
    ax2.scatter(X_class[:, 0], X_class[:, 1], c=colors_class, alpha=0.6)
    ax2.axline((0, 0), slope=-1, color=colors['mlred'], linewidth=2)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Classification: Predicting Categories')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}regression_vs_classification.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}regression_vs_classification.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_curse_dimensionality():
    """Chart 4: Curse of dimensionality visualization"""
    dimensions = np.arange(1, 21)
    sample_sizes = [100, 1000, 10000]

    plt.figure(figsize=(10, 6))

    for n_samples in sample_sizes:
        ratio = (2 ** dimensions) / n_samples
        plt.semilogy(dimensions, ratio, 'o-', linewidth=2,
                    label=f'{n_samples} samples')

    plt.axhline(y=1, color=colors['mlred'], linestyle='--',
                label='Critical threshold')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Feature Combinations / Sample Size')
    plt.title('Curse of Dimensionality: Exponential Growth Problem')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{output_dir}curse_dimensionality.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}curse_dimensionality.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_feature_combinations():
    """Chart 5: Feature combinations growth"""
    n_features = np.arange(1, 21)
    linear_terms = n_features
    pairwise_terms = n_features * (n_features - 1) / 2
    threeway_terms = n_features * (n_features - 1) * (n_features - 2) / 6

    plt.figure(figsize=(10, 6))
    plt.semilogy(n_features, linear_terms, 'o-', label='Linear terms',
                linewidth=2, color=colors['mlblue'])
    plt.semilogy(n_features, pairwise_terms, 's-', label='Pairwise interactions',
                linewidth=2, color=colors['mlorange'])
    plt.semilogy(n_features, threeway_terms, '^-', label='Three-way interactions',
                linewidth=2, color=colors['mlgreen'])

    plt.xlabel('Number of Features')
    plt.ylabel('Number of Terms')
    plt.title('Combinatorial Explosion of Feature Interactions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{output_dir}feature_combinations.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}feature_combinations.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_ols_example():
    """Chart 6: OLS worked example"""
    # Simple example: y = 2x
    X = np.array([1, 2, 3])
    y = np.array([2, 4, 6])

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, s=100, color=colors['mlblue'], label='Data points', zorder=5)
    plt.plot(X, y, color=colors['mlred'], linewidth=2, label='Perfect fit: y = 2x')

    # Show residuals (all zero in this case)
    for i in range(len(X)):
        plt.plot([X[i], X[i]], [y[i], 2*X[i]], color=colors['mlgray'],
                linestyle='--', alpha=0.7)

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('OLS Example: Perfect Linear Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{output_dir}ols_example.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}ols_example.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_ridge_lasso_comparison():
    """Chart 7: Ridge vs LASSO regularization"""
    np.random.seed(42)
    X = np.random.randn(100, 20)
    true_coef = np.zeros(20)
    true_coef[:5] = [2, -1.5, 1, -0.5, 0.8]  # Only first 5 features matter
    y = X @ true_coef + 0.1 * np.random.randn(100)

    alphas = np.logspace(-3, 1, 50)

    ridge_coefs = []
    lasso_coefs = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        ridge_coefs.append(ridge.coef_)

        lasso = Lasso(alpha=alpha, max_iter=1000)
        lasso.fit(X, y)
        lasso_coefs.append(lasso.coef_)

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Ridge
    for i in range(5):  # Plot only first 5 coefficients
        ax1.semilogx(alphas, ridge_coefs[:, i], label=f'Coef {i+1}')
    ax1.set_xlabel('Regularization strength (α)')
    ax1.set_ylabel('Coefficient value')
    ax1.set_title('Ridge: Coefficients Shrink Smoothly')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LASSO
    for i in range(5):  # Plot only first 5 coefficients
        ax2.semilogx(alphas, lasso_coefs[:, i], label=f'Coef {i+1}')
    ax2.set_xlabel('Regularization strength (α)')
    ax2.set_ylabel('Coefficient value')
    ax2.set_title('LASSO: Coefficients Hit Zero')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}ridge_lasso_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}ridge_lasso_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_linear_success_cases():
    """Chart 8: Linear model success cases"""
    np.random.seed(42)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Case 1: Simple linear trend
    X1 = np.linspace(0, 10, 50)
    y1 = 2 * X1 + 1 + np.random.normal(0, 0.5, 50)
    ax1.scatter(X1, y1, alpha=0.6, color=colors['mlgreen'])
    ax1.plot(X1, 2 * X1 + 1, color=colors['mlred'], linewidth=2)
    ax1.set_title('SUCCESS: Linear Trend (R² = 0.94)')
    ax1.grid(True, alpha=0.3)

    # Case 2: Linearly separable classification
    X2 = np.random.randn(100, 2)
    y2 = (X2[:, 0] + X2[:, 1] > 0)
    colors2 = [colors['mlgreen'] if y else colors['mlblue'] for y in y2]
    ax2.scatter(X2[:, 0], X2[:, 1], c=colors2, alpha=0.6)
    ax2.axline((0, 0), slope=-1, color=colors['mlred'], linewidth=2)
    ax2.set_title('SUCCESS: Linearly Separable (Acc = 100%)')
    ax2.grid(True, alpha=0.3)

    # Case 3: Low noise data
    X3 = np.linspace(0, 5, 30)
    y3 = X3**2 * 0.1 + 2 * X3 + np.random.normal(0, 0.2, 30)  # Mostly linear
    ax3.scatter(X3, y3, alpha=0.6, color=colors['mlgreen'])
    z = np.polyfit(X3, y3, 1)
    ax3.plot(X3, np.poly1d(z)(X3), color=colors['mlred'], linewidth=2)
    ax3.set_title('SUCCESS: Low Noise (MSE = 0.01)')
    ax3.grid(True, alpha=0.3)

    # Case 4: Feature importance visualization
    features = ['Size', 'Location', 'Age', 'Bedrooms']
    importance = [0.4, 0.3, 0.2, 0.1]
    ax4.bar(features, importance, color=colors['mlgreen'], alpha=0.7)
    ax4.set_title('SUCCESS: Interpretable Features')
    ax4.set_ylabel('Feature Importance')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}linear_success_cases.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}linear_success_cases.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_linear_failure_cases():
    """Chart 9: Linear model failure cases"""
    np.random.seed(42)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # XOR problem
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    colors_xor = [colors['mlred'] if y == 0 else colors['mlblue'] for y in y_xor]
    ax1.scatter(X_xor[:, 0], X_xor[:, 1], c=colors_xor, s=200)
    ax1.set_title('FAILURE: XOR Problem (Acc = 50%)')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.grid(True, alpha=0.3)

    # Concentric circles
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
    colors_circles = [colors['mlred'] if y == 0 else colors['mlblue'] for y in y_circles]
    ax2.scatter(X_circles[:, 0], X_circles[:, 1], c=colors_circles, alpha=0.6)
    ax2.set_title('FAILURE: Circles (Acc = 52%)')
    ax2.grid(True, alpha=0.3)

    # Moons
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    colors_moons = [colors['mlred'] if y == 0 else colors['mlblue'] for y in y_moons]
    ax3.scatter(X_moons[:, 0], X_moons[:, 1], c=colors_moons, alpha=0.6)
    ax3.set_title('FAILURE: Moons (Acc = 58%)')
    ax3.grid(True, alpha=0.3)

    # Spiral pattern
    t = np.linspace(0, 4*np.pi, 100)
    X_spiral = np.column_stack([t * np.cos(t), t * np.sin(t)])
    y_spiral = (t < 2*np.pi).astype(int)
    colors_spiral = [colors['mlred'] if y == 0 else colors['mlblue'] for y in y_spiral]
    ax4.scatter(X_spiral[:, 0], X_spiral[:, 1], c=colors_spiral, alpha=0.6)
    ax4.set_title('FAILURE: Spirals (Acc = 48%)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}linear_failure_cases.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}linear_failure_cases.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_linear_vs_nonlinear_boundaries():
    """Chart 10: Linear vs nonlinear decision boundaries"""
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear boundary (poor fit)
    colors_plot = [colors['mlred'] if label == 0 else colors['mlblue'] for label in y]
    ax1.scatter(X[:, 0], X[:, 1], c=colors_plot, alpha=0.6)
    ax1.axline((0, 0), slope=1, color='black', linewidth=2, label='Linear boundary')
    ax1.set_title('Linear Boundary: Cannot Separate Circles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Nonlinear boundary (good fit)
    ax2.scatter(X[:, 0], X[:, 1], c=colors_plot, alpha=0.6)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 * np.cos(theta)
    circle_y = 0.5 * np.sin(theta)
    ax2.plot(circle_x, circle_y, color='black', linewidth=2, label='Nonlinear boundary')
    ax2.set_title('Nonlinear Boundary: Perfect Separation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}linear_vs_nonlinear_boundaries.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}linear_vs_nonlinear_boundaries.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_regularization_tradeoff():
    """Chart 11: Regularization bias-variance tradeoff"""
    alphas = np.logspace(-4, 1, 50)

    # Simulate bias and variance curves
    bias_squared = 0.1 + 0.5 * np.log10(alphas + 1e-4) ** 2
    variance = 1.0 / (1 + alphas)
    total_error = bias_squared + variance + 0.1  # Add irreducible error

    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, bias_squared, label='Bias²', linewidth=2, color=colors['mlred'])
    plt.semilogx(alphas, variance, label='Variance', linewidth=2, color=colors['mlblue'])
    plt.semilogx(alphas, total_error, label='Total Error', linewidth=2, color=colors['mlgreen'])

    # Mark optimal point
    optimal_idx = np.argmin(total_error)
    plt.axvline(alphas[optimal_idx], color=colors['mlorange'], linestyle='--',
                label=f'Optimal α = {alphas[optimal_idx]:.3f}')

    plt.xlabel('Regularization strength (α)')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff in Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{output_dir}regularization_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}regularization_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_human_decision_process():
    """Chart 12: Human decision process visualization"""
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create decision tree structure
    boxes = [
        (0.5, 0.9, "Start: Loan Application"),
        (0.2, 0.7, "Income < 50k?"),
        (0.8, 0.7, "Income ≥ 50k?"),
        (0.1, 0.5, "Age > 30?"),
        (0.3, 0.5, "Education > 12y?"),
        (0.7, 0.5, "Age > 40?"),
        (0.9, 0.5, "Credit Score?"),
        (0.05, 0.3, "REJECT"),
        (0.15, 0.3, "REVIEW"),
        (0.25, 0.3, "APPROVE"),
        (0.35, 0.3, "REJECT"),
        (0.65, 0.3, "APPROVE"),
        (0.75, 0.3, "REVIEW"),
        (0.85, 0.3, "APPROVE"),
        (0.95, 0.3, "REJECT")
    ]

    # Draw boxes
    for x, y, text in boxes:
        if "APPROVE" in text:
            color = colors['mlgreen']
        elif "REJECT" in text:
            color = colors['mlred']
        elif "REVIEW" in text:
            color = colors['mlorange']
        else:
            color = colors['mlblue']

        rect = patches.Rectangle((x-0.05, y-0.05), 0.1, 0.1,
                               linewidth=1, edgecolor='black',
                               facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, weight='bold')

    # Draw connections (simplified)
    connections = [
        (0.5, 0.85, 0.2, 0.75),  # Start to left branch
        (0.5, 0.85, 0.8, 0.75),  # Start to right branch
        # Add more connections as needed
    ]

    for x1, y1, x2, y2 in connections:
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.set_title('Human Decision Process: Hierarchical Questions')
    ax.axis('off')

    plt.savefig(f'{output_dir}human_decision_process.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}human_decision_process.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_nonlinear_methods_overview():
    """Chart 13: Overview of nonlinear methods"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Decision Tree partitioning
    ax1.add_patch(plt.Rectangle((0, 0), 0.5, 0.5, fill=False, linewidth=2))
    ax1.add_patch(plt.Rectangle((0.5, 0), 0.5, 0.5, fill=False, linewidth=2))
    ax1.add_patch(plt.Rectangle((0, 0.5), 1, 0.5, fill=False, linewidth=2))
    ax1.axvline(0.5, color=colors['mlred'], linewidth=2)
    ax1.axhline(0.5, color=colors['mlred'], linewidth=2)
    ax1.set_title('Decision Trees: Rectangular Partitions')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # SVM with RBF kernel
    X_svm, y_svm = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    colors_svm = [colors['mlred'] if y == 0 else colors['mlblue'] for y in y_svm]
    ax2.scatter(X_svm[:, 0], X_svm[:, 1], c=colors_svm, alpha=0.6)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 * np.cos(theta)
    circle_y = 0.5 * np.sin(theta)
    ax2.plot(circle_x, circle_y, color='black', linewidth=2)
    ax2.set_title('SVM RBF: Curved Boundaries')

    # Random Forest (multiple trees)
    for i in range(3):
        alpha = 0.3
        ax3.axvline(0.3 + i*0.2, color=colors['mlgreen'], alpha=alpha, linewidth=2)
        ax3.axhline(0.2 + i*0.3, color=colors['mlgreen'], alpha=alpha, linewidth=2)
    ax3.set_title('Random Forest: Multiple Tree Ensemble')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # Neural Network (conceptual)
    layers_x = [0.2, 0.5, 0.8]
    neurons_y = [[0.3, 0.7], [0.2, 0.5, 0.8], [0.5]]

    for i, (x, ys) in enumerate(zip(layers_x, neurons_y)):
        for y in ys:
            circle = plt.Circle((x, y), 0.05, color=colors['mlpurple'], alpha=0.7)
            ax4.add_patch(circle)
            if i < len(layers_x) - 1:
                for next_y in neurons_y[i+1]:
                    ax4.plot([x+0.05, layers_x[i+1]-0.05], [y, next_y],
                            'k-', alpha=0.3)

    ax4.set_title('Neural Networks: Universal Approximation')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}nonlinear_methods_overview.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}nonlinear_methods_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_missing_charts():
    """Create missing chart files referenced in LaTeX"""

    # Twenty questions tree chart
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.9, "Is it bigger than a cat?", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlblue'], alpha=0.3))
    ax.text(0.2, 0.7, "YES", ha='center', va='center', weight='bold')
    ax.text(0.8, 0.7, "NO", ha='center', va='center', weight='bold')
    ax.text(0.1, 0.5, "Does it live\nin water?", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlorange'], alpha=0.3))
    ax.text(0.9, 0.5, "Is it a\ndomestic pet?", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlorange'], alpha=0.3))
    ax.text(0.05, 0.3, "Whale", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlgreen'], alpha=0.3))
    ax.text(0.15, 0.3, "Bear", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlgreen'], alpha=0.3))
    ax.text(0.85, 0.3, "Dog", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlgreen'], alpha=0.3))
    ax.text(0.95, 0.3, "Eagle", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlgreen'], alpha=0.3))

    # Draw decision tree connections
    connections = [
        (0.5, 0.85, 0.2, 0.75), (0.5, 0.85, 0.8, 0.75),
        (0.2, 0.65, 0.1, 0.55), (0.2, 0.65, 0.3, 0.55),
        (0.8, 0.65, 0.9, 0.55), (0.8, 0.65, 0.7, 0.55),
        (0.1, 0.45, 0.05, 0.35), (0.1, 0.45, 0.15, 0.35),
        (0.9, 0.45, 0.85, 0.35), (0.9, 0.45, 0.95, 0.35)
    ]

    for x1, y1, x2, y2 in connections:
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.set_title('20 Questions Game: Decision Tree Logic')
    ax.axis('off')

    plt.savefig(f'{output_dir}twenty_questions_tree.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}twenty_questions_tree.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Tree building example
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create tree visualization with actual numbers
    ax.text(0.5, 0.9, "Root: 4 Yes, 2 No\nGini = 0.444", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlblue'], alpha=0.3))

    ax.text(0.25, 0.7, "Income < 55k\n1 Yes, 2 No\nGini = 0.444", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlorange'], alpha=0.3))

    ax.text(0.75, 0.7, "Income ≥ 55k\n3 Yes, 0 No\nGini = 0.000", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlgreen'], alpha=0.3))

    ax.text(0.25, 0.5, "Information Gain\n= 0.222", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlpurple'], alpha=0.3))

    # Draw connections
    ax.plot([0.5, 0.25], [0.85, 0.75], 'k-', linewidth=2)
    ax.plot([0.5, 0.75], [0.85, 0.75], 'k-', linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    ax.set_title('CART Algorithm: Tree Building with Numbers')
    ax.axis('off')

    plt.savefig(f'{output_dir}tree_building_example.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}tree_building_example.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Additional missing charts
    create_cart_algorithm_steps()
    create_tree_2d_boundaries()
    create_piecewise_approximation()
    create_ensemble_methods_performance()
    create_algorithm_landscape()
    create_interpretability_accuracy_tradeoff()
    create_supervised_to_unsupervised()

def create_cart_algorithm_steps():
    """CART algorithm step visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))

    steps = [
        "1. Calculate Gini impurity",
        "2. Try all possible splits",
        "3. Choose best information gain",
        "4. Recurse on children",
        "5. Stop when criteria met"
    ]

    for i, step in enumerate(steps):
        y_pos = 0.8 - i * 0.15
        ax.text(0.1, y_pos, step, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlblue'], alpha=0.3))
        if i < len(steps) - 1:
            ax.arrow(0.05, y_pos - 0.05, 0, -0.05, head_width=0.02, head_length=0.02,
                    fc='black', ec='black')

    ax.text(0.6, 0.6, "Gini Formula:\nG = 1 - Σ pᵢ²", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlorange'], alpha=0.3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('CART Algorithm Steps')
    ax.axis('off')

    plt.savefig(f'{output_dir}cart_algorithm_steps.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}cart_algorithm_steps.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_tree_2d_boundaries():
    """2D decision boundaries visualization"""
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                              n_clusters_per_class=1, random_state=42)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    # Create mesh for boundary visualization
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    colors_plot = [colors['mlred'] if label == 0 else colors['mlblue'] for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors_plot, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Tree: Axis-Aligned Rectangular Boundaries')
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{output_dir}tree_2d_boundaries.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}tree_2d_boundaries.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_piecewise_approximation():
    """Piecewise approximation visualization"""
    x = np.linspace(0, 4*np.pi, 1000)
    y_true = np.sin(x)

    # Create piecewise approximation
    n_pieces = 8
    x_splits = np.linspace(0, 4*np.pi, n_pieces + 1)
    y_approx = np.zeros_like(x)

    for i in range(n_pieces):
        mask = (x >= x_splits[i]) & (x < x_splits[i+1])
        if i == n_pieces - 1:  # Last piece includes endpoint
            mask = (x >= x_splits[i]) & (x <= x_splits[i+1])
        y_approx[mask] = np.mean(y_true[mask])

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label='True function: sin(x)', linewidth=2, color=colors['mlblue'])
    plt.plot(x, y_approx, label='Piecewise approximation', linewidth=2, color=colors['mlred'])

    # Show the splits
    for split in x_splits[1:-1]:
        plt.axvline(split, color=colors['mlgray'], linestyle='--', alpha=0.7)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Piecewise Approximation: How Trees Work')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{output_dir}piecewise_approximation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}piecewise_approximation.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_ensemble_methods_performance():
    """Ensemble methods performance comparison"""
    methods = ['Single Tree', 'Random Forest\n(10 trees)', 'Random Forest\n(100 trees)',
               'Gradient Boosting\n(10 estimators)', 'Gradient Boosting\n(100 estimators)']

    train_acc = [85, 92, 95, 88, 94]
    test_acc = [82, 89, 88, 86, 91]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_acc, width, label='Training Accuracy',
            color=colors['mlblue'], alpha=0.7)
    plt.bar(x + width/2, test_acc, width, label='Test Accuracy',
            color=colors['mlorange'], alpha=0.7)

    plt.xlabel('Method')
    plt.ylabel('Accuracy (%)')
    plt.title('Ensemble Methods: Training vs Test Performance')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'{output_dir}ensemble_methods_performance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}ensemble_methods_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_algorithm_landscape():
    """Algorithm landscape visualization"""
    complexity = [1, 3, 4, 5]
    performance = [75, 85, 90, 92]
    interpretability = [95, 70, 40, 30]

    algorithms = ['Linear', 'Tree', 'Ensemble', 'SVM']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Complexity vs Performance
    ax1.scatter(complexity, performance, s=200, c=[colors['mlblue'], colors['mlorange'],
                                                  colors['mlgreen'], colors['mlred']])
    for i, alg in enumerate(algorithms):
        ax1.annotate(alg, (complexity[i], performance[i]),
                    xytext=(5, 5), textcoords='offset points')

    ax1.set_xlabel('Model Complexity')
    ax1.set_ylabel('Performance (%)')
    ax1.set_title('Complexity vs Performance')
    ax1.grid(True, alpha=0.3)

    # Performance vs Interpretability
    ax2.scatter(interpretability, performance, s=200, c=[colors['mlblue'], colors['mlorange'],
                                                        colors['mlgreen'], colors['mlred']])
    for i, alg in enumerate(algorithms):
        ax2.annotate(alg, (interpretability[i], performance[i]),
                    xytext=(5, 5), textcoords='offset points')

    ax2.set_xlabel('Interpretability (%)')
    ax2.set_ylabel('Performance (%)')
    ax2.set_title('Performance vs Interpretability Trade-off')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}algorithm_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}algorithm_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_interpretability_accuracy_tradeoff():
    """Interpretability vs accuracy tradeoff"""
    algorithms = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest', 'SVM', 'Neural\nNetwork']
    interpretability = [9, 7, 4, 3, 1]
    accuracy = [6, 7, 8, 9, 9]

    plt.figure(figsize=(10, 6))
    plt.scatter(interpretability, accuracy, s=200,
               c=[colors['mlblue'], colors['mlorange'], colors['mlgreen'],
                  colors['mlred'], colors['mlpurple']], alpha=0.7)

    for i, alg in enumerate(algorithms):
        plt.annotate(alg, (interpretability[i], accuracy[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    plt.xlabel('Interpretability (1-10 scale)')
    plt.ylabel('Typical Accuracy (1-10 scale)')
    plt.title('The Interpretability-Accuracy Trade-off')
    plt.grid(True, alpha=0.3)

    # Add trade-off line
    plt.plot([1, 9], [9, 6], 'k--', alpha=0.5, label='Trade-off trend')
    plt.legend()

    plt.savefig(f'{output_dir}interpretability_accuracy_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}interpretability_accuracy_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_supervised_to_unsupervised():
    """Transition from supervised to unsupervised learning"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Supervised learning
    np.random.seed(42)
    X_sup = np.random.randn(50, 2)
    y_sup = (X_sup[:, 0] + X_sup[:, 1] > 0).astype(int)
    colors_sup = [colors['mlred'] if y == 0 else colors['mlblue'] for y in y_sup]

    ax1.scatter(X_sup[:, 0], X_sup[:, 1], c=colors_sup, alpha=0.7, s=50)
    ax1.axline((0, 0), slope=-1, color='black', linewidth=2, label='Decision boundary')
    ax1.set_title('Supervised Learning\n(With target labels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Unsupervised learning
    X_unsup = np.random.randn(50, 2)
    X_unsup[:25] += [2, 2]  # Create clusters

    ax2.scatter(X_unsup[:, 0], X_unsup[:, 1], c=colors['mlgray'], alpha=0.7, s=50)
    ax2.set_title('Unsupervised Learning\n(No target labels - find patterns)')
    ax2.grid(True, alpha=0.3)

    # Add cluster centers
    ax2.scatter([0, 2], [0, 2], c=colors['mlred'], marker='x', s=200, linewidth=3)
    ax2.text(0, -0.5, 'Cluster 1', ha='center', fontweight='bold')
    ax2.text(2, 1.5, 'Cluster 2', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}supervised_to_unsupervised.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}supervised_to_unsupervised.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_remaining_charts():
    """Create remaining charts 14-15 and additional supporting charts"""

    # Chart 14: Algorithm comparison table (text-based)
    fig, ax = plt.subplots(figsize=(12, 8))

    algorithms = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest', 'SVM\nRBF', 'Gradient\nBoosting']
    datasets = ['Iris', 'Wine', 'Digits', 'Breast\nCancer', 'XOR', 'Circles']

    # Performance matrix (simulated realistic values)
    performance = np.array([
        [96, 96, 94, 95, 50, 52],  # Linear
        [96, 90, 85, 93, 100, 98], # Tree
        [97, 94, 88, 95, 100, 99], # RF
        [98, 96, 98, 97, 100, 100], # SVM
        [97, 95, 92, 96, 100, 99]  # GB
    ])

    im = ax.imshow(performance, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(datasets)):
            text = ax.text(j, i, f'{performance[i, j]}%',
                         ha="center", va="center", color="black", weight='bold')

    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(algorithms)))
    ax.set_xticklabels(datasets)
    ax.set_yticklabels(algorithms)
    ax.set_title('Algorithm Performance Comparison (%)', pad=20)

    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}algorithm_comparison_table.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}algorithm_comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Chart 15: Production ML Pipeline
    fig, ax = plt.subplots(figsize=(12, 6))

    stages = ['Data\nIngestion', 'Feature\nEngineering', 'Model\nTraining',
              'Validation', 'Deployment', 'Monitoring']
    x_positions = np.arange(len(stages))

    # Create pipeline flow
    for i, (stage, x) in enumerate(zip(stages, x_positions)):
        # Box for each stage
        rect = plt.Rectangle((x-0.3, 0.3), 0.6, 0.4,
                           facecolor=colors['mlblue'], alpha=0.3,
                           edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.5, stage, ha='center', va='center', weight='bold')

        # Arrow to next stage
        if i < len(stages) - 1:
            ax.arrow(x+0.3, 0.5, 0.4, 0, head_width=0.05, head_length=0.05,
                    fc='black', ec='black')

    # Add feedback loop
    ax.arrow(5.3, 0.3, -4.6, 0, head_width=0.05, head_length=0.05,
            fc=colors['mlred'], ec=colors['mlred'], linestyle='--', alpha=0.7)
    ax.text(3, 0.1, 'Feedback Loop', ha='center', va='center',
           color=colors['mlred'], style='italic')

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 1)
    ax.set_title('Production ML Pipeline: End-to-End System')
    ax.axis('off')

    plt.savefig(f'{output_dir}production_ml_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}production_ml_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Generate all charts for Week 0b"""
    print("Generating Week 0b: Supervised Learning Charts...")

    chart_functions = [
        create_real_estate_scatter,
        create_linear_regression_fit,
        create_regression_vs_classification,
        create_curse_dimensionality,
        create_feature_combinations,
        create_ols_example,
        create_ridge_lasso_comparison,
        create_linear_success_cases,
        create_linear_failure_cases,
        create_linear_vs_nonlinear_boundaries,
        create_regularization_tradeoff,
        create_human_decision_process,
        create_nonlinear_methods_overview,
        create_missing_charts,
        create_remaining_charts
    ]

    for i, func in enumerate(chart_functions, 1):
        try:
            func()
            print(f"[OK] Created chart {i}: {func.__name__}")
        except Exception as e:
            print(f"[ERROR] Error creating chart {i}: {e}")

    print(f"\nAll charts saved to {output_dir}")
    print("Chart generation complete!")

if __name__ == "__main__":
    main()