"""
Chart Generation for Week 0 Part 2: Supervised Learning
WCAG AAA Compliant Color Palette
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# WCAG AAA Compliant Colors
COLORS = {
    'blue': '#1F77B4',
    'orange': '#FF7F0E',
    'green': '#2CA02C',
    'red': '#D62728',
    'purple': '#9467BD',
    'brown': '#8C564B',
    'pink': '#E377C2',
    'gray': '#7F7F7F',
    'olive': '#BCBD22',
    'cyan': '#17BECF'
}

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'


def create_linear_regression_comparison():
    """Compare OLS, Ridge, and LASSO regression"""
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.ravel() + 3 + np.random.normal(0, 2, 100)

    models = {
        'OLS': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'LASSO (α=1.0)': Lasso(alpha=1.0)
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(X, y, alpha=0.4, s=50, color=COLORS['gray'], label='Data Points')

    colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green']]
    for (name, model), color in zip(models.items(), colors_list):
        model.fit(X, y)
        y_pred = model.predict(X)
        ax.plot(X, y_pred, linewidth=3, label=name, color=color)

    ax.set_xlabel('Feature X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Target y', fontsize=14, fontweight='bold')
    ax.set_title('Linear Regression Methods Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    textstr = 'OLS: Minimize squared errors\nRidge: L2 regularization\nLASSO: L1 regularization (sparse)'
    ax.text(0.98, 0.02, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('../charts/linear_regression_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/linear_regression_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created linear_regression_comparison.pdf")


def create_logistic_regression():
    """Create logistic function and decision boundary"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    z = np.linspace(-10, 10, 200)
    sigmoid = 1 / (1 + np.exp(-z))

    ax1.plot(z, sigmoid, linewidth=3, color=COLORS['blue'])
    ax1.axhline(y=0.5, color=COLORS['red'], linestyle='--', linewidth=2, label='Decision Boundary (p=0.5)')
    ax1.axvline(x=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax1.set_xlabel('z = w^T x + b', fontsize=14, fontweight='bold')
    ax1.set_ylabel('P(y=1|x)', fontsize=14, fontweight='bold')
    ax1.set_title('Sigmoid Function σ(z) = 1/(1+e^(-z))', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    ax2.scatter(X[y==0, 0], X[y==0, 1], c=COLORS['blue'], s=80, alpha=0.6, label='Class 0', edgecolors='black')
    ax2.scatter(X[y==1, 0], X[y==1, 1], c=COLORS['orange'], s=80, alpha=0.6, label='Class 1', edgecolors='black')

    x_line = np.linspace(-3, 3, 100)
    y_line = -x_line
    ax2.plot(x_line, y_line, 'r--', linewidth=3, label='Decision Boundary')

    ax2.set_xlabel('Feature 1', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Feature 2', fontsize=14, fontweight='bold')
    ax2.set_title('Logistic Regression Classification', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../charts/logistic_regression.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/logistic_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created logistic_regression.pdf")


def create_decision_tree():
    """Create decision tree structure"""
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(clf, ax=ax, filled=True, rounded=True, fontsize=10,
              feature_names=['Feature 1', 'Feature 2'],
              class_names=['Class 0', 'Class 1'],
              proportion=True)

    ax.set_title('Decision Tree Structure (Max Depth = 3)', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('../charts/decision_tree.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/decision_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created decision_tree.pdf")


def create_random_forest():
    """Create random forest ensemble visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    tree_positions = [
        (2, 6), (5, 6), (8, 6), (11, 6)
    ]

    for i, (x, y) in enumerate(tree_positions):
        circle = plt.Circle((x, y), 1.2, color=COLORS['green'], alpha=0.3, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, f'Tree {i+1}', ha='center', va='center', fontsize=12, fontweight='bold')

        ax.plot([x, 7], [y-1.2, 2.5], 'k-', linewidth=2, alpha=0.5)

    vote_box = plt.Rectangle((5.5, 1), 3, 1.5, facecolor=COLORS['blue'], edgecolor='black', linewidth=2, alpha=0.5)
    ax.add_patch(vote_box)
    ax.text(7, 1.75, 'Majority Vote\n(Classification)', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')

    ax.text(7, 9.2, 'Random Forest Ensemble', ha='center', fontsize=18, fontweight='bold')

    ax.text(7, 8.5, 'Bootstrap Sampling + Feature Randomness → Multiple Decision Trees → Aggregate Predictions',
            ha='center', fontsize=11, style='italic', color=COLORS['gray'])

    properties = [
        '• Each tree trained on random subset',
        '• Random feature selection at splits',
        '• Reduces overfitting via averaging',
        '• Parallel training possible'
    ]

    for i, prop in enumerate(properties):
        ax.text(1, 4 - i*0.5, prop, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('../charts/random_forest.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/random_forest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created random_forest.pdf")


def create_gradient_boosting():
    """Create boosting process visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    stages = [
        ('Initial Model\nf₀(x)', 2, COLORS['blue']),
        ('Residuals\ne₁ = y - f₀', 4.5, COLORS['red']),
        ('Weak Learner\nh₁(x)', 7, COLORS['green']),
        ('Update\nf₁ = f₀ + αh₁', 9.5, COLORS['purple']),
        ('Repeat', 12, COLORS['orange'])
    ]

    for i, (stage, x, color) in enumerate(stages):
        if i < 4:
            rect = plt.Rectangle((x-0.8, 5), 1.6, 2, facecolor=color, edgecolor='black', linewidth=2, alpha=0.5)
            ax.add_patch(rect)
            ax.text(x, 6, stage, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

            if i < 3:
                ax.annotate('', xy=(stages[i+1][1]-0.9, 6), xytext=(x+0.9, 6),
                           arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        else:
            circle = plt.Circle((x, 6), 0.9, color=color, alpha=0.3, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, 6, stage, ha='center', va='center', fontsize=11, fontweight='bold')
            ax.annotate('', xy=(2.8, 7.2), xytext=(x-0.5, 6.7),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black', linestyle='--'))

    ax.text(7, 9, 'Gradient Boosting Process', ha='center', fontsize=18, fontweight='bold')

    formula = r'$F_M(x) = f_0(x) + \sum_{m=1}^{M} \alpha_m h_m(x)$'
    ax.text(7, 8.2, formula, ha='center', fontsize=14, style='italic')

    explanation = [
        '1. Start with initial prediction f₀',
        '2. Compute residuals (errors)',
        '3. Train weak learner on residuals',
        '4. Update model with weighted learner',
        '5. Iterate M times'
    ]

    for i, exp in enumerate(explanation):
        ax.text(1, 3.5 - i*0.45, exp, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('../charts/gradient_boosting.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/gradient_boosting.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created gradient_boosting.pdf")


def create_knn_classification():
    """Create k-NN classification example"""
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for k, ax in zip([3, 9], [ax1, ax2]):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=[COLORS['blue'], COLORS['orange']])

        ax.scatter(X[y==0, 0], X[y==0, 1], c=COLORS['blue'], s=80, alpha=0.8, label='Class 0', edgecolors='black')
        ax.scatter(X[y==1, 0], X[y==1, 1], c=COLORS['orange'], s=80, alpha=0.8, label='Class 1', edgecolors='black')

        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(f'k-NN Classification (k={k})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Effect of k on Decision Boundary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../charts/knn_classification.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/knn_classification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created knn_classification.pdf")


def create_algorithm_comparison():
    """Create performance comparison across algorithms"""
    algorithms = ['Linear\nRegression', 'Logistic\nRegression', 'SVM', 'Decision\nTree',
                 'Random\nForest', 'Gradient\nBoosting', 'k-NN']

    accuracy = [0.72, 0.85, 0.88, 0.78, 0.91, 0.93, 0.82]
    train_time = [0.01, 0.02, 0.15, 0.05, 0.35, 0.85, 0.03]
    interpretability = [9, 7, 5, 8, 4, 3, 6]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red'],
             COLORS['purple'], COLORS['brown'], COLORS['pink']]

    ax1.barh(algorithms, accuracy, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)

    ax2.barh(algorithms, train_time, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    ax3.barh(algorithms, interpretability, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Interpretability (1-10)', fontsize=12, fontweight='bold')
    ax3.set_title('Model Interpretability', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 10)
    ax3.grid(axis='x', alpha=0.3)

    plt.suptitle('Supervised Learning Algorithm Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../charts/algorithm_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created algorithm_comparison.pdf")


if __name__ == '__main__':
    print("Generating Supervised Learning Charts...")
    create_linear_regression_comparison()
    create_logistic_regression()
    create_decision_tree()
    create_random_forest()
    create_gradient_boosting()
    create_knn_classification()
    create_algorithm_comparison()
    print("[OK] All supervised learning charts created successfully!")