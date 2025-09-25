"""
Create decision boundary visualizations for Week 4 presentation.
Shows how different classification algorithms separate data in 2D space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define consistent color palette
colors = {
    'failed': '#e74c3c',        # Red
    'success': '#27ae60',       # Green
    'boundary': '#34495e',      # Dark gray
    'mesh': '#ecf0f1',          # Light gray
}

def load_data():
    """Load the innovation dataset."""
    # Load numpy arrays
    X_train = np.load('innovation_X_train.npy')
    X_test = np.load('innovation_X_test.npy')
    y_train = np.load('innovation_y_train_binary.npy')
    y_test = np.load('innovation_y_test_binary.npy')
    feature_names = np.load('innovation_feature_names.npy', allow_pickle=True)

    return X_train, X_test, y_train, y_test, feature_names

def create_decision_boundaries():
    """Create decision boundary visualization for multiple classifiers."""
    print("Creating decision boundary visualizations...")

    # Load data
    X_train, X_test, y_train, y_test, _ = load_data()

    # Use subset of data for faster visualization
    n_samples = 2000
    indices = np.random.choice(len(X_train), min(n_samples, len(X_train)), replace=False)
    X = X_train[indices]
    y = y_train[indices]

    # Standardize and reduce to 2D using PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42, C=1.0),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Create mesh for decision boundaries
    h = 0.05  # Step size in mesh (increased for faster computation)
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for idx, (name, clf) in enumerate(classifiers.items()):
        ax = axes[idx]

        # Train classifier on 2D data
        clf.fit(X_2d, y)

        # Predict on mesh
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

        # Plot data points
        scatter = ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1],
                           c=colors['failed'], s=20, alpha=0.6,
                           edgecolors='black', linewidth=0.5, label='Failed')
        ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1],
                  c=colors['success'], s=20, alpha=0.6,
                  edgecolors='black', linewidth=0.5, label='Success')

        # Styling
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.grid(True, alpha=0.3)

        # Add accuracy score
        accuracy = clf.score(X_2d, y)
        ax.text(0.02, 0.98, f'Accuracy: {accuracy:.2%}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add legend to first subplot
    axes[0].legend(loc='lower right', framealpha=0.9)

    plt.suptitle('Decision Boundaries: How Different Algorithms Classify Innovation Success',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    plt.savefig('charts/decision_boundaries.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('charts/decision_boundaries.png', dpi=150, bbox_inches='tight')
    print("Saved: charts/decision_boundaries.pdf")
    plt.close()

def create_linear_vs_nonlinear():
    """Create comparison of linear vs non-linear classifiers."""
    print("Creating linear vs non-linear comparison...")

    # Load data
    X_train, X_test, y_train, y_test, _ = load_data()

    # Use subset of data for faster visualization
    n_samples = 1500
    indices = np.random.choice(len(X_train), min(n_samples, len(X_train)), replace=False)
    X = X_train[indices]
    y = y_train[indices]

    # Standardize and reduce to 2D
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    # Define linear and non-linear classifiers
    classifiers = [
        ('Linear Models', [
            ('Logistic Regression', LogisticRegression(random_state=42)),
            ('Linear SVM', SVC(kernel='linear', probability=True, random_state=42))
        ]),
        ('Non-Linear Models', [
            ('RBF SVM', SVC(kernel='rbf', probability=True, random_state=42)),
            ('Neural Network', MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42))
        ])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create mesh
    h = 0.05  # Coarser mesh for faster computation
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for row_idx, (category, models) in enumerate(classifiers):
        for col_idx, (name, clf) in enumerate(models):
            ax = axes[row_idx, col_idx]

            # Train classifier
            clf.fit(X_2d, y)

            # Predict on mesh
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)

            # Plot
            ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.3)
            ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

            # Plot points
            ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1],
                      c=colors['failed'], s=15, alpha=0.5)
            ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1],
                      c=colors['success'], s=15, alpha=0.5)

            # Styling
            ax.set_title(f'{category}: {name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)

            # Add accuracy
            accuracy = clf.score(X_2d, y)
            ax.text(0.02, 0.98, f'Acc: {accuracy:.2%}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Linear vs Non-Linear Classification Boundaries',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig('charts/linear_vs_nonlinear.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('charts/linear_vs_nonlinear.png', dpi=150, bbox_inches='tight')
    print("Saved: charts/linear_vs_nonlinear.pdf")
    plt.close()

def create_feature_space_visualization():
    """Create visualization of feature space with different projections."""
    print("Creating feature space visualization...")

    # Load data
    X_train, _, y_train, _, feature_names = load_data()

    # Select subset for clarity
    n_samples = 500
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    # Create different projections
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Top 2 features
    ax = axes[0, 0]
    ax.scatter(X_subset[:, 0], X_subset[:, 1],
              c=['#e74c3c' if y == 0 else '#27ae60' for y in y_subset],
              alpha=0.6, s=30)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('Original Feature Space (Top 2 Features)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. PCA projection
    ax = axes[0, 1]
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    ax.scatter(X_pca[:, 0], X_pca[:, 1],
              c=['#e74c3c' if y == 0 else '#27ae60' for y in y_subset],
              alpha=0.6, s=30)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('PCA Projection', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Innovation score vs Market readiness (selected features)
    ax = axes[1, 0]
    # Find indices of specific features
    innovation_idx = 2  # novelty_score
    market_idx = 5      # market_size
    ax.scatter(X_subset[:, innovation_idx], X_subset[:, market_idx],
              c=['#e74c3c' if y == 0 else '#27ae60' for y in y_subset],
              alpha=0.6, s=30)
    ax.set_xlabel('Novelty Score')
    ax.set_ylabel('Market Size')
    ax.set_title('Innovation vs Market Space', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Team experience vs Development time
    ax = axes[1, 1]
    team_idx = 9     # team_experience
    dev_idx = 12     # development_time
    ax.scatter(X_subset[:, team_idx], X_subset[:, dev_idx],
              c=['#e74c3c' if y == 0 else '#27ae60' for y in y_subset],
              alpha=0.6, s=30)
    ax.set_xlabel('Team Experience')
    ax.set_ylabel('Development Time')
    ax.set_title('Team vs Development Space', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Failed', alpha=0.6),
        Patch(facecolor='#27ae60', label='Success', alpha=0.6)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 1.02))

    plt.suptitle('Innovation Success in Different Feature Spaces',
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    # Save
    plt.savefig('charts/feature_space_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('charts/feature_space_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: charts/feature_space_visualization.pdf")
    plt.close()

if __name__ == "__main__":
    print("Generating decision boundary visualizations...")
    create_decision_boundaries()
    create_linear_vs_nonlinear()
    create_feature_space_visualization()
    print("All decision boundary visualizations completed!")