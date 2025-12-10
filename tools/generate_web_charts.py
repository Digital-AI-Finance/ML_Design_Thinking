#!/usr/bin/env python3
"""Generate all charts for Hugo website topic pages."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Standard styling
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.figsize': (10, 6), 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False
})

# Color palette
MLPURPLE = '#3333B2'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'
MLGRAY = '#7F7F7F'

BASE_PATH = Path(__file__).parent.parent / 'static' / 'images' / 'topics'

def save_chart(fig, topic, name):
    """Save chart as PNG."""
    path = BASE_PATH / topic / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Created: {path.name}')


# =============================================================================
# ML FOUNDATIONS
# =============================================================================
def create_ml_foundations_charts():
    print('\nML Foundations:')

    # Chart 1: Learning Paradigms
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Supervised\nLearning', 'Unsupervised\nLearning', 'Reinforcement\nLearning']
    examples = [
        'Classification\nRegression\nForecasting',
        'Clustering\nDimensionality\nReduction',
        'Game Playing\nRobotics\nOptimization'
    ]
    colors = [MLBLUE, MLGREEN, MLORANGE]
    bars = ax.bar(categories, [85, 70, 55], color=colors, width=0.6)
    ax.set_ylabel('Industry Adoption (%)', fontsize=12)
    ax.set_title('Machine Learning Paradigms', fontsize=14, fontweight='bold')
    for bar, ex in zip(bars, examples):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, ex,
                ha='center', va='bottom', fontsize=9, color=MLGRAY)
    ax.set_ylim(0, 110)
    save_chart(fig, 'ml-foundations', 'learning_paradigms')

    # Chart 2: ML vs Traditional Programming
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Traditional
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.add_patch(plt.Rectangle((1, 6), 3, 2, facecolor=MLBLUE, alpha=0.3))
    ax.add_patch(plt.Rectangle((5, 6), 3, 2, facecolor=MLGREEN, alpha=0.3))
    ax.add_patch(plt.Rectangle((3, 2), 3, 2, facecolor=MLORANGE, alpha=0.3))
    ax.text(2.5, 7, 'Data', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(6.5, 7, 'Rules', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(4.5, 3, 'Output', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.annotate('', xy=(4.5, 4.2), xytext=(3.5, 5.8), arrowprops=dict(arrowstyle='->', color=MLGRAY, lw=2))
    ax.annotate('', xy=(4.5, 4.2), xytext=(5.5, 5.8), arrowprops=dict(arrowstyle='->', color=MLGRAY, lw=2))
    ax.set_title('Traditional Programming', fontsize=13, fontweight='bold')
    ax.axis('off')

    # ML
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.add_patch(plt.Rectangle((1, 6), 3, 2, facecolor=MLBLUE, alpha=0.3))
    ax.add_patch(plt.Rectangle((5, 6), 3, 2, facecolor=MLORANGE, alpha=0.3))
    ax.add_patch(plt.Rectangle((3, 2), 3, 2, facecolor=MLGREEN, alpha=0.3))
    ax.text(2.5, 7, 'Data', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(6.5, 7, 'Output', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(4.5, 3, 'Rules\n(Model)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.annotate('', xy=(4.5, 4.2), xytext=(3.5, 5.8), arrowprops=dict(arrowstyle='->', color=MLGRAY, lw=2))
    ax.annotate('', xy=(4.5, 4.2), xytext=(5.5, 5.8), arrowprops=dict(arrowstyle='->', color=MLGRAY, lw=2))
    ax.set_title('Machine Learning', fontsize=13, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    save_chart(fig, 'ml-foundations', 'ml_vs_traditional')

    # Chart 3: Model Performance vs Data
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    traditional = 3 + 0.5 * np.log1p(x)
    ml_simple = 2 + 1.5 * np.log1p(x)
    ml_deep = 1 + 2.5 * np.log1p(x)
    ax.plot(x, traditional, color=MLGRAY, lw=2, label='Traditional Rules')
    ax.plot(x, ml_simple, color=MLBLUE, lw=2, label='Simple ML')
    ax.plot(x, ml_deep, color=MLPURPLE, lw=2, label='Deep Learning')
    ax.set_xlabel('Amount of Data', fontsize=12)
    ax.set_ylabel('Model Performance', fontsize=12)
    ax.set_title('Performance Scaling with Data', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    save_chart(fig, 'ml-foundations', 'performance_scaling')


# =============================================================================
# SUPERVISED LEARNING
# =============================================================================
def create_supervised_learning_charts():
    print('\nSupervised Learning:')

    # Chart 1: Regression vs Classification
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    np.random.seed(42)

    # Regression
    ax = axes[0]
    x = np.linspace(0, 10, 50)
    y = 2 + 0.5 * x + np.random.normal(0, 0.5, 50)
    ax.scatter(x, y, color=MLBLUE, alpha=0.6, s=50)
    ax.plot(x, 2 + 0.5 * x, color=MLRED, lw=2, label='Prediction Line')
    ax.set_xlabel('Feature X')
    ax.set_ylabel('Target Y')
    ax.set_title('Regression: Predict Continuous Value', fontweight='bold')
    ax.legend()

    # Classification
    ax = axes[1]
    x1 = np.random.normal(3, 1, 30)
    y1 = np.random.normal(3, 1, 30)
    x2 = np.random.normal(7, 1, 30)
    y2 = np.random.normal(7, 1, 30)
    ax.scatter(x1, y1, color=MLBLUE, alpha=0.6, s=50, label='Class A')
    ax.scatter(x2, y2, color=MLORANGE, alpha=0.6, s=50, label='Class B')
    ax.plot([0, 10], [10, 0], color=MLRED, lw=2, ls='--', label='Decision Boundary')
    ax.set_xlabel('Feature X1')
    ax.set_ylabel('Feature X2')
    ax.set_title('Classification: Predict Category', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    save_chart(fig, 'supervised-learning', 'regression_vs_classification')

    # Chart 2: Bias-Variance Tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    complexity = np.linspace(0, 10, 100)
    bias = 5 * np.exp(-0.5 * complexity)
    variance = 0.2 * complexity ** 1.5
    total = bias + variance
    ax.plot(complexity, bias, color=MLBLUE, lw=2, label='Bias')
    ax.plot(complexity, variance, color=MLORANGE, lw=2, label='Variance')
    ax.plot(complexity, total, color=MLRED, lw=2, label='Total Error')
    optimal = complexity[np.argmin(total)]
    ax.axvline(optimal, color=MLGREEN, ls='--', lw=2, label=f'Optimal Complexity')
    ax.set_xlabel('Model Complexity', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    save_chart(fig, 'supervised-learning', 'bias_variance')

    # Chart 3: Train/Test Split
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(['Dataset'], [100], color=MLGRAY, alpha=0.3, height=0.5)
    ax.barh(['Dataset'], [80], color=MLBLUE, alpha=0.7, height=0.5, label='Training (80%)')
    ax.barh(['Dataset'], [20], left=80, color=MLORANGE, alpha=0.7, height=0.5, label='Testing (20%)')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage of Data', fontsize=12)
    ax.set_title('Train/Test Split Strategy', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_yticks([])
    save_chart(fig, 'supervised-learning', 'train_test_split')


# =============================================================================
# UNSUPERVISED LEARNING
# =============================================================================
def create_unsupervised_learning_charts():
    print('\nUnsupervised Learning:')

    # Chart 1: Clustering Example
    fig, ax = plt.subplots(figsize=(10, 6))
    np.random.seed(42)
    colors = [MLBLUE, MLORANGE, MLGREEN]
    centers = [(2, 2), (7, 7), (2, 8)]
    for i, (cx, cy) in enumerate(centers):
        x = np.random.normal(cx, 0.8, 40)
        y = np.random.normal(cy, 0.8, 40)
        ax.scatter(x, y, c=colors[i], alpha=0.6, s=50, label=f'Cluster {i+1}')
        ax.scatter(cx, cy, c=colors[i], s=200, marker='*', edgecolors='black', linewidths=1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Clustering: Discovering Groups in Data', fontsize=14, fontweight='bold')
    ax.legend()
    save_chart(fig, 'unsupervised-learning', 'clustering_example')

    # Chart 2: Dimensionality Reduction
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    np.random.seed(42)

    # High dimensional (simulated)
    ax = axes[0]
    dims = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    vals = np.random.rand(10) * 5
    ax.bar(dims, vals, color=MLBLUE, alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('High Dimensional Data (10D)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # Reduced
    ax = axes[1]
    x = np.random.normal(0, 1, 50)
    y = 0.7 * x + np.random.normal(0, 0.3, 50)
    ax.scatter(x, y, c=MLGREEN, alpha=0.6, s=50)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Reduced to 2D (PCA)', fontweight='bold')

    plt.tight_layout()
    save_chart(fig, 'unsupervised-learning', 'dimensionality_reduction')

    # Chart 3: Unsupervised vs Supervised
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = ['Supervised', 'Unsupervised']
    needs_labels = [100, 0]
    ax.barh(categories, needs_labels, color=[MLBLUE, MLGREEN], height=0.5)
    ax.set_xlabel('Requires Labeled Data (%)')
    ax.set_title('Supervised vs Unsupervised Learning', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 120)
    for i, v in enumerate(needs_labels):
        label = 'Yes - needs labels' if v == 100 else 'No - finds patterns'
        ax.text(v + 3, i, label, va='center', fontsize=11)
    save_chart(fig, 'unsupervised-learning', 'supervised_vs_unsupervised')


# =============================================================================
# NEURAL NETWORKS
# =============================================================================
def create_neural_networks_charts():
    print('\nNeural Networks:')

    # Chart 1: Network Architecture
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = [3, 5, 5, 2]
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    colors = [MLBLUE, MLPURPLE, MLPURPLE, MLGREEN]

    for l, (n, name, col) in enumerate(zip(layers, layer_names, colors)):
        x = l * 2.5
        for i in range(n):
            y = (i - n/2 + 0.5) * 1.2
            circle = plt.Circle((x, y), 0.3, color=col, alpha=0.7)
            ax.add_patch(circle)
            if l < len(layers) - 1:
                next_n = layers[l+1]
                for j in range(next_n):
                    next_y = (j - next_n/2 + 0.5) * 1.2
                    ax.plot([x+0.3, x+2.2], [y, next_y], color=MLGRAY, alpha=0.3, lw=0.5)
        ax.text(x, -4, name, ha='center', fontsize=11)

    ax.set_xlim(-1, 9)
    ax.set_ylim(-5, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold', y=0.95)
    save_chart(fig, 'neural-networks', 'network_architecture')

    # Chart 2: Activation Functions
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.linspace(-5, 5, 100)

    # Sigmoid
    ax = axes[0]
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, color=MLBLUE, lw=2)
    ax.axhline(0.5, color=MLGRAY, ls='--', alpha=0.5)
    ax.axvline(0, color=MLGRAY, ls='--', alpha=0.5)
    ax.set_title('Sigmoid', fontweight='bold')
    ax.set_ylim(-0.1, 1.1)

    # ReLU
    ax = axes[1]
    y = np.maximum(0, x)
    ax.plot(x, y, color=MLGREEN, lw=2)
    ax.axhline(0, color=MLGRAY, ls='--', alpha=0.5)
    ax.axvline(0, color=MLGRAY, ls='--', alpha=0.5)
    ax.set_title('ReLU', fontweight='bold')

    # Tanh
    ax = axes[2]
    y = np.tanh(x)
    ax.plot(x, y, color=MLORANGE, lw=2)
    ax.axhline(0, color=MLGRAY, ls='--', alpha=0.5)
    ax.axvline(0, color=MLGRAY, ls='--', alpha=0.5)
    ax.set_title('Tanh', fontweight='bold')
    ax.set_ylim(-1.2, 1.2)

    for ax in axes:
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')

    plt.tight_layout()
    save_chart(fig, 'neural-networks', 'activation_functions')

    # Chart 3: Training Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(1, 51)
    train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.2 + np.random.normal(0, 0.05, 50)
    val_loss = 2.5 * np.exp(-0.06 * epochs) + 0.4 + np.random.normal(0, 0.08, 50)
    ax.plot(epochs, train_loss, color=MLBLUE, lw=2, label='Training Loss')
    ax.plot(epochs, val_loss, color=MLORANGE, lw=2, label='Validation Loss')
    ax.axvline(35, color=MLRED, ls='--', lw=2, label='Early Stopping')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend()
    save_chart(fig, 'neural-networks', 'training_loss')


# =============================================================================
# CLUSTERING
# =============================================================================
def create_clustering_charts():
    print('\nClustering:')

    # Chart 1: K-Means Steps
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    np.random.seed(42)

    # Data
    x = np.concatenate([np.random.normal(2, 0.8, 30), np.random.normal(7, 0.8, 30)])
    y = np.concatenate([np.random.normal(2, 0.8, 30), np.random.normal(6, 0.8, 30)])

    # Step 1: Initial
    ax = axes[0]
    ax.scatter(x, y, c=MLGRAY, alpha=0.6, s=50)
    ax.scatter([3, 6], [5, 3], c=[MLBLUE, MLORANGE], s=200, marker='X', edgecolors='black')
    ax.set_title('Step 1: Random Centroids', fontweight='bold')

    # Step 2: Assign
    ax = axes[1]
    colors = [MLBLUE if xi < 5 else MLORANGE for xi in x]
    ax.scatter(x, y, c=colors, alpha=0.6, s=50)
    ax.scatter([3, 6], [5, 3], c=[MLBLUE, MLORANGE], s=200, marker='X', edgecolors='black')
    ax.set_title('Step 2: Assign Points', fontweight='bold')

    # Step 3: Update
    ax = axes[2]
    ax.scatter(x, y, c=colors, alpha=0.6, s=50)
    ax.scatter([2, 7], [2, 6], c=[MLBLUE, MLORANGE], s=200, marker='X', edgecolors='black')
    ax.set_title('Step 3: Update Centroids', fontweight='bold')

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    plt.tight_layout()
    save_chart(fig, 'clustering', 'kmeans_steps')

    # Chart 2: Elbow Method
    fig, ax = plt.subplots(figsize=(10, 6))
    k_values = range(1, 11)
    inertia = [1000, 400, 200, 120, 100, 90, 85, 82, 80, 78]
    ax.plot(k_values, inertia, 'o-', color=MLBLUE, lw=2, markersize=8)
    ax.axvline(4, color=MLRED, ls='--', lw=2, label='Elbow Point (k=4)')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax.legend()
    save_chart(fig, 'clustering', 'elbow_method')

    # Chart 3: Silhouette Score
    fig, ax = plt.subplots(figsize=(10, 6))
    k_values = range(2, 11)
    silhouette = [0.35, 0.45, 0.62, 0.58, 0.52, 0.48, 0.44, 0.41, 0.38]
    bars = ax.bar(k_values, silhouette, color=MLGREEN, alpha=0.7)
    bars[2].set_color(MLPURPLE)  # Highlight best
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score by Number of Clusters', fontsize=14, fontweight='bold')
    ax.axhline(0.5, color=MLGRAY, ls='--', alpha=0.5, label='Good threshold')
    ax.legend()
    save_chart(fig, 'clustering', 'silhouette_scores')


# =============================================================================
# NLP SENTIMENT
# =============================================================================
def create_nlp_sentiment_charts():
    print('\nNLP Sentiment:')

    # Chart 1: Sentiment Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sentiments = ['Negative', 'Neutral', 'Positive']
    counts = [1250, 2100, 1650]
    colors = [MLRED, MLGRAY, MLGREEN]
    bars = ax.bar(sentiments, counts, color=colors, width=0.6)
    ax.set_ylabel('Number of Reviews', fontsize=12)
    ax.set_title('Sentiment Distribution in Customer Reviews', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=11)
    save_chart(fig, 'nlp-sentiment', 'sentiment_distribution')

    # Chart 2: Word Embeddings Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    np.random.seed(42)
    words = ['good', 'great', 'excellent', 'bad', 'terrible', 'awful', 'happy', 'sad', 'product', 'service']
    positive = ['good', 'great', 'excellent', 'happy']
    negative = ['bad', 'terrible', 'awful', 'sad']

    for word in words:
        if word in positive:
            x, y = np.random.normal(3, 0.5), np.random.normal(3, 0.5)
            color = MLGREEN
        elif word in negative:
            x, y = np.random.normal(7, 0.5), np.random.normal(7, 0.5)
            color = MLRED
        else:
            x, y = np.random.normal(5, 1), np.random.normal(5, 1)
            color = MLBLUE
        ax.scatter(x, y, c=color, s=100, alpha=0.7)
        ax.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel('Dimension 1 (PCA)', fontsize=12)
    ax.set_ylabel('Dimension 2 (PCA)', fontsize=12)
    ax.set_title('Word Embeddings: Similar Words Cluster Together', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    save_chart(fig, 'nlp-sentiment', 'word_embeddings')

    # Chart 3: Text Preprocessing Pipeline
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = ['Raw\nText', 'Tokenize', 'Lowercase', 'Remove\nStopwords', 'Lemmatize', 'Vectors']
    x_pos = np.arange(len(steps)) * 2

    for i, (x, step) in enumerate(zip(x_pos, steps)):
        color = MLBLUE if i < len(steps)-1 else MLGREEN
        ax.add_patch(plt.Rectangle((x-0.7, 0.3), 1.4, 1.4, facecolor=color, alpha=0.3, edgecolor=color, lw=2))
        ax.text(x, 1, step, ha='center', va='center', fontsize=11, fontweight='bold')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x+1.1, 1), xytext=(x+0.9, 1),
                       arrowprops=dict(arrowstyle='->', color=MLGRAY, lw=2))

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    ax.set_title('Text Preprocessing Pipeline', fontsize=14, fontweight='bold', y=0.85)
    save_chart(fig, 'nlp-sentiment', 'preprocessing_pipeline')


# =============================================================================
# CLASSIFICATION
# =============================================================================
def create_classification_charts():
    print('\nClassification:')

    # Chart 1: Decision Boundary
    fig, ax = plt.subplots(figsize=(10, 6))
    np.random.seed(42)
    x1 = np.random.normal(3, 1, 40)
    y1 = np.random.normal(3, 1, 40)
    x2 = np.random.normal(7, 1, 40)
    y2 = np.random.normal(7, 1, 40)
    ax.scatter(x1, y1, c=MLBLUE, s=60, alpha=0.7, label='Class 0')
    ax.scatter(x2, y2, c=MLORANGE, s=60, alpha=0.7, label='Class 1')
    ax.plot([0, 10], [10, 0], color=MLRED, lw=2, ls='--', label='Decision Boundary')
    ax.fill_between([0, 10], [10, 0], [0, 0], alpha=0.1, color=MLBLUE)
    ax.fill_between([0, 10], [10, 0], [10, 10], alpha=0.1, color=MLORANGE)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Decision Boundary Separates Classes', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    save_chart(fig, 'classification', 'decision_boundary')

    # Chart 2: Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array([[85, 15], [10, 90]])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > 50 else 'black'
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=20, color=color)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')
    save_chart(fig, 'classification', 'confusion_matrix')

    # Chart 3: Decision Tree
    fig, ax = plt.subplots(figsize=(10, 6))
    # Root
    ax.add_patch(plt.Rectangle((4, 4.5), 2, 0.8, facecolor=MLBLUE, alpha=0.3, edgecolor=MLBLUE, lw=2))
    ax.text(5, 4.9, 'Age > 30?', ha='center', va='center', fontsize=11, fontweight='bold')
    # Level 1
    ax.add_patch(plt.Rectangle((1.5, 2.5), 2, 0.8, facecolor=MLGREEN, alpha=0.3, edgecolor=MLGREEN, lw=2))
    ax.text(2.5, 2.9, 'Income > 50k?', ha='center', va='center', fontsize=10)
    ax.add_patch(plt.Rectangle((6.5, 2.5), 2, 0.8, facecolor=MLORANGE, alpha=0.3, edgecolor=MLORANGE, lw=2))
    ax.text(7.5, 2.9, 'Buy: Yes', ha='center', va='center', fontsize=10, fontweight='bold')
    # Level 2
    ax.add_patch(plt.Rectangle((0, 0.5), 1.5, 0.8, facecolor=MLRED, alpha=0.3, edgecolor=MLRED, lw=2))
    ax.text(0.75, 0.9, 'Buy: No', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.add_patch(plt.Rectangle((3, 0.5), 1.5, 0.8, facecolor=MLORANGE, alpha=0.3, edgecolor=MLORANGE, lw=2))
    ax.text(3.75, 0.9, 'Buy: Yes', ha='center', va='center', fontsize=9, fontweight='bold')
    # Lines
    ax.plot([5, 2.5], [4.5, 3.3], 'k-', lw=1.5)
    ax.plot([5, 7.5], [4.5, 3.3], 'k-', lw=1.5)
    ax.plot([2.5, 0.75], [2.5, 1.3], 'k-', lw=1.5)
    ax.plot([2.5, 3.75], [2.5, 1.3], 'k-', lw=1.5)
    ax.text(3.5, 4, 'Yes', fontsize=10)
    ax.text(6.2, 4, 'No', fontsize=10)

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Decision Tree: Rule-Based Classification', fontsize=14, fontweight='bold')
    save_chart(fig, 'classification', 'decision_tree')


# =============================================================================
# GENERATIVE AI
# =============================================================================
def create_generative_ai_charts():
    print('\nGenerative AI:')

    # Chart 1: LLM Capabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    capabilities = ['Text\nGeneration', 'Translation', 'Summarization', 'Q&A', 'Code\nGeneration', 'Reasoning']
    scores = [95, 90, 88, 92, 85, 80]
    colors = [MLPURPLE, MLBLUE, MLGREEN, MLORANGE, MLRED, MLGRAY]
    bars = ax.barh(capabilities, scores, color=colors, height=0.6)
    ax.set_xlabel('Capability Score (%)', fontsize=12)
    ax.set_title('Large Language Model Capabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    for bar, score in zip(bars, scores):
        ax.text(score + 2, bar.get_y() + bar.get_height()/2, f'{score}%', va='center', fontsize=10)
    save_chart(fig, 'generative-ai', 'llm_capabilities')

    # Chart 2: Prompt Engineering Flow
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = ['User\nPrompt', 'Context\nInjection', 'LLM\nProcessing', 'Response\nGeneration', 'Output']
    x_pos = np.arange(len(steps)) * 2.5

    for i, (x, step) in enumerate(zip(x_pos, steps)):
        color = [MLBLUE, MLGREEN, MLPURPLE, MLORANGE, MLGREEN][i]
        ax.add_patch(plt.Rectangle((x-0.8, 0.3), 1.6, 1.4, facecolor=color, alpha=0.3, edgecolor=color, lw=2))
        ax.text(x, 1, step, ha='center', va='center', fontsize=10, fontweight='bold')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x+1.3, 1), xytext=(x+1, 1),
                       arrowprops=dict(arrowstyle='->', color=MLGRAY, lw=2))

    ax.set_xlim(-1.5, 12)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    ax.set_title('Prompt Engineering Workflow', fontsize=14, fontweight='bold', y=0.85)
    save_chart(fig, 'generative-ai', 'prompt_flow')

    # Chart 3: Model Size vs Performance
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = ['1B', '7B', '13B', '70B', '175B']
    performance = [65, 78, 85, 92, 95]
    x = np.arange(len(sizes))
    ax.plot(x, performance, 'o-', color=MLPURPLE, lw=2, markersize=12)
    ax.fill_between(x, performance, alpha=0.2, color=MLPURPLE)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Model Size (Parameters)', fontsize=12)
    ax.set_ylabel('Benchmark Performance (%)', fontsize=12)
    ax.set_title('Scaling Laws: Bigger Models Perform Better', fontsize=14, fontweight='bold')
    ax.set_ylim(60, 100)
    save_chart(fig, 'generative-ai', 'scaling_laws')


# =============================================================================
# TOPIC MODELING
# =============================================================================
def create_topic_modeling_charts():
    print('\nTopic Modeling:')

    # Chart 1: Topic Word Distribution
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    topics = [
        ('Topic 1: Sports', ['game', 'team', 'player', 'score', 'win'], [0.15, 0.12, 0.10, 0.08, 0.06]),
        ('Topic 2: Technology', ['data', 'software', 'computer', 'digital', 'tech'], [0.14, 0.11, 0.09, 0.07, 0.05]),
        ('Topic 3: Politics', ['government', 'policy', 'election', 'vote', 'party'], [0.13, 0.10, 0.08, 0.06, 0.05])
    ]
    colors = [MLBLUE, MLGREEN, MLORANGE]

    for ax, (title, words, probs), color in zip(axes, topics, colors):
        ax.barh(words[::-1], probs[::-1], color=color, height=0.6)
        ax.set_xlabel('Probability')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(0, 0.2)

    plt.tight_layout()
    save_chart(fig, 'topic-modeling', 'topic_words')

    # Chart 2: Document-Topic Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    docs = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4', 'Doc 5']
    topic1 = [0.7, 0.1, 0.3, 0.2, 0.5]
    topic2 = [0.2, 0.6, 0.4, 0.1, 0.3]
    topic3 = [0.1, 0.3, 0.3, 0.7, 0.2]

    x = np.arange(len(docs))
    width = 0.25
    ax.bar(x - width, topic1, width, label='Sports', color=MLBLUE)
    ax.bar(x, topic2, width, label='Technology', color=MLGREEN)
    ax.bar(x + width, topic3, width, label='Politics', color=MLORANGE)

    ax.set_ylabel('Topic Proportion')
    ax.set_xticks(x)
    ax.set_xticklabels(docs)
    ax.set_title('Topic Distribution per Document', fontsize=14, fontweight='bold')
    ax.legend()
    save_chart(fig, 'topic-modeling', 'document_topics')

    # Chart 3: Coherence Score
    fig, ax = plt.subplots(figsize=(10, 6))
    n_topics = range(2, 16)
    coherence = [0.35, 0.42, 0.48, 0.55, 0.58, 0.56, 0.52, 0.48, 0.45, 0.43, 0.41, 0.40, 0.39, 0.38]
    ax.plot(n_topics, coherence, 'o-', color=MLPURPLE, lw=2, markersize=8)
    ax.axvline(6, color=MLRED, ls='--', lw=2, label='Optimal (k=6)')
    ax.set_xlabel('Number of Topics', fontsize=12)
    ax.set_ylabel('Coherence Score', fontsize=12)
    ax.set_title('Finding Optimal Number of Topics', fontsize=14, fontweight='bold')
    ax.legend()
    save_chart(fig, 'topic-modeling', 'coherence_score')


# =============================================================================
# RESPONSIBLE AI
# =============================================================================
def create_responsible_ai_charts():
    print('\nResponsible AI:')

    # Chart 1: Fairness Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = ['Group A', 'Group B', 'Group C', 'Group D']
    approval_rates = [0.72, 0.68, 0.45, 0.70]
    colors = [MLGREEN if r > 0.6 else MLRED for r in approval_rates]
    bars = ax.bar(groups, approval_rates, color=colors, width=0.6)
    ax.axhline(0.6, color=MLGRAY, ls='--', lw=2, label='Fairness Threshold')
    ax.set_ylabel('Approval Rate', fontsize=12)
    ax.set_title('Approval Rates by Demographic Group', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, rate in zip(bars, approval_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{rate:.0%}',
                ha='center', va='bottom', fontsize=11)
    save_chart(fig, 'responsible-ai', 'fairness_metrics')

    # Chart 2: SHAP Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ['Income', 'Credit Score', 'Age', 'Employment', 'Debt Ratio', 'Education']
    importance = [0.35, 0.28, 0.15, 0.12, 0.07, 0.03]
    colors = [MLPURPLE if i < 3 else MLBLUE for i in range(len(features))]
    ax.barh(features[::-1], importance[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel('SHAP Importance', fontsize=12)
    ax.set_title('Feature Importance for Model Decisions', fontsize=14, fontweight='bold')
    save_chart(fig, 'responsible-ai', 'shap_importance')

    # Chart 3: AI Ethics Principles
    fig, ax = plt.subplots(figsize=(10, 6))
    principles = ['Fairness', 'Transparency', 'Privacy', 'Accountability', 'Safety']
    scores = [4.2, 3.8, 4.5, 3.5, 4.0]
    colors = [MLGREEN, MLBLUE, MLPURPLE, MLORANGE, MLRED]

    angles = np.linspace(0, 2 * np.pi, len(principles), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += angles[:1]

    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, scores_plot, 'o-', color=MLPURPLE, lw=2)
    ax.fill(angles, scores_plot, alpha=0.25, color=MLPURPLE)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(principles, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_title('AI Ethics Assessment', fontsize=14, fontweight='bold', y=1.1)
    save_chart(fig, 'responsible-ai', 'ethics_radar')


# =============================================================================
# STRUCTURED OUTPUT
# =============================================================================
def create_structured_output_charts():
    print('\nStructured Output:')

    # Chart 1: JSON Success Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['No Schema', 'Prompt\nOnly', 'JSON Mode', 'Structured\nOutput']
    success_rates = [45, 72, 89, 98]
    colors = [MLRED, MLORANGE, MLBLUE, MLGREEN]
    bars = ax.bar(methods, success_rates, color=colors, width=0.6)
    ax.set_ylabel('Valid JSON Rate (%)', fontsize=12)
    ax.set_title('JSON Generation Success by Method', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{rate}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    save_chart(fig, 'structured-output', 'json_success_rate')

    # Chart 2: Validation Pipeline
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = ['LLM\nOutput', 'Parse\nJSON', 'Schema\nValidation', 'Type\nChecking', 'Clean\nData']
    x_pos = np.arange(len(steps)) * 2.5

    for i, (x, step) in enumerate(zip(x_pos, steps)):
        color = [MLBLUE, MLORANGE, MLGREEN, MLPURPLE, MLGREEN][i]
        ax.add_patch(plt.Rectangle((x-0.8, 0.3), 1.6, 1.4, facecolor=color, alpha=0.3, edgecolor=color, lw=2))
        ax.text(x, 1, step, ha='center', va='center', fontsize=10, fontweight='bold')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x+1.3, 1), xytext=(x+1, 1),
                       arrowprops=dict(arrowstyle='->', color=MLGRAY, lw=2))

    ax.set_xlim(-1.5, 12)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    ax.set_title('Output Validation Pipeline', fontsize=14, fontweight='bold', y=0.85)
    save_chart(fig, 'structured-output', 'validation_pipeline')

    # Chart 3: Error Types
    fig, ax = plt.subplots(figsize=(10, 6))
    errors = ['Missing\nFields', 'Wrong\nTypes', 'Invalid\nFormat', 'Extra\nFields', 'Null\nValues']
    counts = [35, 28, 20, 10, 7]
    colors = [MLRED, MLORANGE, MLBLUE, MLGREEN, MLGRAY]
    ax.pie(counts, labels=errors, colors=colors, autopct='%1.0f%%', startangle=90)
    ax.set_title('Common Structured Output Errors', fontsize=14, fontweight='bold')
    save_chart(fig, 'structured-output', 'error_types')


# =============================================================================
# VALIDATION METRICS
# =============================================================================
def create_validation_metrics_charts():
    print('\nValidation Metrics:')

    # Chart 1: ROC Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    fpr = np.linspace(0, 1, 100)
    tpr_good = 1 - (1 - fpr) ** 3
    tpr_fair = 1 - (1 - fpr) ** 1.5
    ax.plot(fpr, tpr_good, color=MLGREEN, lw=2, label='Good Model (AUC=0.92)')
    ax.plot(fpr, tpr_fair, color=MLORANGE, lw=2, label='Fair Model (AUC=0.75)')
    ax.plot([0, 1], [0, 1], color=MLGRAY, ls='--', lw=2, label='Random (AUC=0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_chart(fig, 'validation-metrics', 'roc_curve')

    # Chart 2: Precision-Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = np.linspace(0, 1, 20)
    precision = 0.9 - 0.3 * thresholds + np.random.normal(0, 0.02, 20)
    recall = 1 - 0.8 * thresholds + np.random.normal(0, 0.02, 20)
    ax.plot(thresholds, precision, 'o-', color=MLBLUE, lw=2, label='Precision')
    ax.plot(thresholds, recall, 's-', color=MLORANGE, lw=2, label='Recall')
    ax.axvline(0.5, color=MLRED, ls='--', lw=2, label='Default Threshold')
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    save_chart(fig, 'validation-metrics', 'precision_recall')

    # Chart 3: Cross-Validation
    fig, ax = plt.subplots(figsize=(10, 6))
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    scores = [0.85, 0.88, 0.82, 0.87, 0.84]
    mean_score = np.mean(scores)
    bars = ax.bar(folds, scores, color=MLBLUE, width=0.6, alpha=0.7)
    ax.axhline(mean_score, color=MLRED, ls='--', lw=2, label=f'Mean: {mean_score:.2f}')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    save_chart(fig, 'validation-metrics', 'cross_validation')


# =============================================================================
# A/B TESTING
# =============================================================================
def create_ab_testing_charts():
    print('\nA/B Testing:')

    # Chart 1: Conversion Rates
    fig, ax = plt.subplots(figsize=(10, 6))
    variants = ['Control (A)', 'Variant (B)']
    conversions = [3.2, 4.1]
    errors = [0.3, 0.35]
    colors = [MLBLUE, MLGREEN]
    bars = ax.bar(variants, conversions, color=colors, width=0.5, yerr=errors, capsize=10)
    ax.set_ylabel('Conversion Rate (%)', fontsize=12)
    ax.set_title('A/B Test Results: Conversion Rate', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 6)
    for bar, conv in zip(bars, conversions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{conv}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(0.5, 5.5, '+28% improvement', ha='center', fontsize=11, color=MLGREEN, fontweight='bold')
    save_chart(fig, 'ab-testing', 'conversion_rates')

    # Chart 2: Statistical Significance
    fig, ax = plt.subplots(figsize=(10, 6))
    days = np.arange(1, 15)
    p_values = 0.5 * np.exp(-0.3 * days) + np.random.normal(0, 0.02, 14)
    ax.plot(days, p_values, 'o-', color=MLPURPLE, lw=2, markersize=8)
    ax.axhline(0.05, color=MLRED, ls='--', lw=2, label='Significance Level (p=0.05)')
    ax.fill_between(days, 0, 0.05, alpha=0.2, color=MLGREEN, label='Significant Zone')
    ax.set_xlabel('Days Running', fontsize=12)
    ax.set_ylabel('P-Value', fontsize=12)
    ax.set_title('Statistical Significance Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(1, 14)
    ax.set_ylim(0, 0.6)
    save_chart(fig, 'ab-testing', 'statistical_significance')

    # Chart 3: Sample Size Calculator
    fig, ax = plt.subplots(figsize=(10, 6))
    effect_sizes = [0.5, 1.0, 2.0, 5.0, 10.0]
    sample_sizes = [62000, 16000, 4000, 700, 200]
    ax.plot(effect_sizes, sample_sizes, 'o-', color=MLBLUE, lw=2, markersize=10)
    ax.set_xlabel('Minimum Detectable Effect (%)', fontsize=12)
    ax.set_ylabel('Required Sample Size per Variant', fontsize=12)
    ax.set_title('Sample Size vs Effect Size', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    for x, y in zip(effect_sizes, sample_sizes):
        ax.annotate(f'{y:,}', (x, y), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
    save_chart(fig, 'ab-testing', 'sample_size')


# =============================================================================
# FINANCE APPLICATIONS
# =============================================================================
def create_finance_applications_charts():
    print('\nFinance Applications:')

    # Chart 1: Risk Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    cumulative = np.cumprod(1 + returns) * 100
    ax.plot(cumulative, color=MLBLUE, lw=1.5)
    var_95 = np.percentile(returns, 5) * 100
    ax.axhline(100 + var_95 * 10, color=MLRED, ls='--', lw=2, label=f'VaR 95%: {var_95:.2f}%')
    ax.fill_between(range(len(cumulative)), cumulative, 100, where=cumulative < 100, alpha=0.3, color=MLRED)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.set_title('Portfolio Performance & Value at Risk', fontsize=14, fontweight='bold')
    ax.legend()
    save_chart(fig, 'finance-applications', 'risk_metrics')

    # Chart 2: Portfolio Allocation
    fig, ax = plt.subplots(figsize=(10, 6))
    assets = ['Stocks', 'Bonds', 'Real Estate', 'Commodities', 'Cash']
    allocations = [45, 25, 15, 10, 5]
    colors = [MLBLUE, MLGREEN, MLORANGE, MLPURPLE, MLGRAY]
    wedges, texts, autotexts = ax.pie(allocations, labels=assets, colors=colors, autopct='%1.0f%%',
                                       startangle=90, explode=[0.05, 0, 0, 0, 0])
    ax.set_title('ML-Optimized Portfolio Allocation', fontsize=14, fontweight='bold')
    save_chart(fig, 'finance-applications', 'portfolio_allocation')

    # Chart 3: Prediction Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['ARIMA', 'Random\nForest', 'LSTM', 'Transformer', 'Ensemble']
    accuracy = [62, 71, 78, 82, 85]
    colors = [MLGRAY, MLBLUE, MLGREEN, MLPURPLE, MLORANGE]
    bars = ax.bar(models, accuracy, color=colors, width=0.6)
    ax.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax.set_title('Stock Price Direction Prediction', fontsize=14, fontweight='bold')
    ax.set_ylim(50, 100)
    ax.axhline(50, color=MLGRAY, ls='--', alpha=0.5, label='Random Guess')
    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc}%',
                ha='center', va='bottom', fontsize=10)
    save_chart(fig, 'finance-applications', 'prediction_accuracy')


# =============================================================================
# MAIN
# =============================================================================
def main():
    print('Generating web charts for all topics...')

    create_ml_foundations_charts()
    create_supervised_learning_charts()
    create_unsupervised_learning_charts()
    create_neural_networks_charts()
    create_clustering_charts()
    create_nlp_sentiment_charts()
    create_classification_charts()
    create_generative_ai_charts()
    create_topic_modeling_charts()
    create_responsible_ai_charts()
    create_structured_output_charts()
    create_validation_metrics_charts()
    create_ab_testing_charts()
    create_finance_applications_charts()

    print('\nAll charts generated successfully!')


if __name__ == '__main__':
    main()
