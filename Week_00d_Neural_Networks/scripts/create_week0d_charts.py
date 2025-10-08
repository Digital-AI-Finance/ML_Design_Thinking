#!/usr/bin/env python3
"""
Week 0d Neural Networks Chart Generation
Creates 15 visualizations for the neural networks presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from sklearn.datasets import make_classification, make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create charts directory if it doesn't exist
import os
os.makedirs('../charts', exist_ok=True)

def save_chart(name):
    """Save chart in both PDF and PNG formats"""
    plt.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Chart 1: Hierarchical Features
def create_hierarchical_features():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # Level 1: Raw pixels
    np.random.seed(42)
    pixels = np.random.rand(8, 8)
    axes[0].imshow(pixels, cmap='gray')
    axes[0].set_title('Raw Pixels\n(Meaningless)', fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Level 2: Edges
    edges = np.zeros((8, 8))
    edges[2:6, 1] = 1  # Vertical edge
    edges[1, 2:6] = 1  # Horizontal edge
    axes[1].imshow(edges, cmap='hot')
    axes[1].set_title('Edges\n(Local Features)', fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Level 3: Shapes
    shapes = np.zeros((8, 8))
    # Draw a simple square
    shapes[2:6, 2:6] = 0.5
    shapes[2, 2:6] = 1
    shapes[5, 2:6] = 1
    shapes[2:6, 2] = 1
    shapes[2:6, 5] = 1
    axes[2].imshow(shapes, cmap='hot')
    axes[2].set_title('Shapes\n(Mid-level)', fontsize=10)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Level 4: Objects
    axes[3].text(0.5, 0.5, 'CAR', fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].set_title('Objects\n(High-level)', fontsize=10)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    plt.tight_layout()
    save_chart('hierarchical_features')

# Chart 2: Perceptron Limitation
def create_perceptron_limitation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Linear separable case
    np.random.seed(42)
    X1 = np.random.randn(50, 2) + [1, 1]
    X2 = np.random.randn(50, 2) + [-1, -1]

    ax1.scatter(X1[:, 0], X1[:, 1], c='red', label='Class 1', alpha=0.7)
    ax1.scatter(X2[:, 0], X2[:, 1], c='blue', label='Class 2', alpha=0.7)

    # Draw decision boundary
    x_line = np.linspace(-4, 4, 100)
    y_line = -x_line  # Simple linear boundary
    ax1.plot(x_line, y_line, 'k-', linewidth=2, label='Decision Boundary')

    ax1.set_title('✓ Linear Separable\n(Perceptron Works)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Non-linear case (XOR-like)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    colors = ['red' if y == 0 else 'blue' for y in y_xor]
    ax2.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, alpha=0.8)

    # Try to draw linear boundary (impossible)
    ax2.plot([0.5, 0.5], [-0.2, 1.2], 'k--', linewidth=2, alpha=0.5, label='Impossible boundary')
    ax2.plot([-0.2, 1.2], [0.5, 0.5], 'k--', linewidth=2, alpha=0.5)

    ax2.set_title('✗ Non-linear Pattern\n(Perceptron Fails)', fontsize=12)
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_chart('perceptron_limitation')

# Chart 3: XOR Problem
def create_xor_problem():
    fig, ax = plt.subplots(figsize=(8, 6))

    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Plot points with labels
    colors = ['red', 'blue', 'blue', 'red']
    labels = ['0', '1', '1', '0']

    for i in range(4):
        ax.scatter(X[i, 0], X[i, 1], c=colors[i], s=300, alpha=0.8)
        ax.annotate(f'({X[i,0]}, {X[i,1]}) → {labels[i]}',
                   (X[i, 0], X[i, 1]), xytext=(10, 10),
                   textcoords='offset points', fontsize=12)

    # Try different linear boundaries (all fail)
    x_vals = np.linspace(-0.5, 1.5, 100)

    # Horizontal line
    ax.plot(x_vals, np.ones_like(x_vals) * 0.5, 'k--', alpha=0.5, label='Horizontal')

    # Vertical line
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Vertical')

    # Diagonal line
    ax.plot(x_vals, x_vals, 'purple', linestyle='--', alpha=0.5, label='Diagonal')

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('x₁', fontsize=14)
    ax.set_ylabel('x₂', fontsize=14)
    ax.set_title('XOR Problem: No Linear Separation Possible', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box explaining the problem
    textstr = 'No single line can separate\nred from blue points'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    save_chart('xor_problem')

# Chart 4: Universal Approximation
def create_universal_approximation():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Target function (complex non-linear)
    x = np.linspace(-2, 2, 1000)
    target = np.sin(3*x) * np.exp(-x**2/2) + 0.5*np.cos(5*x)

    # Neural network approximations with different numbers of neurons
    np.random.seed(42)

    def neural_net_approx(x, n_neurons):
        """Simulate neural network approximation"""
        weights_input = np.random.randn(n_neurons, 1) * 2
        biases = np.random.randn(n_neurons) * 2
        weights_output = np.random.randn(n_neurons) * 0.5

        # Forward pass simulation
        hidden = np.tanh(weights_input @ x.reshape(1, -1) + biases.reshape(-1, 1))
        output = weights_output @ hidden
        return output.flatten()

    ax.plot(x, target, 'k-', linewidth=3, label='Target Function', alpha=0.8)

    # Different approximations
    neurons = [5, 10, 20, 50]
    colors = ['red', 'blue', 'green', 'orange']

    for n, color in zip(neurons, colors):
        approx = neural_net_approx(x, n)
        # Scale and shift to roughly match target
        approx = approx * 0.3 + np.mean(target)
        ax.plot(x, approx, color=color, linewidth=2, alpha=0.7,
               label=f'{n} neurons')

    ax.set_xlabel('Input x', fontsize=12)
    ax.set_ylabel('Output f(x)', fontsize=12)
    ax.set_title('Universal Approximation: More Neurons → Better Fit', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_chart('universal_approximation')

# Chart 5: Neurons vs Layers
def create_neurons_vs_layers():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data showing exponential growth in width vs linear in depth
    input_size = np.arange(2, 21)

    # Width requirement (exponential)
    width_neurons = 2**(input_size - 1)

    # Depth requirement (linear)
    depth_layers = input_size

    ax.semilogy(input_size, width_neurons, 'ro-', linewidth=3, markersize=8,
               label='Single Layer (Width)', alpha=0.8)
    ax.semilogy(input_size, depth_layers * 10, 'bo-', linewidth=3, markersize=8,
               label='Multiple Layers (Depth × 10)', alpha=0.8)

    ax.set_xlabel('Problem Size (n bits)', fontsize=12)
    ax.set_ylabel('Number of Parameters Needed', fontsize=12)
    ax.set_title('Curse of Width vs Blessing of Depth', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Exponential explosion!', xy=(15, 2**14), xytext=(12, 2**16),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12, color='red')

    save_chart('neurons_vs_layers')

# Chart 6: MLP Architecture
def create_mlp_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Network structure
    layers = [2, 4, 3, 1]  # Input, Hidden1, Hidden2, Output
    layer_names = ['Input\n(x₁, x₂)', 'Hidden Layer\n(4 neurons)', 'Hidden Layer\n(3 neurons)', 'Output\n(y)']

    # Position neurons
    positions = {}
    for l, (layer_size, name) in enumerate(zip(layers, layer_names)):
        x_pos = l * 3
        for n in range(layer_size):
            y_pos = n - layer_size/2 + 0.5
            positions[(l, n)] = (x_pos, y_pos)

            # Draw neuron
            circle = Circle((x_pos, y_pos), 0.3, facecolor='lightblue',
                          edgecolor='black', linewidth=2)
            ax.add_patch(circle)

    # Draw connections
    for l in range(len(layers)-1):
        for n1 in range(layers[l]):
            for n2 in range(layers[l+1]):
                start = positions[(l, n1)]
                end = positions[(l+1, n2)]
                ax.plot([start[0]+0.3, end[0]-0.3], [start[1], end[1]],
                       'gray', alpha=0.5, linewidth=1)

    # Add layer labels
    for l, name in enumerate(layer_names):
        ax.text(l*3, -3, name, ha='center', va='top', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # Add activation functions
    for l in range(1, len(layers)):
        ax.text(l*3, 2.5, 'σ(·)', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))

    ax.set_xlim(-1, 10)
    ax.set_ylim(-4, 3)
    ax.set_title('Multi-Layer Perceptron Architecture', fontsize=14)
    ax.axis('off')

    save_chart('mlp_architecture')

# Chart 7: XOR Solution
def create_xor_solution():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    # Hidden neuron 1: x1 OR x2
    h1 = [0, 1, 1, 1]  # OR gate
    ax1.scatter(X[:, 0], X[:, 1], c=['red' if h==0 else 'blue' for h in h1], s=200)
    ax1.set_title('Hidden Neuron 1: x₁ OR x₂', fontsize=12)
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.3)

    # Hidden neuron 2: x1 AND x2
    h2 = [0, 0, 0, 1]  # AND gate
    ax2.scatter(X[:, 0], X[:, 1], c=['red' if h==0 else 'blue' for h in h2], s=200)
    ax2.set_title('Hidden Neuron 2: x₁ AND x₂', fontsize=12)
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, alpha=0.3)

    # Feature space (h1, h2)
    H = np.array([[h1[i], h2[i]] for i in range(4)])
    ax3.scatter(H[:, 0], H[:, 1], c=['red' if y==0 else 'blue' for y in y_xor], s=200)
    ax3.set_xlabel('h₁ (OR)', fontsize=12)
    ax3.set_ylabel('h₂ (AND)', fontsize=12)
    ax3.set_title('Feature Space: Now Linearly Separable!', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Draw separating line in feature space
    ax3.plot([0.5, 0.5], [-0.2, 1.2], 'k-', linewidth=3, label='Decision Boundary')
    ax3.legend()

    # Network diagram
    ax4.text(0.1, 0.8, 'x₁', fontsize=14, bbox=dict(boxstyle="circle", facecolor="lightblue"))
    ax4.text(0.1, 0.2, 'x₂', fontsize=14, bbox=dict(boxstyle="circle", facecolor="lightblue"))

    ax4.text(0.5, 0.7, 'OR', fontsize=12, bbox=dict(boxstyle="circle", facecolor="lightgreen"))
    ax4.text(0.5, 0.3, 'AND', fontsize=12, bbox=dict(boxstyle="circle", facecolor="lightgreen"))

    ax4.text(0.9, 0.5, 'XOR', fontsize=14, bbox=dict(boxstyle="circle", facecolor="lightyellow"))

    # Draw connections
    connections = [
        ((0.15, 0.8), (0.45, 0.7)),   # x1 to OR
        ((0.15, 0.8), (0.45, 0.3)),   # x1 to AND
        ((0.15, 0.2), (0.45, 0.7)),   # x2 to OR
        ((0.15, 0.2), (0.45, 0.3)),   # x2 to AND
        ((0.55, 0.7), (0.85, 0.5)),   # OR to output
        ((0.55, 0.3), (0.85, 0.5)),   # AND to output
    ]

    for start, end in connections:
        ax4.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.7)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Network Solution', fontsize=12)
    ax4.axis('off')

    plt.tight_layout()
    save_chart('xor_solution')

# Chart 8: MLP Success Boundaries
def create_mlp_success_boundaries():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Different datasets
    datasets = [
        ("Linear", make_classification(n_samples=200, n_features=2, n_redundant=0,
                                     n_informative=2, n_clusters_per_class=1, random_state=42)),
        ("Moons", make_classification(n_samples=200, n_features=2, n_redundant=0,
                                    n_informative=2, n_clusters_per_class=1, random_state=1)),
        ("Circles", make_blobs(n_samples=200, centers=4, random_state=42)),
        ("XOR-like", make_classification(n_samples=200, n_features=2, n_redundant=0,
                                       n_informative=2, n_clusters_per_class=2, random_state=8)),
        ("Complex", make_classification(n_samples=200, n_features=2, n_redundant=0,
                                      n_informative=2, n_clusters_per_class=1, random_state=15)),
        ("Spirals", make_classification(n_samples=200, n_features=2, n_redundant=0,
                                      n_informative=2, n_clusters_per_class=1, random_state=25))
    ]

    for i, (name, (X, y)) in enumerate(datasets):
        # Train MLP
        mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
        mlp.fit(X, y)

        # Create decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        axes[i].set_title(f'{name}\nAccuracy: {accuracy_score(y, mlp.predict(X)):.2f}', fontsize=10)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_chart('mlp_success_boundaries')

# Chart 9: Vanishing Gradients Data
def create_vanishing_gradients_data():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Performance vs depth data
    layers = np.arange(1, 7)
    accuracy = [87.2, 91.5, 89.3, 82.1, 71.8, 58.3]
    training_time = [10, 15, 25, 40, 60, 90]

    ax1.plot(layers, accuracy, 'ro-', linewidth=3, markersize=8, label='Final Accuracy')
    ax1.set_xlabel('Number of Layers', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Deeper Networks Perform Worse', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add annotation
    ax1.annotate('Peak at 2 layers', xy=(2, 91.5), xytext=(3.5, 95),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12, color='red')

    # Training time
    ax2.bar(layers, training_time, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Number of Layers', fontsize=12)
    ax2.set_ylabel('Training Time (minutes)', fontsize=12)
    ax2.set_title('Training Time Increases Exponentially', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_chart('vanishing_gradients_data')

# Chart 10: Gradient Decay
def create_gradient_decay():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulate gradient magnitudes across layers
    layers = np.arange(1, 11)

    # Sigmoid case (vanishing)
    sigmoid_grads = (0.25) ** (layers - 1)

    # ReLU case (better)
    relu_grads = 0.5 ** (layers - 1)

    # Well-behaved case
    good_grads = 0.9 ** (layers - 1)

    ax.semilogy(layers, sigmoid_grads, 'ro-', linewidth=3, markersize=8,
               label='Sigmoid (≤ 0.25 per layer)', alpha=0.8)
    ax.semilogy(layers, relu_grads, 'bo-', linewidth=3, markersize=8,
               label='ReLU (≈ 0.5 per layer)', alpha=0.8)
    ax.semilogy(layers, good_grads, 'go-', linewidth=3, markersize=8,
               label='Well-behaved (≈ 0.9 per layer)', alpha=0.8)

    ax.set_xlabel('Layer Depth', fontsize=12)
    ax.set_ylabel('Gradient Magnitude', fontsize=12)
    ax.set_title('Gradient Decay: Why Deep Networks Failed', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add threshold line
    ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Vanishing threshold')
    ax.text(6, 1e-5, 'Vanishing threshold', fontsize=10, color='red')

    save_chart('gradient_decay')

# Chart 11: Gradient Flow Layers
def create_gradient_flow_layers():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Network depth
    layers = ['Input', 'L1', 'L2', 'L3', 'L4', 'L5', 'Output']
    x_pos = np.arange(len(layers))

    # Gradient magnitudes (decreasing exponentially)
    gradients = [1.0, 0.25, 0.06, 0.015, 0.004, 0.001, 0.0002]

    # Create gradient flow visualization
    bars = ax.bar(x_pos, gradients, color='lightcoral', alpha=0.7, edgecolor='black')

    # Color code by severity
    for i, bar in enumerate(bars):
        if gradients[i] < 0.01:
            bar.set_color('red')
        elif gradients[i] < 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('green')

    ax.set_xlabel('Network Layer', fontsize=12)
    ax.set_ylabel('Gradient Magnitude', fontsize=12)
    ax.set_title('Gradient Flow: Early Layers Get Tiny Updates', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate('Strong gradients', xy=(6, 0.0002), xytext=(5, 0.01),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=12, color='green')

    ax.annotate('Vanished gradients', xy=(1, 0.25), xytext=(2, 0.1),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12, color='red')

    save_chart('gradient_flow_layers')

# Chart 12: Human Visual Hierarchy
def create_human_visual_hierarchy():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a visual hierarchy representation
    levels = ['V1 (Edges)', 'V2/V4 (Textures)', 'IT (Objects)']
    y_positions = [0.7, 0.4, 0.1]

    # Level 1: Edges
    ax.text(0.1, 0.7, 'V1 Visual Cortex', fontsize=14, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # Draw edge detectors
    edge_patterns = [
        '│', '─', '/', '\\', '┐', '└'
    ]
    for i, pattern in enumerate(edge_patterns):
        ax.text(0.25 + i*0.08, 0.7, pattern, fontsize=20, ha='center',
               bbox=dict(boxstyle="circle", facecolor="white", edgecolor="blue"))

    # Level 2: Textures/Shapes
    ax.text(0.1, 0.4, 'V2/V4 Areas', fontsize=14, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    texture_labels = ['Curves', 'Corners', 'Textures', 'Patterns', 'Colors']
    for i, label in enumerate(texture_labels):
        ax.text(0.25 + i*0.12, 0.4, label, fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green"))

    # Level 3: Objects
    ax.text(0.1, 0.1, 'IT Cortex', fontsize=14, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    objects = ['👤 Face', '🚗 Car', '🏠 House', '🐕 Animal', '✋ Hand']
    for i, obj in enumerate(objects):
        ax.text(0.25 + i*0.12, 0.1, obj, fontsize=12, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="orange"))

    # Draw arrows showing information flow
    for i in range(3):
        if i < 2:
            ax.annotate('', xy=(0.5, y_positions[i+1] + 0.05),
                       xytext=(0.5, y_positions[i] - 0.05),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    ax.set_title('Human Visual Processing Hierarchy', fontsize=16, weight='bold')
    ax.axis('off')

    save_chart('human_visual_hierarchy')

# Chart 13: Architecture Data Matching
def create_architecture_data_matching():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Images -> CNNs
    ax1.imshow(np.random.rand(8, 8), cmap='gray')
    ax1.set_title('Images → CNNs\n(Spatial Locality)', fontsize=12, weight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Sequences -> RNNs
    sequence = ['Hello', 'world', 'how', 'are', 'you']
    for i, word in enumerate(sequence):
        ax2.text(i*0.8, 0.5, word, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        if i < len(sequence)-1:
            ax2.annotate('', xy=((i+1)*0.8-0.2, 0.5), xytext=(i*0.8+0.2, 0.5),
                        arrowprops=dict(arrowstyle='->', lw=2))

    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(0, 1)
    ax2.set_title('Sequences → RNNs\n(Temporal Order)', fontsize=12, weight='bold')
    ax2.axis('off')

    # Graphs -> GNNs
    # Simple graph visualization
    nodes = [(0.2, 0.8), (0.8, 0.8), (0.5, 0.2), (0.2, 0.2), (0.8, 0.2)]
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]

    for node in nodes:
        circle = Circle(node, 0.1, facecolor='lightgreen', edgecolor='black')
        ax3.add_patch(circle)

    for edge in edges:
        start, end = nodes[edge[0]], nodes[edge[1]]
        ax3.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Graphs → GNNs\n(Node Relationships)', fontsize=12, weight='bold')
    ax3.axis('off')

    # Language -> Transformers
    words = ['The', 'cat', 'sat', 'on', 'mat']
    attention_matrix = np.random.rand(5, 5)
    np.fill_diagonal(attention_matrix, 1.0)

    im = ax4.imshow(attention_matrix, cmap='Blues')
    ax4.set_xticks(range(5))
    ax4.set_yticks(range(5))
    ax4.set_xticklabels(words, fontsize=10)
    ax4.set_yticklabels(words, fontsize=10)
    ax4.set_title('Language → Transformers\n(Long-range Dependencies)', fontsize=12, weight='bold')

    plt.tight_layout()
    save_chart('architecture_data_matching')

# Chart 14: Convolution Intuition
def create_convolution_intuition():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Input image
    np.random.seed(42)
    image = np.random.rand(6, 6)
    # Add some structure
    image[2:4, 1:5] = 0.8  # Horizontal edge

    ax1.imshow(image, cmap='gray')
    ax1.set_title('Input Image (6×6)', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Filter
    filter_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ax2.imshow(filter_kernel, cmap='RdBu')
    ax2.set_title('Vertical Edge Filter (3×3)', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add values to filter
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{filter_kernel[i, j]}', ha='center', va='center',
                    fontsize=12, color='white' if abs(filter_kernel[i, j]) > 0.5 else 'black')

    # Convolution process visualization
    # Show sliding window
    ax3.imshow(image, cmap='gray', alpha=0.7)

    # Highlight the 3x3 region being processed
    rect = Rectangle((0.5, 1.5), 3, 3, linewidth=3, edgecolor='red', facecolor='none')
    ax3.add_patch(rect)
    ax3.set_title('Sliding Filter Across Image', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Output feature map (simplified)
    from scipy import ndimage
    output = ndimage.convolve(image, filter_kernel, mode='constant')
    ax4.imshow(output, cmap='hot')
    ax4.set_title('Output Feature Map\n(Edge Response)', fontsize=12)
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.tight_layout()
    save_chart('convolution_intuition')

# Chart 15: Deep Learning Timeline
def create_deep_learning_timeline():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Timeline data
    events = [
        (2012, 'AlexNet', 'CNNs revolutionize ImageNet', 'red'),
        (2014, 'GAN', 'Generative Adversarial Networks', 'blue'),
        (2014, 'Seq2Seq', 'Sequence to Sequence Models', 'green'),
        (2015, 'ResNet', 'Residual Networks solve vanishing gradients', 'orange'),
        (2017, 'Transformer', 'Attention Is All You Need', 'purple'),
        (2018, 'BERT', 'Bidirectional Encoder Representations', 'brown'),
        (2019, 'GPT-2', 'Language model scaling begins', 'pink'),
        (2020, 'GPT-3', '175B parameters, few-shot learning', 'red'),
        (2021, 'DALL-E', 'Text to image generation', 'blue'),
        (2022, 'ChatGPT', 'Conversational AI mainstream', 'green'),
        (2023, 'GPT-4', 'Multimodal AI capabilities', 'gold')
    ]

    years = [event[0] for event in events]
    names = [event[1] for event in events]
    descriptions = [event[2] for event in events]
    colors = [event[3] for event in events]

    # Create timeline
    ax.scatter(years, [1]*len(years), c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)

    # Add event labels
    for i, (year, name, desc, color) in enumerate(events):
        offset = 0.1 if i % 2 == 0 else -0.1
        ax.annotate(f'{name}\n{desc}', xy=(year, 1), xytext=(year, 1 + offset),
                   ha='center', va='bottom' if offset > 0 else 'top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                   fontsize=9, rotation=45 if len(desc) > 20 else 0)

    # Draw timeline line
    ax.plot([2011, 2024], [1, 1], 'k-', linewidth=3, alpha=0.5)

    ax.set_xlim(2011, 2024)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_title('Deep Learning Evolution Timeline', fontsize=16, weight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    save_chart('deep_learning_timeline')

# Additional charts needed for completeness
def create_filter_hierarchy():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Layer 1: Low-level features
    ax = axes[0]
    filters = ['─', '│', '/', '\\', '┌', '┐']
    for i, f in enumerate(filters):
        x, y = i % 3, i // 3
        ax.text(x, y, f, fontsize=20, ha='center', va='center',
               bbox=dict(boxstyle="square", facecolor="lightblue"))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('Layer 1: Edges & Lines', fontsize=12)
    ax.axis('off')

    # Layer 2: Mid-level features
    ax = axes[1]
    patterns = ['Curves', 'Corners', 'Texture', 'Blob']
    for i, p in enumerate(patterns):
        x, y = i % 2, i // 2
        ax.text(x, y, p, fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle="round", facecolor="lightgreen"))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('Layer 2: Textures & Shapes', fontsize=12)
    ax.axis('off')

    # Layer 3: High-level features
    ax = axes[2]
    objects = ['Eye', 'Wheel', 'Face', 'Car']
    for i, obj in enumerate(objects):
        x, y = i % 2, i // 2
        ax.text(x, y, obj, fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle="round", facecolor="lightyellow"))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('Layer 3: Object Parts', fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    save_chart('filter_hierarchy')

def create_cnn_architecture_detailed():
    fig, ax = plt.subplots(figsize=(14, 8))

    # CNN pipeline stages
    stages = ['Input\n32×32×3', 'Conv\n28×28×6', 'Pool\n14×14×6', 'Conv\n10×10×16', 'Pool\n5×5×16', 'FC\n120', 'FC\n84', 'Output\n10']
    x_positions = np.linspace(0, 14, len(stages))

    for i, (stage, x) in enumerate(zip(stages, x_positions)):
        if 'Conv' in stage or 'Pool' in stage or 'Input' in stage:
            # Draw as 3D block
            width = 1.5 if 'Input' in stage else 1.0
            height = 2.0
            rect = Rectangle((x-width/2, 0), width, height,
                           facecolor='lightblue' if 'Conv' in stage else 'lightcoral' if 'Pool' in stage else 'lightgray',
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        else:
            # Draw as circle for FC layers
            circle = Circle((x, 1), 0.5, facecolor='lightgreen', edgecolor='black', linewidth=2)
            ax.add_patch(circle)

        ax.text(x, -0.8, stage, ha='center', va='top', fontsize=10, weight='bold')

        if i < len(stages) - 1:
            ax.arrow(x + 0.8, 1, 0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')

    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 3)
    ax.set_title('CNN Architecture: Feature Extraction → Classification', fontsize=14, weight='bold')
    ax.axis('off')

    save_chart('cnn_architecture_detailed')

def create_convolution_calculation():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Input matrix
    input_matrix = np.array([[1, 2, 1], [0, 1, 2], [1, 0, 1]])
    im1 = ax1.imshow(input_matrix, cmap='Blues')
    ax1.set_title('Input (3×3)', fontsize=12, weight='bold')
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{input_matrix[i,j]}', ha='center', va='center', fontsize=14, weight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Filter
    filter_matrix = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    im2 = ax2.imshow(filter_matrix, cmap='RdBu')
    ax2.set_title('Filter (3×3)', fontsize=12, weight='bold')
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{filter_matrix[i,j]}', ha='center', va='center', fontsize=14, weight='bold',
                    color='white' if abs(filter_matrix[i,j]) > 1 else 'black')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Element-wise multiplication
    product = input_matrix * filter_matrix
    im3 = ax3.imshow(product, cmap='RdYlBu')
    ax3.set_title('Element-wise Product', fontsize=12, weight='bold')
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f'{product[i,j]}', ha='center', va='center', fontsize=14, weight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Final sum
    result = np.sum(product)
    ax4.text(0.5, 0.5, f'Sum = {result}', ha='center', va='center', fontsize=24, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="black", linewidth=2))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Output Value', fontsize=12, weight='bold')
    ax4.axis('off')

    plt.tight_layout()
    save_chart('convolution_calculation')

def create_rnn_transformer_comparison():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # RNN visualization
    words = ['The', 'cat', 'sat', 'on', 'mat']
    x_pos = np.arange(len(words))

    # Draw RNN cells
    for i, word in enumerate(words):
        # Word input
        ax1.text(i, 0, word, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round", facecolor="lightblue"))

        # Hidden state
        ax1.text(i, 1, f'h{i}', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="circle", facecolor="lightgreen"))

        # Arrows
        ax1.arrow(i, 0.3, 0, 0.4, head_width=0.1, head_length=0.1, fc='black', ec='black')

        if i > 0:
            ax1.arrow(i-0.7, 1, 0.4, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')

    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title('RNN: Sequential Processing', fontsize=14, weight='bold')
    ax1.axis('off')

    # Transformer attention visualization
    attention_matrix = np.random.rand(5, 5)
    np.fill_diagonal(attention_matrix, 1.0)
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

    im = ax2.imshow(attention_matrix, cmap='Blues')
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(words)
    ax2.set_yticklabels(words)
    ax2.set_title('Transformer: Parallel Attention', fontsize=14, weight='bold')

    # Add colorbar
    plt.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()
    save_chart('rnn_transformer_comparison')

def create_remaining_charts():
    # Charts that might be missing - let me create placeholders

    # Feature maps and attention
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # CNN feature maps
    feature_map = np.random.rand(8, 8)
    ax1.imshow(feature_map, cmap='hot')
    ax1.set_title('CNN Feature Map\n(Edge Detection)', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Multiple feature maps
    for i in range(3):
        fm = np.random.rand(6, 6)
        ax2.imshow(fm, extent=[i*2, i*2+1.5, 0, 1.5], cmap='viridis', alpha=0.7)
    ax2.set_title('Multiple Feature Maps\n(Different Filters)', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Attention heatmap
    words = ['The', 'cat', 'sat']
    attention = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    im = ax3.imshow(attention, cmap='Blues')
    ax3.set_xticks(range(3))
    ax3.set_yticks(range(3))
    ax3.set_xticklabels(words)
    ax3.set_yticklabels(words)
    ax3.set_title('Attention Heatmap', fontsize=12)

    # Inductive bias reduction
    categories = ['All Functions', 'Linear Functions', 'CNNs', 'RNNs', 'Transformers']
    search_space = [1e10, 1e6, 1e4, 1e3, 1e2]

    ax4.bar(categories, search_space, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    ax4.set_yscale('log')
    ax4.set_ylabel('Search Space Size', fontsize=12)
    ax4.set_title('Inductive Bias Reduces Search Space', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_chart('feature_maps_attention')

    # ImageNet progress
    fig, ax = plt.subplots(figsize=(10, 6))

    years = [2010, 2012, 2013, 2014, 2014.5, 2015, 2017]
    models = ['Traditional', 'AlexNet', 'ZFNet', 'VGGNet', 'GoogLeNet', 'ResNet', 'DenseNet']
    errors = [28.2, 15.3, 14.8, 7.3, 6.7, 3.6, 2.2]

    ax.plot(years, errors, 'ro-', linewidth=3, markersize=8)
    ax.axhline(y=5.1, color='green', linestyle='--', linewidth=2, label='Human Performance')

    for year, model, error in zip(years, models, errors):
        ax.annotate(f'{model}\n{error}%', xy=(year, error), xytext=(5, 5),
                   textcoords='offset points', fontsize=9, rotation=0)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Top-5 Error Rate (%)', fontsize=12)
    ax.set_title('ImageNet Challenge: Progress Over Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_chart('imagenet_progress')

    # Design principles
    fig, ax = plt.subplots(figsize=(10, 8))

    principles = ['Locality', 'Hierarchy', 'Invariance', 'Efficiency']
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    # Create a 2x2 grid of principles
    positions = [(0.2, 0.7), (0.7, 0.7), (0.2, 0.2), (0.7, 0.2)]

    for principle, color, pos in zip(principles, colors, positions):
        circle = Circle(pos, 0.15, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], principle, ha='center', va='center', fontsize=14, weight='bold')

    # Add descriptions
    descriptions = [
        'Nearby elements\nare related',
        'Build complexity\ngradually',
        'Robust to\nirrelevant changes',
        'Parameter sharing\n& optimization'
    ]

    for desc, pos in zip(descriptions, positions):
        ax.text(pos[0], pos[1]-0.25, desc, ha='center', va='center', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Architecture Design Principles', fontsize=16, weight='bold')
    ax.axis('off')

    save_chart('design_principles')

    # Modern applications
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Computer Vision
    apps_cv = ['Object Detection', 'Image Segmentation', 'Medical Imaging', 'Autonomous Driving']
    ax1.pie([1, 1, 1, 1], labels=apps_cv, autopct='', startangle=90, colors=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax1.set_title('Computer Vision Applications', fontsize=12, weight='bold')

    # NLP
    apps_nlp = ['Translation', 'Text Generation', 'Q&A', 'Code Generation']
    ax2.pie([1, 1, 1, 1], labels=apps_nlp, autopct='', startangle=90, colors=['lightpink', 'lightgray', 'lightcyan', 'wheat'])
    ax2.set_title('NLP Applications', fontsize=12, weight='bold')

    # Multimodal
    ax3.text(0.5, 0.7, 'Image Captioning', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightblue"))
    ax3.text(0.5, 0.5, 'Visual Q&A', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgreen"))
    ax3.text(0.5, 0.3, 'Video Understanding', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Multimodal AI', fontsize=12, weight='bold')
    ax3.axis('off')

    # Summary preview
    ax4.text(0.5, 0.8, 'From Recognition', ha='center', va='center', fontsize=14, weight='bold',
            bbox=dict(boxstyle="round", facecolor="lightblue"))
    ax4.text(0.5, 0.5, '→', ha='center', va='center', fontsize=20, weight='bold')
    ax4.text(0.5, 0.2, 'To Generation', ha='center', va='center', fontsize=14, weight='bold',
            bbox=dict(boxstyle="round", facecolor="lightgreen"))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Next: Generative AI', fontsize=12, weight='bold')
    ax4.axis('off')

    plt.tight_layout()
    save_chart('modern_applications')

    save_chart('summary_preview')

    # Create inductive bias reduction chart
    fig, ax = plt.subplots(figsize=(10, 6))

    approaches = ['Fully\nConnected', 'With\nArchitecture']
    search_sizes = [1e12, 1e6]
    colors = ['red', 'green']

    bars = ax.bar(approaches, search_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel('Effective Search Space Size', fontsize=12)
    ax.set_title('Architecture Reduces Search Space Exponentially', fontsize=14, weight='bold')

    # Add value labels
    for bar, size in zip(bars, search_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
               f'{size:.0e}', ha='center', va='bottom', fontsize=12, weight='bold')

    ax.grid(True, alpha=0.3)
    save_chart('inductive_bias_reduction')

# Generate all charts
if __name__ == "__main__":
    print("Generating Week 0d Neural Networks charts...")

    chart_functions = [
        create_hierarchical_features,
        create_perceptron_limitation,
        create_xor_problem,
        create_universal_approximation,
        create_neurons_vs_layers,
        create_mlp_architecture,
        create_xor_solution,
        create_mlp_success_boundaries,
        create_vanishing_gradients_data,
        create_gradient_decay,
        create_gradient_flow_layers,
        create_human_visual_hierarchy,
        create_architecture_data_matching,
        create_convolution_intuition,
        create_deep_learning_timeline,
        create_filter_hierarchy,
        create_cnn_architecture_detailed,
        create_convolution_calculation,
        create_rnn_transformer_comparison,
        create_remaining_charts
    ]

    for i, func in enumerate(chart_functions, 1):
        print(f"Creating chart {i}/20: {func.__name__}")
        func()

    print("All charts generated successfully!")
    print("Charts saved to ../charts/ directory")