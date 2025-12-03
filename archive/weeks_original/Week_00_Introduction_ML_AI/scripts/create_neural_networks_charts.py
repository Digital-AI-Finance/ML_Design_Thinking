"""
Chart Generation for Week 0 Part 4: Neural Networks
WCAG AAA Compliant Color Palette
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc
import numpy as np

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


def create_perceptron_model():
    """Create perceptron architecture and decision boundary"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')

    input_nodes = [(2, y) for y in [6, 5, 4]]
    input_labels = ['x₁', 'x₂', 'x₃']

    for (x, y), label in zip(input_nodes, input_labels):
        circle = Circle((x, y), 0.4, color=COLORS['blue'], edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y, label, ha='center', va='center', fontsize=13, fontweight='bold', color='white')

    output_x, output_y = 7, 5
    circle = Circle((output_x, output_y), 0.5, color=COLORS['red'], edgecolor='black', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(output_x, output_y, 'y', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    weights = ['w₁', 'w₂', 'w₃']
    for (x_in, y_in), weight in zip(input_nodes, weights):
        arrow = FancyArrowPatch((x_in + 0.4, y_in), (output_x - 0.5, output_y),
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color=COLORS['gray'], alpha=0.7)
        ax1.add_patch(arrow)

        mid_x = (x_in + output_x) / 2
        mid_y = (y_in + output_y) / 2
        ax1.text(mid_x, mid_y + 0.3, weight, fontsize=11, style='italic', color=COLORS['gray'])

    ax1.text(output_x, output_y - 1.2, 'b', fontsize=11, style='italic', color=COLORS['gray'])

    ax1.text(5, 7.5, 'Perceptron Architecture', ha='center', fontsize=14, fontweight='bold')

    formula = r'$y = \sigma(w^T x + b)$'
    ax1.text(5, 1.5, formula, ha='center', fontsize=13, style='italic')

    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    ax2.scatter(X[y==0, 0], X[y==0, 1], c=COLORS['blue'], s=80, alpha=0.6,
               edgecolors='black', label='Class 0')
    ax2.scatter(X[y==1, 0], X[y==1, 1], c=COLORS['orange'], s=80, alpha=0.6,
               edgecolors='black', label='Class 1')

    x_line = np.linspace(-3, 3, 100)
    y_line = -x_line
    ax2.plot(x_line, y_line, 'r--', linewidth=3, label='Decision Boundary')

    ax2.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax2.set_title('Linear Decision Boundary', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('The Perceptron Model', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../charts/perceptron_model.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/perceptron_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created perceptron_model.pdf")


def create_mlp_architecture():
    """Create multi-layer perceptron structure"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    layer_configs = [
        ('Input\nLayer', 2, 7, COLORS['blue']),
        ('Hidden\nLayer 1', 5, 9, COLORS['orange']),
        ('Hidden\nLayer 2', 8, 7, COLORS['green']),
        ('Output\nLayer', 11, 5, COLORS['red'])
    ]

    node_positions = []

    for name, x, n_neurons, color in layer_configs:
        y_positions = np.linspace(2, 8, n_neurons)
        positions = []

        for y in y_positions:
            circle = Circle((x, y), 0.3, color=color, alpha=0.6, edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)
            positions.append((x, y))

        node_positions.append(positions)
        ax.text(x, 9.3, name, ha='center', va='top', fontsize=12, fontweight='bold')

    for i in range(len(node_positions) - 1):
        for x1, y1 in node_positions[i]:
            for x2, y2 in node_positions[i + 1]:
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5, alpha=0.2)

    annotations = [
        (3.5, 1.2, 'ReLU'),
        (6.5, 1.2, 'ReLU'),
        (9.5, 1.2, 'Softmax')
    ]

    for x, y, text in annotations:
        ax.text(x, y, text, ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.3))

    ax.text(7, 9.7, 'Multi-Layer Perceptron (MLP)', ha='center', fontsize=16, fontweight='bold')

    formula = r'$h^{(l+1)} = \sigma(W^{(l)} h^{(l)} + b^{(l)})$'
    ax.text(7, 0.5, formula, ha='center', fontsize=13, style='italic')

    plt.tight_layout()
    plt.savefig('../charts/mlp_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/mlp_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created mlp_architecture.pdf")


def create_backpropagation():
    """Create backpropagation computation graph"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    nodes = [
        ('x', 2, 5, COLORS['blue']),
        ('w', 2, 7, COLORS['blue']),
        ('z = wx', 5, 6, COLORS['orange']),
        ('a = σ(z)', 8, 6, COLORS['green']),
        ('L(a, y)', 11, 6, COLORS['red'])
    ]

    for label, x, y, color in nodes:
        box = FancyBboxPatch((x - 0.7, y - 0.4), 1.4, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black',
                            linewidth=2, alpha=0.6)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

    forward_arrows = [
        ((2.7, 5), (4.3, 5.7)),
        ((2.7, 7), (4.3, 6.3)),
        ((5.7, 6), (7.3, 6)),
        ((8.7, 6), (10.3, 6))
    ]

    for start, end in forward_arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color='black')
        ax.add_patch(arrow)

    ax.text(6.5, 7.5, 'Forward Pass', ha='center', fontsize=14,
           fontweight='bold', color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))

    backward_arrows = [
        ((10.3, 5.5), (8.7, 5.5)),
        ((7.3, 5.5), (5.7, 5.5)),
        ((4.3, 5.5), (2.7, 5.5))
    ]

    for start, end in backward_arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color=COLORS['red'], linestyle='--')
        ax.add_patch(arrow)

    gradient_labels = [
        (9.5, 5, r'$\frac{\partial L}{\partial a}$'),
        (6.5, 5, r'$\frac{\partial L}{\partial z}$'),
        (3.5, 5, r'$\frac{\partial L}{\partial w}$')
    ]

    for x, y, label in gradient_labels:
        ax.text(x, y, label, ha='center', fontsize=11, style='italic', color=COLORS['red'])

    ax.text(6.5, 2.5, 'Backward Pass (Gradient Flow)', ha='center', fontsize=14,
           fontweight='bold', color=COLORS['red'],
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.3))

    ax.text(7, 9.5, 'Backpropagation: Chain Rule in Action', ha='center',
           fontsize=16, fontweight='bold')

    formula = r'$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$'
    ax.text(7, 1, formula, ha='center', fontsize=13, style='italic')

    plt.tight_layout()
    plt.savefig('../charts/backpropagation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/backpropagation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created backpropagation.pdf")


def create_activation_functions():
    """Compare activation functions"""
    x = np.linspace(-5, 5, 200)

    functions = {
        'Sigmoid': (1 / (1 + np.exp(-x)), COLORS['blue']),
        'Tanh': (np.tanh(x), COLORS['orange']),
        'ReLU': (np.maximum(0, x), COLORS['green']),
        'Leaky ReLU': (np.where(x > 0, x, 0.1 * x), COLORS['red'])
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (name, (y, color)) in zip(axes.flat, functions.items()):
        ax.plot(x, y, linewidth=3, color=color)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('z', fontsize=12, fontweight='bold')
        ax.set_ylabel('σ(z)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if name == 'Sigmoid':
            formula = r'$\sigma(z) = \frac{1}{1+e^{-z}}$'
        elif name == 'Tanh':
            formula = r'$\sigma(z) = \tanh(z)$'
        elif name == 'ReLU':
            formula = r'$\sigma(z) = \max(0, z)$'
        else:
            formula = r'$\sigma(z) = \max(0.1z, z)$'

        ax.text(0.05, 0.95, formula, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Activation Functions Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../charts/activation_functions.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/activation_functions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created activation_functions.pdf")


def create_cnn_architecture():
    """Create CNN architecture visualization"""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    layers = [
        ('Input\n28×28×1', 1.5, 4, 4, COLORS['blue']),
        ('Conv\n24×24×32', 4, 3.5, 3.5, COLORS['orange']),
        ('Pool\n12×12×32', 6.5, 3, 3, COLORS['green']),
        ('Conv\n8×8×64', 9, 2.5, 2.5, COLORS['orange']),
        ('Pool\n4×4×64', 11.5, 2, 2, COLORS['green']),
        ('Flatten\n1024', 14, 0.5, 3, COLORS['purple'])
    ]

    for i, (label, x, w, h, color) in enumerate(layers):
        rect = Rectangle((x - w/2, 4 - h/2), w, h,
                        facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, 4, label, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        if i < len(layers) - 1:
            next_x = layers[i + 1][1]
            arrow = FancyArrowPatch((x + w/2, 4), (next_x - layers[i+1][2]/2, 4),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='black')
            ax.add_patch(arrow)

    fc_x = 15
    for i, y in enumerate([2, 3, 4, 5, 6]):
        circle = Circle((fc_x, y), 0.15, color=COLORS['red'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)

    ax.text(fc_x, 1, 'Output\n10 classes', ha='center', fontsize=10, fontweight='bold')

    ax.text(8, 7.5, 'Convolutional Neural Network Architecture', ha='center',
           fontsize=16, fontweight='bold')

    operations = [
        (4, 1, 'Conv: 5×5 filters'),
        (6.5, 1, 'MaxPool: 2×2'),
        (9, 1, 'Conv: 5×5 filters'),
        (11.5, 1, 'MaxPool: 2×2')
    ]

    for x, y, text in operations:
        ax.text(x, y, text, ha='center', fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('../charts/cnn_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/cnn_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created cnn_architecture.pdf")


def create_rnn_architecture():
    """Create RNN and LSTM architecture"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')

    time_steps = [2, 4, 6, 8]
    for i, x in enumerate(time_steps):
        box = FancyBboxPatch((x - 0.6, 4 - 0.6), 1.2, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=COLORS['blue'], edgecolor='black',
                            linewidth=2, alpha=0.6)
        ax1.add_patch(box)
        ax1.text(x, 4, f'RNN\nt={i}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        ax1.annotate('', xy=(x, 2.5), xytext=(x, 3.4),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['orange']))
        ax1.text(x, 2, f'x_{i}', ha='center', fontsize=11, fontweight='bold')

        ax1.annotate('', xy=(x, 6.5), xytext=(x, 5.4),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['green']))
        ax1.text(x, 7, f'h_{i}', ha='center', fontsize=11, fontweight='bold')

        if i < len(time_steps) - 1:
            ax1.annotate('', xy=(time_steps[i+1] - 0.7, 4), xytext=(x + 0.7, 4),
                        arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    ax1.text(5, 7.8, 'Recurrent Neural Network', ha='center',
            fontsize=13, fontweight='bold')

    formula = r'$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$'
    ax1.text(5, 1, formula, ha='center', fontsize=11, style='italic')

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')

    lstm_x = 5
    lstm_y = 4

    main_box = Rectangle((lstm_x - 1.5, lstm_y - 1.5), 3, 3,
                         facecolor=COLORS['purple'], edgecolor='black',
                         linewidth=2, alpha=0.3)
    ax2.add_patch(main_box)

    gates = [
        ('f_t', lstm_x - 0.8, lstm_y + 0.8, COLORS['red']),
        ('i_t', lstm_x, lstm_y + 0.8, COLORS['orange']),
        ('o_t', lstm_x + 0.8, lstm_y + 0.8, COLORS['green']),
        ('c_t', lstm_x, lstm_y, COLORS['blue'])
    ]

    for label, x, y, color in gates:
        circle = Circle((x, y), 0.3, color=color, edgecolor='black',
                       linewidth=1.5, alpha=0.7)
        ax2.add_patch(circle)
        ax2.text(x, y, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    ax2.annotate('', xy=(lstm_x, 2), xytext=(lstm_x, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['orange']))
    ax2.text(lstm_x, 1.5, 'x_t', ha='center', fontsize=11, fontweight='bold')

    ax2.annotate('', xy=(lstm_x, 7), xytext=(lstm_x, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['green']))
    ax2.text(lstm_x, 7.5, 'h_t', ha='center', fontsize=11, fontweight='bold')

    ax2.text(5, 7.8, 'LSTM Cell', ha='center', fontsize=13, fontweight='bold')

    gate_labels = 'f: forget, i: input, o: output, c: cell state'
    ax2.text(5, 1, gate_labels, ha='center', fontsize=9, style='italic')

    plt.suptitle('Recurrent Neural Networks', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../charts/rnn_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/rnn_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created rnn_architecture.pdf")


def create_training_curves():
    """Create training dynamics visualization"""
    epochs = np.arange(1, 51)

    train_loss = 2.0 * np.exp(-0.08 * epochs) + 0.1
    val_loss = 2.0 * np.exp(-0.06 * epochs) + 0.3

    train_acc = 1 - np.exp(-0.09 * epochs)
    val_acc = 1 - np.exp(-0.07 * epochs) - 0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, train_loss, linewidth=3, color=COLORS['blue'],
            label='Training Loss')
    ax1.plot(epochs, val_loss, linewidth=3, color=COLORS['orange'],
            label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, linewidth=3, color=COLORS['blue'],
            label='Training Accuracy')
    ax2.plot(epochs, val_acc, linewidth=3, color=COLORS['orange'],
            label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.suptitle('Training Dynamics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../charts/training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created training_curves.pdf")


def create_deep_learning_timeline():
    """Create historical milestones in deep learning"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(1950, 2030)
    ax.set_ylim(0, 10)
    ax.axis('off')

    milestones = [
        (1958, 'Perceptron', COLORS['blue']),
        (1986, 'Backprop', COLORS['orange']),
        (1998, 'LeNet-5', COLORS['green']),
        (2006, 'Deep Belief Nets', COLORS['red']),
        (2012, 'AlexNet', COLORS['purple']),
        (2014, 'GANs', COLORS['brown']),
        (2017, 'Transformers', COLORS['pink']),
        (2020, 'GPT-3', COLORS['cyan']),
        (2023, 'GPT-4', COLORS['olive'])
    ]

    for year, name, color in milestones:
        ax.scatter(year, 5, s=300, c=color, edgecolors='black', linewidth=2, zorder=3)
        ax.text(year, 6.5, name, ha='center', fontsize=11, fontweight='bold',
               rotation=45)
        ax.text(year, 3.5, str(year), ha='center', fontsize=10, style='italic')

    ax.plot([1950, 2025], [5, 5], 'k-', linewidth=2, alpha=0.3)

    ax.text(1987, 8.5, 'Deep Learning Evolution', ha='center',
           fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/deep_learning_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/deep_learning_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created deep_learning_timeline.pdf")


if __name__ == '__main__':
    print("Generating Neural Networks Charts...")
    create_perceptron_model()
    create_mlp_architecture()
    create_backpropagation()
    create_activation_functions()
    create_cnn_architecture()
    create_rnn_architecture()
    create_training_curves()
    create_deep_learning_timeline()
    print("[OK] All neural networks charts created successfully!")