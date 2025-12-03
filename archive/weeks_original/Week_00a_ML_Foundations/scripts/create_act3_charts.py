"""
Generate charts for Week 0a Act 3: The Breakthrough
Following pedagogical framework: Three solutions to nonlinearity
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Create output directory
os.makedirs('../charts', exist_ok=True)

# Template color palette
mlblue = (0/255, 102/255, 204/255)
mlpurple = (51/255, 51/255, 178/255)
mlorange = (255/255, 127/255, 14/255)
mlgreen = (44/255, 160/255, 44/255)
mlred = (214/255, 39/255, 40/255)
mlgray = (127/255, 127/255, 127/255)
mllavender = (173/255, 173/255, 224/255)

plt.style.use('seaborn-v0_8-whitegrid')

def create_feature_engineering_xor():
    """
    Chart 1: XOR transformation from 2D to 3D
    Shows how feature engineering makes problem linearly separable
    """
    fig = plt.figure(figsize=(14, 6))

    # Left: Original 2D space (not separable)
    ax1 = fig.add_subplot(121)

    xor_x = np.array([0, 0, 1, 1])
    xor_y = np.array([0, 1, 0, 1])
    xor_labels = np.array([0, 1, 1, 0])

    colors = [mlblue if label == 0 else mlred for label in xor_labels]
    markers = ['o' if label == 0 else 's' for label in xor_labels]

    for i in range(4):
        ax1.scatter(xor_x[i], xor_y[i], s=400, c=[colors[i]], marker=markers[i],
                   edgecolors='black', linewidth=3, zorder=3)

    # Failed linear separator
    ax1.plot([0, 1], [0.5, 0.5], '--', color=mlgray, linewidth=2, alpha=0.5, label='Failed Linear Boundary')

    ax1.set_xlabel('$x_1$', fontsize=13, fontweight='bold')
    ax1.set_ylabel('$x_2$', fontsize=13, fontweight='bold')
    ax1.set_title('Original 2D Space\n(Not Linearly Separable)', fontsize=14, fontweight='bold', color=mlred)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.3, 1.3)
    ax1.set_aspect('equal')

    # Label points
    labels_text = ['(0,0)\nBlue', '(0,1)\nRed', '(1,0)\nRed', '(1,1)\nBlue']
    positions = [(0, -0.15), (0, 1.15), (1, -0.15), (1, 1.15)]
    for i, (x, y) in enumerate(positions):
        ax1.text(xor_x[i], xor_y[i] + (0.15 if i == 1 or i == 3 else -0.15), labels_text[i],
                ha='center', fontsize=9, fontweight='bold')

    # Right: Transformed 3D space (separable)
    ax2 = fig.add_subplot(122, projection='3d')

    # Add third feature: x1 * x2
    xor_z = xor_x * xor_y

    for i in range(4):
        ax2.scatter(xor_x[i], xor_y[i], xor_z[i], s=400, c=[colors[i]], marker=markers[i],
                   edgecolors='black', linewidth=3, zorder=3, depthshade=False)

    # Separating plane
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 10), np.linspace(-0.2, 1.2, 10))
    zz = 0.3 * np.ones_like(xx)  # Plane at z = 0.3
    ax2.plot_surface(xx, yy, zz, alpha=0.3, color=mlgreen, label='Separating Plane')

    ax2.set_xlabel('$x_1$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('$x_2$', fontsize=12, fontweight='bold')
    ax2.set_zlabel('$x_1 \\times x_2$', fontsize=12, fontweight='bold')
    ax2.set_title('Transformed 3D Space\n(Linearly Separable!)', fontsize=14, fontweight='bold', color=mlgreen)
    ax2.set_xlim(-0.3, 1.3)
    ax2.set_ylim(-0.3, 1.3)
    ax2.set_zlim(-0.2, 1.2)

    plt.suptitle('Feature Engineering: Making XOR Linearly Separable',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/feature_engineering_xor.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/feature_engineering_xor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: feature_engineering_xor.pdf/png")


def create_curse_of_dimensionality():
    """
    Chart 2: Curse of dimensionality
    Shows exponential growth of feature space
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Feature count explosion
    original_dims = np.array([10, 100, 1000, 10000])
    polynomial_2 = original_dims * (original_dims + 1) / 2
    polynomial_3 = original_dims * (original_dims + 1) * (original_dims + 2) / 6

    ax1.semilogy(original_dims, original_dims, 'o-', color=mlblue, linewidth=3,
                markersize=10, label='Original Features')
    ax1.semilogy(original_dims, polynomial_2, 's-', color=mlorange, linewidth=3,
                markersize=10, label='Quadratic Features')
    ax1.semilogy(original_dims, polynomial_3, 'D-', color=mlred, linewidth=3,
                markersize=10, label='Cubic Features')

    # Annotate key points
    ax1.annotate('100 pixels\n→ 5M quadratic features',
                xy=(100, polynomial_2[1]), xytext=(200, 1e5),
                arrowprops=dict(arrowstyle='->', color=mlpurple, lw=2),
                fontsize=10, fontweight='bold', color=mlpurple)

    ax1.set_xlabel('Original Dimensions', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Features (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Count Explosion', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')

    # Right: Computational cost
    features = np.array([100, 1000, 10000, 100000, 1000000, 10000000])
    training_time = features ** 2 / 1e9  # Rough O(n²) scaling in seconds
    memory_gb = features ** 2 * 8 / 1e9  # 8 bytes per float

    ax2_time = ax2
    ax2_time.loglog(features, training_time, 'o-', color=mlblue, linewidth=3,
                   markersize=10, label='Training Time (hours)')
    ax2_time.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax2_time.set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold', color=mlblue)
    ax2_time.tick_params(axis='y', labelcolor=mlblue)
    ax2_time.grid(True, alpha=0.3, which='both')

    ax2_mem = ax2_time.twinx()
    ax2_mem.loglog(features, memory_gb, 's-', color=mlred, linewidth=3,
                  markersize=10, label='Memory (GB)')
    ax2_mem.set_ylabel('Memory Required (GB)', fontsize=12, fontweight='bold', color=mlred)
    ax2_mem.tick_params(axis='y', labelcolor=mlred)

    # Mark "impossible" region
    ax2_time.axhline(y=24, color=mlgray, linestyle='--', linewidth=2, alpha=0.5)
    ax2_time.text(200, 30, 'Impractical (> 1 day)', fontsize=10, color=mlgray, fontweight='bold')

    ax2_time.set_title('Computational Cost Explosion', fontsize=14, fontweight='bold')

    plt.suptitle('The Curse of Dimensionality: Why Manual Features Fail',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/curse_of_dimensionality.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/curse_of_dimensionality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: curse_of_dimensionality.pdf/png")


def create_svm_rbf_kernel():
    """
    Chart 3: SVM with RBF kernel solving XOR
    Shows nonlinear decision boundary in 2D
    """
    from sklearn.svm import SVC

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # XOR data
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    # Left: Linear kernel (fails)
    svm_linear = SVC(kernel='linear', C=1.0)
    svm_linear.fit(X_train, y_train)

    # Create mesh
    h = 0.02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z_linear = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_linear = Z_linear.reshape(xx.shape)

    ax1.contourf(xx, yy, Z_linear, alpha=0.3, levels=1, colors=[mlblue, mlred])
    ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
               s=300, c=[mlblue], marker='o', edgecolors='black', linewidth=3, label='Class 0')
    ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               s=300, c=[mlred], marker='s', edgecolors='black', linewidth=3, label='Class 1')

    ax1.set_xlabel('$x_1$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$x_2$', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Kernel: FAILS\n(50% accuracy)', fontsize=13, fontweight='bold', color=mlred)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # Right: RBF kernel (succeeds)
    svm_rbf = SVC(kernel='rbf', gamma=2, C=1.0)
    svm_rbf.fit(X_train, y_train)

    Z_rbf = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_rbf = Z_rbf.reshape(xx.shape)

    ax2.contourf(xx, yy, Z_rbf, alpha=0.3, levels=1, colors=[mlblue, mlred])
    ax2.contour(xx, yy, Z_rbf, levels=[0.5], colors='black', linewidths=3)
    ax2.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
               s=300, c=[mlblue], marker='o', edgecolors='black', linewidth=3, label='Class 0')
    ax2.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               s=300, c=[mlred], marker='s', edgecolors='black', linewidth=3, label='Class 1')

    ax2.set_xlabel('$x_1$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('$x_2$', fontsize=12, fontweight='bold')
    ax2.set_title('RBF Kernel: SUCCESS\n(100% accuracy)', fontsize=13, fontweight='bold', color=mlgreen)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.suptitle('SVM with RBF Kernel: The Kernel Trick Solves XOR',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/svm_rbf_kernel.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/svm_rbf_kernel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: svm_rbf_kernel.pdf/png")


def create_neural_network_architecture():
    """
    Chart 4: Neural network architecture visualization
    Shows multi-layer perceptron solving XOR
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Layer positions
    input_layer = [(1, 3), (1, 1)]
    hidden_layer = [(3, 3.5), (3, 2.5), (3, 1.5), (3, 0.5)]
    output_layer = [(5, 2)]

    # Draw connections (weights)
    for inp in input_layer:
        for hid in hidden_layer:
            ax.plot([inp[0], hid[0]], [inp[1], hid[1]], '-', color=mlgray, alpha=0.3, linewidth=1)

    for hid in hidden_layer:
        for out in output_layer:
            ax.plot([hid[0], out[0]], [hid[1], out[1]], '-', color=mlgray, alpha=0.3, linewidth=1)

    # Highlight one path in color
    ax.plot([input_layer[0][0], hidden_layer[0][0]], [input_layer[0][1], hidden_layer[0][1]],
           '-', color=mlblue, linewidth=3, alpha=0.7)
    ax.plot([hidden_layer[0][0], output_layer[0][0]], [hidden_layer[0][1], output_layer[0][1]],
           '-', color=mlgreen, linewidth=3, alpha=0.7)

    # Draw neurons
    # Input layer
    for i, pos in enumerate(input_layer):
        ax.scatter(pos[0], pos[1], s=800, c=[mlblue], edgecolors='black', linewidth=3, zorder=3)
        ax.text(pos[0], pos[1], f'$x_{i+1}$', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')

    # Hidden layer
    for i, pos in enumerate(hidden_layer):
        ax.scatter(pos[0], pos[1], s=800, c=[mlgreen], edgecolors='black', linewidth=3, zorder=3)
        ax.text(pos[0], pos[1], f'$h_{i+1}$', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')

    # Output layer
    for i, pos in enumerate(output_layer):
        ax.scatter(pos[0], pos[1], s=800, c=[mlorange], edgecolors='black', linewidth=3, zorder=3)
        ax.text(pos[0], pos[1], '$y$', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')

    # Layer labels
    ax.text(1, -0.5, 'Input Layer\n(2 neurons)', ha='center', fontsize=12, fontweight='bold')
    ax.text(3, -0.5, 'Hidden Layer\n(4 neurons)', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, -0.5, 'Output Layer\n(1 neuron)', ha='center', fontsize=12, fontweight='bold')

    # Activation functions
    ax.text(2, 4.2, r'$\sigma(W^{[1]}x + b^{[1]})$', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(4, 4.2, r'$\sigma(W^{[2]}h + b^{[2]})$', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Example computation
    textstr = 'Forward Pass Example:\n' + \
              'Input: $x = [0, 1]$\n' + \
              'Hidden: $h = \\sigma(Wx) = [0.7, 0.3, 0.8, 0.2]$\n' + \
              'Output: $y = \\sigma(Wh) = 0.95$ (Red)'
    props = dict(boxstyle='round', facecolor=mllavender, alpha=0.3)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_xlim(0, 6)
    ax.set_ylim(-1, 4.5)
    ax.axis('off')
    ax.set_title('Multi-Layer Perceptron Architecture: Learning XOR',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/neural_network_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/neural_network_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: neural_network_architecture.pdf/png")


def create_deep_learning_revolution():
    """
    Chart 5: Deep learning revolution timeline
    Shows key milestones from 2012-2023
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Timeline events
    events = [
        (2012, 'AlexNet', 'ImageNet error:\n26% → 16%\n8 layers', mlblue),
        (2014, 'VGGNet', '19 layers\nSystematic design', mlorange),
        (2015, 'ResNet', '152 layers\nSkip connections\nHuman-level', mlgreen),
        (2017, 'Transformer', 'Attention mechanism\nParallelizable\nFoundation of NLP', mlpurple),
        (2018, 'BERT & GPT-2', 'Pre-training +\nFine-tuning\n1.5B params', mlred),
        (2020, 'GPT-3', '175B parameters\nFew-shot learning\nEmergent abilities', mlblue),
        (2022, 'ChatGPT', 'Conversational AI\nRLHF training\nMass adoption', mlorange),
        (2023, 'GPT-4', '~1.8T parameters\nMultimodal\nProfessional-level', mlgreen)
    ]

    # Draw timeline
    years = [e[0] for e in events]
    ax.plot([min(years)-0.5, max(years)+0.5], [0, 0], 'k-', linewidth=3)

    # Plot events
    for i, (year, name, desc, color) in enumerate(events):
        # Alternate up and down
        y_pos = 1.5 if i % 2 == 0 else -1.5
        y_text = 2.2 if i % 2 == 0 else -2.2

        # Marker
        ax.plot(year, 0, 'o', markersize=15, color=color, markeredgecolor='black',
               markeredgewidth=2, zorder=3)

        # Connecting line
        ax.plot([year, year], [0, y_pos], '--', color=color, linewidth=2, alpha=0.5)

        # Text box
        ax.text(year, y_text, f'{name}\\n{desc}', ha='center', va='center',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor='black'))

    # Axes
    ax.set_xlim(2011, 2024)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_title('The Deep Learning Revolution: Key Milestones (2012-2023)',
                fontsize=16, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3)

    # Add impact metrics
    textstr = 'Impact Metrics:\n' + \
              '2012: 26% ImageNet error\n' + \
              '2023: 3% ImageNet error (superhuman)\n' + \
              'Parameters: 60M → 1.8T (30,000× increase)\n' + \
              'Training cost: $10K → $100M'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig('../charts/deep_learning_revolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/deep_learning_revolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: deep_learning_revolution.pdf/png")


def create_three_approaches_comparison():
    """
    Chart 6: Comparison of three approaches
    Performance vs complexity trade-offs
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy vs Dataset Size
    data_sizes = np.array([100, 500, 1000, 5000, 10000, 50000, 100000])

    # Polynomial features plateau early
    poly_acc = 70 + 20 * (1 - np.exp(-data_sizes / 2000))

    # SVM plateaus medium
    svm_acc = 75 + 20 * (1 - np.exp(-data_sizes / 5000))

    # Neural networks scale best
    nn_acc = 60 + 35 * (1 - np.exp(-data_sizes / 15000))

    ax1.semilogx(data_sizes, poly_acc, 'o-', color=mlblue, linewidth=3,
                markersize=8, label='Polynomial Features')
    ax1.semilogx(data_sizes, svm_acc, 's-', color=mlorange, linewidth=3,
                markersize=8, label='SVM (RBF Kernel)')
    ax1.semilogx(data_sizes, nn_acc, 'D-', color=mlgreen, linewidth=3,
                markersize=8, label='Neural Networks')

    ax1.set_xlabel('Training Dataset Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy vs Data: Neural Networks Scale Best', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(55, 100)

    # 2. Training Time vs Dataset Size
    poly_time = data_sizes / 10000  # Fast
    svm_time = (data_sizes ** 2) / 1e8  # O(n²)
    nn_time = data_sizes / 2000  # Linear but higher constant

    ax2.loglog(data_sizes, poly_time, 'o-', color=mlblue, linewidth=3,
              markersize=8, label='Polynomial Features')
    ax2.loglog(data_sizes, svm_time, 's-', color=mlorange, linewidth=3,
              markersize=8, label='SVM (RBF Kernel)')
    ax2.loglog(data_sizes, nn_time, 'D-', color=mlgreen, linewidth=3,
              markersize=8, label='Neural Networks')

    ax2.set_xlabel('Training Dataset Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Training Time: SVM Scales Poorly', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # 3. Performance on different problem types
    problems = ['Linear\\nSeparable', 'Polynomial\\nFeatures', 'Complex\\nNonlinear', 'High-Dim\\nImages']
    poly_perf = [95, 90, 60, 30]
    svm_perf = [98, 95, 85, 70]
    nn_perf = [95, 92, 95, 98]

    x = np.arange(len(problems))
    width = 0.25

    ax3.bar(x - width, poly_perf, width, label='Polynomial', color=mlblue, edgecolor='black')
    ax3.bar(x, svm_perf, width, label='SVM', color=mlorange, edgecolor='black')
    ax3.bar(x + width, nn_perf, width, label='Neural Network', color=mlgreen, edgecolor='black')

    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Performance by Problem Type', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(problems, fontsize=9)
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_ylim(0, 105)

    # 4. Interpretability vs Flexibility
    interpretability = [9, 6, 3]
    flexibility = [4, 7, 10]
    ease_of_use = [8, 5, 4]
    names = ['Polynomial\\nFeatures', 'SVM\\n(RBF)', 'Neural\\nNetworks']

    x = np.arange(len(names))
    width = 0.25

    ax4.bar(x - width, interpretability, width, label='Interpretability', color=mlblue, edgecolor='black')
    ax4.bar(x, flexibility, width, label='Flexibility', color=mlorange, edgecolor='black')
    ax4.bar(x + width, ease_of_use, width, label='Ease of Use', color=mlgreen, edgecolor='black')

    ax4.set_ylabel('Rating (1-10)', fontsize=11, fontweight='bold')
    ax4.set_title('Qualitative Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.set_ylim(0, 11)

    plt.suptitle('Comparing Three Approaches to Nonlinear Learning',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/three_approaches_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/three_approaches_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: three_approaches_comparison.pdf/png")


if __name__ == '__main__':
    print("Generating Week 0a Act 3 charts...")
    print("-" * 50)

    create_feature_engineering_xor()
    create_curse_of_dimensionality()
    create_svm_rbf_kernel()
    create_neural_network_architecture()
    create_deep_learning_revolution()
    create_three_approaches_comparison()

    print("-" * 50)
    print("All Act 3 charts generated successfully!")
    print("Location: ../charts/")