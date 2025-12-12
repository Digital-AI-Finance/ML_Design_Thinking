"""Neural Network Architecture - Simple MLP visualization"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

fig, ax = plt.subplots(figsize=(10, 6))

# Layer configuration
layers = [3, 5, 4, 2]  # Input, Hidden1, Hidden2, Output
layer_names = ['Input\n(Features)', 'Hidden 1', 'Hidden 2', 'Output\n(Predictions)']
colors = ['#3498db', '#2ecc71', '#2ecc71', '#c0392b']

# Positioning
x_positions = [1, 3, 5, 7]
y_center = 3
node_radius = 0.25

# Draw connections first (so they're behind nodes)
for l in range(len(layers) - 1):
    x1 = x_positions[l]
    x2 = x_positions[l + 1]
    for i in range(layers[l]):
        y1 = y_center + (i - (layers[l] - 1) / 2) * 0.8
        for j in range(layers[l + 1]):
            y2 = y_center + (j - (layers[l + 1] - 1) / 2) * 0.8
            ax.plot([x1 + node_radius, x2 - node_radius], [y1, y2],
                   'gray', alpha=0.2, linewidth=0.5, zorder=1)

# Draw nodes
for l, (n_nodes, x, color, name) in enumerate(zip(layers, x_positions, colors, layer_names)):
    for i in range(n_nodes):
        y = y_center + (i - (n_nodes - 1) / 2) * 0.8
        circle = mpatches.Circle((x, y), node_radius, facecolor=color,
                                  edgecolor='white', linewidth=2, zorder=2)
        ax.add_patch(circle)

    # Layer label
    ax.text(x, 0.3, name, ha='center', va='top', fontsize=9, fontweight='bold')

# Add forward propagation annotation
ax.annotate('', xy=(6.5, 4.5), xytext=(1.5, 4.5),
           arrowprops=dict(arrowstyle='->', color='#9467bd', lw=2))
ax.text(4, 4.8, 'Forward Propagation: $a^{(l)} = \\sigma(W^{(l)}a^{(l-1)} + b^{(l)})$',
       ha='center', fontsize=10, color='#9467bd')

# Activation function note
ax.text(4, 0.8, 'Activation: ReLU, Sigmoid, Tanh', ha='center', fontsize=8, color='gray', style='italic')

ax.set_xlim(0, 8)
ax.set_ylim(0, 5.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Neural Network Architecture', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("16_neural_network_arch created")
