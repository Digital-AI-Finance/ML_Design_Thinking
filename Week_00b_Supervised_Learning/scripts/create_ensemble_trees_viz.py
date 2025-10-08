"""
Create ensemble tree visualization using actual graphviz
Shows 3 diverse decision trees to illustrate Random Forest diversity
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.datasets import make_regression
import graphviz
from PIL import Image
import io

# Generate synthetic data
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=4, noise=10, random_state=42)
feature_names = ['sqft', 'bedrooms', 'location', 'age']

# Create 3 decision trees with different random states (simulating bootstrap)
trees = []
for i in range(3):
    # Create tree with different random state
    tree = DecisionTreeRegressor(max_depth=3, random_state=i)
    # Use bootstrap sample
    indices = np.random.RandomState(i).choice(len(X), size=len(X), replace=True)
    tree.fit(X[indices], y[indices])
    trees.append(tree)

# Export trees to graphviz format
dot_data_list = []
for i, tree in enumerate(trees):
    dot_data = export_graphviz(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        precision=0,
        out_file=None
    )
    dot_data_list.append(dot_data)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Random Forest: Multiple Diverse Trees', fontsize=16, fontweight='bold', y=0.98)

# Render each tree
for idx, (dot_data, ax) in enumerate(zip(dot_data_list, axes)):
    # Render graphviz to PNG in memory
    graph = graphviz.Source(dot_data, format='png')
    png_bytes = graph.pipe(format='png')

    # Load PNG into PIL Image
    img = Image.open(io.BytesIO(png_bytes))

    # Display in matplotlib
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Tree {idx+1} (Bootstrap Sample {idx+1})',
                fontsize=12, fontweight='bold', pad=5)

# Add annotation
fig.text(0.5, 0.01,
         'Each tree uses different features and splits due to bootstrap sampling - averaging reduces variance',
         ha='center', fontsize=11, style='italic', color='#555555')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Save
plt.savefig('../charts/ensemble_trees_graphviz.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/ensemble_trees_graphviz.png', dpi=150, bbox_inches='tight')
print("Created ensemble_trees_graphviz.pdf and .png using graphviz")
plt.close()
