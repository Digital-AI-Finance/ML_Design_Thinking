import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(14, 8))

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlpink': '#e377c2'
}

np.random.seed(42)

# Generate 5000 innovation ideas
n_samples = 5000
n_clusters = 5

# Create initial scattered data (left panel)
X_scattered = np.random.randn(n_samples, 2) * 4
ax1 = axes[0]
ax1.scatter(X_scattered[:, 0], X_scattered[:, 1], 
           c='gray', alpha=0.3, s=1)
ax1.set_title('5000 Innovation Ideas\n(Chaotic State)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Feature 1: Technical Complexity')
ax1.set_ylabel('Feature 2: Market Readiness')
ax1.set_xlim(-12, 12)
ax1.set_ylim(-12, 12)
ax1.text(0, -14, 'Raw, unstructured data', ha='center', fontsize=10, style='italic')

# Create converging data (middle panel)
ax2 = axes[1]
# Generate data that's starting to cluster
X_converging = []
centers = [(4, 4), (-4, 4), (0, -5), (-5, -3), (5, -3)]
for center in centers:
    cluster_data = np.random.randn(n_samples//5, 2) * 1.5 + center
    X_converging.append(cluster_data)
X_converging = np.vstack(X_converging)

# Add connecting lines to show movement
for i in range(100):
    idx = np.random.randint(0, n_samples)
    ax2.plot([X_scattered[idx, 0], X_converging[idx, 0]], 
            [X_scattered[idx, 1], X_converging[idx, 1]], 
            'gray', alpha=0.1, linewidth=0.5)

ax2.scatter(X_converging[:, 0], X_converging[:, 1], 
           c='purple', alpha=0.4, s=2)
ax2.set_title('ML Algorithm Processing\n(Finding Patterns)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Feature 1: Technical Complexity')
ax2.set_ylabel('Feature 2: Market Readiness')
ax2.set_xlim(-12, 12)
ax2.set_ylim(-12, 12)
# Add animated-style arrows
for i in range(3):
    ax2.annotate('', xy=(np.random.uniform(-8, 8), np.random.uniform(-8, 8)),
                xytext=(np.random.uniform(-10, 10), np.random.uniform(-10, 10)),
                arrowprops=dict(arrowstyle='->', color='purple', alpha=0.3, lw=2))
ax2.text(0, -14, 'Convergence in progress', ha='center', fontsize=10, style='italic')

# Create final clustered data (right panel)
ax3 = axes[2]
# Use K-means for real clustering
kmeans = KMeans(n_clusters=5, random_state=42)
X_final = []
cluster_colors = []
for i, center in enumerate(centers):
    cluster_data = np.random.randn(n_samples//5, 2) * 0.8 + center
    X_final.append(cluster_data)
    cluster_colors.extend([list(colors.values())[i]] * (n_samples//5))
X_final = np.vstack(X_final)
labels = kmeans.fit_predict(X_final)

# Plot clusters with different colors
for i in range(5):
    mask = labels == i
    ax3.scatter(X_final[mask, 0], X_final[mask, 1], 
               c=list(colors.values())[i], alpha=0.6, s=3,
               label=f'Innovation Type {i+1}')

# Add cluster centers
centers_final = kmeans.cluster_centers_
ax3.scatter(centers_final[:, 0], centers_final[:, 1], 
           c='black', s=200, alpha=0.8, marker='*',
           edgecolors='white', linewidth=2)

# Add circles around clusters
for i, center in enumerate(centers_final):
    circle = Circle(center, 2.5, fill=False, 
                   edgecolor=list(colors.values())[i], 
                   linewidth=2, alpha=0.5, linestyle='--')
    ax3.add_patch(circle)

ax3.set_title('5 Innovation Patterns Discovered\n(Organized Knowledge)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Feature 1: Technical Complexity')
ax3.set_ylabel('Feature 2: Market Readiness')
ax3.set_xlim(-12, 12)
ax3.set_ylim(-12, 12)
ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax3.text(0, -14, 'Clear patterns emerged', ha='center', fontsize=10, style='italic')

# Add main title
fig.suptitle('The Convergence Flow: From Chaos to Clarity', fontsize=16, fontweight='bold', y=0.98)

# Add arrows between panels
fig.text(0.36, 0.5, '➔', fontsize=30, ha='center', color=colors['mlpurple'])
fig.text(0.64, 0.5, '➔', fontsize=30, ha='center', color=colors['mlpurple'])

# Add bottom text
fig.text(0.5, 0.02, 'Machine Learning transforms scattered innovation ideas into actionable patterns', 
         ha='center', fontsize=11, color=colors['mlpurple'], fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('convergence_flow.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('convergence_flow.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Convergence flow chart created successfully!")