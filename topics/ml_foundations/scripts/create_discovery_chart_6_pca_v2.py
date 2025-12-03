"""
Chart 6: The Dimensionality Reduction (Simplified - 3 panels)
Shows PCA projection from 3D to 2D with reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
mlblue = '#1f77b4'
mlorange = '#ff7f0e'
mlgreen = '#2ca02c'
mlred = '#d62728'
mlpurple = '#9467bd'

# Generate 3D data lying mostly on a plane
np.random.seed(42)
n_points = 50

# Data lies along diagonal: y≈x, z≈0.5x with small noise
t = np.linspace(0, 10, n_points)
X_3d = np.column_stack([
    t + np.random.randn(n_points) * 0.3,
    t + np.random.randn(n_points) * 0.3,
    0.5 * t + np.random.randn(n_points) * 0.2
])

# Perform PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_3d)

# Create 3-panel figure
fig = plt.figure(figsize=(18, 6))

# Panel 1: Original 3D data
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=mlblue, s=50, alpha=0.6, edgecolors='black')

# Draw diagonal plane
xx, yy = np.meshgrid(range(0, 12), range(0, 12))
zz = 0.5 * xx
ax1.plot_surface(xx, yy, zz, alpha=0.2, color=mlgreen)

ax1.set_xlabel('X', fontsize=12, fontweight='bold')
ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
ax1.set_zlabel('Z', fontsize=12, fontweight='bold')
ax1.set_title('ORIGINAL DATA (3D)\nPoints near diagonal plane',
             fontsize=13, fontweight='bold', color=mlblue, pad=20)

# Add variance text
var_x = np.var(X_3d[:, 0])
var_y = np.var(X_3d[:, 1])
var_z = np.var(X_3d[:, 2])

ax1.text2D(0.05, 0.95, f'Requires 3 numbers\nper point (x, y, z)',
          transform=ax1.transAxes, fontsize=10,
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 2: PCA 2D projection
ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=mlorange, s=50, alpha=0.6, edgecolors='black')

ax2.set_xlabel('Principal Component 1', fontsize=12, fontweight='bold')
ax2.set_ylabel('Principal Component 2', fontsize=12, fontweight='bold')
ax2.set_title('PCA PROJECTION (2D)\nCompressed representation',
             fontsize=13, fontweight='bold', color=mlorange, pad=20)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Add variance explained
var_ratio_pc1 = pca.explained_variance_ratio_[0] * 100
var_ratio_pc2 = pca.explained_variance_ratio_[1] * 100

ax2.text(0.05, 0.95, f'Requires only 2 numbers\nper point (PC1, PC2)\n\nPC1: {var_ratio_pc1:.1f}% of variance\nPC2: {var_ratio_pc2:.1f}% of variance',
        transform=ax2.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        verticalalignment='top')

# Panel 3: Reconstruction
X_reconstructed_2d = pca.inverse_transform(np.column_stack([X_pca[:, 0], X_pca[:, 1], np.zeros(n_points)]))

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=mlblue, s=50, alpha=0.4,
           edgecolors='black', label='Original', marker='o')
ax3.scatter(X_reconstructed_2d[:, 0], X_reconstructed_2d[:, 1], X_reconstructed_2d[:, 2],
           c=mlred, s=50, alpha=0.6, edgecolors='black', label='Reconstructed', marker='s')

# Draw reconstruction errors
for i in range(0, n_points, 5):  # Show every 5th line to avoid clutter
    ax3.plot([X_3d[i, 0], X_reconstructed_2d[i, 0]],
            [X_3d[i, 1], X_reconstructed_2d[i, 1]],
            [X_3d[i, 2], X_reconstructed_2d[i, 2]],
            'r--', alpha=0.3, linewidth=0.5)

ax3.set_xlabel('X', fontsize=12, fontweight='bold')
ax3.set_ylabel('Y', fontsize=12, fontweight='bold')
ax3.set_zlabel('Z', fontsize=12, fontweight='bold')
ax3.set_title('RECONSTRUCTION (3D)\nFrom 2D back to 3D',
             fontsize=13, fontweight='bold', color=mlred, pad=20)
ax3.legend(fontsize=10, loc='upper left')

# Calculate reconstruction error
recon_error = np.mean(np.sqrt(np.sum((X_3d - X_reconstructed_2d)**2, axis=1)))
info_loss = 100 - var_ratio_pc1 - var_ratio_pc2

ax3.text2D(0.05, 0.95, f'Average Error: {recon_error:.3f}\nInfo Loss: {info_loss:.1f}%\n\nClose match shows\ngood compression',
          transform=ax3.transAxes, fontsize=10,
          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Main title
fig.suptitle('DIMENSIONALITY REDUCTION: Compressing 3D Data to 2D',
            fontsize=16, fontweight='bold', y=0.98)

# Add explanatory text at bottom
fig.text(0.5, 0.02,
        'DATA COMPRESSION PRINCIPLE: Points lying near a flat surface in 3D can be represented with 2D coordinates.\n' +
        'Left: Original data requires 3 numbers per point  |  Middle: Compressed to 2 numbers per point  |  Right: Reconstruction quality check',
        ha='center', va='bottom', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6, edgecolor='black', linewidth=2))

plt.tight_layout(rect=[0, 0.06, 1, 0.96])

# Save
plt.savefig('../charts/discovery_chart_6_pca_v2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/discovery_chart_6_pca_v2.png', dpi=150, bbox_inches='tight')
print("Chart 6 V2 (Simplified PCA) created successfully!")
print(f"PC1 explains {var_ratio_pc1:.1f}% of variance")
print(f"PC2 explains {var_ratio_pc2:.1f}% of variance")
print(f"Total variance retained (2D): {var_ratio_pc1+var_ratio_pc2:.1f}%")
print(f"Average reconstruction error: {recon_error:.3f}")

plt.show()
