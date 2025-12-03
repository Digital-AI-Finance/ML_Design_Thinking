"""
Chart 6: The Dimensionality Revelation (Projection Series)
Shows PCA projection from 3D to 2D with variance accounting
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

# Create figure
fig = plt.figure(figsize=(16, 12))

# Top left: Original 3D data
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=mlblue, s=50, alpha=0.6, edgecolors='black')

# Draw diagonal plane
xx, yy = np.meshgrid(range(0, 12), range(0, 12))
zz = 0.5 * xx
ax1.plot_surface(xx, yy, zz, alpha=0.2, color=mlgreen)

ax1.set_xlabel('X', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y', fontsize=11, fontweight='bold')
ax1.set_zlabel('Z', fontsize=11, fontweight='bold')
ax1.set_title('ORIGINAL DATA (3D)\nPoints near diagonal plane',
             fontsize=12, fontweight='bold')

# Add variance text
var_x = np.var(X_3d[:, 0])
var_y = np.var(X_3d[:, 1])
var_z = np.var(X_3d[:, 2])
total_var_original = var_x + var_y + var_z

ax1.text2D(0.05, 0.95, f'Var(X): {var_x:.2f}\nVar(Y): {var_y:.2f}\nVar(Z): {var_z:.2f}\nTotal: {total_var_original:.2f}',
          transform=ax1.transAxes, fontsize=9,
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Top middle: PCA 2D projection
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=mlorange, s=50, alpha=0.6, edgecolors='black')

ax2.set_xlabel('PC1', fontsize=11, fontweight='bold')
ax2.set_ylabel('PC2', fontsize=11, fontweight='bold')
ax2.set_title('PCA PROJECTION (2D)\nMaximum variance directions',
             fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Add variance explained
var_pc1 = pca.explained_variance_[0]
var_pc2 = pca.explained_variance_[1]
var_pc3 = pca.explained_variance_[2]
var_ratio_pc1 = pca.explained_variance_ratio_[0] * 100
var_ratio_pc2 = pca.explained_variance_ratio_[1] * 100

ax2.text(0.05, 0.95, f'PC1: {var_ratio_pc1:.1f}% of variance\nPC2: {var_ratio_pc2:.1f}% of variance\nTotal retained: {var_ratio_pc1+var_ratio_pc2:.1f}%',
        transform=ax2.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        verticalalignment='top')

# Top right: Reconstruction
X_reconstructed_2d = pca.inverse_transform(np.column_stack([X_pca[:, 0], X_pca[:, 1], np.zeros(n_points)]))

ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=mlblue, s=50, alpha=0.4,
           edgecolors='black', label='Original', marker='o')
ax3.scatter(X_reconstructed_2d[:, 0], X_reconstructed_2d[:, 1], X_reconstructed_2d[:, 2],
           c=mlred, s=50, alpha=0.6, edgecolors='black', label='Reconstructed', marker='s')

# Draw reconstruction errors
for i in range(n_points):
    ax3.plot([X_3d[i, 0], X_reconstructed_2d[i, 0]],
            [X_3d[i, 1], X_reconstructed_2d[i, 1]],
            [X_3d[i, 2], X_reconstructed_2d[i, 2]],
            'r--', alpha=0.2, linewidth=0.5)

ax3.set_xlabel('X', fontsize=11, fontweight='bold')
ax3.set_ylabel('Y', fontsize=11, fontweight='bold')
ax3.set_zlabel('Z', fontsize=11, fontweight='bold')
ax3.set_title('RECONSTRUCTION (3D)\nFrom 2D back to 3D',
             fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)

# Calculate reconstruction error
recon_error = np.mean(np.sqrt(np.sum((X_3d - X_reconstructed_2d)**2, axis=1)))
ax3.text2D(0.05, 0.95, f'Avg Error: {recon_error:.3f}\nInfo Loss: {100-var_ratio_pc1-var_ratio_pc2:.1f}%',
          transform=ax3.transAxes, fontsize=10,
          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Bottom left: Scree plot
ax4 = fig.add_subplot(2, 3, 4)
components = ['PC1', 'PC2', 'PC3']
variance_ratios = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(variance_ratios)

bars = ax4.bar(components, variance_ratios, color=[mlblue, mlorange, mlgreen], alpha=0.7, edgecolor='black')
ax4_twin = ax4.twinx()
ax4_twin.plot(components, cumulative_variance, 'ro-', linewidth=3, markersize=10, label='Cumulative')

ax4.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
ax4_twin.set_ylabel('Cumulative Variance (%)', fontsize=11, fontweight='bold', color='red')
ax4.set_title('SCREE PLOT: Variance per Component', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 100)
ax4_twin.set_ylim(0, 105)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, variance_ratios)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Bottom middle: Projection demonstration table
ax5 = fig.add_subplot(2, 3, 5)
ax5.axis('off')

# Sample points
sample_points = [0, 15, 30]
table_data = []
table_data.append(['Point', 'Original (x,y,z)', 'PC1, PC2', 'Reconstructed'])

for i in sample_points:
    orig = f'({X_3d[i,0]:.1f},{X_3d[i,1]:.1f},{X_3d[i,2]:.1f})'
    proj = f'({X_pca[i,0]:.1f},{X_pca[i,1]:.1f})'
    recon = f'({X_reconstructed_2d[i,0]:.1f},{X_reconstructed_2d[i,1]:.1f},{X_reconstructed_2d[i,2]:.1f})'
    table_data.append([f'#{i+1}', orig, proj, recon])

table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.12, 0.32, 0.25, 0.32])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Color header row
for i in range(4):
    table[(0, i)].set_facecolor(mlblue)
    table[(0, i)].set_text_props(weight='bold', color='white')

ax5.set_title('PROJECTION EXAMPLES', fontsize=12, fontweight='bold', pad=20)

# Bottom right: Information/compression trade-off
ax6 = fig.add_subplot(2, 3, 6)

dimensions = [3, 2, 1]
info_retained = [100, var_ratio_pc1+var_ratio_pc2, var_ratio_pc1]
storage_savings = [0, 33.3, 66.7]

ax6_twin = ax6.twinx()

bars1 = ax6.bar(np.array(dimensions) - 0.2, info_retained, 0.4,
               label='Info Retained (%)', color=mlgreen, alpha=0.7, edgecolor='black')
bars2 = ax6_twin.bar(np.array(dimensions) + 0.2, storage_savings, 0.4,
                     label='Storage Saved (%)', color=mlred, alpha=0.7, edgecolor='black')

ax6.set_xlabel('Dimensions Kept', fontsize=11, fontweight='bold')
ax6.set_ylabel('Information Retained (%)', fontsize=11, fontweight='bold', color=mlgreen)
ax6_twin.set_ylabel('Storage Savings (%)', fontsize=11, fontweight='bold', color=mlred)
ax6.set_title('COMPRESSION TRADE-OFF', fontsize=12, fontweight='bold')
ax6.set_xticks(dimensions)
ax6.set_ylim(0, 110)
ax6_twin.set_ylim(0, 110)
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars1, info_retained):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color=mlgreen)

for bar, val in zip(bars2, storage_savings):
    ax6_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color=mlred)

# Main title
fig.suptitle('DIMENSIONALITY REDUCTION: From 3D to 2D with PCA',
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
plt.savefig('../charts/discovery_chart_6_pca.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/discovery_chart_6_pca.png', dpi=150, bbox_inches='tight')
print("Chart 6 (PCA Dimensionality Revelation) created successfully!")
print(f"PC1 explains {var_ratio_pc1:.1f}% of variance")
print(f"PC2 explains {var_ratio_pc2:.1f}% of variance")
print(f"Total variance retained (2D): {var_ratio_pc1+var_ratio_pc2:.1f}%")
print(f"Average reconstruction error: {recon_error:.3f}")

plt.show()
