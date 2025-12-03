"""
Chart 3: The Impossible Boundary (4-panel progression)
Shows datasets from linearly separable to XOR impossible
"""

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
mlblue = '#1f77b4'
mlred = '#d62728'
mlgray = '#7f7f7f'

# Create figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Dataset A: Perfectly linearly separable
np.random.seed(42)
ax = axes[0]
X_red_A = np.random.randn(15, 2) * 0.5 + [2, 2]
X_blue_A = np.random.randn(15, 2) * 0.5 + [7, 7]

ax.scatter(X_red_A[:, 0], X_red_A[:, 1], c=mlred, s=100, alpha=0.7,
          edgecolors='black', linewidth=1, label='Red class')
ax.scatter(X_blue_A[:, 0], X_blue_A[:, 1], c=mlblue, s=100, alpha=0.7,
          edgecolors='black', linewidth=1, label='Blue class')

# Draw separating line
x_line = np.linspace(0, 10, 100)
y_line = x_line  # y = x
ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label='Perfect separation')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
ax.set_ylabel('Feature Y', fontsize=11, fontweight='bold')
ax.set_title('DATASET A\nLinearly Separable', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

ax.text(0.05, 0.05, 'Errors: 0/30 (0%)\nAchievable: Perfect',
       transform=ax.transAxes, fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Dataset B: Nearly separable (few outliers)
ax = axes[1]
X_red_B = np.random.randn(14, 2) * 0.6 + [2.5, 2.5]
X_blue_B = np.random.randn(14, 2) * 0.6 + [6.5, 6.5]
# Add outliers
X_red_B = np.vstack([X_red_B, [[5.5, 5.5], [6, 6]]])
X_blue_B = np.vstack([X_blue_B, [[3.5, 3.5]]])

ax.scatter(X_red_B[:, 0], X_red_B[:, 1], c=mlred, s=100, alpha=0.7,
          edgecolors='black', linewidth=1, label='Red class')
ax.scatter(X_blue_B[:, 0], X_blue_B[:, 1], c=mlblue, s=100, alpha=0.7,
          edgecolors='black', linewidth=1, label='Blue class')

# Draw best linear separator
x_line = np.linspace(0, 10, 100)
y_line = 0.9*x_line + 0.5
ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label='Best line')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
ax.set_ylabel('Feature Y', fontsize=11, fontweight='bold')
ax.set_title('DATASET B\nNearly Separable', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

ax.text(0.05, 0.05, 'Errors: 3/33 (9%)\nAchievable: ~5-10%',
       transform=ax.transAxes, fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Dataset C: Circular boundary needed
ax = axes[2]
np.random.seed(123)
# Generate circular pattern
theta_red = np.random.rand(15) * 2 * np.pi
r_red = np.random.rand(15) * 2 + 0.5
X_red_C = np.column_stack([r_red * np.cos(theta_red) + 5,
                            r_red * np.sin(theta_red) + 5])

theta_blue = np.random.rand(15) * 2 * np.pi
r_blue = np.random.rand(15) * 2 + 3.5
X_blue_C = np.column_stack([r_blue * np.cos(theta_blue) + 5,
                             r_blue * np.sin(theta_blue) + 5])

ax.scatter(X_red_C[:, 0], X_red_C[:, 1], c=mlred, s=100, alpha=0.7,
          edgecolors='black', linewidth=1, label='Red class')
ax.scatter(X_blue_C[:, 0], X_blue_C[:, 1], c=mlblue, s=100, alpha=0.7,
          edgecolors='black', linewidth=1, label='Blue class')

# Draw circle boundary
circle = plt.Circle((5, 5), 3, fill=False, edgecolor='purple', linewidth=2,
                   linestyle='--', label='Circular boundary')
ax.add_patch(circle)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
ax.set_ylabel('Feature Y', fontsize=11, fontweight='bold')
ax.set_title('DATASET C\nCurved Boundary', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

ax.text(0.05, 0.05, 'Linear: ~15/30 errors\nNonlinear: 0 errors',
       transform=ax.transAxes, fontsize=9,
       bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

# Dataset D: XOR pattern (impossible for linear)
ax = axes[3]
# Create XOR pattern
X_red_D = np.array([
    [1, 1], [1.5, 1.2], [1.2, 1.5], [1.8, 1.8],
    [8, 8], [8.5, 8.2], [8.2, 8.5], [8.8, 8.8],
    [1, 8], [1.5, 8.2], [1.2, 8.5], [1.8, 8.8],
    [8, 1], [8.5, 1.2], [8.2, 1.5], [8.8, 1.8]
])

X_blue_D = np.array([
    [1, 8], [1.5, 7.8], [1.2, 7.5], [1.8, 7.2],
    [8, 1], [8.5, 0.8], [8.2, 0.5], [8.8, 0.2],
    [1, 1], [1.5, 0.8], [1.2, 0.5], [1.8, 0.2],
    [8, 8], [8.5, 7.8], [8.2, 7.5], [8.8, 7.2]
])

# Add noise
np.random.seed(42)
X_red_D = X_red_D + np.random.randn(16, 2) * 0.3
X_blue_D = X_blue_D + np.random.randn(16, 2) * 0.3

# Create proper XOR pattern
X_red_D_new = []
X_blue_D_new = []
for i in range(10):
    for j in range(10):
        x, y = i + np.random.rand()*0.8, j + np.random.rand()*0.8
        if (x < 5 and y < 5) or (x >= 5 and y >= 5):
            X_red_D_new.append([x, y])
        else:
            X_blue_D_new.append([x, y])

X_red_D = np.array(X_red_D_new)
X_blue_D = np.array(X_blue_D_new)

ax.scatter(X_red_D[:, 0], X_red_D[:, 1], c=mlred, s=100, alpha=0.7,
          edgecolors='black', linewidth=1, label='Red class')
ax.scatter(X_blue_D[:, 0], X_blue_D[:, 1], c=mlblue, s=100, alpha=0.7,
          edgecolors='black', linewidth=1, label='Blue class')

# Draw XOR decision boundary
ax.axvline(x=5, color='purple', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=5, color='purple', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
ax.set_ylabel('Feature Y', fontsize=11, fontweight='bold')
ax.set_title('DATASET D\nXOR Pattern', fontsize=12, fontweight='bold', color='purple')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

ax.text(0.05, 0.05, 'Linear: IMPOSSIBLE\n(Min 50% error)\nNeed 2+ lines',
       transform=ax.transAxes, fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.tight_layout()

# Save
plt.savefig('../charts/discovery_chart_3_boundaries.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/discovery_chart_3_boundaries.png', dpi=150, bbox_inches='tight')
print("Chart 3 (Impossible Boundary) created successfully!")

plt.show()
