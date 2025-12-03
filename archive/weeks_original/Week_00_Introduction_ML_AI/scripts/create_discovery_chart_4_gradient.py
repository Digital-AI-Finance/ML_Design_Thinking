"""
Chart 4: The Gradient Descent Landscape (Topographic Map)
Shows optimization landscape with multiple local minima
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
mlblue = '#1f77b4'
mlorange = '#ff7f0e'
mlgreen = '#2ca02c'
mlred = '#d62728'
mlpurple = '#9467bd'

# Create error landscape with multiple local minima
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
X_grid, Y_grid = np.meshgrid(x, y)

# Complex landscape with 3 valleys
Z = (np.sin(X_grid * 0.8) * 2 +
     np.sin(Y_grid * 0.8) * 2 +
     0.3 * X_grid +
     0.2 * Y_grid +
     (X_grid - 5)**2 * 0.05 +
     (Y_grid - 5)**2 * 0.05 +
     3)

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Plot contour map
contour_filled = ax.contourf(X_grid, Y_grid, Z, levels=25, cmap='terrain', alpha=0.7)
contour_lines = ax.contour(X_grid, Y_grid, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

# Simulate gradient descent from two starting points
def gradient_descent(start, learning_rate=0.3, iterations=50):
    """Simple gradient descent with numerical gradients"""
    path = [start]
    x_curr, y_curr = start

    for _ in range(iterations):
        # Numerical gradient
        h = 0.01
        grad_x = (np.sin((x_curr + h) * 0.8) * 2 + 0.3 + 2 * (x_curr + h - 5) * 0.05 -
                 (np.sin(x_curr * 0.8) * 2 + 0.3 + 2 * (x_curr - 5) * 0.05)) / h
        grad_y = (np.sin((y_curr + h) * 0.8) * 2 + 0.2 + 2 * (y_curr + h - 5) * 0.05 -
                 (np.sin(y_curr * 0.8) * 2 + 0.2 + 2 * (y_curr - 5) * 0.05)) / h

        # Update
        x_curr = x_curr - learning_rate * grad_x
        y_curr = y_curr - learning_rate * grad_y

        # Bounds
        x_curr = np.clip(x_curr, 0, 10)
        y_curr = np.clip(y_curr, 0, 10)

        path.append((x_curr, y_curr))

        # Check convergence
        if len(path) > 2:
            if np.sqrt((path[-1][0] - path[-2][0])**2 +
                      (path[-1][1] - path[-2][1])**2) < 0.01:
                break

    return np.array(path)

# Path A: Starts at (2, 8), ends in local minimum
path_A = gradient_descent((2, 8), learning_rate=0.25, iterations=100)
# Path B: Starts at (7, 8), ends in different minimum
path_B = gradient_descent((7, 8), learning_rate=0.25, iterations=100)

# Plot paths
ax.plot(path_A[:, 0], path_A[:, 1], 'o-', color=mlred, linewidth=3,
       markersize=6, label='Path A (Local minimum)', zorder=5)
ax.plot(path_B[:, 0], path_B[:, 1], 's-', color=mlblue, linewidth=3,
       markersize=6, label='Path B (Different minimum)', zorder=5)

# Mark start and end points
ax.scatter(path_A[0, 0], path_A[0, 1], c='red', s=300, marker='*',
          edgecolors='black', linewidth=2, zorder=6, label='Start A')
ax.scatter(path_A[-1, 0], path_A[-1, 1], c='darkred', s=200, marker='X',
          edgecolors='black', linewidth=2, zorder=6)

ax.scatter(path_B[0, 0], path_B[0, 1], c='blue', s=300, marker='*',
          edgecolors='black', linewidth=2, zorder=6, label='Start B')
ax.scatter(path_B[-1, 0], path_B[-1, 1], c='darkblue', s=200, marker='X',
          edgecolors='black', linewidth=2, zorder=6)

# Calculate final errors
def error_func(x, y):
    return (np.sin(x * 0.8) * 2 +
            np.sin(y * 0.8) * 2 +
            0.3 * x +
            0.2 * y +
            (x - 5)**2 * 0.05 +
            (y - 5)**2 * 0.05 +
            3)

error_A = error_func(path_A[-1, 0], path_A[-1, 1])
error_B = error_func(path_B[-1, 0], path_B[-1, 1])

# Find global minimum (approximate)
min_idx = np.unravel_index(Z.argmin(), Z.shape)
global_min_x = X_grid[min_idx]
global_min_y = Y_grid[min_idx]
global_min_error = Z[min_idx]

ax.scatter(global_min_x, global_min_y, c='gold', s=400, marker='*',
          edgecolors='black', linewidth=3, zorder=7, label='Global minimum')

# Annotations
ax.annotate(f'Local Min\nError: {error_A:.2f}',
           xy=path_A[-1], xytext=(path_A[-1, 0]+1, path_A[-1, 1]+1),
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))

ax.annotate(f'Local Min\nError: {error_B:.2f}',
           xy=path_B[-1], xytext=(path_B[-1, 0]-2, path_B[-1, 1]+1),
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))

ax.annotate(f'Global Min\nError: {global_min_error:.2f}',
           xy=(global_min_x, global_min_y),
           xytext=(global_min_x-2, global_min_y-2),
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='gold', alpha=0.9),
           arrowprops=dict(arrowstyle='->', lw=3, color='orange'))

# Labels and title
ax.set_xlabel('Parameter theta1', fontsize=13, fontweight='bold')
ax.set_ylabel('Parameter theta2', fontsize=13, fontweight='bold')
ax.set_title('ERROR LANDSCAPE: Gradient Descent with Multiple Minima',
            fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Add colorbar
cbar = plt.colorbar(contour_filled, ax=ax, label='Error Value')
cbar.set_label('Error Value', fontsize=11, fontweight='bold')

# Add text box with key insights
textstr = '\n'.join([
    'KEY INSIGHTS:',
    f'• Path A converges in {len(path_A)} steps',
    f'• Path B converges in {len(path_B)} steps',
    '• Different starts → different solutions',
    '• Neither finds global minimum!',
    '',
    'ELEVATION = ERROR VALUE',
    'Lower = Better'
])
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
       verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()

# Save
plt.savefig('../charts/discovery_chart_4_gradient.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/discovery_chart_4_gradient.png', dpi=150, bbox_inches='tight')
print("Chart 4 (Gradient Descent Landscape) created successfully!")
print(f"Path A: {len(path_A)} iterations, final error: {error_A:.2f}")
print(f"Path B: {len(path_B)} iterations, final error: {error_B:.2f}")
print(f"Global minimum error: {global_min_error:.2f}")

plt.show()
