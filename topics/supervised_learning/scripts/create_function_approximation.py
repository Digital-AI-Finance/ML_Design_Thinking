"""
Create function approximation visualization
Shows how ensembles smooth predictions: single tree → few trees → many trees
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Generate nonlinear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = np.sin(X).ravel() + 0.5 * np.cos(2*X).ravel()  # True nonlinear function
y_noise = y_true + np.random.normal(0, 0.3, X.shape[0])  # Add noise

# Train models
tree_single = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_single.fit(X, y_noise)
pred_single = tree_single.predict(X)

rf_3 = RandomForestRegressor(n_estimators=3, max_depth=5, random_state=42)
rf_3.fit(X, y_noise)
pred_3 = rf_3.predict(X)

rf_100 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_100.fit(X, y_noise)
pred_100 = rf_100.predict(X)

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Ensemble Smoothing: From Step Functions to Smooth Approximations',
             fontsize=14, fontweight='bold')

# Color scheme
color_true = '#2C3E50'
color_single = '#E74C3C'
color_3 = '#F39C12'
color_100 = '#27AE60'
color_data = '#95A5A6'

# Plot 1: Single Tree (Jagged)
ax1 = axes[0]
ax1.scatter(X, y_noise, alpha=0.3, s=20, color=color_data, label='Training Data')
ax1.plot(X, y_true, 'k--', linewidth=2, label='True Function', alpha=0.5)
ax1.plot(X, pred_single, color=color_single, linewidth=2.5, label='Single Tree Prediction')
ax1.set_title('Single Decision Tree: Step-Function Approximation (High Variance)',
              fontsize=11, fontweight='bold')
ax1.set_ylabel('Prediction', fontsize=10)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 10)

# Add annotation
ax1.text(0.5, 0.05, 'Jagged predictions - high variance, overfits training data',
         transform=ax1.transAxes, fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 2: 3 Trees (Less Jagged)
ax2 = axes[1]
ax2.scatter(X, y_noise, alpha=0.3, s=20, color=color_data, label='Training Data')
ax2.plot(X, y_true, 'k--', linewidth=2, label='True Function', alpha=0.5)
ax2.plot(X, pred_3, color=color_3, linewidth=2.5, label='3-Tree Ensemble Prediction')
ax2.set_title('Small Ensemble (3 Trees): Partial Smoothing',
              fontsize=11, fontweight='bold')
ax2.set_ylabel('Prediction', fontsize=10)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)

# Add annotation
ax2.text(0.5, 0.05, 'Averaging 3 trees reduces variance - smoother but still some steps',
         transform=ax2.transAxes, fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 3: 100 Trees (Smooth)
ax3 = axes[2]
ax3.scatter(X, y_noise, alpha=0.3, s=20, color=color_data, label='Training Data')
ax3.plot(X, y_true, 'k--', linewidth=2, label='True Function', alpha=0.5)
ax3.plot(X, pred_100, color=color_100, linewidth=2.5, label='100-Tree Ensemble Prediction')
ax3.set_title('Large Ensemble (100 Trees): Smooth Approximation (Low Variance)',
              fontsize=11, fontweight='bold')
ax3.set_xlabel('Feature Value', fontsize=10)
ax3.set_ylabel('Prediction', fontsize=10)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 10)

# Add annotation
ax3.text(0.5, 0.05, 'Averaging 100 trees creates smooth, accurate approximation - low variance',
         transform=ax3.transAxes, fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save
plt.savefig('../charts/ensemble_function_approximation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/ensemble_function_approximation.png', dpi=150, bbox_inches='tight')
print("Created ensemble_function_approximation.pdf and .png")
plt.close()
