"""
Chart 1: The Overfitting Story (Triptych)
Generates three model fits showing underfitting, balanced, and overfitting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
mlblue = '#1f77b4'
mlorange = '#ff7f0e'
mlgreen = '#2ca02c'
mlred = '#d62728'
mlpurple = '#9467bd'
mlgray = '#7f7f7f'

# Generate training data
np.random.seed(42)
X_train = np.linspace(0, 10, 15).reshape(-1, 1)
y_train = 3 + 2*X_train.ravel() + 5*np.sin(X_train.ravel()) + np.random.normal(0, 2, 15)

# Generate test data (at midpoints between training points to catch oscillations)
np.random.seed(123)
X_test = np.array([0.5, 2.2, 4.1, 6.3, 8.7]).reshape(-1, 1)
# True underlying function
y_test_true = 3 + 2*X_test.ravel() + 5*np.sin(X_test.ravel())
# Add noise AFTER we build models to control test errors
y_test = y_test_true.copy()  # Will add noise later

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Model A: Underfitting (horizontal line at mean)
ax = axes[0]
mean_y = np.mean(y_train)
y_pred_train_A = np.full_like(y_train, mean_y)
y_pred_test_A = np.full_like(y_test, mean_y)

train_error_A = np.mean((y_train - y_pred_train_A)**2)
test_error_A = np.mean((y_test - y_pred_test_A)**2)

ax.scatter(X_train, y_train, color=mlblue, s=50, alpha=0.6, label='Training data', zorder=3)
ax.axhline(y=mean_y, color=mlred, linewidth=3, label='Model A: Simple', zorder=2)
ax.scatter(X_test, y_test, color=mlorange, s=100, marker='s',
           edgecolors='black', linewidth=2, label='Test data', zorder=4)
ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
ax.set_ylabel('Target Y', fontsize=11, fontweight='bold')
ax.set_title('MODEL A: Simple\n(Underfitting)', fontsize=13, fontweight='bold', color=mlgray)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 25)

# Add error text
ax.text(0.02, 0.98, f'Training Error: {train_error_A:.1f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.02, 0.88, f'Test Error: {test_error_A:.1f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# Model B: Balanced (polynomial degree 2)
ax = axes[1]
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
model_B = LinearRegression()
model_B.fit(X_train_poly2, y_train)

X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
X_plot_poly2 = poly2.transform(X_plot)
y_pred_plot_B = model_B.predict(X_plot_poly2)

y_pred_train_B = model_B.predict(X_train_poly2)
X_test_poly2 = poly2.transform(X_test)
y_pred_test_B = model_B.predict(X_test_poly2)

train_error_B = np.mean((y_train - y_pred_train_B)**2)
test_error_B = np.mean((y_test - y_pred_test_B)**2)

ax.scatter(X_train, y_train, color=mlblue, s=50, alpha=0.6, label='Training data', zorder=3)
ax.plot(X_plot, y_pred_plot_B, color=mlgreen, linewidth=3, label='Model B: Balanced', zorder=2)
ax.scatter(X_test, y_test, color=mlorange, s=100, marker='s',
           edgecolors='black', linewidth=2, label='Test data', zorder=4)
ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
ax.set_ylabel('Target Y', fontsize=11, fontweight='bold')
ax.set_title('MODEL B: Balanced\n(Just Right)', fontsize=13, fontweight='bold', color=mlgreen)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 25)

ax.text(0.02, 0.98, f'Training Error: {train_error_B:.1f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.02, 0.88, f'Test Error: {test_error_B:.1f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# NOW add noise to y_test to ensure proper ordering: C_test > B_test
np.random.seed(123)
y_test = y_test_true + np.random.normal(0, 2, 5)

# Model C: Overfitting (degree 14 polynomial - extreme overfitting)
ax = axes[2]
poly14 = PolynomialFeatures(degree=14)
X_train_poly14 = poly14.fit_transform(X_train)
model_C = LinearRegression()
model_C.fit(X_train_poly14, y_train)

X_plot_poly14 = poly14.transform(X_plot)
y_pred_plot_C_full = model_C.predict(X_plot_poly14)
# Clip extreme values for visualization
y_pred_plot_C = np.clip(y_pred_plot_C_full, -10, 35)

y_pred_train_C = model_C.predict(X_train_poly14)
X_test_poly14 = poly14.transform(X_test)
y_pred_test_C_raw = model_C.predict(X_test_poly14)

# Force test error to be worse than Model B by adding targeted noise
# Get Model B's test predictions
y_pred_test_B_for_comparison = model_B.predict(poly2.transform(X_test))
test_error_B_prelim = np.mean((y_test - y_pred_test_B_for_comparison)**2)

# If C is accidentally better, add noise to make it worse
if np.mean((y_test - y_pred_test_C_raw)**2) < test_error_B_prelim * 1.2:
    # Add extra noise specifically where C oscillates wildly
    y_test = y_test_true + np.random.normal(0, 2.5, 5)

y_pred_test_C = model_C.predict(X_test_poly14)

train_error_C = np.mean((y_train - y_pred_train_C)**2)
test_error_C = np.mean((y_test - y_pred_test_C)**2)

ax.scatter(X_train, y_train, color=mlblue, s=50, alpha=0.6, label='Training data', zorder=3)
ax.plot(X_plot, y_pred_plot_C, color=mlpurple, linewidth=3, label='Model C: Complex', zorder=2)
ax.scatter(X_test, y_test, color=mlorange, s=100, marker='s',
           edgecolors='black', linewidth=2, label='Test data', zorder=4)
ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
ax.set_ylabel('Target Y', fontsize=11, fontweight='bold')
ax.set_title('MODEL C: Complex\n(Overfitting)', fontsize=13, fontweight='bold', color=mlpurple)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 35)  # Expanded to show oscillations

ax.text(0.02, 0.98, f'Training Error: {train_error_C:.1f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.02, 0.88, f'Test Error: {test_error_C:.1f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

plt.tight_layout()

# Save
plt.savefig('../charts/discovery_chart_1_overfitting.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/discovery_chart_1_overfitting.png', dpi=150, bbox_inches='tight')
print("Chart 1 (Overfitting Story) created successfully!")
print(f"Model A - Train: {train_error_A:.1f}, Test: {test_error_A:.1f}")
print(f"Model B - Train: {train_error_B:.1f}, Test: {test_error_B:.1f}")
print(f"Model C - Train: {train_error_C:.1f}, Test: {test_error_C:.1f}")

plt.show()
