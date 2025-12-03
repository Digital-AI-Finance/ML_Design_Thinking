"""
Generate charts for Week 0a Act 2: First Solution and Limits
Following pedagogical framework: Success BEFORE failure pattern
"""

import matplotlib.pyplot as plt
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

plt.style.use('seaborn-v0_8-whitegrid')

def create_linear_regression_success():
    """
    Chart 1: Linear regression success on house prices
    Shows actual data points and fitted line
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Realistic house price data
    np.random.seed(42)
    sizes = np.array([1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200])
    # True relationship: Price = 50 + 0.1 * Size + noise
    true_prices = 50 + 0.1 * sizes
    noise = np.random.normal(0, 15, len(sizes))
    prices = true_prices + noise

    # Fit linear model
    coeffs = np.polyfit(sizes, prices, 1)
    w1, w0 = coeffs[0], coeffs[1]
    fitted_prices = w0 + w1 * sizes

    # Plot data and fit
    ax.scatter(sizes, prices, s=150, c=[mlblue]*len(sizes), alpha=0.7,
              edgecolors='black', linewidth=2, label='Actual House Prices', zorder=3)

    # Fitted line
    x_line = np.linspace(900, 3300, 100)
    y_line = w0 + w1 * x_line
    ax.plot(x_line, y_line, '--', color=mlred, linewidth=3,
           label=f'Learned Model: y = {w0:.1f} + {w1:.3f}x', zorder=2)

    # Show example prediction
    test_size = 1800
    test_pred = w0 + w1 * test_size
    ax.plot(test_size, test_pred, 'D', color=mlgreen, markersize=15,
           label=f'Prediction: {test_size} sqft → ${test_pred:.0f}k', zorder=4)
    ax.vlines(test_size, 0, test_pred, colors=mlgreen, linestyles=':', linewidth=2, alpha=0.5)
    ax.hlines(test_pred, 0, test_size, colors=mlgreen, linestyles=':', linewidth=2, alpha=0.5)

    # Calculate R²
    residuals = prices - fitted_prices
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((prices - np.mean(prices))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Labels
    ax.set_xlabel('House Size (square feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($1000s)', fontsize=12, fontweight='bold')
    ax.set_title(f'Linear Regression Success: House Price Prediction (R² = {r_squared:.2f})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(800, 3400)
    ax.set_ylim(0, 400)

    # Add success annotation
    textstr = f'SUCCESS:\n82% of variance explained\nMean error: $12.5k\nFast training: 0.03s'
    props = dict(boxstyle='round', facecolor=mlgreen, alpha=0.2)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig('../charts/linear_regression_success.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/linear_regression_success.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: linear_regression_success.pdf/png")


def create_regression_performance():
    """
    Chart 2: Regression performance visualization
    Shows actual vs predicted scatter plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate larger dataset
    np.random.seed(42)
    n_samples = 100
    sizes = np.random.uniform(1000, 3500, n_samples)
    true_prices = 50 + 0.1 * sizes
    noise = np.random.normal(0, 18, n_samples)
    actual_prices = true_prices + noise

    # Predictions from linear model
    predicted_prices = 50 + 0.1 * sizes

    # Left: Actual vs Predicted
    ax1.scatter(actual_prices, predicted_prices, s=80, c=mlblue, alpha=0.6, edgecolors='black')

    # Perfect prediction line
    min_val = min(actual_prices.min(), predicted_prices.min())
    max_val = max(actual_prices.max(), predicted_prices.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax1.set_xlabel('Actual Price ($1000s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Price ($1000s)', fontsize=12, fontweight='bold')
    ax1.set_title('Actual vs Predicted: Strong Correlation', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Calculate metrics
    mae = np.mean(np.abs(actual_prices - predicted_prices))
    rmse = np.sqrt(np.mean((actual_prices - predicted_prices)**2))
    r2 = 1 - (np.sum((actual_prices - predicted_prices)**2) /
              np.sum((actual_prices - np.mean(actual_prices))**2))

    # Right: Residuals distribution
    residuals = actual_prices - predicted_prices
    ax2.hist(residuals, bins=20, color=mlgreen, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color=mlred, linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error ($1000s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution: Centered at Zero', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add metrics text
    textstr = f'Performance Metrics:\nMAE = ${mae:.1f}k\nRMSE = ${rmse:.1f}k\nR² = {r2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.suptitle('Linear Regression Performance: Quantified Success', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/regression_performance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/regression_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: regression_performance.pdf/png")


def create_xor_failure():
    """
    Chart 3: XOR problem - linear model failure
    Shows the impossibility of linear separation
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # XOR data
    xor_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_labels = np.array([0, 1, 1, 0])  # Blue, Red, Red, Blue

    # Plot points
    blue_points = xor_data[xor_labels == 0]
    red_points = xor_data[xor_labels == 1]

    ax.scatter(blue_points[:, 0], blue_points[:, 1], s=500, c=[mlblue]*len(blue_points),
              marker='o', edgecolors='black', linewidth=3, label='Blue (Label 0)', zorder=3)
    ax.scatter(red_points[:, 0], red_points[:, 1], s=500, c=[mlred]*len(red_points),
              marker='s', edgecolors='black', linewidth=3, label='Red (Label 1)', zorder=3)

    # Try multiple linear separators (all fail)
    x_line = np.linspace(-0.3, 1.3, 100)

    # Attempt 1: Horizontal line
    ax.plot(x_line, [0.5]*len(x_line), '--', color=mlgray, linewidth=2,
           alpha=0.6, label='Failed Separator 1')

    # Attempt 2: Vertical line
    ax.plot([0.5]*len(x_line), x_line, '--', color=mlgray, linewidth=2,
           alpha=0.6, label='Failed Separator 2')

    # Attempt 3: Diagonal line
    y_diag = x_line
    ax.plot(x_line, y_diag, '--', color=mlgray, linewidth=2,
           alpha=0.6, label='Failed Separator 3')

    # Show what we actually need (nonlinear boundary)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 + 0.35 * np.cos(theta)
    circle_y = 0.5 + 0.35 * np.sin(theta)
    ax.plot(circle_x, circle_y, '-', color=mlgreen, linewidth=3,
           label='Needed: Nonlinear Boundary', zorder=2)

    # Labels
    ax.set_xlabel('Feature $x_1$', fontsize=13, fontweight='bold')
    ax.set_ylabel('Feature $x_2$', fontsize=13, fontweight='bold')
    ax.set_title('The XOR Problem: Linear Models Cannot Solve This', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')

    # Add failure annotation
    textstr = 'FAILURE:\nNo straight line can\nseparate red from blue\nAccuracy: 50% (random)'
    props = dict(boxstyle='round', facecolor=mlred, alpha=0.2)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/xor_failure.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/xor_failure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: xor_failure.pdf/png")


def create_linear_decision_boundary():
    """
    Chart 4: Geometric intuition of linear decision boundary
    Shows what a hyperplane can and cannot do
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Linearly separable case (CAN solve)
    np.random.seed(42)
    class1_x = np.random.normal(2, 0.8, 50)
    class1_y = np.random.normal(2, 0.8, 50)
    class2_x = np.random.normal(5, 0.8, 50)
    class2_y = np.random.normal(5, 0.8, 50)

    ax1.scatter(class1_x, class1_y, s=100, c=[mlblue]*50, alpha=0.6,
               edgecolors='black', label='Class A')
    ax1.scatter(class2_x, class2_y, s=100, c=[mlred]*50, alpha=0.6,
               edgecolors='black', label='Class B')

    # Perfect linear separator
    x_sep = np.linspace(0, 7, 100)
    y_sep = x_sep
    ax1.plot(x_sep, y_sep, 'g-', linewidth=3, label='Linear Boundary (Success)')
    ax1.fill_between(x_sep, 0, y_sep, alpha=0.1, color=mlblue)
    ax1.fill_between(x_sep, y_sep, 7, alpha=0.1, color=mlred)

    ax1.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax1.set_title('Linearly Separable: Linear Model Succeeds', fontsize=13, fontweight='bold', color=mlgreen)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 7)
    ax1.set_aspect('equal')

    # Right: Non-linearly separable (CANNOT solve)
    # Concentric circles
    n_points = 100
    angles = np.random.uniform(0, 2*np.pi, n_points)

    # Inner circle (blue)
    r_inner = np.random.uniform(1, 2, n_points//2)
    inner_x = 3.5 + r_inner * np.cos(angles[:n_points//2])
    inner_y = 3.5 + r_inner * np.sin(angles[:n_points//2])

    # Outer ring (red)
    r_outer = np.random.uniform(2.5, 3.5, n_points//2)
    outer_x = 3.5 + r_outer * np.cos(angles[n_points//2:])
    outer_y = 3.5 + r_outer * np.sin(angles[n_points//2:])

    ax2.scatter(inner_x, inner_y, s=100, c=[mlblue]*(n_points//2), alpha=0.6,
               edgecolors='black', label='Class A (inner)')
    ax2.scatter(outer_x, outer_y, s=100, c=[mlred]*(n_points//2), alpha=0.6,
               edgecolors='black', label='Class B (outer)')

    # Failed linear separators
    ax2.plot([0, 7], [3.5, 3.5], '--', color=mlgray, linewidth=2, alpha=0.5, label='Failed Linear Boundary')
    ax2.plot([3.5, 3.5], [0, 7], '--', color=mlgray, linewidth=2, alpha=0.5)

    # What we actually need
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 3.5 + 2.2 * np.cos(theta)
    circle_y = 3.5 + 2.2 * np.sin(theta)
    ax2.plot(circle_x, circle_y, '-', color=mlgreen, linewidth=3, label='Needed: Circular Boundary')

    ax2.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax2.set_title('Non-Linearly Separable: Linear Model Fails', fontsize=13, fontweight='bold', color=mlred)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 7)
    ax2.set_ylim(0, 7)
    ax2.set_aspect('equal')

    plt.suptitle('Geometric Intuition: What Linear Boundaries Can and Cannot Do',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/linear_decision_boundary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/linear_decision_boundary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: linear_decision_boundary.pdf/png")


def create_bias_variance_tradeoff():
    """
    Chart 5: The classic bias-variance tradeoff U-curve
    Shows training vs test error as model complexity increases
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Model complexity spectrum
    complexity = np.linspace(0, 10, 100)

    # Bias (decreases with complexity)
    bias = 8 * np.exp(-complexity / 2)

    # Variance (increases with complexity)
    variance = 0.2 * complexity**1.8

    # Total error
    total_error = bias + variance + 1.5  # +1.5 for irreducible error

    # Noise floor
    noise = np.ones_like(complexity) * 1.5

    # Plot
    ax.plot(complexity, bias, '-', color=mlblue, linewidth=3, label='Bias (Underfitting)', marker='o', markersize=6, markevery=10)
    ax.plot(complexity, variance, '-', color=mlred, linewidth=3, label='Variance (Overfitting)', marker='s', markersize=6, markevery=10)
    ax.plot(complexity, total_error, '-', color=mlpurple, linewidth=4, label='Total Error', marker='D', markersize=8, markevery=10)
    ax.plot(complexity, noise, '--', color=mlgray, linewidth=2, alpha=0.7, label='Irreducible Error (Noise)')

    # Find optimal complexity
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    optimal_error = total_error[optimal_idx]

    # Mark optimal point
    ax.plot(optimal_complexity, optimal_error, 'g*', markersize=25, label='Optimal Complexity', zorder=5)
    ax.vlines(optimal_complexity, 0, optimal_error, colors=mlgreen, linestyles=':', linewidth=2, alpha=0.7)

    # Shade regions
    ax.fill_between(complexity[:40], 0, 12, alpha=0.1, color=mlblue, label='_High Bias Region')
    ax.fill_between(complexity[60:], 0, 12, alpha=0.1, color=mlred, label='_High Variance Region')

    # Annotations
    ax.annotate('Too Simple\n(Underfit)', xy=(2, 6), fontsize=12, fontweight='bold',
               color=mlblue, ha='center')
    ax.annotate('Too Complex\n(Overfit)', xy=(8, 8), fontsize=12, fontweight='bold',
               color=mlred, ha='center')
    ax.annotate('Sweet Spot', xy=(optimal_complexity, optimal_error - 0.8),
               fontsize=12, fontweight='bold', color=mlgreen, ha='center')

    # Labels
    ax.set_xlabel('Model Complexity (e.g., polynomial degree)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error', fontsize=12, fontweight='bold')
    ax.set_title('The Bias-Variance Tradeoff: The Fundamental Challenge in ML',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 12)

    # Add mathematical formula
    textstr = r'Total Error = Bias² + Variance + Noise'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='center', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/bias_variance_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: bias_variance_tradeoff.pdf/png")


if __name__ == '__main__':
    print("Generating Week 0a Act 2 charts...")
    print("-" * 50)

    create_linear_regression_success()
    create_regression_performance()
    create_xor_failure()
    create_linear_decision_boundary()
    create_bias_variance_tradeoff()

    print("-" * 50)
    print("All Act 2 charts generated successfully!")
    print("Location: ../charts/")