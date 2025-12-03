"""
Additional finance-specific visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)

# Color scheme
colors = {
    'finblue': '#004990',
    'fingreen': '#009246',
    'finred': '#D70000',
    'fingold': '#FFB81C',
    'fingray': '#595959'
}

def create_vc_dimension():
    """VC Dimension visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 2 points - can shatter
    ax = axes[0]
    ax.scatter([0.3, 0.7], [0.5, 0.5], s=100, c=['red', 'blue'])
    ax.axvline(x=0.5, color='black', linestyle='--')
    ax.set_title('2 Points: Can Shatter', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 3 points - can shatter (2D)
    ax = axes[1]
    ax.scatter([0.3, 0.7, 0.5], [0.3, 0.3, 0.7], s=100, c=['red', 'blue', 'red'])
    ax.plot([0, 1], [0.5, 0.5], 'k--')
    ax.set_title('3 Points in 2D: Can Shatter', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 4 points - cannot shatter (2D linear)
    ax = axes[2]
    ax.scatter([0.2, 0.8, 0.2, 0.8], [0.2, 0.2, 0.8, 0.8], 
              s=100, c=['red', 'blue', 'blue', 'red'])
    ax.set_title('4 Points in 2D: Cannot Shatter (XOR)', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.05, 'VC Dimension of Linear = 3', ha='center', fontweight='bold')
    
    plt.suptitle('VC Dimension: Shattering Capability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_svm_classification():
    """SVM classification with margin"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate linearly separable data
    np.random.seed(42)
    X1 = np.random.randn(20, 2) + np.array([2, 2])
    X2 = np.random.randn(20, 2) + np.array([5, 5])
    
    ax.scatter(X1[:, 0], X1[:, 1], c=colors['finblue'], s=50, label='Class 1')
    ax.scatter(X2[:, 0], X2[:, 1], c=colors['finred'], s=50, label='Class 2')
    
    # Decision boundary
    x_line = np.linspace(0, 7, 100)
    y_line = x_line - 0.5
    ax.plot(x_line, y_line, 'k-', linewidth=2, label='Decision Boundary')
    
    # Margin lines
    ax.plot(x_line, y_line + 1, 'k--', alpha=0.5, label='Margin')
    ax.plot(x_line, y_line - 1, 'k--', alpha=0.5)
    
    # Support vectors
    sv_indices = [5, 15, 25, 35]  # Mock support vectors
    all_points = np.vstack([X1, X2])
    for idx in sv_indices:
        ax.scatter(all_points[idx, 0], all_points[idx, 1], 
                  s=200, facecolors='none', edgecolors='green', linewidth=2)
    
    ax.set_xlabel('Feature 1 (e.g., Debt Ratio)', fontsize=11)
    ax.set_ylabel('Feature 2 (e.g., Income)', fontsize=11)
    ax.set_title('SVM: Maximum Margin Classification for Credit Risk', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_efficient_frontier():
    """Efficient frontier visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate random portfolios
    n_portfolios = 5000
    returns = []
    risks = []
    
    for _ in range(n_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        
        # Mock returns and covariance
        asset_returns = np.array([0.08, 0.12, 0.15, 0.10])
        cov_matrix = np.array([[0.05, 0.02, 0.01, 0.01],
                              [0.02, 0.08, 0.02, 0.01],
                              [0.01, 0.02, 0.10, 0.03],
                              [0.01, 0.01, 0.03, 0.06]])
        
        portfolio_return = np.sum(weights * asset_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        returns.append(portfolio_return)
        risks.append(portfolio_risk)
    
    # Plot all portfolios
    scatter = ax.scatter(risks, returns, c=np.array(returns)/np.array(risks), 
                        cmap='viridis', alpha=0.5, s=10)
    
    # Highlight efficient frontier (mock)
    frontier_risks = np.linspace(min(risks), max(risks), 50)
    frontier_returns = 0.05 + 2.5 * frontier_risks - 3 * frontier_risks**2
    ax.plot(frontier_risks, frontier_returns, 'r-', linewidth=3, 
           label='Efficient Frontier')
    
    # Mark special portfolios
    ax.scatter([0.15], [0.09], marker='*', s=500, c=colors['fingold'], 
              label='Minimum Variance', edgecolors='black', linewidth=2)
    ax.scatter([0.25], [0.12], marker='^', s=300, c=colors['fingreen'], 
              label='Maximum Sharpe', edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Risk (Standard Deviation)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Efficient Frontier with ML-Enhanced Covariance Estimation', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    return fig

# Generate all charts
if __name__ == "__main__":
    # VC Dimension
    fig1 = create_vc_dimension()
    plt.savefig('vc_dimension.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('vc_dimension.png', dpi=150, bbox_inches='tight')
    print("Created vc_dimension charts")
    
    # SVM Classification
    fig2 = create_svm_classification()
    plt.savefig('svm_classification.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('svm_classification.png', dpi=150, bbox_inches='tight')
    print("Created svm_classification charts")
    
    # Efficient Frontier
    fig3 = create_efficient_frontier()
    plt.savefig('efficient_frontier.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('efficient_frontier.png', dpi=150, bbox_inches='tight')
    print("Created efficient_frontier charts")
    
    plt.close('all')