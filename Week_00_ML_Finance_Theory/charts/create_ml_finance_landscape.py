"""
ML Finance Landscape: Opening Power Chart
Shows the $10 trillion impact of ML across finance sectors
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Rectangle
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)

# Define colors
colors = {
    'finblue': '#004990',
    'fingreen': '#009246',
    'finred': '#D70000',
    'fingold': '#FFB81C',
    'fingray': '#595959',
    'finpurple': '#8064A2',
    'finorange': '#F15A22',
    'fincyan': '#00AEEF'
}

def create_ml_finance_landscape():
    """Create comprehensive ML finance landscape visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 95, 'Machine Learning in Finance: $10 Trillion Market Impact',
           fontsize=20, fontweight='bold', ha='center')
    
    # Define sectors with their ML applications and market size
    sectors = [
        {
            'name': 'Trading & Execution',
            'pos': (20, 70),
            'size': 2500,  # Billions
            'color': colors['finblue'],
            'applications': ['HFT', 'Algo Trading', 'Market Making', 'Smart Routing']
        },
        {
            'name': 'Risk Management',
            'pos': (50, 70),
            'size': 1800,
            'color': colors['finred'],
            'applications': ['VaR', 'Credit Risk', 'Operational Risk', 'Liquidity']
        },
        {
            'name': 'Portfolio Management',
            'pos': (80, 70),
            'size': 2100,
            'color': colors['fingreen'],
            'applications': ['Robo-Advisors', 'Factor Models', 'Optimization', 'Allocation']
        },
        {
            'name': 'Banking & Lending',
            'pos': (20, 40),
            'size': 1500,
            'color': colors['finpurple'],
            'applications': ['Credit Scoring', 'Loan Approval', 'Collections', 'Pricing']
        },
        {
            'name': 'Fraud & Compliance',
            'pos': (50, 40),
            'size': 800,
            'color': colors['finorange'],
            'applications': ['AML', 'KYC', 'Transaction Monitoring', 'Cyber Security']
        },
        {
            'name': 'Insurance',
            'pos': (80, 40),
            'size': 1300,
            'color': colors['fincyan'],
            'applications': ['Underwriting', 'Claims', 'Pricing', 'Risk Assessment']
        }
    ]
    
    # Draw sectors as circles with size proportional to market impact
    for sector in sectors:
        # Calculate radius based on market size
        radius = np.sqrt(sector['size']) / 5
        
        # Draw main circle
        circle = Circle(sector['pos'], radius, 
                       facecolor=sector['color'], alpha=0.3,
                       edgecolor=sector['color'], linewidth=3)
        ax.add_patch(circle)
        
        # Sector name
        ax.text(sector['pos'][0], sector['pos'][1] + radius + 3, sector['name'],
               fontsize=12, fontweight='bold', ha='center', color=sector['color'])
        
        # Market size
        ax.text(sector['pos'][0], sector['pos'][1], f'${sector["size"]}B',
               fontsize=14, fontweight='bold', ha='center', va='center')
        
        # Applications
        angle_step = 360 / len(sector['applications'])
        for i, app in enumerate(sector['applications']):
            angle = np.radians(i * angle_step)
            x = sector['pos'][0] + (radius + 5) * np.cos(angle)
            y = sector['pos'][1] + (radius + 5) * np.sin(angle)
            ax.text(x, y, app, fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add growth indicators
    ax.text(50, 20, 'Annual Growth Rates', fontsize=14, fontweight='bold', ha='center')
    
    growth_data = [
        ('ML in Trading', 45, colors['finblue']),
        ('ML in Risk', 38, colors['finred']),
        ('ML in Portfolio', 52, colors['fingreen']),
        ('ML in Banking', 41, colors['finpurple']),
        ('ML in Fraud', 67, colors['finorange']),
        ('ML in Insurance', 35, colors['fincyan'])
    ]
    
    x_start = 20
    for i, (label, growth, color) in enumerate(growth_data):
        x = x_start + i * 10
        # Bar
        rect = Rectangle((x, 5), 8, growth/5, facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        # Label
        ax.text(x + 4, 3, label, fontsize=8, ha='center', rotation=45)
        # Value
        ax.text(x + 4, 5 + growth/5 + 1, f'{growth}%', fontsize=9, ha='center',
               fontweight='bold', color=color)
    
    # Add key statistics
    stats_box = FancyBboxPatch((5, 85), 30, 10,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['fingold'], alpha=0.2,
                               edgecolor=colors['fingold'], linewidth=2)
    ax.add_patch(stats_box)
    
    ax.text(20, 92, 'Key Statistics', fontsize=11, fontweight='bold', ha='center')
    ax.text(20, 89, '• 75% of trades algorithmic', fontsize=9, ha='center')
    ax.text(20, 87, '• 40% cost reduction in ops', fontsize=9, ha='center')
    
    # Add technology stack
    tech_box = FancyBboxPatch((65, 85), 30, 10,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['fingray'], alpha=0.2,
                              edgecolor=colors['fingray'], linewidth=2)
    ax.add_patch(tech_box)
    
    ax.text(80, 92, 'Core Technologies', fontsize=11, fontweight='bold', ha='center')
    ax.text(80, 89, '• Deep Learning • XGBoost', fontsize=9, ha='center')
    ax.text(80, 87, '• Reinforcement Learning • NLP', fontsize=9, ha='center')
    
    return fig

# Create additional supporting charts
def create_bias_variance_tradeoff():
    """Create bias-variance tradeoff visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Dartboard analogy
    for ax, title, bias, variance in zip(axes,
                                         ['High Bias\nLow Variance', 
                                          'Balanced',
                                          'Low Bias\nHigh Variance'],
                                         [True, False, False],
                                         [False, False, True]):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        
        # Draw target
        for r in [0.5, 1.0, 1.5, 2.0]:
            circle = Circle((0, 0), r, fill=False, edgecolor='gray', linewidth=1)
            ax.add_patch(circle)
        
        # Draw center
        ax.scatter(0, 0, c='red', s=100, marker='x', linewidth=3)
        
        # Generate points
        np.random.seed(42)
        if bias:
            # High bias: systematic offset
            points = np.random.randn(20, 2) * 0.2 + np.array([1.0, 0.5])
        elif variance:
            # High variance: spread out
            points = np.random.randn(20, 2) * 0.8
        else:
            # Balanced: clustered near target
            points = np.random.randn(20, 2) * 0.3 + np.array([0.1, 0.1])
        
        ax.scatter(points[:, 0], points[:, 1], c=colors['finblue'], 
                  alpha=0.6, s=50)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Bias-Variance Tradeoff: The Fundamental ML Challenge',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_paradigm_charts():
    """Create simple paradigm illustrations"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Supervised
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input-output pairs
    for i in range(5):
        y = 8 - i * 1.5
        ax.arrow(2, y, 4, 0, head_width=0.3, head_length=0.3, 
                fc=colors['finblue'], ec=colors['finblue'])
        ax.text(1, y, f'X{i+1}', fontsize=10, ha='center', va='center')
        ax.text(7, y, f'Y{i+1}', fontsize=10, ha='center', va='center')
    
    ax.text(5, 9, 'Supervised Learning', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1, 'Learn: f(X) = Y', fontsize=10, ha='center', style='italic')
    
    # Unsupervised
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Clusters
    np.random.seed(42)
    for i, color in enumerate([colors['fingreen'], colors['finred'], colors['finblue']]):
        cluster = np.random.randn(15, 2) * 0.5 + np.array([3 + i*2, 5])
        ax.scatter(cluster[:, 0], cluster[:, 1], c=color, alpha=0.6, s=30)
    
    ax.text(5, 9, 'Unsupervised Learning', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1, 'Find: Structure in X', fontsize=10, ha='center', style='italic')
    
    # Reinforcement
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Agent-Environment loop
    agent = Circle((3, 5), 1, facecolor=colors['finorange'], alpha=0.7)
    env = Rectangle((6, 4), 2, 2, facecolor=colors['fincyan'], alpha=0.7)
    ax.add_patch(agent)
    ax.add_patch(env)
    
    # Arrows
    ax.arrow(4, 5.5, 1.5, 0, head_width=0.2, head_length=0.2, 
            fc='black', ec='black')
    ax.arrow(6.5, 4.5, -1.5, 0, head_width=0.2, head_length=0.2, 
            fc='black', ec='black')
    
    ax.text(3, 5, 'Agent', fontsize=10, ha='center', va='center', color='white')
    ax.text(7, 5, 'Env', fontsize=10, ha='center', va='center', color='white')
    ax.text(5, 6, 'Action', fontsize=8, ha='center')
    ax.text(5, 4, 'Reward', fontsize=8, ha='center')
    
    ax.text(5, 9, 'Reinforcement Learning', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1, 'Learn: Optimal Policy π*', fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # Create main landscape
    fig1 = create_ml_finance_landscape()
    plt.savefig('ml_finance_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('ml_finance_landscape.png', dpi=150, bbox_inches='tight')
    print("Created ml_finance_landscape.pdf/png")
    
    # Create bias-variance visualization
    fig2 = create_bias_variance_tradeoff()
    plt.savefig('bias_variance_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    print("Created bias_variance_tradeoff.pdf/png")
    
    # Create paradigm charts
    fig3 = create_paradigm_charts()
    for i, paradigm in enumerate(['supervised', 'unsupervised', 'reinforcement']):
        plt.savefig(f'{paradigm}_paradigm.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{paradigm}_paradigm.png', dpi=150, bbox_inches='tight')
    print("Created paradigm charts")
    
    plt.close('all')