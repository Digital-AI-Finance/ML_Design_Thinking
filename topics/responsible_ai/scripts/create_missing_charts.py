"""
Create 8 missing charts for Week 7: Responsible AI and Ethical Innovation
Charts follow template_beamer_final.tex color scheme (mllavender/mlpurple)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.patches as mpatches

# Template colors
mlblue = '#0066CC'
mlpurple = '#3333B2'
mllavender = '#ADADE0'
mlorange = '#FF7F0E'
mlgreen = '#2CA02C'
mlred = '#D62728'
mlgray = '#7F7F7F'

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette([mlblue, mlorange, mlgreen, mlred, mlpurple, mllavender])

def save_chart(fig, name):
    """Save chart as both PDF and PNG"""
    fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    print(f"Created: {name}")
    plt.close(fig)

# Chart 1: Bias Amplification Timeline
def create_bias_amplification():
    fig, ax = plt.subplots(figsize=(10, 6))

    t = np.linspace(0, 10, 100)
    B0 = 0.05  # Initial bias 5%

    alphas = [0, 0.1, 0.26, 0.5]
    labels = ['α=0 (no feedback)', 'α=0.1 (weak)', 'α=0.26 (observed)', 'α=0.5 (strong)']
    colors = [mlgray, mlblue, mlorange, mlred]

    for alpha, label, color in zip(alphas, labels, colors):
        B_t = B0 * (1 + alpha)**t
        ax.plot(t, B_t * 100, label=label, linewidth=2.5, color=color)

    ax.axhline(y=5, color=mlgray, linestyle='--', alpha=0.5, label='Initial 5%')
    ax.axhline(y=20, color=mlred, linestyle=':', alpha=0.5, label='Critical 20%')

    ax.set_xlabel('Time (iterations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bias (%)', fontsize=12, fontweight='bold')
    ax.set_title('Bias Amplification Over Time: Feedback Loop Dynamics',
                 fontsize=14, fontweight='bold', color=mlpurple, pad=15)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 30)

    # Annotations
    ax.annotate('4x amplification\nafter 10 iterations',
                xy=(10, 20.25), xytext=(7, 25),
                arrowprops=dict(arrowstyle='->', color=mlorange, lw=2),
                fontsize=10, color=mlorange, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=mlorange))

    save_chart(fig, 'bias_amplification_timeline')

# Chart 2: Intersectionality Matrix
def create_intersectionality_matrix():
    fig, ax = plt.subplots(figsize=(10, 7))

    attributes = np.arange(1, 11)
    subgroups = 2**attributes
    samples_needed = subgroups * 30  # 30 per subgroup minimum

    ax.semilogy(attributes, subgroups, 'o-', color=mlpurple, linewidth=3,
                markersize=10, label='Subgroups ($2^n$)')
    ax.semilogy(attributes, samples_needed, 's-', color=mlorange, linewidth=3,
                markersize=10, label='Samples needed (30/subgroup)')

    # Reality line
    ax.axhline(y=10000, color=mlred, linestyle='--', linewidth=2, label='Typical dataset (10K)')

    ax.set_xlabel('Number of Attributes (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Intersectionality Explosion: Exponential Subgroup Growth',
                 fontsize=14, fontweight='bold', color=mlpurple, pad=15)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Highlight 6 attributes
    ax.axvline(x=6, color=mlgreen, linestyle=':', alpha=0.7, linewidth=2)
    ax.annotate('6 attributes:\n490,140 subgroups\n14.7M samples needed',
                xy=(6, 100000), xytext=(7.5, 10000),
                arrowprops=dict(arrowstyle='->', color=mlgreen, lw=2),
                fontsize=10, color=mlgreen, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=mlgreen))

    save_chart(fig, 'intersectionality_matrix')

# Chart 3: Fairness Metrics Taxonomy
def create_fairness_metrics_taxonomy():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'Complete Fairness Metrics Taxonomy',
            ha='center', fontsize=16, fontweight='bold', color=mlpurple)

    # Three families
    families = [
        ('Independence\n(Group Fairness)', 1.5, 7.5, mlblue),
        ('Separation\n(Conditional)', 6, 7.5, mlorange),
        ('Sufficiency\n(Calibration)', 10.5, 7.5, mlgreen)
    ]

    for name, x, y, color in families:
        box = FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle='round,pad=0.1',
                            facecolor=color, edgecolor='black', alpha=0.3, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=11,
                fontweight='bold', color='black')

    # Metrics under each family
    independence_metrics = [
        'Demographic Parity',
        'Statistical Parity',
        '80% Rule'
    ]

    separation_metrics = [
        'Equal Opportunity',
        'Equalized Odds',
        'FPR Parity'
    ]

    sufficiency_metrics = [
        'Calibration',
        'PPV Parity',
        'Predictive Parity'
    ]

    y_start = 5.5
    for i, metric in enumerate(independence_metrics):
        ax.text(1.5, y_start - i*0.8, f'• {metric}', fontsize=9, color=mlblue)

    for i, metric in enumerate(separation_metrics):
        ax.text(6, y_start - i*0.8, f'• {metric}', fontsize=9, color=mlorange)

    for i, metric in enumerate(sufficiency_metrics):
        ax.text(10.5, y_start - i*0.8, f'• {metric}', fontsize=9, color=mlgreen)

    # Causal fairness box
    causal_box = FancyBboxPatch((4, 1.5), 4, 1.2, boxstyle='round,pad=0.1',
                               facecolor=mlpurple, edgecolor='black', alpha=0.2, linewidth=2)
    ax.add_patch(causal_box)
    ax.text(6, 2.3, 'Causal Fairness', ha='center', fontsize=11,
            fontweight='bold', color=mlpurple)
    ax.text(6, 1.8, '• Counterfactual Fairness\n• Path-Specific Effects',
            ha='center', fontsize=9, color=mlpurple)

    # Legend
    ax.text(6, 0.5, '12 major metrics organized by type | Chouldechova (2017): Cannot satisfy all simultaneously',
            ha='center', fontsize=9, style='italic', color=mlgray)

    save_chart(fig, 'fairness_metrics_taxonomy')

# Chart 4: Calibration Reliability Diagram
def create_calibration_reliability():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Before calibration
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Uncalibrated: overconfident
    np.random.seed(42)
    empirical_before = bin_centers + np.random.normal(0, 0.15, len(bin_centers))
    empirical_before = np.clip(empirical_before, 0, 1)

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax1.plot(bin_centers, empirical_before, 'o-', color=mlred, linewidth=2.5,
             markersize=8, label='Observed frequency')
    ax1.fill_between(bin_centers, bin_centers, empirical_before, alpha=0.3, color=mlred)

    ax1.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Observed Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Before Calibration\n(Overconfident)', fontsize=12, fontweight='bold', color=mlred)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # After calibration
    empirical_after = bin_centers + np.random.normal(0, 0.03, len(bin_centers))
    empirical_after = np.clip(empirical_after, 0, 1)

    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax2.plot(bin_centers, empirical_after, 'o-', color=mlgreen, linewidth=2.5,
             markersize=8, label='Observed frequency')
    ax2.fill_between(bin_centers, bin_centers, empirical_after, alpha=0.3, color=mlgreen)

    ax2.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Observed Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('After Calibration\n(Well-calibrated)', fontsize=12, fontweight='bold', color=mlgreen)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle('Calibration Reliability Diagram: Before vs After',
                 fontsize=14, fontweight='bold', color=mlpurple, y=1.02)

    plt.tight_layout()
    save_chart(fig, 'calibration_reliability_diagram')

# Chart 5: Pareto Frontier Detailed
def create_pareto_frontier():
    fig, ax = plt.subplots(figsize=(10, 7))

    # Generate Pareto frontier points
    lambdas = np.array([0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    accuracy = np.array([85.0, 84.8, 84.3, 83.5, 82.3, 79.1, 74.2, 68.5])
    dp_violation = np.array([30.0, 28.1, 22.4, 12.8, 4.8, 1.2, 0.3, 0.0])

    # Plot Pareto frontier
    ax.plot(accuracy, dp_violation, 'o-', color=mlpurple, linewidth=3,
            markersize=10, label='Pareto frontier')

    # Highlight sweet spot (λ=0.3)
    idx_sweet = 4
    ax.plot(accuracy[idx_sweet], dp_violation[idx_sweet], 'o', color=mlgreen,
            markersize=15, label='Sweet spot (λ=0.3)', zorder=10)

    # Label key points
    for i, lam in enumerate(lambdas):
        if lam in [0, 0.3, 10]:
            ax.annotate(f'λ={lam}\n({accuracy[i]:.1f}%, {dp_violation[i]:.1f}%)',
                       xy=(accuracy[i], dp_violation[i]),
                       xytext=(accuracy[i]-2, dp_violation[i]+3),
                       fontsize=9, color=mlpurple if lam != 0.3 else mlgreen,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=mlpurple if lam != 0.3 else mlgreen))

    # Infeasible region
    ax.fill_between([68, 85.5], [0, 0], [35, 35], alpha=0.1, color=mlgray,
                    label='Infeasible region')

    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('DP Violation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Fairness-Accuracy Trade-off Space',
                 fontsize=14, fontweight='bold', color=mlpurple, pad=15)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(67, 86)
    ax.set_ylim(-2, 35)

    # ROI annotation
    ax.text(80, 25, 'Best ROI:\nλ=0.3 → 9.3x\nbias reduction\nper accuracy point',
            fontsize=10, color=mlgreen, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=mlgreen, linewidth=2))

    save_chart(fig, 'pareto_frontier_detailed')

# Chart 6: Adversarial Architecture
def create_adversarial_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'Adversarial Debiasing Architecture',
            ha='center', fontsize=16, fontweight='bold', color=mlpurple)

    # Input
    input_box = Rectangle((0.5, 3.5), 1.2, 1, facecolor=mllavender, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.1, 4, 'Input\nX', ha='center', va='center', fontsize=10, fontweight='bold')

    # Predictor network
    pred_box = Rectangle((3, 3), 2, 2, facecolor=mlblue, edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(pred_box)
    ax.text(4, 4.5, 'Predictor\nPθ', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(4, 3.8, 'Maximize\naccuracy', ha='center', va='center', fontsize=8, color='white')

    # Adversary network
    adv_box = Rectangle((3, 0.5), 2, 1.5, facecolor=mlred, edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(adv_box)
    ax.text(4, 1.25, 'Adversary\nAφ', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(4, 0.8, 'Predict A', ha='center', va='center', fontsize=8, color='white')

    # Outputs
    y_box = Rectangle((7, 3.5), 1.2, 1, facecolor=mlgreen, edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(y_box)
    ax.text(7.6, 4, 'Output\nŶ', ha='center', va='center', fontsize=10, fontweight='bold')

    a_box = Rectangle((7, 0.5), 1.2, 1, facecolor=mlred, edgecolor='black', linewidth=2, alpha=0.3)
    ax.add_patch(a_box)
    ax.text(7.6, 1, 'Pred\nÂ', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    ax.arrow(1.7, 4, 1.1, 0, head_width=0.2, head_length=0.2, fc=mlblue, ec=mlblue, linewidth=2)
    ax.arrow(5, 4, 1.8, 0, head_width=0.2, head_length=0.2, fc=mlgreen, ec=mlgreen, linewidth=2)
    ax.arrow(4, 3, 0, -1.3, head_width=0.2, head_length=0.2, fc=mlred, ec=mlred, linewidth=2, linestyle='--')
    ax.arrow(4, 2, 0, -0.3, head_width=0.2, head_length=0.2, fc=mlred, ec=mlred, linewidth=2)
    ax.arrow(5, 1.25, 1.8, -0.25, head_width=0.2, head_length=0.2, fc=mlred, ec=mlred, linewidth=2)

    # Loss functions
    ax.text(10, 5.5, 'Predictor Loss:', fontsize=10, fontweight='bold', color=mlblue)
    ax.text(10, 5, r'$L_P = -\mathrm{Acc}(Y, \hat{Y})$', fontsize=9, color=mlblue)

    ax.text(10, 3.5, 'Adversary Loss:', fontsize=10, fontweight='bold', color=mlred)
    ax.text(10, 3, r'$L_A = -\mathrm{Acc}(A, \hat{A})$', fontsize=9, color=mlred)

    ax.text(10, 1.5, 'Combined:', fontsize=10, fontweight='bold', color=mlpurple)
    ax.text(10, 1, r'$\min_\theta \max_\phi L_P - \lambda L_A$', fontsize=9, color=mlpurple)

    # Gradient reversal
    ax.text(4, 2.5, 'Gradient\nReversal', ha='center', fontsize=8,
            style='italic', color=mlred,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=mlred))

    save_chart(fig, 'adversarial_architecture')

# Chart 7: Production Fairness Stack
def create_production_stack():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'Production Fairness Stack: 5-Layer Architecture',
            ha='center', fontsize=16, fontweight='bold', color=mlpurple)

    # 5 layers
    layers = [
        ('Layer 1: Data Validation', 8, ['Drift detection', 'Bias scanning', 'Quality checks'], mlblue),
        ('Layer 2: Training', 6.5, ['Constrained optimization', 'Adversarial debiasing', 'Reweighing'], mlorange),
        ('Layer 3: Evaluation', 5, ['Fairness metrics', 'A/B testing', 'Statistical tests'], mlgreen),
        ('Layer 4: Deployment', 3.5, ['Serving infrastructure', 'Real-time monitoring', 'Rollback triggers'], mlpurple),
        ('Layer 5: Audit Trail', 2, ['Logging', 'Explainability', 'Compliance reports'], mlgray)
    ]

    for i, (name, y, items, color) in enumerate(layers):
        # Layer box
        box = FancyBboxPatch((1, y-0.6), 10, 1.2, boxstyle='round,pad=0.05',
                            facecolor=color, edgecolor='black', alpha=0.3, linewidth=2)
        ax.add_patch(box)

        # Layer name
        ax.text(1.5, y, name, fontsize=12, fontweight='bold', va='center', color='black')

        # Items
        item_text = ' | '.join(items)
        ax.text(6.5, y-0.25, item_text, fontsize=9, va='center', color='black')

    # Arrows between layers
    for i in range(len(layers)-1):
        y_from = layers[i][1] - 0.7
        y_to = layers[i+1][1] + 0.6
        ax.arrow(6, y_from, 0, y_to-y_from+0.1, head_width=0.3, head_length=0.15,
                fc=mlgray, ec=mlgray, linewidth=2, alpha=0.5)

    # Tools annotations
    ax.text(6, 0.5, 'Tools: Snowflake/BigQuery (data) | Fairlearn/PyTorch (training) | TF Serving (inference) | Prometheus (monitoring)',
            ha='center', fontsize=9, style='italic', color=mlgray)

    save_chart(fig, 'production_fairness_stack')

# Chart 8: Causal DAG for Fairness
def create_causal_dag():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Causal DAG: Direct vs Indirect Discrimination',
            ha='center', fontsize=16, fontweight='bold', color=mlpurple)

    # Nodes
    nodes = {
        'A': (2, 7, 'Protected\nAttribute\n(A)', mlred),
        'X': (5, 7, 'Features\n(X)', mlblue),
        'Y': (8, 7, 'True\nOutcome\n(Y)', mlgreen),
        'D': (5, 4, 'Decision\n(D)', mlorange),
        'U': (8, 4, 'Unobserved\nConfounder\n(U)', mlgray)
    }

    for node, (x, y, label, color) in nodes.items():
        circle = Circle((x, y), 0.5, facecolor=color, edgecolor='black', linewidth=2, alpha=0.4)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

    # Edges
    edges = [
        ('A', 'X', 'Proxy\nvariables', mlred, 'solid'),
        ('A', 'D', 'Direct\ndiscrimination', mlred, 'solid'),
        ('X', 'D', 'Legitimate', mlblue, 'solid'),
        ('Y', 'D', 'Target', mlgreen, 'dashed'),
        ('U', 'Y', 'Confounding', mlgray, 'dashed'),
        ('U', 'A', 'Confounding', mlgray, 'dashed')
    ]

    edge_coords = {
        ('A', 'X'): (2.5, 7, 4.5, 7),
        ('A', 'D'): (2.3, 6.6, 4.7, 4.4),
        ('X', 'D'): (5, 6.5, 5, 4.5),
        ('Y', 'D'): (7.7, 6.6, 5.3, 4.4),
        ('U', 'Y'): (8, 4.5, 8, 6.5),
        ('U', 'A'): (7.5, 4.3, 2.5, 6.7)
    }

    for (from_node, to_node, label, color, style) in edges:
        x1, y1, x2, y2 = edge_coords[(from_node, to_node)]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5,
                               color=color, linestyle=style)
        ax.add_patch(arrow)

        # Label
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.3, mid_y+0.2, label, fontsize=8, color=color,
                style='italic', bbox=dict(boxstyle='round,pad=0.2',
                facecolor='white', edgecolor=color, alpha=0.8))

    # Paths
    ax.text(5, 2.5, 'Direct path: A → D (red solid)', fontsize=10, color=mlred, fontweight='bold')
    ax.text(5, 2, 'Indirect path: A → X → D (red + blue)', fontsize=10, color=mlpurple, fontweight='bold')
    ax.text(5, 1.5, 'Legitimate path: X → D (blue, no A involved)', fontsize=10, color=mlblue, fontweight='bold')

    ax.text(5, 0.5, 'Counterfactual fairness requires blocking A → D and A → X → D paths',
            ha='center', fontsize=9, style='italic', color=mlgray)

    save_chart(fig, 'causal_dag_fairness')

# Main execution
if __name__ == "__main__":
    print("Creating 8 missing charts for Week 7...")
    print("-" * 50)

    create_bias_amplification()
    create_intersectionality_matrix()
    create_fairness_metrics_taxonomy()
    create_calibration_reliability()
    create_pareto_frontier()
    create_adversarial_architecture()
    create_production_stack()
    create_causal_dag()

    print("-" * 50)
    print("All 8 charts created successfully!")
    print("Charts saved in ../charts/ as both PDF and PNG")
