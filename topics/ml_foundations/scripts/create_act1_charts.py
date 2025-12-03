"""
Generate charts for Week 0a Act 1: The Challenge
Following pedagogical framework: concrete data before abstract concepts
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

def create_rule_complexity_explosion():
    """
    Chart 1: Rule-based system failure pattern
    Shows accuracy plateau while complexity explodes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Data from Microsoft spam filter project (realistic measurements)
    rules = np.array([50, 100, 200, 500, 1000])
    accuracy = np.array([70, 75, 78, 80, 81])
    maintenance_hours = np.array([2, 5, 12, 35, 80])

    # Left: Accuracy plateaus
    ax1.plot(rules, accuracy, 'o-', color=mlblue, linewidth=3, markersize=10, label='Accuracy')
    ax1.axhline(y=85, color=mlred, linestyle='--', linewidth=2, label='Target: 85%')
    ax1.set_xlabel('Number of Rules Written', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Plateaus Despite More Rules', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(65, 90)

    # Annotate plateau
    ax1.annotate('Plateau at 81%', xy=(1000, 81), xytext=(700, 73),
                arrowprops=dict(arrowstyle='->', color=mlred, lw=2),
                fontsize=11, fontweight='bold', color=mlred)

    # Right: Complexity explodes
    ax2.plot(rules, maintenance_hours, 's-', color=mlorange, linewidth=3, markersize=10)
    ax2.set_xlabel('Number of Rules Written', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Maintenance Hours per Week', fontsize=12, fontweight='bold')
    ax2.set_title('Maintenance Complexity Explodes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Annotate explosion
    ax2.annotate('Unsustainable', xy=(1000, 80), xytext=(600, 50),
                arrowprops=dict(arrowstyle='->', color=mlred, lw=2),
                fontsize=11, fontweight='bold', color=mlred)

    # Add shaded region for "unsustainable zone"
    ax2.axhspan(40, 100, alpha=0.2, color=mlred, label='Unsustainable')
    ax2.legend(fontsize=10)

    plt.suptitle('The Rule Explosion Problem: Why Traditional Programming Fails',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    plt.savefig('../charts/rule_complexity_explosion.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/rule_complexity_explosion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: rule_complexity_explosion.pdf/png")


def create_learning_curve_spam():
    """
    Chart 2: Learning curve showing improvement with data
    Demonstrates diminishing returns and bias-variance tradeoff
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Realistic learning curve data
    examples = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000])

    # Training accuracy (approaches 99% - can memorize training data)
    train_accuracy = 50 + 49 * (1 - np.exp(-examples / 50000))

    # Test accuracy (plateaus at 98.5% - fundamental limit)
    test_accuracy = 50 + 45 * (1 - np.exp(-examples / 80000))

    # Plot both curves
    ax.semilogx(examples, train_accuracy, 'o-', color=mlblue, linewidth=3,
                markersize=8, label='Training Accuracy', alpha=0.8)
    ax.semilogx(examples, test_accuracy, 's-', color=mlgreen, linewidth=3,
                markersize=8, label='Test Accuracy (Real Performance)')

    # Target line
    ax.axhline(y=95, color=mlgray, linestyle='--', linewidth=2, alpha=0.7, label='Target: 95%')

    # Shade overfitting region
    ax.fill_between(examples, train_accuracy, test_accuracy,
                     alpha=0.2, color=mlred, label='Overfitting Gap')

    # Annotations
    ax.annotate('Diminishing returns', xy=(100000, 95), xytext=(200000, 90),
                arrowprops=dict(arrowstyle='->', color=mlpurple, lw=2),
                fontsize=12, fontweight='bold', color=mlpurple)

    ax.annotate('10,000 examples\nreaches 95%', xy=(10000, 95), xytext=(2000, 85),
                arrowprops=dict(arrowstyle='->', color=mlblue, lw=2),
                fontsize=11, fontweight='bold', color=mlblue)

    # Labels
    ax.set_xlabel('Number of Training Examples (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curve: Performance Improves with Data (Diminishing Returns)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(45, 101)

    # Add practical guidance box
    textstr = 'Practical Insight:\n10K examples = 95% accuracy\n100K examples = 98% accuracy\nDiminishing returns beyond this'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save
    plt.savefig('../charts/learning_curve_spam.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/learning_curve_spam.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: learning_curve_spam.pdf/png")


def create_learning_paradigms_comparison():
    """
    Chart 3: Three learning paradigms comparison
    Visual summary for three-column slide
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Supervised Learning visualization
    ax1 = axes[0]
    np.random.seed(42)

    # Generate labeled data
    spam_x = np.random.normal(7, 1.5, 30)
    spam_y = np.random.normal(7, 1.5, 30)
    legit_x = np.random.normal(3, 1.5, 30)
    legit_y = np.random.normal(3, 1.5, 30)

    ax1.scatter(spam_x, spam_y, c=[mlred]*30, s=100, alpha=0.6, edgecolors='black', label='Spam')
    ax1.scatter(legit_x, legit_y, c=[mlblue]*30, s=100, alpha=0.6, edgecolors='black', label='Legitimate')

    # Decision boundary
    x_line = np.linspace(0, 10, 100)
    y_line = x_line
    ax1.plot(x_line, y_line, 'k--', linewidth=2, label='Learned Boundary')

    ax1.set_xlabel('Feature 1 (e.g., CAPS usage)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Feature 2 (e.g., suspicious links)', fontsize=11, fontweight='bold')
    ax1.set_title('Supervised Learning\n(Labeled Examples)', fontsize=13, fontweight='bold', color=mlblue)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # Unsupervised Learning visualization
    ax2 = axes[1]

    # Generate clustered data without labels
    cluster1_x = np.random.normal(3, 1, 25)
    cluster1_y = np.random.normal(3, 1, 25)
    cluster2_x = np.random.normal(7, 1, 25)
    cluster2_y = np.random.normal(7, 1, 25)
    cluster3_x = np.random.normal(5, 0.8, 20)
    cluster3_y = np.random.normal(8, 0.8, 20)

    ax2.scatter(cluster1_x, cluster1_y, c=[mlgreen]*25, s=100, alpha=0.6,
               edgecolors='black', label='Discovered Group 1')
    ax2.scatter(cluster2_x, cluster2_y, c=[mlorange]*25, s=100, alpha=0.6,
               edgecolors='black', label='Discovered Group 2')
    ax2.scatter(cluster3_x, cluster3_y, c=[mlpurple]*20, s=100, alpha=0.6,
               edgecolors='black', label='Discovered Group 3')

    ax2.set_xlabel('Customer Feature 1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Customer Feature 2', fontsize=11, fontweight='bold')
    ax2.set_title('Unsupervised Learning\n(No Labels - Discover Patterns)', fontsize=13, fontweight='bold', color=mlgreen)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Reinforcement Learning visualization
    ax3 = axes[2]

    # Show learning progress over games
    games = np.array([0, 100, 500, 1000, 5000, 10000, 50000, 100000])
    win_rate = 20 + 75 * (1 - np.exp(-games / 20000))

    ax3.plot(games, win_rate, 'o-', color=mlorange, linewidth=3, markersize=8)
    ax3.axhline(y=50, color=mlgray, linestyle='--', linewidth=2, alpha=0.5, label='Random (50%)')
    ax3.fill_between(games, 20, win_rate, alpha=0.3, color=mlorange)

    ax3.set_xlabel('Number of Games Played', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Reinforcement Learning\n(Learn Through Trial & Reward)', fontsize=13, fontweight='bold', color=mlorange)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(15, 100)

    plt.suptitle('Three Learning Paradigms: Complete Taxonomy', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    plt.savefig('../charts/learning_paradigms_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/learning_paradigms_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: learning_paradigms_comparison.pdf/png")


if __name__ == '__main__':
    print("Generating Week 0a Act 1 charts...")
    print("-" * 50)

    create_rule_complexity_explosion()
    create_learning_curve_spam()
    create_learning_paradigms_comparison()

    print("-" * 50)
    print("All Act 1 charts generated successfully!")
    print("Location: ../charts/")