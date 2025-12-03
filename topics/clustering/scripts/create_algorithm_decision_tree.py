"""
Generate clustering algorithm selection decision tree chart
Following Week 9's validation_depth_decision.pdf pattern
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Use seaborn style
plt.style.use('seaborn-v0_8-whitegrid')

def save_chart(name):
    """Save chart as both PDF and PNG"""
    plt.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Created {name}.pdf and {name}.png')

def chart_clustering_algorithm_decision():
    """Decision tree for clustering algorithm selection"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'When to Use Which Clustering Algorithm: Decision Framework',
            ha='center', fontsize=16, fontweight='bold')

    # Root question
    root_box = FancyBboxPatch((5.5, 8.2), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(root_box)
    ax.text(7, 8.6, 'What is your priority?', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Three main branches
    # Branch 1: Speed & Simplicity -> K-means
    ax.arrow(6.5, 8.2, -2, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    speed_box = FancyBboxPatch((2.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor='#2196F3', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(speed_box)
    ax.text(4, 6.2, 'SPEED & SIMPLICITY', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    ax.text(4, 5.4, 'Fast results needed\nRound-ish clusters\nKnow approx. K\nFirst exploration',
            ha='center', va='top', fontsize=9)

    ax.arrow(4, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_kmeans = FancyBboxPatch((2.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                 facecolor='#64B5F6', edgecolor='black', linewidth=2)
    ax.add_patch(result_kmeans)
    ax.text(4, 4.5, 'K-MEANS', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(4, 4.1, 'O(nKd) complexity\nSpherical clusters\nRequires K parameter\nFastest option',
            ha='center', va='center', fontsize=8)

    # Branch 2: Arbitrary Shapes -> DBSCAN
    ax.arrow(7, 8.2, 0, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    shape_box = FancyBboxPatch((5.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='#4CAF50', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(shape_box)
    ax.text(7, 6.2, 'ARBITRARY SHAPES', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    ax.text(7, 5.4, 'Non-convex clusters\nOutliers present\nVarying densities\nDon\'t know K',
            ha='center', va='top', fontsize=9)

    ax.arrow(7, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_dbscan = FancyBboxPatch((5.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                facecolor='#81C784', edgecolor='black', linewidth=2)
    ax.add_patch(result_dbscan)
    ax.text(7, 4.5, 'DBSCAN', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(7, 4.1, 'Density-based\nHandles any shape\nIdentifies noise\nRequires eps/minPts',
            ha='center', va='center', fontsize=8)

    # Branch 3: Hierarchy Needed -> Hierarchical
    ax.arrow(7.5, 8.2, 2, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    hier_box = FancyBboxPatch((8.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='#FF9800', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(hier_box)
    ax.text(10, 6.2, 'HIERARCHY NEEDED', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    ax.text(10, 5.4, 'Multi-level groups\nExplore granularity\nTaxonomy building\nSmaller dataset',
            ha='center', va='top', fontsize=9)

    ax.arrow(10, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_hier = FancyBboxPatch((8.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                facecolor='#FFB74D', edgecolor='black', linewidth=2)
    ax.add_patch(result_hier)
    ax.text(10, 4.5, 'HIERARCHICAL', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(10, 4.1, 'Dendrogram output\nNested clusters\nCut at any level\nO(n²) or O(n³)',
            ha='center', va='center', fontsize=8)

    # Additional considerations box
    consider_box = FancyBboxPatch((0.5, 0.5), 13, 2.5, boxstyle="round,pad=0.1",
                                  facecolor='#F0F0F0', edgecolor='black', linewidth=2)
    ax.add_patch(consider_box)
    ax.text(7, 2.7, 'Additional Considerations', ha='center', va='center',
            fontsize=12, fontweight='bold')

    considerations_text = """
Dataset Size: Very large (>100K points) → MiniBatch K-means; Small (<10K) → Hierarchical feasible
Outliers Critical: Fraud detection, anomaly detection → DBSCAN preferred
Soft Assignments Needed: Mixed populations, uncertainty quantification → GMM (Gaussian Mixture)
High Dimensions: d>20 → Curse of dimensionality affects distance; Consider dimensionality reduction first
Reproducibility: Random init sensitivity → Use K-means++ or fixed seed; DBSCAN/Hierarchical deterministic
Production Deployment: Streaming data → BIRCH; Real-time → K-means; Batch → Any algorithm suitable
    """
    ax.text(7, 1.5, considerations_text, ha='center', va='center', fontsize=8,
            family='monospace')

    # Bottom principle
    ax.text(7, 0.2, 'Principle: Start simple (K-means), upgrade if needed (DBSCAN for shapes, Hierarchical for structure)',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    save_chart('clustering_algorithm_decision')

if __name__ == '__main__':
    chart_clustering_algorithm_decision()
    print('Clustering algorithm decision tree chart created successfully!')
