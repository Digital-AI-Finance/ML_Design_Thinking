"""
Generate charts for Week 0a Act 4: Synthesis and Impact
Following pedagogical framework: Bring everything together
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
mllavender = (173/255, 173/255, 224/255)

plt.style.use('seaborn-v0_8-whitegrid')

def create_ml_pipeline_complete():
    """
    Chart 1: Complete ML pipeline from data to deployment
    Shows the full lifecycle with feedback loops
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Pipeline stages (y-coordinates)
    stages = [
        (2, 8, 'Data Collection', mlblue, 'Web scraping, sensors,\ndatabases, APIs'),
        (6, 8, 'Data Cleaning', mlblue, 'Handle missing values,\noutliers, duplicates'),
        (10, 8, 'Data Splitting', mlblue, 'Train 70%\nValidation 15%\nTest 15%'),

        (2, 5, 'Algorithm Selection', mlgreen, 'Linear, SVM, or\nNeural Network?'),
        (6, 5, 'Model Training', mlgreen, 'Optimize parameters\nMinimize loss'),
        (10, 5, 'Validation', mlgreen, 'Tune hyperparameters\nPrevent overfitting'),

        (2, 2, 'Testing', mlorange, 'Measure real\nperformance'),
        (6, 2, 'Deployment', mlorange, 'API, cloud, or\nedge device'),
        (10, 2, 'Monitoring', mlorange, 'Detect drift\nMeasure impact'),
        (14, 2, 'Retraining', mlorange, 'Update with\nnew data')
    ]

    # Draw stages
    box_width = 2.5
    box_height = 1.2

    for x, y, label, color, details in stages:
        # Main box
        box = FancyBboxPatch((x - box_width/2, y - box_height/2), box_width, box_height,
                            boxstyle="round,pad=0.1", edgecolor='black', facecolor=color,
                            linewidth=2, alpha=0.6, zorder=2)
        ax.add_patch(box)

        # Label
        ax.text(x, y + 0.25, label, ha='center', va='center',
               fontsize=11, fontweight='bold', color='white', zorder=3)

        # Details
        ax.text(x, y - 0.25, details, ha='center', va='center',
               fontsize=8, color='white', zorder=3)

    # Draw arrows between stages
    arrows = [
        # Data phase
        (2 + box_width/2, 8, 6 - box_width/2, 8, 'black'),
        (6 + box_width/2, 8, 10 - box_width/2, 8, 'black'),

        # Model phase
        (2 + box_width/2, 5, 6 - box_width/2, 5, 'black'),
        (6 + box_width/2, 5, 10 - box_width/2, 5, 'black'),

        # Deployment phase
        (2 + box_width/2, 2, 6 - box_width/2, 2, 'black'),
        (6 + box_width/2, 2, 10 - box_width/2, 2, 'black'),
        (10 + box_width/2, 2, 14 - box_width/2, 2, 'black'),

        # Vertical connections
        (10, 8 - box_height/2, 2, 5 + box_height/2, mlpurple),  # Split to algo
        (10, 5 - box_height/2, 2, 2 + box_height/2, mlpurple),  # Validation to test

        # Feedback loops
        (14, 2 - box_height/2, 2, 8 + box_height/2, mlred),  # Retrain to collect (arc)
    ]

    for x1, y1, x2, y2, color in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=30, linewidth=2.5,
                               color=color, zorder=1, alpha=0.7)
        ax.add_patch(arrow)

    # Phase labels
    ax.text(6, 9.5, 'DATA PHASE', ha='center', fontsize=14, fontweight='bold',
           color=mlblue, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(6, 6.5, 'MODEL PHASE', ha='center', fontsize=14, fontweight='bold',
           color=mlgreen, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(8, 0.5, 'DEPLOYMENT PHASE', ha='center', fontsize=14, fontweight='bold',
           color=mlorange, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Key insight box
    textstr = 'KEY INSIGHT:\n\nML is a continuous cycle.\nMonitoring and retraining\nare as important as\ninitial model training.'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black', linewidth=2)
    ax.text(14.5, 6, textstr, ha='left', va='center', fontsize=10,
           bbox=props, fontweight='bold')

    # Feedback loop label
    ax.text(8, 4.5, 'Feedback Loop:\nRetrain when\nperformance degrades',
           ha='center', fontsize=9, color=mlred, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('The Complete Machine Learning Pipeline: From Data to Production',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('../charts/ml_pipeline_complete.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/ml_pipeline_complete.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: ml_pipeline_complete.pdf/png")


def create_algorithm_decision_tree():
    """
    Chart 2: Decision tree for algorithm selection
    Helps users choose the right approach
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Decision tree nodes
    # Format: (x, y, text, color, shape)
    nodes = [
        # Root
        (7, 9, 'Start:\nWhat is your\ndata size?', mlgray, 'round', 1.2),

        # First level (data size)
        (2, 7, 'Small\n< 1K samples', mlblue, 'round', 1.0),
        (7, 7, 'Medium\n1K-100K samples', mlgreen, 'round', 1.0),
        (12, 7, 'Large\n> 100K samples', mlorange, 'round', 1.0),

        # Second level (small data)
        (1, 5, 'Linear\nRelationship?', mlblue, 'round', 0.8),
        (3, 5, 'Need\nInterpretability?', mlblue, 'round', 0.8),

        # Recommendations for small data
        (0.5, 3, 'Linear\nRegression', mlblue, 'box', 0.7),
        (1.5, 3, 'Polynomial\nFeatures', mlblue, 'box', 0.7),
        (3, 3, 'Decision\nTree', mlblue, 'box', 0.7),

        # Second level (medium data)
        (6, 5, 'Structured\nData?', mlgreen, 'round', 0.8),
        (8, 5, 'Need Global\nOptimum?', mlgreen, 'round', 0.8),

        # Recommendations for medium data
        (5.5, 3, 'SVM\n(Linear)', mlgreen, 'box', 0.7),
        (7, 3, 'SVM\n(RBF Kernel)', mlgreen, 'box', 0.7),
        (8.5, 3, 'Random\nForest', mlgreen, 'box', 0.7),

        # Second level (large data)
        (11, 5, 'Data Type?', mlorange, 'round', 0.8),
        (13, 5, 'Computational\nResources?', mlorange, 'round', 0.8),

        # Recommendations for large data
        (10.5, 3, 'MLP\n(Dense)', mlorange, 'box', 0.7),
        (11.5, 3, 'CNN\n(Images)', mlorange, 'box', 0.7),
        (12.5, 3, 'Transformer\n(Text)', mlorange, 'box', 0.7),
        (13.5, 3, 'Gradient\nBoosting', mlorange, 'box', 0.7),
    ]

    # Draw nodes
    for x, y, label, color, shape, size in nodes:
        if shape == 'round':
            circle = plt.Circle((x, y), size/2, color=color, alpha=0.6, edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(circle)
        else:  # box
            box = FancyBboxPatch((x - size/2, y - size/3), size, size * 0.66,
                                boxstyle="round,pad=0.05", edgecolor='black', facecolor=color,
                                linewidth=2, alpha=0.7, zorder=2)
            ax.add_patch(box)

        # Text
        if shape == 'round':
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=9 if size < 1 else 10, fontweight='bold', color='white', zorder=3)
        else:
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white', zorder=3)

    # Draw edges
    edges = [
        # Root to first level
        (7, 9 - 0.6, 2, 7 + 0.5, mlgray),
        (7, 9 - 0.6, 7, 7 + 0.5, mlgray),
        (7, 9 - 0.6, 12, 7 + 0.5, mlgray),

        # Small data branches
        (2, 7 - 0.5, 1, 5 + 0.4, mlblue),
        (2, 7 - 0.5, 3, 5 + 0.4, mlblue),
        (1, 5 - 0.4, 0.5, 3 + 0.35, mlblue),
        (1, 5 - 0.4, 1.5, 3 + 0.35, mlblue),
        (3, 5 - 0.4, 3, 3 + 0.35, mlblue),

        # Medium data branches
        (7, 7 - 0.5, 6, 5 + 0.4, mlgreen),
        (7, 7 - 0.5, 8, 5 + 0.4, mlgreen),
        (6, 5 - 0.4, 5.5, 3 + 0.35, mlgreen),
        (6, 5 - 0.4, 7, 3 + 0.35, mlgreen),
        (8, 5 - 0.4, 8.5, 3 + 0.35, mlgreen),

        # Large data branches
        (12, 7 - 0.5, 11, 5 + 0.4, mlorange),
        (12, 7 - 0.5, 13, 5 + 0.4, mlorange),
        (11, 5 - 0.4, 10.5, 3 + 0.35, mlorange),
        (11, 5 - 0.4, 11.5, 3 + 0.35, mlorange),
        (11, 5 - 0.4, 12.5, 3 + 0.35, mlorange),
        (13, 5 - 0.4, 13.5, 3 + 0.35, mlorange),
    ]

    for x1, y1, x2, y2, color in edges:
        ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=2, alpha=0.6, zorder=1)

    # Add decision labels on edges
    ax.text(1.5, 7.5, 'Small', fontsize=8, ha='center', color=mlblue, fontweight='bold')
    ax.text(7, 8, 'Medium', fontsize=8, ha='center', color=mlgreen, fontweight='bold')
    ax.text(10, 7.5, 'Large', fontsize=8, ha='center', color=mlorange, fontweight='bold')

    # Key insights box
    textstr = 'DECISION FACTORS:\n\n• Data size\n• Interpretability needs\n• Data type (tabular/images/text)\n• Computational resources\n• Linearity of relationships\n\nTRY MULTIPLE:\nOften best to compare\nseveral approaches!'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, ha='left', va='top',
           fontsize=9, bbox=props, fontweight='bold')

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(2, 10)
    ax.axis('off')
    ax.set_title('Algorithm Selection Decision Tree: Choose the Right Approach',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('../charts/algorithm_decision_tree.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/algorithm_decision_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: algorithm_decision_tree.pdf/png")


if __name__ == '__main__':
    print("Generating Week 0a Act 4 charts...")
    print("-" * 50)

    create_ml_pipeline_complete()
    create_algorithm_decision_tree()

    print("-" * 50)
    print("All Act 4 charts generated successfully!")
    print("Location: ../charts/")