#!/usr/bin/env python3
"""
Create topic modeling visualizations for Week 5 presentation.
Generates all charts referenced in the slides.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import pandas as pd
from scipy.stats import dirichlet
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme matching the presentation
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f',
    'mlyellow': '#f1c40f',
    'mlcyan': '#17becf'
}

def create_topic_discovery_landscape():
    """Create the opening visualization showing topic discovery."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate random points for documents
    np.random.seed(42)
    n_docs = 500

    # Create 5 topic clusters
    topics = []
    topic_labels = ['Innovation', 'Sustainability', 'Digital', 'Health', 'Education']
    topic_colors = [colors['mlblue'], colors['mlgreen'], colors['mlorange'],
                   colors['mlred'], colors['mlpurple']]

    for i in range(5):
        center_x = np.random.uniform(0.2, 0.8)
        center_y = np.random.uniform(0.2, 0.8)
        cluster_x = np.random.normal(center_x, 0.1, n_docs//5)
        cluster_y = np.random.normal(center_y, 0.1, n_docs//5)
        topics.append((cluster_x, cluster_y))

        # Plot cluster
        ax.scatter(cluster_x, cluster_y, alpha=0.3, s=20, c=topic_colors[i])

        # Add topic label
        ax.text(center_x, center_y, topic_labels[i], fontsize=14,
               fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Semantic Dimension 1', fontsize=12)
    ax.set_ylabel('Semantic Dimension 2', fontsize=12)
    ax.set_title('Topic Discovery Landscape: From 10,000 Ideas to 5 Themes', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../charts/topic_discovery_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topic_discovery_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_topic_word_distribution():
    """Create visualization of topic-word distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Sample words for each topic
    topics_data = {
        'Topic 1: Technology': ['AI', 'machine', 'learning', 'data', 'algorithm', 'model', 'neural', 'deep'],
        'Topic 2: Sustainability': ['green', 'eco', 'renewable', 'carbon', 'energy', 'sustainable', 'climate', 'waste'],
        'Topic 3: Health': ['wellness', 'fitness', 'mental', 'health', 'care', 'medical', 'therapy', 'patient']
    }

    for idx, (topic_name, words) in enumerate(topics_data.items()):
        ax = axes[idx]

        # Generate probabilities
        probs = np.random.dirichlet(np.ones(len(words)) * 2)
        probs = np.sort(probs)[::-1]

        # Create bar plot
        bars = ax.barh(range(len(words)), probs, color=list(colors.values())[idx])
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Probability', fontsize=10)
        ax.set_title(topic_name, fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(probs) * 1.1)

        # Add probability values
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(prob + 0.005, i, f'{prob:.3f}', va='center', fontsize=9)

    plt.suptitle('Topic-Word Probability Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../charts/topic_word_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topic_word_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_innovation_funnel():
    """Create the innovation funnel with topic stages."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Funnel stages
    stages = [
        ('Raw Ideas\n10,000', 1.0, colors['mlpurple']),
        ('Topics Found\n50', 0.7, colors['mlblue']),
        ('Prioritized\n10', 0.4, colors['mlorange']),
        ('Developed\n3', 0.2, colors['mlgreen'])
    ]

    y_pos = 0.8
    for i, (label, width, color) in enumerate(stages):
        # Draw trapezoid for funnel
        left = 0.5 - width/2
        right = 0.5 + width/2

        if i < len(stages) - 1:
            next_width = stages[i+1][1]
            vertices = [
                (left, y_pos),
                (right, y_pos),
                (0.5 + next_width/2, y_pos - 0.2),
                (0.5 - next_width/2, y_pos - 0.2)
            ]
        else:
            vertices = [
                (left, y_pos),
                (right, y_pos),
                (right, y_pos - 0.15),
                (left, y_pos - 0.15)
            ]

        from matplotlib.patches import Polygon
        poly = Polygon(vertices, facecolor=color, alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(poly)

        # Add text
        ax.text(0.5, y_pos - 0.08, label, ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

        y_pos -= 0.2

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Innovation Funnel: From Ideas to Solutions', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../charts/innovation_funnel_topics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/innovation_funnel_topics.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_lda_plate_notation():
    """Create simplified LDA plate notation diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Outer plate (documents)
    doc_rect = FancyBboxPatch((0.1, 0.2), 0.8, 0.6,
                              boxstyle="round,pad=0.02",
                              linewidth=2, edgecolor='black',
                              facecolor='none')
    ax.add_patch(doc_rect)
    ax.text(0.85, 0.25, 'D documents', fontsize=10, style='italic')

    # Inner plate (words)
    word_rect = FancyBboxPatch((0.4, 0.3), 0.4, 0.4,
                               boxstyle="round,pad=0.02",
                               linewidth=2, edgecolor='gray',
                               facecolor='none', linestyle='--')
    ax.add_patch(word_rect)
    ax.text(0.75, 0.35, 'N words', fontsize=10, style='italic')

    # Nodes
    nodes = [
        (0.2, 0.65, 'α', colors['mlblue']),
        (0.3, 0.5, 'θ', 'white'),
        (0.5, 0.5, 'z', 'white'),
        (0.65, 0.5, 'w', colors['mlgray']),
        (0.5, 0.8, 'β', colors['mlorange'])
    ]

    for x, y, label, color in nodes:
        circle = Circle((x, y), 0.06, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold')

    # Arrows
    arrows = [
        ((0.26, 0.65), (0.28, 0.54)),  # α → θ
        ((0.36, 0.5), (0.44, 0.5)),     # θ → z
        ((0.5, 0.74), (0.5, 0.56)),     # β → w
        ((0.56, 0.5), (0.59, 0.5))      # z → w
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('LDA Graphical Model', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../charts/lda_plate_notation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/lda_plate_notation.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_algorithm_comparison():
    """Create algorithm comparison matrix."""
    fig, ax = plt.subplots(figsize=(12, 8))

    algorithms = ['LDA', 'NMF', 'LSA', 'BERTopic']
    metrics = ['Speed', 'Interpretability', 'Accuracy', 'Scalability', 'Short Text']

    # Scores (0-1)
    scores = np.array([
        [0.6, 0.9, 0.7, 0.7, 0.5],  # LDA
        [0.8, 0.8, 0.7, 0.8, 0.8],  # NMF
        [0.9, 0.6, 0.6, 0.9, 0.6],  # LSA
        [0.5, 0.7, 0.9, 0.6, 0.9]   # BERTopic
    ])

    # Create heatmap
    im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticklabels(algorithms, fontsize=11)

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{scores[i, j]:.1f}',
                         ha="center", va="center", color="black", fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score', fontsize=11)

    ax.set_title('Topic Modeling Algorithm Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/algorithm_comparison_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/algorithm_comparison_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_topic_coherence_plot():
    """Create topic coherence vs number of topics plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_topics = np.arange(5, 31, 1)
    coherence = 0.35 + 0.25 * np.exp(-((n_topics - 15) ** 2) / 50)
    coherence += np.random.normal(0, 0.01, len(n_topics))

    ax.plot(n_topics, coherence, 'o-', color=colors['mlblue'], linewidth=2, markersize=6)

    # Mark optimal point
    optimal_idx = np.argmax(coherence)
    ax.plot(n_topics[optimal_idx], coherence[optimal_idx], 'o',
           color=colors['mlred'], markersize=12, zorder=5)
    ax.annotate(f'Optimal: {n_topics[optimal_idx]} topics',
               xy=(n_topics[optimal_idx], coherence[optimal_idx]),
               xytext=(n_topics[optimal_idx] + 3, coherence[optimal_idx] - 0.02),
               arrowprops=dict(arrowstyle='->', color=colors['mlred']),
               fontsize=10, fontweight='bold')

    ax.set_xlabel('Number of Topics', fontsize=12)
    ax.set_ylabel('Coherence Score', fontsize=12)
    ax.set_title('Finding Optimal Number of Topics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.65)

    plt.tight_layout()
    plt.savefig('../charts/topic_coherence_plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topic_coherence_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_topic_quality_dashboard():
    """Create topic quality metrics dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Coherence by topic
    ax = axes[0, 0]
    topics = [f'T{i}' for i in range(1, 11)]
    coherences = np.random.uniform(0.4, 0.7, 10)
    bars = ax.bar(topics, coherences, color=[colors['mlgreen'] if c > 0.5 else colors['mlorange'] for c in coherences])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Coherence Score', fontsize=10)
    ax.set_title('Topic Coherence', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.8)

    # Distinctiveness matrix
    ax = axes[0, 1]
    n_topics = 5
    dist_matrix = np.random.uniform(0.6, 0.9, (n_topics, n_topics))
    np.fill_diagonal(dist_matrix, 1.0)
    im = ax.imshow(dist_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(n_topics))
    ax.set_yticks(range(n_topics))
    ax.set_xticklabels([f'T{i+1}' for i in range(n_topics)])
    ax.set_yticklabels([f'T{i+1}' for i in range(n_topics)])
    ax.set_title('Topic Distinctiveness', fontsize=11, fontweight='bold')

    # Coverage pie chart
    ax = axes[1, 0]
    sizes = [65, 20, 10, 5]
    labels = ['Covered', 'Partial', 'Minimal', 'Not covered']
    colors_pie = [colors['mlgreen'], colors['mlyellow'], colors['mlorange'], colors['mlred']]
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title('Document Coverage', fontsize=11, fontweight='bold')

    # Topic size distribution
    ax = axes[1, 1]
    topic_sizes = np.random.exponential(100, 10)
    topic_sizes = np.sort(topic_sizes)[::-1]
    ax.barh(range(len(topic_sizes)), topic_sizes, color=colors['mlpurple'])
    ax.set_yticks(range(len(topic_sizes)))
    ax.set_yticklabels([f'Topic {i+1}' for i in range(len(topic_sizes))])
    ax.set_xlabel('Number of Documents', fontsize=10)
    ax.set_title('Topic Size Distribution', fontsize=11, fontweight='bold')

    plt.suptitle('Topic Quality Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/topic_quality_dashboard.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topic_quality_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Generate all charts for Week 5."""
    print("Generating Week 5 Topic Modeling charts...")

    # Create charts directory if it doesn't exist
    import os
    os.makedirs('../charts', exist_ok=True)

    # Generate all visualizations
    create_topic_discovery_landscape()
    print("[OK] Topic discovery landscape")

    create_topic_word_distribution()
    print("[OK] Topic word distributions")

    create_innovation_funnel()
    print("[OK] Innovation funnel")

    create_lda_plate_notation()
    print("[OK] LDA plate notation")

    create_algorithm_comparison()
    print("[OK] Algorithm comparison matrix")

    create_topic_coherence_plot()
    print("[OK] Topic coherence plot")

    create_topic_quality_dashboard()
    print("[OK] Topic quality dashboard")

    print("\nAll charts generated successfully!")

if __name__ == "__main__":
    main()