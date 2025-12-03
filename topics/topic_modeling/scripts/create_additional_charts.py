"""
Create additional visualization charts for Week 5: Topic Modeling & Ideation
Charts that complement the narrative flow of discovering hidden patterns
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.lines as mlines

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define color palette (matching template_beamer_layout.tex)
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f',
    'mlyellow': '#FFD700',
    'mlcyan': '#17becf',
    'mllavender': '#9999CC',
    'mllavender2': '#B3B3E0',
    'mllavender3': '#CCCCF0',
    'mllavender4': '#E6E6F5'
}

def create_hidden_patterns_revealed():
    """Create the iceberg visualization of hidden vs visible patterns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Traditional Analysis
    ax1.set_title('Traditional Analysis', fontsize=14, fontweight='bold')
    categories = ['Price', 'Quality', 'Service', 'Speed', 'Other']
    visible = [25, 20, 15, 10, 30]

    bars = ax1.bar(categories, visible, color=colors['mlgray'], alpha=0.7)
    ax1.set_ylabel('% of Feedback', fontsize=12)
    ax1.set_ylim(0, 100)

    # Add text showing what's missing
    ax1.text(0.5, 60, 'Missing 70% of insights',
             transform=ax1.transAxes, fontsize=11, color=colors['mlred'],
             horizontalalignment='center', style='italic')

    # Right: Topic Modeling Discovery
    ax2.set_title('Topic Modeling Discovery', fontsize=14, fontweight='bold')

    # Create layers of discovery
    y_positions = np.linspace(0, 90, 20)
    widths = np.random.uniform(3, 15, 20)
    x_positions = np.random.uniform(0, 10, 20)

    for i, (x, y, w) in enumerate(zip(x_positions, y_positions, widths)):
        color = plt.cm.viridis(i/20)
        ax2.barh(y, w, left=x, height=3, color=color, alpha=0.6)

    ax2.set_xlabel('Topic Strength', fontsize=12)
    ax2.set_ylabel('Hidden Themes Discovered', fontsize=12)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 100)

    # Add annotations
    ax2.text(0.5, 0.9, '20 themes found', transform=ax2.transAxes,
             fontsize=11, color=colors['mlgreen'], fontweight='bold',
             horizontalalignment='center')

    plt.tight_layout()
    plt.savefig('../charts/hidden_patterns_revealed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/hidden_patterns_revealed.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_topic_discovery_landscape():
    """Create landscape visualization of topic discovery process"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create mountain-like landscape representing document space
    x = np.linspace(0, 10, 1000)

    # Multiple peaks representing different topics
    y = (0.8 * np.exp(-(x-2)**2/0.5) +
         1.2 * np.exp(-(x-4)**2/0.3) +
         0.6 * np.exp(-(x-6)**2/0.4) +
         0.9 * np.exp(-(x-8)**2/0.35) +
         0.3 * np.sin(x*3) * np.exp(-x/10))

    ax.fill_between(x, 0, y, color=colors['mllavender'], alpha=0.3)
    ax.plot(x, y, color=colors['mlpurple'], linewidth=2)

    # Mark discovered topics as peaks
    peak_positions = [2, 4, 6, 8]
    peak_heights = [0.8, 1.2, 0.6, 0.9]
    peak_labels = ['Customer\nService', 'Product\nQuality', 'Shipping\nIssues', 'Innovation\nNeeds']

    for pos, height, label in zip(peak_positions, peak_heights, peak_labels):
        ax.plot(pos, height, 'o', markersize=12, color=colors['mlgreen'])
        ax.annotate(label, xy=(pos, height), xytext=(pos, height+0.3),
                   ha='center', fontsize=10, color=colors['mlgreen'],
                   fontweight='bold', arrowprops=dict(arrowstyle='->', color=colors['mlgreen']))

    ax.set_xlabel('Document Space', fontsize=12)
    ax.set_ylabel('Topic Strength', fontsize=12)
    ax.set_title('Topic Discovery Landscape: Finding Peaks in Data Mountains', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig('../charts/topic_discovery_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topic_discovery_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_topic_word_distribution():
    """Create visualization of word probabilities within a topic"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Words and their probabilities for a "Customer Service" topic
    words = ['service', 'support', 'helpful', 'response', 'quick',
             'team', 'issue', 'resolved', 'thank', 'excellent',
             'problem', 'email', 'chat', 'phone', 'wait']
    probs = [0.15, 0.12, 0.10, 0.08, 0.07,
             0.06, 0.05, 0.05, 0.04, 0.04,
             0.03, 0.03, 0.03, 0.02, 0.02]

    # Normalize to show remaining probability
    shown_prob = sum(probs)
    probs.append(1 - shown_prob)
    words.append('...')

    # Create bars
    bars = ax.bar(range(len(words)), probs, color=colors['mlblue'], alpha=0.7)

    # Color top 5 words differently
    for i in range(5):
        bars[i].set_color(colors['mlorange'])
        bars[i].set_alpha(0.9)

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Topic Word Distribution: "Customer Service" Theme', fontsize=14, fontweight='bold')

    # Add annotation
    ax.axhline(y=0.05, color=colors['mlred'], linestyle='--', alpha=0.5)
    ax.text(len(words)-1, 0.05, 'Core words threshold',
            ha='right', va='bottom', color=colors['mlred'], fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../charts/topic_word_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topic_word_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_topic_coherence_plot():
    """Create plot showing coherence scores vs number of topics"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Number of topics to test
    num_topics = np.arange(5, 51, 5)

    # Simulated coherence scores (typically peaks around 15-25 topics)
    coherence = (0.35 + 0.15 * np.exp(-(num_topics-20)**2/100)
                 - 0.001 * num_topics + 0.02 * np.random.randn(len(num_topics)))

    # Plot main line
    ax.plot(num_topics, coherence, 'o-', color=colors['mlblue'],
            linewidth=2, markersize=8, label='Coherence Score')

    # Mark optimal point
    optimal_idx = np.argmax(coherence)
    ax.plot(num_topics[optimal_idx], coherence[optimal_idx], 'o',
            markersize=15, color=colors['mlgreen'], zorder=5)
    ax.annotate('Optimal: 20 topics',
                xy=(num_topics[optimal_idx], coherence[optimal_idx]),
                xytext=(num_topics[optimal_idx]+5, coherence[optimal_idx]+0.02),
                arrowprops=dict(arrowstyle='->', color=colors['mlgreen']),
                fontsize=11, color=colors['mlgreen'], fontweight='bold')

    # Add regions
    ax.axvspan(5, 10, alpha=0.2, color=colors['mlred'], label='Too General')
    ax.axvspan(35, 50, alpha=0.2, color=colors['mlorange'], label='Too Specific')
    ax.axvspan(15, 25, alpha=0.2, color=colors['mlgreen'], label='Sweet Spot')

    ax.set_xlabel('Number of Topics', fontsize=12)
    ax.set_ylabel('Topic Coherence Score', fontsize=12)
    ax.set_title('Finding the Right Number of Topics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('../charts/topic_coherence_plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topic_coherence_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_algorithm_comparison():
    """Create comparison chart of different topic modeling algorithms"""
    fig, ax = plt.subplots(figsize=(12, 8))

    algorithms = ['LDA', 'NMF', 'LSA', 'BERTopic']
    metrics = ['Speed', 'Interpretability', 'Accuracy', 'Scalability', 'Context-Aware']

    # Scores for each algorithm (out of 5)
    scores = {
        'LDA': [3, 5, 4, 4, 2],
        'NMF': [5, 4, 3, 5, 2],
        'LSA': [4, 3, 3, 4, 1],
        'BERTopic': [2, 4, 5, 3, 5]
    }

    x = np.arange(len(metrics))
    width = 0.2

    colors_list = [colors['mlred'], colors['mlblue'], colors['mlgreen'], colors['mlpurple']]

    for i, (algo, color) in enumerate(zip(algorithms, colors_list)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, scores[algo], width, label=algo, color=color, alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score (1-5)', fontsize=12)
    ax.set_title('Algorithm Comparison: Choose Based on Your Needs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 5.5)

    # Add annotations for best use cases
    ax.text(0.02, 0.98, 'Best for:', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    ax.text(0.02, 0.93, '• LDA: General exploration', transform=ax.transAxes, fontsize=10, va='top', color=colors['mlred'])
    ax.text(0.02, 0.88, '• NMF: Clear topics', transform=ax.transAxes, fontsize=10, va='top', color=colors['mlblue'])
    ax.text(0.02, 0.83, '• LSA: Quick analysis', transform=ax.transAxes, fontsize=10, va='top', color=colors['mlgreen'])
    ax.text(0.02, 0.78, '• BERT: Accuracy', transform=ax.transAxes, fontsize=10, va='top', color=colors['mlpurple'])

    plt.tight_layout()
    plt.savefig('../charts/algorithm_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_dynamic_topics_timeline():
    """Create visualization of topics evolving over time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Time periods
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

    # Topics and their prevalence over time
    topics = {
        'Quality Issues': [10, 25, 30, 15, 10, 5],
        'Shipping Delays': [5, 10, 15, 35, 20, 10],
        'Great Service': [30, 25, 20, 15, 25, 35],
        'Price Concerns': [15, 10, 5, 10, 20, 25],
        'New Features': [5, 10, 15, 20, 25, 30]
    }

    # Create stacked area chart
    x = np.arange(len(months))
    y_stack = np.zeros(len(months))

    colors_list = [colors['mlred'], colors['mlorange'], colors['mlgreen'],
                   colors['mlblue'], colors['mlpurple']]

    for (topic, values), color in zip(topics.items(), colors_list):
        ax.fill_between(x, y_stack, y_stack + values, label=topic, color=color, alpha=0.7)
        y_stack += values

    # Mark critical events
    ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.7)
    ax.text(3.5, 95, 'Supply Chain Crisis', rotation=90,
            va='top', ha='right', color='red', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Topic Prevalence (%)', fontsize=12)
    ax.set_title('Dynamic Topics: Tracking Changes Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('../charts/dynamic_topics_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/dynamic_topics_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_innovation_opportunity_map():
    """Create opportunity map for Spotify micro-moods case"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create quadrant plot
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Define opportunities
    opportunities = {
        'Monday Motivation': (0.7, 0.8, colors['mlgreen']),
        'Workout Energy': (0.9, 0.6, colors['mlgreen']),
        'Study Focus': (0.6, 0.9, colors['mlgreen']),
        'Rainy Day': (0.3, 0.7, colors['mlblue']),
        'Party Mode': (0.8, 0.4, colors['mlorange']),
        'Sleep Time': (-0.2, 0.5, colors['mlpurple']),
        'Commute Mix': (0.5, 0.3, colors['mlblue']),
        'Cooking Jazz': (-0.4, 0.6, colors['mlpurple']),
        'Romance': (0.2, 0.5, colors['mlred']),
        'Nostalgia': (-0.6, 0.8, colors['mlred']),
        'Productivity': (0.7, 0.7, colors['mlgreen']),
        'Meditation': (-0.7, 0.9, colors['mlpurple'])
    }

    for mood, (x, y, color) in opportunities.items():
        size = np.random.uniform(200, 800)  # Random size for market potential
        ax.scatter(x, y, s=size, alpha=0.6, color=color)
        ax.annotate(mood, xy=(x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    ax.set_xlabel('← Traditional | Innovative →', fontsize=12)
    ax.set_ylabel('← Niche | Mainstream →', fontsize=12)
    ax.set_title('Innovation Opportunity Map: 1,500 Micro-Moods Discovered', fontsize=14, fontweight='bold')

    # Add quadrant labels
    ax.text(0.5, 0.9, 'High Impact\nInnovations', transform=ax.transAxes,
            fontsize=11, ha='center', color=colors['mlgreen'], fontweight='bold')
    ax.text(0.05, 0.9, 'Experimental\nNiche', transform=ax.transAxes,
            fontsize=11, ha='center', color=colors['mlpurple'], fontweight='bold')

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('../charts/innovation_opportunity_map.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/innovation_opportunity_map.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_topics_to_opportunities():
    """Create funnel showing topics converting to opportunities"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Funnel data
    stages = ['Raw Feedback\n100,000 items',
              'Topics Found\n50 themes',
              'Validated\n20 strong',
              'Actionable\n10 ideas',
              'Prioritized\n5 projects',
              'Launched\n2 products']

    values = [100000, 50, 20, 10, 5, 2]
    colors_list = [colors['mlgray'], colors['mllavender'], colors['mlblue'],
                   colors['mlorange'], colors['mlgreen'], colors['mlpurple']]

    # Create funnel
    y_pos = np.arange(len(stages))

    for i, (stage, value, color) in enumerate(zip(stages, values, colors_list)):
        # Calculate width based on log scale
        width = np.log10(value + 1)
        ax.barh(i, width, color=color, alpha=0.7, edgecolor='white', linewidth=2)

        # Add text
        ax.text(width/2, i, stage, ha='center', va='center',
                fontweight='bold', fontsize=10)

    ax.set_yticks([])
    ax.set_xlim(0, 6)
    ax.set_xlabel('Process Refinement →', fontsize=12)
    ax.set_title('From Feedback Chaos to Product Launch', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add success rate
    ax.text(0.98, 0.02, 'Success Rate: 2%', transform=ax.transAxes,
            fontsize=11, ha='right', color=colors['mlred'], fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/topics_to_opportunities.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topics_to_opportunities.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_innovation_funnel_topics():
    """Create innovation funnel visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Funnel stages
    stages = [
        ('Data Collection', 100, colors['mlgray']),
        ('Topic Discovery', 70, colors['mllavender2']),
        ('Pattern Analysis', 40, colors['mlblue']),
        ('Opportunity Mapping', 20, colors['mlorange']),
        ('Concept Development', 10, colors['mlgreen']),
        ('Product Launch', 3, colors['mlpurple'])
    ]

    # Create funnel shape
    for i, (stage, width, color) in enumerate(stages):
        y = len(stages) - i - 1

        # Create trapezoid for each stage
        left = (100 - width) / 2
        right = left + width

        # Draw the stage
        ax.fill([left, right, right + 5, left - 5],
                [y, y, y-0.9, y-0.9],
                color=color, alpha=0.7, edgecolor='white', linewidth=2)

        # Add text
        ax.text(50, y-0.45, f'{stage}\n{width}%',
                ha='center', va='center', fontweight='bold', fontsize=10)

    # Add side annotations
    ax.text(105, 5.5, '500+ Ideas', fontsize=10, color=colors['mlgray'])
    ax.text(105, 4.5, '20-30 Topics', fontsize=10, color=colors['mllavender'])
    ax.text(105, 3.5, 'Patterns', fontsize=10, color=colors['mlblue'])
    ax.text(105, 2.5, '5-10 Opps', fontsize=10, color=colors['mlorange'])
    ax.text(105, 1.5, 'Concepts', fontsize=10, color=colors['mlgreen'])
    ax.text(105, 0.5, '1-3 Products', fontsize=10, color=colors['mlpurple'])

    ax.set_xlim(-10, 120)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    ax.set_title('The Innovation Funnel: From Data to Products',
                 fontsize=14, fontweight='bold', pad=20)

    # Add success metrics
    ax.text(0.5, -0.05, 'Success Rate: 60% vs 10% traditional',
            transform=ax.transAxes, ha='center', fontsize=11,
            color=colors['mlgreen'], fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/innovation_funnel_topics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/innovation_funnel_topics.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Generate all charts"""
    print("Creating Week 5 additional charts...")

    create_hidden_patterns_revealed()
    print("[OK] Hidden patterns revealed chart created")

    create_topic_discovery_landscape()
    print("[OK] Topic discovery landscape created")

    create_topic_word_distribution()
    print("[OK] Topic word distribution created")

    create_topic_coherence_plot()
    print("[OK] Topic coherence plot created")

    create_algorithm_comparison()
    print("[OK] Algorithm comparison created")

    create_dynamic_topics_timeline()
    print("[OK] Dynamic topics timeline created")

    create_innovation_opportunity_map()
    print("[OK] Innovation opportunity map created")

    create_topics_to_opportunities()
    print("[OK] Topics to opportunities funnel created")

    create_innovation_funnel_topics()
    print("[OK] Innovation funnel created")

    print("\nAll charts generated successfully!")

if __name__ == "__main__":
    main()