#!/usr/bin/env python3
"""
Create additional visualizations for Week 5 Topic Modeling presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Wedge
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme
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

def create_lda_document_topics():
    """Create visualization of document-topic distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sample documents
    docs = ['Review 1', 'Review 2', 'Review 3', 'Review 4']
    topics = ['Food', 'Service', 'Atmosphere']

    # Document-topic proportions
    data = np.array([
        [0.7, 0.2, 0.1],  # Review 1
        [0.2, 0.6, 0.2],  # Review 2
        [0.1, 0.2, 0.7],  # Review 3
        [0.4, 0.3, 0.3],  # Review 4
    ])

    # Create stacked bar chart
    bottom = np.zeros(len(docs))
    topic_colors = [colors['mlblue'], colors['mlorange'], colors['mlgreen']]

    for i, topic in enumerate(topics):
        ax.bar(docs, data[:, i], bottom=bottom, label=topic, color=topic_colors[i])
        bottom += data[:, i]

    ax.set_ylabel('Topic Proportion', fontsize=12)
    ax.set_title('Document-Topic Distribution Example', fontsize=14, fontweight='bold')
    ax.legend(title='Topics', loc='upper right')
    ax.set_ylim(0, 1)

    # Add percentage labels
    for i, doc in enumerate(docs):
        y_offset = 0
        for j, topic in enumerate(topics):
            if data[i, j] > 0.05:
                ax.text(i, y_offset + data[i, j]/2, f'{data[i, j]:.0%}',
                       ha='center', va='center', fontsize=10, color='white', fontweight='bold')
            y_offset += data[i, j]

    plt.tight_layout()
    plt.savefig('../charts/lda_document_topics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/lda_document_topics.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_nmf_decomposition():
    """Create NMF matrix decomposition visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Remove axes
    ax.axis('off')

    # Matrix V (documents × terms)
    ax.add_patch(Rectangle((0.05, 0.3), 0.15, 0.4, facecolor=colors['mlgray'], alpha=0.5))
    ax.text(0.125, 0.5, 'V\n(Docs×Terms)\n10000×5000', ha='center', va='center', fontsize=11, fontweight='bold')

    # Equals sign
    ax.text(0.25, 0.5, '≈', fontsize=20, ha='center', va='center')

    # Matrix W (documents × topics)
    ax.add_patch(Rectangle((0.35, 0.3), 0.15, 0.4, facecolor=colors['mlblue'], alpha=0.5))
    ax.text(0.425, 0.5, 'W\n(Docs×Topics)\n10000×20', ha='center', va='center', fontsize=11, fontweight='bold')

    # Times sign
    ax.text(0.55, 0.5, '×', fontsize=20, ha='center', va='center')

    # Matrix H (topics × terms)
    ax.add_patch(Rectangle((0.65, 0.4), 0.25, 0.2, facecolor=colors['mlorange'], alpha=0.5))
    ax.text(0.775, 0.5, 'H\n(Topics×Terms)\n20×5000', ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('NMF: Non-negative Matrix Factorization', fontsize=14, fontweight='bold', y=0.85)

    plt.tight_layout()
    plt.savefig('../charts/nmf_decomposition.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/nmf_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_topics_to_opportunities():
    """Create topics to opportunities transformation visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Left side - Topics
    topics_y = np.linspace(0.2, 0.8, 3)
    topics = ['Battery Issues', 'User Interface', 'Connectivity']

    for i, (y, topic) in enumerate(zip(topics_y, topics)):
        ax.add_patch(Circle((0.2, y), 0.08, facecolor=colors['mlblue'], alpha=0.6))
        ax.text(0.2, y, 'T', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(0.05, y, topic, ha='right', va='center', fontsize=10)

    # Middle - Translation
    ax.arrow(0.35, 0.5, 0.2, 0, head_width=0.05, head_length=0.02, fc=colors['mlgray'], ec=colors['mlgray'])
    ax.text(0.45, 0.55, 'Analysis', ha='center', fontsize=10, style='italic')

    # Right side - Opportunities
    opps_y = np.linspace(0.15, 0.85, 4)
    opportunities = [
        'Wireless Charging',
        'Power Banks',
        'Gesture Control',
        '5G Integration'
    ]

    for i, (y, opp) in enumerate(zip(opps_y, opportunities)):
        ax.add_patch(Rectangle((0.65, y-0.03), 0.3, 0.06, facecolor=colors['mlgreen'], alpha=0.6))
        ax.text(0.8, y, opp, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw connections
    connections = [
        (0, 0), (0, 1), (1, 2), (2, 3)
    ]
    for topic_idx, opp_idx in connections:
        ax.plot([0.28, 0.65], [topics_y[topic_idx], opps_y[opp_idx]],
               'k--', alpha=0.3, linewidth=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('From Topics to Design Opportunities', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../charts/topics_to_opportunities.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/topics_to_opportunities.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_innovation_opportunity_map():
    """Create innovation opportunity mapping visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Generate random opportunities
    np.random.seed(42)
    n_opportunities = 30

    frequency = np.random.uniform(0, 100, n_opportunities)
    impact = np.random.uniform(0, 100, n_opportunities)

    # Categorize into quadrants
    colors_quad = []
    for f, i in zip(frequency, impact):
        if f > 50 and i > 50:
            colors_quad.append(colors['mlgreen'])  # High-High
        elif f < 50 and i > 50:
            colors_quad.append(colors['mlorange'])  # Low-High
        elif f > 50 and i < 50:
            colors_quad.append(colors['mlyellow'])  # High-Low
        else:
            colors_quad.append(colors['mlgray'])    # Low-Low

    # Create scatter plot
    scatter = ax.scatter(frequency, impact, c=colors_quad, s=200, alpha=0.6, edgecolors='black', linewidth=1)

    # Add quadrant lines
    ax.axhline(y=50, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.3)

    # Label quadrants
    ax.text(75, 75, 'Priority\nInnovations', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(25, 75, 'Differentiation\nOpportunities', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(75, 25, 'Quick Wins', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(25, 25, 'Low Priority', ha='center', va='center', fontsize=11, fontweight='bold', alpha=0.5)

    ax.set_xlabel('Topic Frequency (%)', fontsize=12)
    ax.set_ylabel('Business Impact (%)', fontsize=12)
    ax.set_title('Innovation Opportunity Mapping', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../charts/innovation_opportunity_map.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/innovation_opportunity_map.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_persona_topic_heatmap():
    """Create persona-topic alignment heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    personas = ['Tech Enthusiasts', 'Eco-conscious', 'Budget-minded', 'Premium seekers', 'Early adopters']
    topics = ['AI/Automation', 'Sustainability', 'Value/Price', 'Quality', 'Innovation', 'Design', 'Ethics']

    # Create random alignment scores
    np.random.seed(42)
    data = np.random.uniform(0.2, 1.0, (len(personas), len(topics)))

    # Make some logical adjustments
    data[0, 0] = 0.95  # Tech Enthusiasts - AI
    data[1, 1] = 0.92  # Eco-conscious - Sustainability
    data[2, 2] = 0.88  # Budget-minded - Value
    data[3, 3] = 0.90  # Premium seekers - Quality

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(topics)))
    ax.set_yticks(np.arange(len(personas)))
    ax.set_xticklabels(topics, rotation=45, ha='right')
    ax.set_yticklabels(personas)

    # Add text annotations
    for i in range(len(personas)):
        for j in range(len(topics)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                         ha="center", va="center", color="black" if data[i, j] < 0.5 else "white", fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Alignment Score', fontsize=11)

    ax.set_title('Persona-Topic Alignment Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../charts/persona_topic_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/persona_topic_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_trend_evolution():
    """Create trend evolution over time visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create trend data
    ai_ethics = 20 + np.cumsum(np.random.randn(12) * 2 + 1.5)
    circular_economy = 15 + np.cumsum(np.random.randn(12) * 1.5 + 1.2)
    mental_health = 10 + np.cumsum(np.random.randn(12) * 2 + 2)
    remote_work = 30 - np.cumsum(np.random.randn(12) * 1 + 0.5)

    ax.plot(months, ai_ethics, marker='o', linewidth=2, label='AI Ethics', color=colors['mlblue'])
    ax.plot(months, circular_economy, marker='s', linewidth=2, label='Circular Economy', color=colors['mlgreen'])
    ax.plot(months, mental_health, marker='^', linewidth=2, label='Mental Health Tech', color=colors['mlorange'])
    ax.plot(months, remote_work, marker='d', linewidth=2, label='Remote Work', color=colors['mlgray'], linestyle='--')

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Topic Prevalence (%)', fontsize=12)
    ax.set_title('Emerging Topic Trends Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 60)

    # Add trend arrows
    ax.annotate('', xy=(11.5, ai_ethics[-1]), xytext=(10.5, ai_ethics[-3]),
               arrowprops=dict(arrowstyle='->', color=colors['mlgreen'], lw=2))
    ax.annotate('', xy=(11.5, remote_work[-1]), xytext=(10.5, remote_work[-3]),
               arrowprops=dict(arrowstyle='->', color=colors['mlred'], lw=2))

    plt.tight_layout()
    plt.savefig('../charts/trend_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/trend_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_workshop_results():
    """Create workshop results visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pie chart for topics discovered
    sizes = [18, 15, 12, 11, 9, 35]
    labels = ['Healthcare AI', 'Sustainable Tech', 'EdTech', 'FinTech', 'Food Delivery', 'Others']
    colors_pie = [colors['mlblue'], colors['mlgreen'], colors['mlorange'],
                 colors['mlpurple'], colors['mlcyan'], colors['mlgray']]

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90,
                                       explode=[0.05, 0.05, 0, 0, 0, 0])

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax.set_title('Workshop Results: Topic Distribution in Startup Ideas', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/workshop_results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/workshop_results.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Generate additional charts for Week 5."""
    print("Generating additional Week 5 charts...")

    create_lda_document_topics()
    print("[OK] LDA document topics")

    create_nmf_decomposition()
    print("[OK] NMF decomposition")

    create_topics_to_opportunities()
    print("[OK] Topics to opportunities")

    create_innovation_opportunity_map()
    print("[OK] Innovation opportunity map")

    create_persona_topic_heatmap()
    print("[OK] Persona topic heatmap")

    create_trend_evolution()
    print("[OK] Trend evolution")

    create_workshop_results()
    print("[OK] Workshop results")

    print("\nAdditional charts generated successfully!")

if __name__ == "__main__":
    main()