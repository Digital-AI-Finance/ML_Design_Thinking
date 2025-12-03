"""
Generate additional NLP and design-focused charts for Week 3
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Wedge
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

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
    'mlyellow': '#ffbb78',
    'mlcyan': '#17becf'
}

def create_language_emotion_flow():
    """Create language to emotion flow diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create flow from text to insights
    levels = [
        ('User Text', ['Reviews', 'Support', 'Social']),
        ('Processing', ['Clean', 'Tokenize', 'Embed']),
        ('Analysis', ['Sentiment', 'Emotion', 'Topics']),
        ('Insights', ['Pain Points', 'Delights', 'Needs'])
    ]
    
    x_positions = [0.15, 0.4, 0.6, 0.85]
    
    for i, (title, items) in enumerate(levels):
        x = x_positions[i]
        
        # Title
        ax.text(x, 0.9, title, ha='center', fontsize=14, fontweight='bold')
        
        # Items
        for j, item in enumerate(items):
            y = 0.7 - j * 0.25
            
            if i == 0:
                color = colors['mlblue']
            elif i == len(levels) - 1:
                color = colors['mlgreen']
            else:
                color = colors['mlorange']
            
            rect = FancyBboxPatch((x - 0.06, y - 0.08), 0.12, 0.12,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, alpha=0.3,
                                  edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, item, ha='center', va='center', fontsize=10)
            
            # Draw connections
            if i < len(levels) - 1:
                for k in range(len(items)):
                    ax.plot([x + 0.06, x_positions[i+1] - 0.06],
                           [y, 0.7 - k * 0.25],
                           color=colors['mlgray'], alpha=0.3, linewidth=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('From Language to Design Insights', fontsize=16, fontweight='bold')
    
    return fig

def create_context_sentiment_examples():
    """Create context-dependent sentiment examples"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    examples = [
        ("This is sick!", "Gaming Community", "Positive", colors['mlgreen']),
        ("This is sick!", "Healthcare Forum", "Negative", colors['mlred']),
        ("It's fine...", "After Complaint", "Negative", colors['mlred']),
        ("It's fine!", "First Experience", "Positive", colors['mlgreen']),
        ("Whatever", "Teen Response", "Neutral", colors['mlgray']),
        ("Whatever", "Support Chat", "Negative", colors['mlred'])
    ]
    
    for i, (text, context, sentiment, color) in enumerate(examples):
        row = i // 3
        col = i % 3
        x = 0.15 + col * 0.3
        y = 0.7 - row * 0.4
        
        # Draw box
        rect = FancyBboxPatch((x - 0.12, y - 0.15), 0.24, 0.25,
                              boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.2,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y + 0.05, f'"{text}"', ha='center', fontsize=12, fontweight='bold')
        ax.text(x, y - 0.02, context, ha='center', fontsize=9, style='italic')
        ax.text(x, y - 0.08, sentiment, ha='center', fontsize=10, color=color, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Context Changes Everything', fontsize=16, fontweight='bold')
    
    return fig

def create_emotion_wheel():
    """Create emotion wheel visualization"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Primary emotions
    emotions = {
        'Joy': (colors['mlyellow'], 0),
        'Trust': (colors['mlgreen'], 45),
        'Fear': (colors['mlpurple'], 90),
        'Surprise': (colors['mlcyan'], 135),
        'Sadness': (colors['mlblue'], 180),
        'Disgust': (colors['mlbrown'], 225),
        'Anger': (colors['mlred'], 270),
        'Anticipation': (colors['mlorange'], 315)
    }
    
    # Draw emotion segments
    for emotion, (color, angle) in emotions.items():
        wedge = Wedge((0, 0), 1, angle, angle + 45,
                     facecolor=color, alpha=0.6, edgecolor='white', linewidth=2)
        ax.add_patch(wedge)
        
        # Add labels
        angle_rad = np.radians(angle + 22.5)
        x = 0.7 * np.cos(angle_rad)
        y = 0.7 * np.sin(angle_rad)
        ax.text(x, y, emotion, ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')
    
    # Add inner circle
    inner_circle = Circle((0, 0), 0.3, facecolor='white', edgecolor='gray', linewidth=2)
    ax.add_patch(inner_circle)
    ax.text(0, 0, 'EMOTIONS', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    ax.set_title('Plutchik\'s Wheel of Emotions', fontsize=16, fontweight='bold')
    
    return fig

def create_nlp_challenge_pyramid():
    """Create NLP challenge pyramid"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Pyramid levels
    levels = [
        ('Volume', '1M+ texts/day', colors['mlblue'], 1.0),
        ('Languages', '35+ languages', colors['mlorange'], 0.75),
        ('Context', 'Domain specific', colors['mlpurple'], 0.5),
        ('Nuance', 'Sarcasm, emotion', colors['mlred'], 0.25)
    ]
    
    y_base = 0.2
    for i, (label, desc, color, width) in enumerate(levels):
        y = y_base + i * 0.15
        
        # Draw trapezoid
        x_left = 0.5 - width/2
        x_right = 0.5 + width/2
        
        poly = plt.Polygon([(x_left, y), (x_right, y),
                           (x_right - 0.05, y + 0.15), (x_left + 0.05, y + 0.15)],
                          facecolor=color, alpha=0.6, edgecolor=color, linewidth=2)
        ax.add_patch(poly)
        
        # Add text
        ax.text(0.5, y + 0.075, label, ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
        ax.text(0.5, y + 0.03, desc, ha='center', va='center',
               fontsize=9, color='white', style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('The NLP Challenge: Scale Meets Nuance', fontsize=16, fontweight='bold')
    
    return fig

def create_nlp_impact_metrics():
    """Create NLP business impact metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Customer Satisfaction
    ax1 = axes[0, 0]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    before = [65, 66, 64, 67, 65, 66]
    after = [68, 72, 75, 78, 82, 85]
    
    x = np.arange(len(months))
    width = 0.35
    
    ax1.bar(x - width/2, before, width, label='Before NLP', color=colors['mlgray'], alpha=0.7)
    ax1.bar(x + width/2, after, width, label='After NLP', color=colors['mlgreen'], alpha=0.7)
    ax1.set_ylabel('CSAT Score', fontsize=10)
    ax1.set_title('Customer Satisfaction', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Response Time
    ax2 = axes[0, 1]
    categories = ['Critical', 'High', 'Medium', 'Low']
    response_before = [24, 18, 12, 8]
    response_after = [8, 6, 4, 3]
    
    x = np.arange(len(categories))
    ax2.barh(x - width/2, response_before, width, label='Before', color=colors['mlred'], alpha=0.7)
    ax2.barh(x + width/2, response_after, width, label='After', color=colors['mlgreen'], alpha=0.7)
    ax2.set_xlabel('Hours', fontsize=10)
    ax2.set_title('Support Response Time', fontsize=12, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(categories)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # Issue Detection
    ax3 = axes[1, 0]
    detection_data = [23, 40, 15, 22]
    labels = ['Manual\nReview', 'NLP\nDetected', 'Customer\nReported', 'Missed']
    colors_pie = [colors['mlgray'], colors['mlgreen'], colors['mlorange'], colors['mlred']]
    
    ax3.pie(detection_data, labels=labels, colors=colors_pie, autopct='%1.1f%%',
           startangle=90)
    ax3.set_title('Issue Detection Sources', fontsize=12, fontweight='bold')
    
    # ROI Metrics
    ax4 = axes[1, 1]
    metrics = ['Cost\nSavings', 'Time\nSaved', 'Issues\nPrevented', 'Customer\nRetention']
    values = [35, 60, 45, 25]
    colors_bar = [colors['mlgreen'], colors['mlblue'], colors['mlorange'], colors['mlpurple']]
    
    bars = ax4.bar(metrics, values, color=colors_bar, alpha=0.7)
    ax4.set_ylabel('Improvement (%)', fontsize=10)
    ax4.set_title('NLP ROI Metrics', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'+{val}%', ha='center', fontsize=9)
    
    plt.suptitle('NLP Impact on Business Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_bert_architecture():
    """Create simplified BERT architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Input layer
    input_words = ['[CLS]', 'The', 'product', 'is', 'great', '[SEP]']
    for i, word in enumerate(input_words):
        x = 0.15 + i * 0.13
        rect = Rectangle((x - 0.05, 0.1), 0.1, 0.08,
                        facecolor=colors['mlblue'], alpha=0.3,
                        edgecolor=colors['mlblue'], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.14, word, ha='center', va='center', fontsize=9)
    
    # Embeddings
    ax.text(0.5, 0.25, 'Token + Position + Segment Embeddings', 
           ha='center', fontsize=11, fontweight='bold')
    
    # Transformer layers
    n_layers = 4
    for layer in range(n_layers):
        y = 0.35 + layer * 0.13
        
        # Self-attention
        rect1 = Rectangle((0.2, y), 0.25, 0.08,
                         facecolor=colors['mlorange'], alpha=0.3,
                         edgecolor=colors['mlorange'], linewidth=2)
        ax.add_patch(rect1)
        ax.text(0.325, y + 0.04, 'Multi-Head\nAttention', ha='center', va='center', fontsize=8)
        
        # Feed forward
        rect2 = Rectangle((0.55, y), 0.25, 0.08,
                         facecolor=colors['mlgreen'], alpha=0.3,
                         edgecolor=colors['mlgreen'], linewidth=2)
        ax.add_patch(rect2)
        ax.text(0.675, y + 0.04, 'Feed\nForward', ha='center', va='center', fontsize=8)
        
        # Connections
        ax.arrow(0.45, y + 0.04, 0.08, 0, head_width=0.01, head_length=0.01,
                fc=colors['mlgray'], ec=colors['mlgray'])
        
        # Layer label
        ax.text(0.1, y + 0.04, f'Layer {layer + 1}', ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    # Output
    ax.text(0.5, 0.88, 'Contextualized Representations', 
           ha='center', fontsize=11, fontweight='bold')
    
    # Task-specific heads
    ax.text(0.25, 0.95, 'Classification', ha='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlred'], alpha=0.3))
    ax.text(0.5, 0.95, 'Token Classification', ha='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlpurple'], alpha=0.3))
    ax.text(0.75, 0.95, 'Question Answering', ha='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlcyan'], alpha=0.3))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('BERT Architecture Overview', fontsize=16, fontweight='bold')
    
    return fig

def create_amazon_case_overview():
    """Create Amazon case study overview"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Review volume
    ax1 = axes[0, 0]
    hours = np.arange(24)
    volume = 80000 + 20000 * np.sin((hours - 6) * np.pi / 12) + np.random.randn(24) * 5000
    volume[volume < 0] = 1000
    
    ax1.plot(hours, volume/1000, color=colors['mlblue'], linewidth=2)
    ax1.fill_between(hours, 0, volume/1000, alpha=0.3, color=colors['mlblue'])
    ax1.set_xlabel('Hour of Day', fontsize=10)
    ax1.set_ylabel('Reviews (thousands)', fontsize=10)
    ax1.set_title('Daily Review Volume', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Language distribution
    ax2 = axes[0, 1]
    languages = ['English', 'Spanish', 'German', 'French', 'Japanese', 'Other']
    percentages = [45, 15, 10, 8, 7, 15]
    colors_lang = [colors['mlblue'], colors['mlorange'], colors['mlgreen'],
                  colors['mlred'], colors['mlpurple'], colors['mlgray']]
    
    ax2.pie(percentages, labels=languages, colors=colors_lang, autopct='%1.1f%%',
           startangle=90)
    ax2.set_title('Review Languages', fontsize=12, fontweight='bold')
    
    # Category distribution
    ax3 = axes[1, 0]
    categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Toys']
    reviews = [350, 280, 220, 180, 150]
    
    bars = ax3.barh(categories, reviews, color=colors['mlorange'], alpha=0.7)
    ax3.set_xlabel('Daily Reviews (thousands)', fontsize=10)
    ax3.set_title('Reviews by Category', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Processing pipeline
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    pipeline = ['Collect\n2M reviews', 'Filter\nSpam', 'Analyze\nSentiment',
               'Extract\nInsights', 'Update\nProducts']
    
    for i, step in enumerate(pipeline):
        y = 0.5
        x = 0.1 + i * 0.2
        
        if i == 0:
            color = colors['mlblue']
        elif i == len(pipeline) - 1:
            color = colors['mlgreen']
        else:
            color = colors['mlorange']
        
        circle = Circle((x, y), 0.08, facecolor=color, alpha=0.3,
                       edgecolor=color, linewidth=2)
        ax4.add_patch(circle)
        ax4.text(x, y, step, ha='center', va='center', fontsize=8)
        
        if i < len(pipeline) - 1:
            ax4.arrow(x + 0.08, y, 0.04, 0, head_width=0.02, head_length=0.01,
                     fc=colors['mlgray'], ec=colors['mlgray'])
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Processing Pipeline', fontsize=12, fontweight='bold')
    
    plt.suptitle('Amazon Review Intelligence System', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Additional charts for remaining slides
def create_voice_of_customer():
    """Create Voice of Customer framework visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create Sankey diagram for VoC flow
    sankey = Sankey(ax=ax, scale=0.01, offset=0.3, format='%.0f',
                   gap=0.5, radius=0.2, shoulder=0.03, margin=0.5)
    
    sankey.add(flows=[100, -30, -40, -20, -10],
              orientations=[0, -1, 1, -1, 0],
              labels=['Customer\nFeedback', 'Feature\nRequests', 'Bug\nReports',
                     'Praise', 'Insights'],
              pathlengths=[0.25, 0.5, 0.5, 0.5, 0.25],
              trunklength=1.5,
              facecolor=colors['mlblue'],
              edgecolor=colors['mlblue'])
    
    diagrams = sankey.finish()
    
    ax.set_title('Voice of Customer Flow', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig

def create_data_sources_pyramid():
    """Create data sources pyramid"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Data source levels
    sources = [
        ('Social Media', '500K+ posts/day', colors['mlcyan'], 1.0),
        ('Product Reviews', '200K+ reviews/day', colors['mlblue'], 0.8),
        ('Support Tickets', '50K+ tickets/day', colors['mlorange'], 0.6),
        ('Surveys', '10K+ responses/day', colors['mlgreen'], 0.4),
        ('Focus Groups', '100+ sessions/month', colors['mlpurple'], 0.2)
    ]
    
    y_base = 0.15
    for i, (source, volume, color, width) in enumerate(sources):
        y = y_base + i * 0.15
        
        # Draw trapezoid
        x_left = 0.5 - width/2
        x_right = 0.5 + width/2
        
        if i < len(sources) - 1:
            next_width = sources[i+1][3]
            x_next_left = 0.5 - next_width/2
            x_next_right = 0.5 + next_width/2
        else:
            x_next_left = 0.48
            x_next_right = 0.52
        
        poly = plt.Polygon([(x_left, y), (x_right, y),
                           (x_next_right, y + 0.15), (x_next_left, y + 0.15)],
                          facecolor=color, alpha=0.6, edgecolor=color, linewidth=2)
        ax.add_patch(poly)
        
        # Add text
        ax.text(0.5, y + 0.075, source, ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')
        ax.text(0.5, y + 0.03, volume, ha='center', va='center',
               fontsize=8, color='white', style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Text Data Sources Hierarchy', fontsize=16, fontweight='bold')
    
    return fig

# Generate all additional charts
if __name__ == "__main__":
    import os
    os.makedirs('../charts', exist_ok=True)
    
    print("Generating additional NLP charts...")
    
    charts = [
        ('language_emotion_flow', create_language_emotion_flow),
        ('context_sentiment_examples', create_context_sentiment_examples),
        ('emotion_wheel', create_emotion_wheel),
        ('nlp_challenge_pyramid', create_nlp_challenge_pyramid),
        ('nlp_impact_metrics', create_nlp_impact_metrics),
        ('bert_architecture', create_bert_architecture),
        ('amazon_case_overview', create_amazon_case_overview),
        ('voice_of_customer', create_voice_of_customer),
        ('data_sources_pyramid', create_data_sources_pyramid)
    ]
    
    for name, func in charts:
        print(f"Creating {name}...")
        fig = func()
        fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("All additional charts created successfully!")
    plt.close('all')