"""
Generate placeholder charts for remaining missing visualizations
These are simpler charts for layout templates and miscellaneous references
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

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

def create_generic_chart(title, chart_type='bar'):
    """Create a generic chart with given title"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if chart_type == 'bar':
        categories = ['A', 'B', 'C', 'D', 'E']
        values = np.random.rand(5) * 100
        ax.bar(categories, values, color=[colors['mlblue'], colors['mlorange'], 
                                         colors['mlgreen'], colors['mlpurple'], colors['mlred']], alpha=0.7)
        ax.set_ylabel('Value', fontsize=12)
    elif chart_type == 'line':
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.exp(-x/10) + np.random.randn(100) * 0.1
        ax.plot(x, y, color=colors['mlblue'], linewidth=2)
        ax.fill_between(x, y, alpha=0.3, color=colors['mlblue'])
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
    elif chart_type == 'scatter':
        x = np.random.randn(100)
        y = x * 2 + np.random.randn(100)
        ax.scatter(x, y, c=np.random.randint(0, 3, 100), cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
    elif chart_type == 'pie':
        sizes = [30, 25, 20, 15, 10]
        labels = ['Category A', 'Category B', 'Category C', 'Category D', 'Other']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
              colors=[colors['mlblue'], colors['mlorange'], colors['mlgreen'], 
                     colors['mlpurple'], colors['mlgray']])
    elif chart_type == 'heatmap':
        data = np.random.rand(5, 5)
        sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3)
    
    return fig

def create_data_to_emotion_bridge():
    """Create data to emotion bridge visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Data side
    rect1 = FancyBboxPatch((0.1, 0.3), 0.25, 0.4,
                           boxstyle="round,pad=0.02",
                           facecolor=colors['mlblue'], alpha=0.3)
    ax.add_patch(rect1)
    ax.text(0.225, 0.5, 'Raw Data\n• Numbers\n• Metrics\n• Statistics',
           ha='center', va='center', fontsize=10)
    
    # Bridge
    for i in range(5):
        y = 0.35 + i * 0.06
        ax.arrow(0.35, y, 0.3, 0, head_width=0.02, head_length=0.02,
                fc=colors['mlorange'], ec=colors['mlorange'], alpha=0.3 + i*0.1)
    
    ax.text(0.5, 0.5, 'NLP Bridge', ha='center', fontsize=12, fontweight='bold')
    
    # Emotion side
    rect2 = FancyBboxPatch((0.65, 0.3), 0.25, 0.4,
                           boxstyle="round,pad=0.02",
                           facecolor=colors['mlgreen'], alpha=0.3)
    ax.add_patch(rect2)
    ax.text(0.775, 0.5, 'Human Emotions\n• Joy\n• Frustration\n• Satisfaction',
           ha='center', va='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Bridging Data and Emotions', fontsize=16, fontweight='bold')
    
    return fig

def create_review_mining_process():
    """Create review mining process visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    stages = ['Collection', 'Filtering', 'Analysis', 'Insights', 'Actions']
    values = [1000, 800, 600, 300, 100]
    
    # Funnel chart
    for i, (stage, val) in enumerate(zip(stages, values)):
        width = val / 1000
        left = 0.5 - width/2
        rect = Rectangle((left, i * 0.15 + 0.2), width, 0.12,
                        facecolor=colors['mlblue'], alpha=0.3 + i*0.15,
                        edgecolor=colors['mlblue'], linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, i * 0.15 + 0.26, f'{stage}: {val}', ha='center', va='center',
               fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Review Mining Process Funnel', fontsize=16, fontweight='bold')
    
    return fig

def create_support_ticket_analysis():
    """Create support ticket analysis dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Ticket categories
    ax1 = axes[0, 0]
    categories = ['Technical', 'Billing', 'Feature', 'Bug', 'Other']
    counts = [350, 200, 150, 300, 100]
    ax1.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90,
           colors=[colors['mlred'], colors['mlorange'], colors['mlgreen'],
                  colors['mlpurple'], colors['mlgray']])
    ax1.set_title('Ticket Categories', fontsize=12)
    
    # Sentiment trend
    ax2 = axes[0, 1]
    days = np.arange(30)
    sentiment = 0.6 + 0.2 * np.sin(days * 0.2) + np.random.randn(30) * 0.05
    ax2.plot(days, sentiment, color=colors['mlblue'], linewidth=2)
    ax2.fill_between(days, sentiment, alpha=0.3, color=colors['mlblue'])
    ax2.set_xlabel('Days', fontsize=10)
    ax2.set_ylabel('Avg Sentiment', fontsize=10)
    ax2.set_title('Sentiment Trend', fontsize=12)
    ax2.grid(alpha=0.3)
    
    # Priority matrix
    ax3 = axes[1, 0]
    urgency = np.random.rand(50) * 10
    impact = np.random.rand(50) * 10
    ax3.scatter(urgency, impact, c=urgency*impact, cmap='RdYlGn_r', s=100, alpha=0.6)
    ax3.set_xlabel('Urgency', fontsize=10)
    ax3.set_ylabel('Impact', fontsize=10)
    ax3.set_title('Priority Matrix', fontsize=12)
    ax3.grid(alpha=0.3)
    
    # Resolution time
    ax4 = axes[1, 1]
    priorities = ['Critical', 'High', 'Medium', 'Low']
    times = [2, 8, 24, 48]
    ax4.barh(priorities, times, color=[colors['mlred'], colors['mlorange'],
                                       colors['mlyellow'], colors['mlgreen']], alpha=0.7)
    ax4.set_xlabel('Hours', fontsize=10)
    ax4.set_title('Avg Resolution Time', fontsize=12)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Support Ticket Analysis Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_social_media_sentiment():
    """Create social media sentiment visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Time series with events
    hours = np.arange(24)
    baseline = 0.6
    sentiment = baseline + np.random.randn(24) * 0.1
    
    # Add event impacts
    sentiment[8:10] -= 0.3  # Negative event
    sentiment[14:16] += 0.2  # Positive event
    
    ax.plot(hours, sentiment, color=colors['mlblue'], linewidth=2)
    ax.fill_between(hours, sentiment, baseline, where=(sentiment > baseline),
                    alpha=0.3, color=colors['mlgreen'], label='Positive')
    ax.fill_between(hours, sentiment, baseline, where=(sentiment <= baseline),
                    alpha=0.3, color=colors['mlred'], label='Negative')
    
    # Mark events
    ax.annotate('Product Issue', xy=(9, sentiment[9]), xytext=(9, 0.2),
               arrowprops=dict(arrowstyle='->', color=colors['mlred']))
    ax.annotate('Positive PR', xy=(15, sentiment[15]), xytext=(15, 0.9),
               arrowprops=dict(arrowstyle='->', color=colors['mlgreen']))
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title('Social Media Sentiment Timeline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig

def create_pain_point_heatmap():
    """Create customer pain point heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Journey stages and pain points
    stages = ['Awareness', 'Consideration', 'Purchase', 'Onboarding', 'Usage', 'Support']
    pain_points = ['Confusion', 'Frustration', 'Anxiety', 'Difficulty', 'Dissatisfaction']
    
    # Generate intensity data
    data = np.random.rand(len(pain_points), len(stages))
    # Make some areas more problematic
    data[1, 2] = 0.9  # Frustration at Purchase
    data[3, 3] = 0.85  # Difficulty at Onboarding
    data[4, 5] = 0.8  # Dissatisfaction at Support
    
    sns.heatmap(data, xticklabels=stages, yticklabels=pain_points,
                annot=True, fmt='.2f', cmap='RdYlGn_r', vmin=0, vmax=1, ax=ax)
    
    ax.set_title('Customer Pain Point Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Customer Journey Stage', fontsize=12)
    ax.set_ylabel('Pain Point Type', fontsize=12)
    
    return fig

def create_delight_moments():
    """Create customer delight moments visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Journey with delight peaks
    x = np.linspace(0, 10, 100)
    baseline = 5
    satisfaction = baseline + np.sin(x) * 2
    
    # Add delight peaks
    peaks = [2, 5, 8]
    for peak in peaks:
        idx = int(peak * 10)
        satisfaction[idx-5:idx+5] += 3 * np.exp(-0.5 * ((np.arange(10) - 5) / 2) ** 2)
    
    ax.plot(x, satisfaction, color=colors['mlblue'], linewidth=2)
    ax.fill_between(x, satisfaction, baseline, where=(satisfaction > baseline + 1),
                    alpha=0.3, color=colors['mlgreen'])
    
    # Mark delight moments
    for peak in peaks:
        ax.plot(peak, satisfaction[int(peak * 10)], 'o', markersize=12,
               color=colors['mlorange'])
        ax.annotate('Delight!', xy=(peak, satisfaction[int(peak * 10)]),
                   xytext=(peak, satisfaction[int(peak * 10)] + 1),
                   fontsize=10, ha='center', color=colors['mlorange'])
    
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Customer Journey', fontsize=12)
    ax.set_ylabel('Satisfaction Level', fontsize=12)
    ax.set_title('Customer Delight Moments', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig

def create_emotional_personalization():
    """Create emotional personalization visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # User segments
    ax1 = axes[0]
    segments = ['Enthusiasts', 'Pragmatists', 'Skeptics', 'Beginners', 'Experts']
    emotions = ['Joy', 'Trust', 'Anticipation', 'Fear', 'Interest']
    
    # Create radar chart data
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    for i, segment in enumerate(segments[:3]):
        values = np.random.rand(len(emotions)) * 0.5 + 0.3
        values = np.concatenate((values, [values[0]]))
        ax1.plot(angles, values, 'o-', linewidth=2, label=segment, alpha=0.7)
        ax1.fill(angles, values, alpha=0.25)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(emotions)
    ax1.set_ylim(0, 1)
    ax1.set_title('Emotional Profiles by Segment', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True)
    
    # Personalization strategies
    ax2 = axes[1]
    strategies = ['Tone', 'Content', 'Timing', 'Channel', 'Frequency']
    effectiveness = [85, 92, 78, 70, 65]
    
    bars = ax2.bar(strategies, effectiveness, color=[colors['mlblue'], colors['mlorange'],
                                                     colors['mlgreen'], colors['mlpurple'],
                                                     colors['mlred']], alpha=0.7)
    ax2.set_ylabel('Effectiveness (%)', fontsize=10)
    ax2.set_title('Personalization Strategy Impact', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values
    for bar, val in zip(bars, effectiveness):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}%', ha='center', fontsize=9)
    
    plt.suptitle('Emotional Personalization Framework', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_impact_measurement():
    """Create impact measurement dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Before/After metrics
    ax1 = axes[0, 0]
    metrics = ['NPS', 'CSAT', 'CES', 'Retention']
    before = [30, 65, 70, 75]
    after = [55, 82, 85, 88]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, before, width, label='Before', color=colors['mlgray'], alpha=0.7)
    ax1.bar(x + width/2, after, width, label='After', color=colors['mlgreen'], alpha=0.7)
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_title('Key Metrics Impact', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Sentiment journey
    ax2 = axes[0, 1]
    stages = ['Start', 'Mid', 'End']
    old_journey = [0.5, 0.4, 0.45]
    new_journey = [0.6, 0.7, 0.85]
    
    ax2.plot(stages, old_journey, 'o-', label='Old Journey', color=colors['mlred'], linewidth=2)
    ax2.plot(stages, new_journey, 's-', label='New Journey', color=colors['mlgreen'], linewidth=2)
    ax2.set_ylabel('Sentiment', fontsize=10)
    ax2.set_title('Journey Sentiment', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Channel performance
    ax3 = axes[1, 0]
    channels = ['Email', 'Chat', 'Phone', 'Social']
    performance = [75, 88, 65, 92]
    ax3.barh(channels, performance, color=[colors['mlblue'], colors['mlorange'],
                                          colors['mlpurple'], colors['mlcyan']], alpha=0.7)
    ax3.set_xlabel('Satisfaction (%)', fontsize=10)
    ax3.set_title('Channel Performance', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # ROI calculation
    ax4 = axes[1, 1]
    ax4.axis('off')
    roi_text = """
    Investment: $100,000
    Revenue Gain: $450,000
    Cost Savings: $200,000
    
    Total ROI: 550%
    Payback: 3 months
    """
    ax4.text(0.5, 0.5, roi_text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlgreen'], alpha=0.2))
    ax4.set_title('ROI Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('NLP Impact Measurement Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Create simple placeholders for remaining charts
def create_simple_placeholder(title):
    """Create a simple placeholder chart"""
    return create_generic_chart(title, chart_type=np.random.choice(['bar', 'line', 'scatter', 'pie', 'heatmap']))

def create_all_placeholder_charts():
    """Generate all placeholder charts"""
    
    # Specific charts
    specific_charts = [
        ('data_to_emotion_bridge', create_data_to_emotion_bridge),
        ('review_mining_process', create_review_mining_process),
        ('support_ticket_analysis', create_support_ticket_analysis),
        ('social_media_sentiment', create_social_media_sentiment),
        ('pain_point_heatmap', create_pain_point_heatmap),
        ('delight_moments', create_delight_moments),
        ('emotional_personalization', create_emotional_personalization),
        ('impact_measurement', create_impact_measurement),
    ]
    
    # Simple placeholders
    placeholder_names = [
        'competency_radar', 'hybrid_results', 'ideation_comparison',
        'learning_progress', 'method_comparison', 'metrics_trend',
        'ml_enhanced_results', 'ml_impact', 'module_completion',
        'score_distribution', 'skill_correlation', 'success_factors',
        'time_allocation', 'traditional_results', 'user_clusters'
    ]
    
    print("Generating placeholder charts for Week 3...")
    
    import os
    os.makedirs('../charts', exist_ok=True)
    
    # Create specific charts
    for name, func in specific_charts:
        print(f"Creating {name}...")
        fig = func()
        fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Create simple placeholders
    for name in placeholder_names:
        print(f"Creating placeholder for {name}...")
        fig = create_simple_placeholder(name.replace('_', ' ').title())
        fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("All placeholder charts created successfully!")

if __name__ == "__main__":
    create_all_placeholder_charts()