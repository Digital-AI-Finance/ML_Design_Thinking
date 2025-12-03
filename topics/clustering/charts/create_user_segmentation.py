"""
Week 2: User Segmentation Visualization
Shows how clusters translate to actionable user personas with characteristics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f',
    'mlyellow': '#bcbd22',
    'mlcyan': '#17becf'
}

# Define persona colors
persona_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

def generate_user_behavior_data(n_samples=3000):
    """Generate realistic user behavior data with meaningful features"""
    
    # Define 5 user segments with realistic behavior patterns
    segments = {
        'Casual Browsers': {
            'center': [20, 30],  # Low engagement, low frequency
            'size': 800,
            'std': [8, 10]
        },
        'Power Users': {
            'center': [85, 90],  # High engagement, high frequency
            'size': 400,
            'std': [10, 8]
        },
        'Social Sharers': {
            'center': [60, 75],  # Medium-high engagement, high sharing
            'size': 600,
            'std': [12, 10]
        },
        'Content Creators': {
            'center': [70, 50],  # High creation, medium frequency
            'size': 500,
            'std': [10, 15]
        },
        'Window Shoppers': {
            'center': [40, 20],  # Medium browse, low purchase
            'size': 700,
            'std': [15, 8]
        }
    }
    
    X = []
    true_labels = []
    personas = []
    
    for i, (persona, params) in enumerate(segments.items()):
        # Generate data for this segment
        n = params['size']
        center = params['center']
        std = params['std']
        
        # Generate features
        engagement = np.random.normal(center[0], std[0], n)
        frequency = np.random.normal(center[1], std[1], n)
        
        # Clip to realistic ranges [0, 100]
        engagement = np.clip(engagement, 0, 100)
        frequency = np.clip(frequency, 0, 100)
        
        data = np.column_stack([engagement, frequency])
        X.extend(data)
        true_labels.extend([i] * n)
        personas.extend([persona] * n)
    
    return np.array(X), np.array(true_labels), list(segments.keys())

def create_segmentation_main():
    """Create main segmentation visualization with personas"""
    X, true_labels, persona_names = generate_user_behavior_data()
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Raw clustering
    for i in range(5):
        mask = cluster_labels == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=persona_colors[i], 
                   alpha=0.6, s=20, edgecolors='black', linewidth=0.5,
                   label=f'Cluster {i+1}')
    
    # Plot centers
    ax1.scatter(centers[:, 0], centers[:, 1], c='black', s=300, 
               marker='*', edgecolors='white', linewidth=2, zorder=10)
    
    ax1.set_xlabel('Engagement Score (0-100)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Visit Frequency (0-100)', fontsize=12, fontweight='bold')
    ax1.set_title('User Behavior Clustering', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    
    # Right plot: Named personas with characteristics
    ax2.scatter(X[:, 0], X[:, 1], c=[persona_colors[i] for i in cluster_labels],
               alpha=0.3, s=10, edgecolors='none')
    
    # Map clusters to personas based on center positions
    cluster_to_persona = {}
    for i, center in enumerate(centers):
        # Find closest persona based on engagement/frequency patterns
        if center[0] < 30 and center[1] < 40:  # Low both
            cluster_to_persona[i] = 'Casual Browsers'
        elif center[0] > 75 and center[1] > 75:  # High both
            cluster_to_persona[i] = 'Power Users'
        elif center[0] > 55 and center[1] > 60:  # Medium-high both
            cluster_to_persona[i] = 'Social Sharers'
        elif center[0] > 60 and center[1] < 60:  # High engagement, medium frequency
            cluster_to_persona[i] = 'Content Creators'
        else:  # Medium engagement, low frequency
            cluster_to_persona[i] = 'Window Shoppers'
    
    # Add persona labels with characteristics
    persona_chars = {
        'Casual Browsers': ['• Visit 1-2x/month', '• Quick sessions', '• Price sensitive'],
        'Power Users': ['• Daily active', '• Premium features', '• High retention'],
        'Social Sharers': ['• Share content', '• Community active', '• Influencers'],
        'Content Creators': ['• Create content', '• Seek tools', '• Quality focused'],
        'Window Shoppers': ['• Browse often', '• Rarely purchase', '• Research mode']
    }
    
    # Plot centers with persona labels
    for i, center in enumerate(centers):
        persona = cluster_to_persona[i]
        
        # Plot center
        ax2.scatter(center[0], center[1], c=persona_colors[i], s=500,
                   marker='o', edgecolors='black', linewidth=2, zorder=10,
                   alpha=0.9)
        
        # Add persona name
        ax2.text(center[0], center[1], persona.split()[0], 
                fontsize=9, fontweight='bold', ha='center', va='center',
                color='white')
        
        # Add characteristics box
        chars = persona_chars[persona]
        char_text = '\n'.join(chars)
        
        # Position box based on cluster location
        if center[0] < 50:
            box_x = center[0] + 15
        else:
            box_x = center[0] - 25
        
        if center[1] < 50:
            box_y = center[1] + 10
        else:
            box_y = center[1] - 15
        
        bbox = dict(boxstyle='round,pad=0.5', facecolor=persona_colors[i],
                   alpha=0.9, edgecolor='black', linewidth=1.5)
        ax2.text(box_x, box_y, char_text, fontsize=8,
                bbox=bbox, color='white', fontweight='bold')
    
    ax2.set_xlabel('Engagement Score (0-100)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Visit Frequency (0-100)', fontsize=12, fontweight='bold')
    ax2.set_title('User Personas from Clustering', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    
    plt.suptitle('From Data Points to Actionable User Personas',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def create_persona_profiles():
    """Create detailed persona profile cards"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    personas = [
        {
            'name': 'Casual Browsers',
            'size': '35%',
            'engagement': 20,
            'frequency': 30,
            'value': 25,
            'growth': 15,
            'retention': 40,
            'traits': ['Price conscious', 'Limited time', 'Basic needs', 'Mobile first'],
            'opportunities': ['Onboarding', 'Quick wins', 'Mobile optimize']
        },
        {
            'name': 'Power Users',
            'size': '15%',
            'engagement': 85,
            'frequency': 90,
            'value': 95,
            'growth': 60,
            'retention': 95,
            'traits': ['Feature hungry', 'Early adopters', 'Advocates', 'Premium'],
            'opportunities': ['Beta testing', 'Advanced features', 'Community']
        },
        {
            'name': 'Social Sharers',
            'size': '20%',
            'engagement': 60,
            'frequency': 75,
            'value': 70,
            'growth': 80,
            'retention': 75,
            'traits': ['Influencers', 'Network effect', 'Content spreaders', 'Trendy'],
            'opportunities': ['Referral programs', 'Share features', 'Social proof']
        },
        {
            'name': 'Content Creators',
            'size': '15%',
            'engagement': 70,
            'frequency': 50,
            'value': 80,
            'growth': 70,
            'retention': 85,
            'traits': ['Quality focused', 'Tool seekers', 'Professional', 'Creative'],
            'opportunities': ['Creator tools', 'Analytics', 'Monetization']
        },
        {
            'name': 'Window Shoppers',
            'size': '25%',
            'engagement': 40,
            'frequency': 20,
            'value': 30,
            'growth': 50,
            'retention': 35,
            'traits': ['Researchers', 'Comparison', 'Price sensitive', 'Cautious'],
            'opportunities': ['Trust signals', 'Reviews', 'Free trials']
        }
    ]
    
    for idx, (ax, persona) in enumerate(zip(axes[:5], personas)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Background color
        rect = FancyBboxPatch((0.5, 0.5), 9, 9,
                              boxstyle="round,pad=0.1",
                              facecolor=persona_colors[idx],
                              alpha=0.2,
                              edgecolor=persona_colors[idx],
                              linewidth=3)
        ax.add_patch(rect)
        
        # Title
        ax.text(5, 8.5, persona['name'], fontsize=14, fontweight='bold',
               ha='center', color=persona_colors[idx])
        ax.text(5, 7.8, f"Segment Size: {persona['size']} of users",
               fontsize=10, ha='center', style='italic')
        
        # Metrics radar
        metrics = ['Engage', 'Freq', 'Value', 'Growth', 'Retain']
        values = [persona['engagement'], persona['frequency'], 
                 persona['value'], persona['growth'], persona['retention']]
        
        # Mini bar chart
        bar_y = 6.5
        bar_width = 1.5
        for i, (metric, value) in enumerate(zip(metrics, values)):
            x = 1.5 + i * 1.5
            # Background bar
            ax.add_patch(plt.Rectangle((x, bar_y - 2), bar_width * 0.8, 2,
                                      facecolor='lightgray', alpha=0.3))
            # Value bar
            ax.add_patch(plt.Rectangle((x, bar_y - 2), bar_width * 0.8, 2 * value/100,
                                      facecolor=persona_colors[idx], alpha=0.8))
            ax.text(x + bar_width*0.4, bar_y - 2.5, metric, fontsize=7,
                   ha='center', rotation=0)
            ax.text(x + bar_width*0.4, bar_y + 0.2, f'{value}', fontsize=7,
                   ha='center', fontweight='bold')
        
        # Traits
        ax.text(5, 3.8, 'Key Traits:', fontsize=10, fontweight='bold', ha='center')
        traits_text = ' • '.join(persona['traits'])
        ax.text(5, 3.2, traits_text, fontsize=8, ha='center', wrap=True)
        
        # Opportunities
        ax.text(5, 2.2, 'Opportunities:', fontsize=10, fontweight='bold', ha='center')
        opps_text = ' • '.join(persona['opportunities'])
        ax.text(5, 1.6, opps_text, fontsize=8, ha='center',
               color=colors['mlgreen'], fontweight='bold')
    
    # Summary in last subplot
    ax = axes[5]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 8, 'Segmentation Impact', fontsize=14, fontweight='bold', ha='center')
    
    impacts = [
        '→ 5 distinct user groups identified',
        '→ Clear behavioral patterns',
        '→ Targeted strategies per segment',
        '→ Personalized user experiences',
        '→ Resource allocation optimized',
        '→ 40% improvement in engagement'
    ]
    
    for i, impact in enumerate(impacts):
        ax.text(5, 6.5 - i*0.8, impact, fontsize=10, ha='center',
               color=colors['mlblue'] if i < 5 else colors['mlgreen'],
               fontweight='bold' if i == 5 else 'normal')
    
    plt.suptitle('User Persona Profiles: Deep Understanding from Clustering',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_journey_comparison():
    """Create journey comparison across personas"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Define journey stages
    stages = ['Discover', 'Explore', 'Engage', 'Convert', 'Retain']
    
    # Journey patterns for each persona (conversion rates)
    journeys = {
        'Casual Browsers': [100, 60, 30, 10, 5],
        'Power Users': [100, 95, 90, 85, 95],
        'Social Sharers': [100, 80, 70, 50, 60]
    }
    
    colors_journey = [persona_colors[0], persona_colors[1], persona_colors[2]]
    
    # Plot 1: Funnel comparison
    ax1 = axes[0]
    x = np.arange(len(stages))
    width = 0.25
    
    for i, (persona, values) in enumerate(journeys.items()):
        offset = (i - 1) * width
        ax1.bar(x + offset, values, width, label=persona,
               color=colors_journey[i], alpha=0.8)
    
    ax1.set_xlabel('Journey Stage', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Users (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Journey Progression by Persona', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Drop-off rates
    ax2 = axes[1]
    
    for i, (persona, values) in enumerate(journeys.items()):
        drop_offs = [values[j] - values[j+1] for j in range(len(values)-1)]
        x_pos = np.arange(len(drop_offs))
        ax2.plot(x_pos, drop_offs, 'o-', label=persona,
                color=colors_journey[i], linewidth=2, markersize=8)
    
    ax2.set_xlabel('Transition Point', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Drop-off Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Drop-off Analysis', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Discover→Explore', 'Explore→Engage', 
                        'Engage→Convert', 'Convert→Retain'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimization opportunities
    ax3 = axes[2]
    
    opportunities = [
        ('Casual', 'Explore', 40, 'Simplify onboarding'),
        ('Casual', 'Convert', 20, 'Better value prop'),
        ('Social', 'Convert', 20, 'Social features'),
        ('Social', 'Retain', 10, 'Community building'),
        ('Power', 'All', 5, 'Advanced features')
    ]
    
    y_pos = np.arange(len(opportunities))
    impacts = [opp[2] for opp in opportunities]
    colors_opp = [colors_journey[0] if 'Casual' in opp[0] else 
                  colors_journey[2] if 'Social' in opp[0] else 
                  colors_journey[1] for opp in opportunities]
    
    bars = ax3.barh(y_pos, impacts, color=colors_opp, alpha=0.8)
    
    # Add labels
    for i, (persona, stage, impact, action) in enumerate(opportunities):
        ax3.text(impact + 1, i, action, fontsize=9, va='center')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"{opp[0]} - {opp[1]}" for opp in opportunities])
    ax3.set_xlabel('Potential Impact (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Optimization Opportunities', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('User Journey Analysis Across Personas',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Create main segmentation visualization
    fig1 = create_segmentation_main()
    plt.savefig('user_segmentation_main.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('user_segmentation_main.png', dpi=150, bbox_inches='tight')
    print("Created user_segmentation_main.pdf/png")
    
    # Create persona profiles
    fig2 = create_persona_profiles()
    plt.savefig('persona_profiles.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('persona_profiles.png', dpi=150, bbox_inches='tight')
    print("Created persona_profiles.pdf/png")
    
    # Create journey comparison
    fig3 = create_journey_comparison()
    plt.savefig('journey_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('journey_comparison.png', dpi=150, bbox_inches='tight')
    print("Created journey_comparison.pdf/png")
    
    plt.close('all')