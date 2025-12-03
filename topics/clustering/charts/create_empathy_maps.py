"""
Week 2: Empathy Maps from Clustering
Shows how cluster data translates to design thinking empathy maps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import matplotlib.lines as mlines

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

# Persona colors
persona_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

def create_empathy_map_template():
    """Create a standard empathy map template"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Center circle for persona
    center_circle = Circle((5, 5), 1.5, facecolor='white', 
                          edgecolor=colors['mlblue'], linewidth=3)
    ax.add_patch(center_circle)
    
    # Empathy map sections
    sections = [
        {'angle': 0, 'label': 'THINK & FEEL', 'y_text': 7.5},
        {'angle': 60, 'label': 'HEAR', 'y_text': 6.5},
        {'angle': 120, 'label': 'SEE', 'y_text': 5.5},
        {'angle': 180, 'label': 'SAY & DO', 'y_text': 4.5},
        {'angle': 240, 'label': 'PAIN', 'y_text': 3.5},
        {'angle': 300, 'label': 'GAIN', 'y_text': 2.5}
    ]
    
    # Draw sections
    for i, section in enumerate(sections):
        start_angle = section['angle'] - 30
        end_angle = section['angle'] + 30
        
        wedge = Wedge((5, 5), 3.5, start_angle, end_angle, 
                     width=2, facecolor=persona_colors[i % len(persona_colors)],
                     alpha=0.3, edgecolor='black', linewidth=2)
        ax.add_patch(wedge)
        
        # Add section labels
        angle_rad = np.radians(section['angle'])
        label_x = 5 + 2.5 * np.cos(angle_rad)
        label_y = 5 + 2.5 * np.sin(angle_rad)
        
        ax.text(label_x, label_y, section['label'], fontsize=10,
               fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Title
    ax.text(5, 9, 'Empathy Map Template', fontsize=16, fontweight='bold', ha='center')
    ax.text(5, 8.5, 'Understanding Users Through Multiple Lenses', 
           fontsize=11, ha='center', style='italic')
    
    # Center persona label
    ax.text(5, 5, 'USER\nPERSONA', fontsize=12, fontweight='bold',
           ha='center', va='center', color=colors['mlblue'])
    
    return fig

def create_cluster_to_empathy():
    """Show how cluster data maps to empathy map insights"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    personas = [
        {
            'name': 'Casual Browser',
            'cluster_stats': {
                'Engagement': 25,
                'Frequency': 30,
                'Session_Time': 15,
                'Features_Used': 20,
                'Content_Created': 5
            },
            'empathy': {
                'think_feel': ['Overwhelmed', 'Price conscious', 'Time limited'],
                'hear': ['Simple is better', 'Free alternatives', 'Quick solutions'],
                'see': ['Complex interfaces', 'Premium walls', 'Too many options'],
                'say_do': ['Just browsing', 'Compare prices', 'Quick visits'],
                'pain': ['Complexity', 'Cost', 'Time investment'],
                'gain': ['Simplicity', 'Value', 'Quick wins']
            }
        },
        {
            'name': 'Power User',
            'cluster_stats': {
                'Engagement': 90,
                'Frequency': 95,
                'Session_Time': 85,
                'Features_Used': 95,
                'Content_Created': 80
            },
            'empathy': {
                'think_feel': ['Efficiency matters', 'Need control', 'Innovation'],
                'hear': ['New features', 'Beta access', 'Pro tips'],
                'see': ['Opportunities', 'Inefficiencies', 'Patterns'],
                'say_do': ['Suggest features', 'Help others', 'Explore all'],
                'pain': ['Limitations', 'Slow updates', 'Basic features'],
                'gain': ['Productivity', 'Recognition', 'Early access']
            }
        },
        {
            'name': 'Social Sharer',
            'cluster_stats': {
                'Engagement': 65,
                'Frequency': 70,
                'Session_Time': 50,
                'Features_Used': 60,
                'Content_Created': 75
            },
            'empathy': {
                'think_feel': ['Community', 'Influence', 'Trendy'],
                'hear': ['Viral content', 'Social proof', 'Trends'],
                'see': ['Share buttons', 'Likes/comments', 'Networks'],
                'say_do': ['Share often', 'Comment', 'Tag friends'],
                'pain': ['Isolation', 'No reach', 'Old content'],
                'gain': ['Connections', 'Influence', 'Recognition']
            }
        }
    ]
    
    for idx, (ax, persona) in enumerate(zip(axes, personas)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, persona['name'], fontsize=13, fontweight='bold',
               ha='center', color=persona_colors[idx])
        
        # Left side: Cluster metrics (data-driven)
        ax.text(2, 8.5, 'Cluster Data', fontsize=10, fontweight='bold',
               ha='center', color=colors['mlblue'])
        
        y_pos = 7.5
        for metric, value in persona['cluster_stats'].items():
            # Mini bar
            bar_width = value / 100 * 2
            rect = patches.Rectangle((1, y_pos - 0.15), bar_width, 0.3,
                                    facecolor=persona_colors[idx], alpha=0.7)
            ax.add_patch(rect)
            
            # Label
            metric_clean = metric.replace('_', ' ')
            ax.text(0.8, y_pos, metric_clean, fontsize=8, ha='right', va='center')
            ax.text(3.2, y_pos, f'{value}%', fontsize=8, ha='left', va='center',
                   fontweight='bold')
            y_pos -= 0.8
        
        # Arrow from data to insights
        arrow = patches.FancyArrowPatch((3.5, 5), (4.5, 5),
                                       connectionstyle="arc3,rad=0",
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=2, color=colors['mlgreen'])
        ax.add_patch(arrow)
        ax.text(4, 5.3, 'ML', fontsize=8, ha='center', fontweight='bold',
               color=colors['mlgreen'])
        
        # Right side: Empathy insights (human-centered)
        ax.text(7, 8.5, 'Empathy Insights', fontsize=10, fontweight='bold',
               ha='center', color=colors['mlpurple'])
        
        y_pos = 7.5
        empathy_items = [
            ('Think/Feel', persona['empathy']['think_feel'][0]),
            ('Hear', persona['empathy']['hear'][0]),
            ('See', persona['empathy']['see'][0]),
            ('Say/Do', persona['empathy']['say_do'][0]),
            ('Pain', persona['empathy']['pain'][0]),
            ('Gain', persona['empathy']['gain'][0])
        ]
        
        for category, insight in empathy_items:
            ax.text(5.5, y_pos, f'{category}:', fontsize=8, 
                   fontweight='bold', ha='left', va='center')
            ax.text(6.8, y_pos, insight, fontsize=8, ha='left', va='center',
                   style='italic', color=colors['mlpurple'])
            y_pos -= 0.6
        
        # Bottom note
        ax.text(5, 1, 'Data → Insights → Empathy', fontsize=9,
               ha='center', style='italic', color=colors['mlgray'])
    
    plt.suptitle('From Clustering Metrics to Empathy Understanding',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_detailed_empathy_map():
    """Create a detailed empathy map for one persona with clustering insights"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11, 'Power User Empathy Map', fontsize=18, fontweight='bold',
           ha='center', color=persona_colors[1])
    ax.text(7, 10.4, 'Built from Clustering Analysis (n=400, 15% of users)',
           fontsize=11, ha='center', style='italic')
    
    # Center persona circle
    center = Circle((7, 6), 1.8, facecolor=persona_colors[1], alpha=0.3,
                   edgecolor=persona_colors[1], linewidth=3)
    ax.add_patch(center)
    
    # Persona stats in center
    ax.text(7, 6.8, 'POWER USER', fontsize=12, fontweight='bold',
           ha='center', va='center')
    ax.text(7, 6.2, 'Engagement: 90%', fontsize=9, ha='center')
    ax.text(7, 5.8, 'Frequency: 95%', fontsize=9, ha='center')
    ax.text(7, 5.4, 'Value: High', fontsize=9, ha='center')
    ax.text(7, 5.0, 'Retention: 95%', fontsize=9, ha='center')
    
    # Empathy map sections with insights
    sections = [
        {
            'title': 'THINKS & FEELS',
            'x': 7, 'y': 9,
            'insights': [
                'Efficiency is everything',
                'Needs advanced control',
                'Frustrated by limitations',
                'Proud of expertise',
                'Seeks innovation'
            ],
            'color': colors['mlblue']
        },
        {
            'title': 'HEARS',
            'x': 2, 'y': 7,
            'insights': [
                'Industry best practices',
                'New feature releases',
                'Community discussions',
                'Competitor offerings',
                'Power user tips'
            ],
            'color': colors['mlorange']
        },
        {
            'title': 'SEES',
            'x': 12, 'y': 7,
            'insights': [
                'Optimization opportunities',
                'System inefficiencies',
                'Hidden features',
                'Workflow patterns',
                'Data insights'
            ],
            'color': colors['mlgreen']
        },
        {
            'title': 'SAYS & DOES',
            'x': 7, 'y': 3,
            'insights': [
                'Provides feedback actively',
                'Helps other users',
                'Creates advanced workflows',
                'Tests beta features',
                'Shares knowledge'
            ],
            'color': colors['mlpurple']
        },
        {
            'title': 'PAIN POINTS',
            'x': 2, 'y': 3,
            'insights': [
                'Speed limitations',
                'Missing integrations',
                'Repetitive tasks',
                'Basic user features',
                'Update delays'
            ],
            'color': colors['mlred']
        },
        {
            'title': 'GAINS',
            'x': 12, 'y': 3,
            'insights': [
                'Maximum productivity',
                'Time savings',
                'Recognition as expert',
                'Early access to features',
                'Influence on roadmap'
            ],
            'color': colors['mlgreen']
        }
    ]
    
    # Draw sections
    for section in sections:
        # Box for section
        box = FancyBboxPatch((section['x'] - 2, section['y'] - 1.2),
                            4, 2, boxstyle="round,pad=0.1",
                            facecolor=section['color'], alpha=0.1,
                            edgecolor=section['color'], linewidth=2)
        ax.add_patch(box)
        
        # Title
        ax.text(section['x'], section['y'] + 0.5, section['title'],
               fontsize=11, fontweight='bold', ha='center',
               color=section['color'])
        
        # Insights
        y_offset = 0.1
        for insight in section['insights']:
            ax.text(section['x'], section['y'] - y_offset, f'• {insight}',
                   fontsize=9, ha='center', va='center')
            y_offset += 0.3
        
        # Line to center
        line = mlines.Line2D([section['x'], 7], [section['y'] - 1, 6.5 if section['y'] > 6 else 5.5],
                           color=section['color'], alpha=0.3, linewidth=1,
                           linestyle='--')
        ax.add_line(line)
    
    # Data source box
    data_box = FancyBboxPatch((0.5, 0.2), 3, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['mlgray'], alpha=0.1,
                             edgecolor=colors['mlgray'], linewidth=2)
    ax.add_patch(data_box)
    
    ax.text(2, 1.3, 'Data Sources:', fontsize=9, fontweight='bold', ha='center')
    ax.text(2, 0.9, '• Behavioral clustering', fontsize=8, ha='center')
    ax.text(2, 0.6, '• Usage analytics', fontsize=8, ha='center')
    ax.text(2, 0.3, '• Feature adoption', fontsize=8, ha='center')
    
    # Opportunities box
    opp_box = FancyBboxPatch((10.5, 0.2), 3, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['mlyellow'], alpha=0.1,
                            edgecolor=colors['mlyellow'], linewidth=2)
    ax.add_patch(opp_box)
    
    ax.text(12, 1.3, 'Opportunities:', fontsize=9, fontweight='bold', ha='center')
    ax.text(12, 0.9, '• Advanced features', fontsize=8, ha='center')
    ax.text(12, 0.6, '• API access', fontsize=8, ha='center')
    ax.text(12, 0.3, '• Beta programs', fontsize=8, ha='center')
    
    return fig

def create_empathy_evolution():
    """Show how empathy maps evolve with more clustering data"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    stages = [
        {
            'title': 'Initial Assumptions',
            'subtitle': 'Before Clustering',
            'data_points': 0,
            'confidence': 30,
            'insights': [
                'Generic user needs',
                'Basic pain points',
                'Common features',
                'Standard journey'
            ],
            'color': colors['mlgray']
        },
        {
            'title': 'Data-Informed',
            'subtitle': 'After Basic Clustering',
            'data_points': 1000,
            'confidence': 70,
            'insights': [
                '5 distinct segments',
                'Behavioral patterns',
                'Usage correlations',
                'Segment preferences'
            ],
            'color': colors['mlblue']
        },
        {
            'title': 'Deep Understanding',
            'subtitle': 'Advanced Analysis',
            'data_points': 10000,
            'confidence': 95,
            'insights': [
                'Micro-segments',
                'Predictive behaviors',
                'Hidden motivations',
                'Innovation opportunities'
            ],
            'color': colors['mlgreen']
        }
    ]
    
    for idx, (ax, stage) in enumerate(zip(axes, stages)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9, stage['title'], fontsize=13, fontweight='bold',
               ha='center', color=stage['color'])
        ax.text(5, 8.4, stage['subtitle'], fontsize=10, ha='center',
               style='italic')
        
        # Central empathy map (simplified)
        center_size = 1.5 + idx * 0.3  # Growing size
        center = Circle((5, 5.5), center_size, 
                       facecolor=stage['color'], alpha=0.2,
                       edgecolor=stage['color'], linewidth=2)
        ax.add_patch(center)
        
        # Segments around (more defined over time)
        n_segments = 2 + idx * 2
        for i in range(n_segments):
            angle = i * (360 / n_segments)
            angle_rad = np.radians(angle)
            x = 5 + 2.5 * np.cos(angle_rad)
            y = 5.5 + 2.5 * np.sin(angle_rad)
            
            segment_circle = Circle((x, y), 0.3 + idx * 0.1,
                                  facecolor=persona_colors[i % len(persona_colors)],
                                  alpha=0.3 + idx * 0.2,
                                  edgecolor='black', linewidth=1)
            ax.add_patch(segment_circle)
        
        # Stats
        ax.text(5, 3, f'Data Points: {stage["data_points"]:,}', 
               fontsize=10, ha='center', fontweight='bold')
        
        # Confidence bar
        bar_width = stage['confidence'] / 100 * 4
        conf_bar = patches.Rectangle((5 - bar_width/2, 2.3), bar_width, 0.3,
                                    facecolor=stage['color'], alpha=0.7)
        ax.add_patch(conf_bar)
        ax.text(5, 2.1, f'Confidence: {stage["confidence"]}%',
               fontsize=9, ha='center')
        
        # Insights
        ax.text(5, 1.5, 'Key Insights:', fontsize=9, fontweight='bold',
               ha='center')
        y_pos = 1.0
        for insight in stage['insights']:
            ax.text(5, y_pos, f'• {insight}', fontsize=8, ha='center',
                   color=stage['color'])
            y_pos -= 0.3
    
    plt.suptitle('Evolution of Empathy Understanding Through Clustering',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Create empathy map template
    fig1 = create_empathy_map_template()
    plt.savefig('empathy_map_template.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('empathy_map_template.png', dpi=150, bbox_inches='tight')
    print("Created empathy_map_template.pdf/png")
    
    # Create cluster to empathy mapping
    fig2 = create_cluster_to_empathy()
    plt.savefig('cluster_to_empathy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cluster_to_empathy.png', dpi=150, bbox_inches='tight')
    print("Created cluster_to_empathy.pdf/png")
    
    # Create detailed empathy map
    fig3 = create_detailed_empathy_map()
    plt.savefig('empathy_map_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('empathy_map_detailed.png', dpi=150, bbox_inches='tight')
    print("Created empathy_map_detailed.pdf/png")
    
    # Create empathy evolution
    fig4 = create_empathy_evolution()
    plt.savefig('empathy_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('empathy_evolution.png', dpi=150, bbox_inches='tight')
    print("Created empathy_evolution.pdf/png")
    
    plt.close('all')