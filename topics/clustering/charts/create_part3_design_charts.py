"""
Create all missing Part 3 Design charts for Week 1 slides
Focus on innovation patterns, design thinking, and human-centered applications
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import networkx as nx
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlpink': '#e377c2',
    'mlbrown': '#8c564b',
    'mlgray': '#7f7f7f',
    'mlyellow': '#bcbd22',
    'mlcyan': '#17becf'
}

np.random.seed(42)

# Chart 1: Innovation Archetypes
print("Creating innovation_archetypes.pdf...")
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

archetypes = [
    {'name': 'Disruptor', 'traits': ['High Risk', 'High Reward', 'Market Changer'], 
     'icon': '*', 'color': colors['mlred']},
    {'name': 'Optimizer', 'traits': ['Efficiency', 'Cost Reduction', 'Process Focus'],
     'icon': 'o', 'color': colors['mlblue']},
    {'name': 'Explorer', 'traits': ['Research', 'New Markets', 'Experimentation'],
     'icon': '^', 'color': colors['mlgreen']},
    {'name': 'Builder', 'traits': ['Platform', 'Ecosystem', 'Infrastructure'],
     'icon': 's', 'color': colors['mlorange']},
    {'name': 'Connector', 'traits': ['Partnerships', 'Integration', 'Networks'],
     'icon': 'D', 'color': colors['mlpurple']},
    {'name': 'Guardian', 'traits': ['Security', 'Compliance', 'Risk Mitigation'],
     'icon': 'p', 'color': colors['mlbrown']}
]

for idx, archetype in enumerate(archetypes):
    ax = axes[idx]
    
    # Create persona card
    card = FancyBboxPatch((0.1, 0.3), 0.8, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor=archetype['color'], alpha=0.1,
                          edgecolor=archetype['color'], linewidth=2)
    ax.add_patch(card)
    
    # Add icon
    ax.scatter([0.5], [0.75], s=500, marker=archetype['icon'], 
              c=archetype['color'], alpha=0.7, edgecolors='white', linewidth=2)
    
    # Add name
    ax.text(0.5, 0.55, archetype['name'], fontsize=12, fontweight='bold',
           ha='center', va='center')
    
    # Add traits
    for i, trait in enumerate(archetype['traits']):
        ax.text(0.5, 0.4 - i*0.08, f'• {trait}', fontsize=9,
               ha='center', va='center', color='gray')
    
    # Add data distribution
    x = np.random.normal(0.5, 0.15, 50)
    y = np.random.uniform(0.05, 0.25, 50)
    ax.scatter(x, y, s=10, alpha=0.3, c=archetype['color'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

fig.suptitle('Innovation Archetypes: 6 Distinct Patterns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('innovation_archetypes.pdf', dpi=300, bbox_inches='tight')
plt.savefig('innovation_archetypes.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Innovation Pattern Maps
print("Creating innovation_pattern_maps.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))

# Create journey stages
stages = ['Ideation', 'Validation', 'Development', 'Launch', 'Growth', 'Maturity']
y_positions = np.linspace(8, 1, len(stages))

# Draw timeline
ax.plot([1, 1], [0, 9], 'k-', linewidth=2, alpha=0.3)

# Create patterns for each stage
patterns = {
    'Ideation': {'clusters': 5, 'spread': 2.0, 'color': colors['mlpurple']},
    'Validation': {'clusters': 4, 'spread': 1.5, 'color': colors['mlblue']},
    'Development': {'clusters': 3, 'spread': 1.2, 'color': colors['mlorange']},
    'Launch': {'clusters': 3, 'spread': 1.0, 'color': colors['mlred']},
    'Growth': {'clusters': 2, 'spread': 0.8, 'color': colors['mlgreen']},
    'Maturity': {'clusters': 2, 'spread': 0.5, 'color': colors['mlbrown']}
}

for i, stage in enumerate(stages):
    y = y_positions[i]
    pattern = patterns[stage]
    
    # Stage marker
    ax.scatter([1], [y], s=200, c=pattern['color'], zorder=5)
    ax.text(0.5, y, stage, fontsize=11, fontweight='bold', ha='right', va='center')
    
    # Create cluster patterns
    for j in range(pattern['clusters']):
        x = np.random.normal(3 + j*0.8, pattern['spread']*0.3, 20)
        y_cluster = np.random.normal(y, pattern['spread']*0.2, 20)
        ax.scatter(x, y_cluster, s=20, alpha=0.4, c=pattern['color'])
    
    # Connect stages
    if i < len(stages) - 1:
        ax.annotate('', xy=(1, y_positions[i+1]), xytext=(1, y),
                   arrowprops=dict(arrowstyle='->', lw=2, alpha=0.5, color='gray'))

# Add insights
ax.text(7, 8, 'Many divergent ideas', fontsize=9, style='italic', color=colors['mlpurple'])
ax.text(7, 6.5, 'Testing & filtering', fontsize=9, style='italic', color=colors['mlblue'])
ax.text(7, 5, 'Focus & refinement', fontsize=9, style='italic', color=colors['mlorange'])
ax.text(7, 3.5, 'Market entry', fontsize=9, style='italic', color=colors['mlred'])
ax.text(7, 2, 'Scaling patterns', fontsize=9, style='italic', color=colors['mlgreen'])
ax.text(7, 0.5, 'Optimization', fontsize=9, style='italic', color=colors['mlbrown'])

ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
ax.set_title('Innovation Journey: Pattern Evolution Over Time', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('innovation_pattern_maps.pdf', dpi=300, bbox_inches='tight')
plt.savefig('innovation_pattern_maps.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Innovation Taxonomy
print("Creating innovation_taxonomy.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))

# Create hierarchical tree structure
tree_data = {
    'Innovation': {
        'Product': {
            'Incremental': ['Feature Update', 'Performance', 'Quality'],
            'Radical': ['New Category', 'Breakthrough', 'Disruption']
        },
        'Process': {
            'Efficiency': ['Automation', 'Optimization', 'Lean'],
            'Transformation': ['Digitization', 'AI/ML', 'Industry 4.0']
        },
        'Business Model': {
            'Revenue': ['Subscription', 'Freemium', 'Platform'],
            'Value': ['Ecosystem', 'Sharing Economy', 'Circular']
        }
    }
}

# Draw tree
def draw_node(ax, x, y, text, level, color):
    if level == 0:
        size = 3000
        fc = color
        alpha = 0.3
    elif level == 1:
        size = 2000
        fc = color
        alpha = 0.4
    elif level == 2:
        size = 1000
        fc = color
        alpha = 0.5
    else:
        size = 500
        fc = color
        alpha = 0.6
    
    ax.scatter(x, y, s=size, c=fc, alpha=alpha, edgecolors='white', linewidth=2)
    ax.text(x, y, text, fontsize=10-level, fontweight='bold' if level < 2 else 'normal',
           ha='center', va='center')
    return x, y

# Root
root_x, root_y = draw_node(ax, 5, 8, 'Innovation', 0, colors['mlpurple'])

# Level 1
l1_positions = [(2, 5), (5, 5), (8, 5)]
l1_names = ['Product', 'Process', 'Business Model']
l1_colors = [colors['mlblue'], colors['mlorange'], colors['mlgreen']]

for (x, y), name, color in zip(l1_positions, l1_names, l1_colors):
    node_x, node_y = draw_node(ax, x, y, name, 1, color)
    ax.plot([root_x, node_x], [root_y, node_y], 'k-', alpha=0.3, linewidth=1)
    
    # Level 2
    if name == 'Product':
        l2_pos = [(1, 3), (3, 3)]
        l2_names = ['Incremental', 'Radical']
    elif name == 'Process':
        l2_pos = [(4, 3), (6, 3)]
        l2_names = ['Efficiency', 'Transformation']
    else:
        l2_pos = [(7, 3), (9, 3)]
        l2_names = ['Revenue', 'Value']
    
    for (x2, y2), name2 in zip(l2_pos, l2_names):
        node2_x, node2_y = draw_node(ax, x2, y2, name2, 2, color)
        ax.plot([node_x, node2_x], [node_y, node2_y], 'k-', alpha=0.3, linewidth=1)

ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.set_title('Innovation Taxonomy: Hierarchical Classification', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('innovation_taxonomy.pdf', dpi=300, bbox_inches='tight')
plt.savefig('innovation_taxonomy.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Innovation Lifecycle
print("Creating innovation_lifecycle.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))

# Create circular lifecycle
angles = np.linspace(0, 2*np.pi, 7)[:-1]
radius = 3
center = (5, 5)

stages = ['Discover', 'Define', 'Develop', 'Deliver', 'Deploy', 'Measure']
stage_colors = [colors['mlpurple'], colors['mlblue'], colors['mlorange'], 
               colors['mlgreen'], colors['mlred'], colors['mlbrown']]

# Draw cycle
for i, (angle, stage, color) in enumerate(zip(angles, stages, stage_colors)):
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    
    # Stage circle
    circle = Circle((x, y), 0.8, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, stage, fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Arrow to next stage
    next_angle = angles[(i+1) % 6]
    next_x = center[0] + radius * np.cos(next_angle)
    next_y = center[1] + radius * np.sin(next_angle)
    
    # Calculate arrow position
    arrow_start_x = x + 0.6 * np.cos(angle + np.pi/6)
    arrow_start_y = y + 0.6 * np.sin(angle + np.pi/6)
    arrow_end_x = next_x - 0.6 * np.cos(next_angle - np.pi/6)
    arrow_end_y = next_y - 0.6 * np.sin(next_angle - np.pi/6)
    
    arrow = FancyArrowPatch((arrow_start_x, arrow_start_y), 
                           (arrow_end_x, arrow_end_y),
                           arrowstyle='->', mutation_scale=20,
                           color='gray', alpha=0.5, linewidth=2)
    ax.add_patch(arrow)

# Center text
ax.text(center[0], center[1], 'Continuous\nInnovation\nCycle', 
       fontsize=12, fontweight='bold', ha='center', va='center',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add iteration indicators
for angle in np.linspace(0, 2*np.pi, 12):
    x = center[0] + (radius + 1.5) * np.cos(angle)
    y = center[1] + (radius + 1.5) * np.sin(angle)
    ax.scatter(x, y, s=20, c='gray', alpha=0.3)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('Innovation Lifecycle: Continuous Improvement Process', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('innovation_lifecycle.pdf', dpi=300, bbox_inches='tight')
plt.savefig('innovation_lifecycle.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Innovation Ecosystem
print("Creating innovation_ecosystem.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))

# Create network graph
G = nx.Graph()

# Define nodes
node_categories = {
    'Core': ['Innovation Hub', 'R&D Center', 'Design Lab'],
    'Partners': ['Universities', 'Startups', 'Tech Giants', 'Suppliers'],
    'Enablers': ['Investors', 'Accelerators', 'Government', 'Mentors'],
    'Market': ['Customers', 'Early Adopters', 'Distributors', 'Retailers']
}

# Add nodes and edges
pos = {}
node_colors = []
node_sizes = []

# Position nodes
angle_offset = 0
for category, nodes in node_categories.items():
    if category == 'Core':
        # Central nodes
        for i, node in enumerate(nodes):
            angle = i * 2 * np.pi / len(nodes)
            pos[node] = (5 + 0.8*np.cos(angle), 5 + 0.8*np.sin(angle))
            G.add_node(node)
            node_colors.append(colors['mlpurple'])
            node_sizes.append(1500)
    else:
        # Outer nodes
        base_angle = angle_offset
        for i, node in enumerate(nodes):
            angle = base_angle + i * (2*np.pi/12)
            radius = 3
            pos[node] = (5 + radius*np.cos(angle), 5 + radius*np.sin(angle))
            G.add_node(node)
            if category == 'Partners':
                node_colors.append(colors['mlblue'])
            elif category == 'Enablers':
                node_colors.append(colors['mlorange'])
            else:
                node_colors.append(colors['mlgreen'])
            node_sizes.append(800)
        angle_offset += 2*np.pi/3

# Add edges
for core_node in node_categories['Core']:
    for category, nodes in node_categories.items():
        if category != 'Core':
            for node in nodes[:2]:  # Connect to first 2 of each category
                G.add_edge(core_node, node)

# Draw network
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

# Add legend
legend_elements = [
    mpatches.Patch(color=colors['mlpurple'], alpha=0.7, label='Core Innovation'),
    mpatches.Patch(color=colors['mlblue'], alpha=0.7, label='Partners'),
    mpatches.Patch(color=colors['mlorange'], alpha=0.7, label='Enablers'),
    mpatches.Patch(color=colors['mlgreen'], alpha=0.7, label='Market')
]
ax.legend(handles=legend_elements, loc='upper right')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('Innovation Ecosystem: Network of Relationships', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('innovation_ecosystem.pdf', dpi=300, bbox_inches='tight')
plt.savefig('innovation_ecosystem.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 6: Journey Map Clusters
print("Creating journey_map_clusters.pdf...")
fig, ax = plt.subplots(figsize=(14, 8))

# Define journey stages and touchpoints
stages = ['Awareness', 'Consideration', 'Purchase', 'Onboarding', 'Usage', 'Loyalty']
touchpoints = {
    'Awareness': ['Social Media', 'Ads', 'Word of Mouth'],
    'Consideration': ['Website', 'Reviews', 'Demos'],
    'Purchase': ['Online Store', 'Sales Team', 'Partners'],
    'Onboarding': ['Tutorial', 'Support', 'Documentation'],
    'Usage': ['Product', 'Features', 'Updates'],
    'Loyalty': ['Community', 'Rewards', 'Advocacy']
}

# Create clusters for each stage
x_positions = np.linspace(1, 12, len(stages))
cluster_colors = [colors['mlpurple'], colors['mlblue'], colors['mlorange'],
                 colors['mlgreen'], colors['mlred'], colors['mlbrown']]

for i, (stage, x_pos, color) in enumerate(zip(stages, x_positions, cluster_colors)):
    # Stage header
    ax.text(x_pos, 7, stage, fontsize=11, fontweight='bold', ha='center')
    
    # Emotion curve
    if i == 0:
        emotion = 5
    elif i == 1:
        emotion = 4
    elif i == 2:
        emotion = 3
    elif i == 3:
        emotion = 4
    elif i == 4:
        emotion = 5
    else:
        emotion = 6
    
    ax.scatter(x_pos, emotion, s=300, c=color, alpha=0.5)
    
    # Touchpoints
    for j, tp in enumerate(touchpoints[stage]):
        y = 2 - j*0.5
        ax.scatter(x_pos, y, s=100, c=color, alpha=0.7)
        ax.text(x_pos + 0.2, y, tp, fontsize=8, va='center')
    
    # Connect emotion points
    if i > 0:
        ax.plot([x_positions[i-1], x_pos], 
               [prev_emotion, emotion], 
               'k-', alpha=0.3, linewidth=2)
    prev_emotion = emotion

# Add emotion line label
ax.text(0.5, 5, 'Customer\nEmotion', fontsize=9, fontweight='bold', ha='center', va='center')

ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.set_title('Customer Journey Map with Cluster Touchpoints', fontsize=14, fontweight='bold')
ax.axis('off')

# Add grid for reference
for y in range(1, 7):
    ax.axhline(y=y, color='gray', alpha=0.1, linestyle='--')

plt.tight_layout()
plt.savefig('journey_map_clusters.pdf', dpi=300, bbox_inches='tight')
plt.savefig('journey_map_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 7: Opportunity Heatmap
print("Creating opportunity_heatmap.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))

# Create heatmap data
markets = ['Healthcare', 'Finance', 'Retail', 'Education', 'Manufacturing', 'Energy']
technologies = ['AI/ML', 'IoT', 'Blockchain', 'AR/VR', 'Cloud', 'Quantum']

# Generate opportunity scores (0-100)
np.random.seed(42)
opportunity_matrix = np.array([
    [95, 70, 40, 60, 85, 30],  # Healthcare
    [90, 60, 85, 30, 90, 50],  # Finance
    [85, 80, 50, 75, 85, 20],  # Retail
    [80, 65, 30, 85, 90, 25],  # Education
    [75, 95, 45, 70, 80, 35],  # Manufacturing
    [70, 90, 60, 40, 75, 60]   # Energy
])

# Create heatmap
im = ax.imshow(opportunity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set ticks
ax.set_xticks(np.arange(len(technologies)))
ax.set_yticks(np.arange(len(markets)))
ax.set_xticklabels(technologies)
ax.set_yticklabels(markets)

# Rotate the tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(markets)):
    for j in range(len(technologies)):
        score = opportunity_matrix[i, j]
        text_color = 'white' if score > 50 else 'black'
        ax.text(j, i, f'{score}', ha='center', va='center', 
               color=text_color, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Opportunity Score', rotation=270, labelpad=15)

ax.set_title('Innovation Opportunity Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Technologies', fontsize=11)
ax.set_ylabel('Market Sectors', fontsize=11)

plt.tight_layout()
plt.savefig('opportunity_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('opportunity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 8: Behavior Patterns
print("Creating behavior_patterns.pdf...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

behavior_types = ['Early Adopters', 'Mainstream Users', 'Laggards']
behavior_colors = [colors['mlgreen'], colors['mlblue'], colors['mlorange']]

for idx, (behavior, color, ax) in enumerate(zip(behavior_types, behavior_colors, axes)):
    # Generate behavior data
    if behavior == 'Early Adopters':
        # High engagement, high variability
        engagement = np.random.beta(2, 1, 200) * 100
        frequency = np.random.gamma(2, 2, 200) * 10
    elif behavior == 'Mainstream Users':
        # Moderate engagement, normal distribution
        engagement = np.random.normal(50, 15, 200)
        frequency = np.random.normal(50, 10, 200)
    else:
        # Low engagement, skewed
        engagement = np.random.beta(1, 3, 200) * 100
        frequency = np.random.beta(1, 2, 200) * 100
    
    # Apply clustering
    data = np.column_stack([engagement, frequency])
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Plot clusters
    for i in range(3):
        mask = labels == i
        ax.scatter(engagement[mask], frequency[mask], 
                  alpha=0.6, s=30, label=f'Segment {i+1}')
    
    ax.set_xlabel('Engagement Level')
    ax.set_ylabel('Usage Frequency')
    ax.set_title(f'{behavior}\nBehavior Patterns', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

fig.suptitle('User Behavior Pattern Clustering', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('behavior_patterns.pdf', dpi=300, bbox_inches='tight')
plt.savefig('behavior_patterns.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 9: Design Priority Matrix
print("Creating design_priority_matrix.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))

# Define items with impact and effort scores
items = [
    {'name': 'AI Chatbot', 'impact': 85, 'effort': 70, 'category': 'Tech'},
    {'name': 'Mobile App', 'impact': 90, 'effort': 60, 'category': 'Product'},
    {'name': 'Data Analytics', 'impact': 75, 'effort': 40, 'category': 'Tech'},
    {'name': 'User Training', 'impact': 60, 'effort': 30, 'category': 'Service'},
    {'name': 'API Platform', 'impact': 80, 'effort': 85, 'category': 'Tech'},
    {'name': 'Design System', 'impact': 70, 'effort': 50, 'category': 'Design'},
    {'name': 'Customer Portal', 'impact': 65, 'effort': 45, 'category': 'Product'},
    {'name': 'Process Automation', 'impact': 85, 'effort': 55, 'category': 'Process'},
    {'name': 'Community Building', 'impact': 55, 'effort': 35, 'category': 'Service'},
    {'name': 'AR Features', 'impact': 45, 'effort': 80, 'category': 'Tech'},
    {'name': 'Personalization', 'impact': 75, 'effort': 65, 'category': 'Product'},
    {'name': 'Security Upgrade', 'impact': 95, 'effort': 75, 'category': 'Tech'}
]

# Define quadrants
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

# Quadrant labels
ax.text(25, 75, 'Quick Wins\n(High Impact, Low Effort)', 
       ha='center', va='center', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=colors['mlgreen'], alpha=0.1))
ax.text(75, 75, 'Major Projects\n(High Impact, High Effort)', 
       ha='center', va='center', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=colors['mlorange'], alpha=0.1))
ax.text(25, 25, 'Fill-ins\n(Low Impact, Low Effort)', 
       ha='center', va='center', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=colors['mlgray'], alpha=0.1))
ax.text(75, 25, 'Question Marks\n(Low Impact, High Effort)', 
       ha='center', va='center', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=colors['mlred'], alpha=0.1))

# Plot items
category_colors = {
    'Tech': colors['mlpurple'],
    'Product': colors['mlblue'],
    'Service': colors['mlgreen'],
    'Design': colors['mlorange'],
    'Process': colors['mlbrown']
}

for item in items:
    color = category_colors[item['category']]
    ax.scatter(item['effort'], item['impact'], s=200, c=color, alpha=0.7,
              edgecolors='white', linewidth=2)
    ax.annotate(item['name'], (item['effort'], item['impact']),
               xytext=(5, 5), textcoords='offset points',
               fontsize=8, alpha=0.8)

# Add legend
legend_elements = [mpatches.Patch(color=color, alpha=0.7, label=cat) 
                  for cat, color in category_colors.items()]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

ax.set_xlabel('Implementation Effort →', fontsize=12)
ax.set_ylabel('Business Impact →', fontsize=12)
ax.set_title('Design Priority Matrix: Impact vs Effort Analysis', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('design_priority_matrix.pdf', dpi=300, bbox_inches='tight')
plt.savefig('design_priority_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 10: Stakeholder Network
print("Creating stakeholder_network.pdf...")
fig, ax = plt.subplots(figsize=(14, 10))

# Create stakeholder network
G = nx.Graph()

# Define stakeholder categories with influence and interest
stakeholders = {
    'Primary': {
        'CEO': {'influence': 100, 'interest': 90},
        'Product Team': {'influence': 85, 'interest': 95},
        'Customers': {'influence': 70, 'interest': 100}
    },
    'Secondary': {
        'Engineering': {'influence': 80, 'interest': 70},
        'Marketing': {'influence': 60, 'interest': 80},
        'Sales': {'influence': 65, 'interest': 85},
        'Support': {'influence': 50, 'interest': 75}
    },
    'External': {
        'Investors': {'influence': 90, 'interest': 60},
        'Partners': {'influence': 55, 'interest': 70},
        'Regulators': {'influence': 75, 'interest': 40},
        'Community': {'influence': 40, 'interest': 65}
    }
}

# Position nodes based on influence and interest
pos = {}
node_colors = []
node_sizes = []

for category, members in stakeholders.items():
    for name, metrics in members.items():
        G.add_node(name)
        # Position based on influence (x) and interest (y)
        pos[name] = (metrics['influence']/10, metrics['interest']/10)
        
        # Color by category
        if category == 'Primary':
            node_colors.append(colors['mlred'])
            node_sizes.append(1500)
        elif category == 'Secondary':
            node_colors.append(colors['mlblue'])
            node_sizes.append(1000)
        else:
            node_colors.append(colors['mlgreen'])
            node_sizes.append(800)

# Add edges based on relationships
edges = [
    ('CEO', 'Product Team'), ('CEO', 'Engineering'), ('CEO', 'Investors'),
    ('Product Team', 'Customers'), ('Product Team', 'Engineering'),
    ('Product Team', 'Marketing'), ('Product Team', 'Support'),
    ('Customers', 'Support'), ('Customers', 'Community'),
    ('Engineering', 'Partners'), ('Marketing', 'Sales'),
    ('Sales', 'Customers'), ('Investors', 'Partners'),
    ('Regulators', 'CEO'), ('Community', 'Marketing')
]
G.add_edges_from(edges)

# Draw network
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                      alpha=0.7, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

# Add quadrant labels
ax.text(2.5, 9, 'Low Influence\nHigh Interest', ha='center', va='center', 
       fontsize=9, alpha=0.5)
ax.text(7.5, 9, 'High Influence\nHigh Interest\n(KEY PLAYERS)', ha='center', va='center',
       fontsize=9, fontweight='bold', color=colors['mlred'])
ax.text(2.5, 2, 'Low Influence\nLow Interest', ha='center', va='center',
       fontsize=9, alpha=0.5)
ax.text(7.5, 2, 'High Influence\nLow Interest', ha='center', va='center',
       fontsize=9, alpha=0.5)

# Add grid lines
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=5, color='gray', linestyle='--', alpha=0.3)

# Legend
legend_elements = [
    mpatches.Patch(color=colors['mlred'], alpha=0.7, label='Primary Stakeholders'),
    mpatches.Patch(color=colors['mlblue'], alpha=0.7, label='Secondary Stakeholders'),
    mpatches.Patch(color=colors['mlgreen'], alpha=0.7, label='External Stakeholders')
]
ax.legend(handles=legend_elements, loc='lower right')

ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.set_xlabel('Influence Level →', fontsize=12)
ax.set_ylabel('Interest Level →', fontsize=12)
ax.set_title('Stakeholder Network Analysis', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('stakeholder_network.pdf', dpi=300, bbox_inches='tight')
plt.savefig('stakeholder_network.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll Part 3 Design charts created successfully!")
print("Created files:")
print("- innovation_archetypes.pdf/png")
print("- innovation_pattern_maps.pdf/png")
print("- innovation_taxonomy.pdf/png")
print("- innovation_lifecycle.pdf/png")
print("- innovation_ecosystem.pdf/png")
print("- journey_map_clusters.pdf/png")
print("- opportunity_heatmap.pdf/png")
print("- behavior_patterns.pdf/png")
print("- design_priority_matrix.pdf/png")
print("- stakeholder_network.pdf/png")