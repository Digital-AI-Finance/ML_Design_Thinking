"""
Create idea selection metrics visualization for Week 6
Shows how to evaluate and select ideas from the evolution tree
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with 3 subplots
fig = plt.figure(figsize=(16, 10))

# Create grid for subplots
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 0.8])
ax1 = fig.add_subplot(gs[0, :])  # Top spanning both columns
ax2 = fig.add_subplot(gs[1, 0])   # Bottom left
ax3 = fig.add_subplot(gs[1, 1])   # Bottom right

# Top: Idea Funnel
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.axis('off')

# Funnel stages
stages = [
    (2, 6, 3, '100 Ideas\nGenerated', '#e3f2fd'),
    (3.5, 4.5, 2.5, '25 Ideas\nScreened', '#bbdefb'),
    (4.5, 3, 2, '10 Ideas\nEvaluated', '#90caf9'),
    (5.5, 1.5, 1.5, '3 Ideas\nPrototyped', '#64b5f6'),
    (6.5, 0.5, 1, '1 Idea\nSelected', '#42a5f5')
]

# Draw funnel
for i, (x, y, width, label, color) in enumerate(stages):
    # Draw trapezoid/rectangle for each stage
    if i < len(stages) - 1:
        next_x, next_y, next_width, _, _ = stages[i + 1]
        vertices = [
            (x - width/2, y + 0.4),
            (x + width/2, y + 0.4),
            (next_x + next_width/2, next_y + 0.4),
            (next_x - next_width/2, next_y + 0.4)
        ]
        funnel = mpatches.Polygon(vertices, facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(funnel)
    else:
        rect = FancyBboxPatch((x - width/2, y), width, 0.8,
                              boxstyle='round,pad=0.02',
                              facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
    
    # Add label
    ax1.text(x, y + 0.4, label, ha='center', va='center', 
            fontsize=13, fontweight='bold')
    
    # Add filtering criteria
    if i < len(stages) - 1:
        criteria = ['Feasibility', 'Impact', 'Novelty', 'Resources', 'Risk'][i]
        ax1.text(8, y, f'Filter: {criteria}', fontsize=11, style='italic')

ax1.set_title('Idea Selection Funnel: From Many to One', fontsize=18, fontweight='bold')

# Bottom left: Scoring Matrix
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Create scatter plot of ideas
np.random.seed(42)
n_ideas = 30
feasibility = np.random.uniform(2, 9, n_ideas)
impact = np.random.uniform(2, 9, n_ideas)
novelty = np.random.uniform(20, 100, n_ideas)  # For size
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, n_ideas))

scatter = ax2.scatter(feasibility, impact, s=novelty*5, c=colors, alpha=0.6, 
                     edgecolors='black', linewidth=1)

# Highlight top ideas
top_indices = np.argsort(feasibility * impact)[-3:]
for idx in top_indices:
    ax2.scatter(feasibility[idx], impact[idx], s=novelty[idx]*5, 
               facecolors='none', edgecolors='red', linewidth=3)
    ax2.annotate(f'Top {len(top_indices) - np.where(top_indices == idx)[0][0]}',
                xy=(feasibility[idx], impact[idx]),
                xytext=(feasibility[idx] + 0.5, impact[idx] + 0.5),
                fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Add quadrant lines
ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.5)

# Quadrant labels
ax2.text(2.5, 8.5, 'High Impact\nLow Feasibility', ha='center', fontsize=11, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
ax2.text(7.5, 8.5, 'High Impact\nHigh Feasibility', ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
ax2.text(2.5, 1.5, 'Low Impact\nLow Feasibility', ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))
ax2.text(7.5, 1.5, 'Low Impact\nHigh Feasibility', ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

ax2.set_xlabel('Feasibility Score', fontsize=14)
ax2.set_ylabel('Impact Score', fontsize=14)
ax2.set_title('Idea Scoring Matrix', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add legend for size
ax2.text(0.5, 9.5, 'Size = Novelty', fontsize=11, style='italic')

# Bottom right: Evaluation Criteria
ax3.axis('off')

# Create radar chart style evaluation
criteria = ['Feasibility', 'Impact', 'Novelty', 'Cost', 'Time', 'Risk']
n_criteria = len(criteria)

# Example scores for best idea
scores = [8, 9, 7, 6, 7, 5]
max_score = 10

# Create circular layout
angles = np.linspace(0, 2*np.pi, n_criteria, endpoint=False).tolist()
scores_norm = [s/max_score for s in scores]
angles += angles[:1]
scores_norm += scores_norm[:1]

# Plot radar
ax_radar = plt.subplot(gs[1, 1], projection='polar')
ax_radar.plot(angles, scores_norm, 'o-', linewidth=2, color='#42a5f5')
ax_radar.fill(angles, scores_norm, alpha=0.25, color='#42a5f5')
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(criteria, size=12)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_radar.set_yticklabels(['2', '4', '6', '8', '10'], size=10)
ax_radar.grid(True)
ax_radar.set_title('Best Idea Profile', fontsize=16, fontweight='bold', pad=20)

# Add overall score
overall_score = np.mean(scores)
ax3.text(0.5, 0.1, f'Overall Score: {overall_score:.1f}/10', 
        ha='center', fontsize=14, fontweight='bold',
        transform=ax3.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))

# Main title
fig.suptitle('Idea Selection Metrics: Pruning the Evolution Tree', 
            fontsize=20, fontweight='bold', y=0.98)

plt.tight_layout()

# Save the figure
plt.savefig('idea_metrics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('idea_metrics.png', dpi=150, bbox_inches='tight')

print("Idea metrics visualization created successfully!")