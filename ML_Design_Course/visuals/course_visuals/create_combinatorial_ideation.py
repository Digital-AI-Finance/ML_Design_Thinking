"""
Create combinatorial ideation visualization for Week 6
Shows how AI combines unrelated concepts for innovation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.lines as mlines

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Concept categories
left_concepts = ['Fitness', 'Music', 'Food', 'Travel', 'Gaming']
right_concepts = ['Social', 'Learning', 'Shopping', 'Health', 'Finance']

# Colors for each category
left_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
right_colors = ['#DDA0DD', '#98D8C8', '#FFB6C1', '#87CEEB', '#F0E68C']

# Draw left concepts
left_x = 2
for i, (concept, color) in enumerate(zip(left_concepts, left_colors)):
    y = 8 - i * 1.5
    circle = Circle((left_x, y), 0.8, facecolor=color, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(left_x, y, concept, ha='center', va='center', 
           fontsize=13, fontweight='bold')

# Draw right concepts  
right_x = 12
for i, (concept, color) in enumerate(zip(right_concepts, right_colors)):
    y = 8 - i * 1.5
    circle = Circle((right_x, y), 0.8, facecolor=color, alpha=0.7,
                   edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(right_x, y, concept, ha='center', va='center',
           fontsize=13, fontweight='bold')

# Draw AI combination box in center
ai_box = FancyBboxPatch((5, 3), 4, 4,
                        boxstyle="round,pad=0.1",
                        facecolor='lightgray', alpha=0.3,
                        edgecolor='black', linewidth=2)
ax.add_patch(ai_box)

ax.text(7, 6.5, 'AI Combinator', ha='center', fontsize=16, fontweight='bold')
ax.text(7, 6, '(GPT/LLM)', ha='center', fontsize=12, style='italic')

# Example combinations with innovation ideas
combinations = [
    ('Fitness', 'Social', 'Virtual workout buddies with real-time form correction'),
    ('Music', 'Learning', 'Songs that teach programming concepts'),
    ('Food', 'Finance', 'Meal planning based on budget optimization'),
    ('Travel', 'Health', 'Destinations recommended by wellness goals'),
    ('Gaming', 'Shopping', 'AR treasure hunts in retail stores')
]

# Draw selected combination lines and results
selected_pairs = [(0, 0), (1, 1), (2, 4), (3, 3), (4, 2)]  # Indices to connect

for (left_idx, right_idx), (left_c, right_c, innovation) in zip(selected_pairs, combinations):
    left_y = 8 - left_idx * 1.5
    right_y = 8 - right_idx * 1.5
    
    # Draw connecting lines through AI box
    # Left to AI box
    ax.plot([left_x + 0.8, 5], [left_y, 5], 
           color=left_colors[left_idx], alpha=0.5, linewidth=2, linestyle='--')
    # AI box to right
    ax.plot([9, right_x - 0.8], [5, right_y],
           color=right_colors[right_idx], alpha=0.5, linewidth=2, linestyle='--')

# Innovation outputs below
output_y = 2.5
ax.text(7, output_y, 'Innovation Examples:', fontsize=14, fontweight='bold', ha='center')

# Create output boxes
for i, (_, _, innovation) in enumerate(combinations[:3]):  # Show top 3
    y_pos = output_y - 0.5 - i * 0.6
    
    # Output box
    output_box = FancyBboxPatch((2.5, y_pos - 0.25), 9, 0.4,
                                boxstyle="round,pad=0.02",
                                facecolor='lightyellow', alpha=0.8,
                                edgecolor='orange', linewidth=1)
    ax.add_patch(output_box)
    
    ax.text(7, y_pos, f'→ {innovation}', ha='center', fontsize=11)

# Add formula
formula_text = 'Innovation = Concept A + Concept B + AI Context'
ax.text(7, 7.8, formula_text, ha='center', fontsize=13,
       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Statistics box
stats_text = (
    "Combinatorial Power:\n"
    "5 × 5 = 25 direct combinations\n"
    "With context variations: 250+ ideas\n"
    "Success rate: 15-20% viable"
)
ax.text(1, 0.5, stats_text, fontsize=11,
       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.9))

# Title
ax.text(7, 9.5, 'Combinatorial Ideation: Unexpected Connections', 
       fontsize=18, fontweight='bold', ha='center')

plt.tight_layout()

# Save the figure
plt.savefig('combinatorial_ideation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('combinatorial_ideation.png', dpi=150, bbox_inches='tight')

print("Combinatorial ideation visualization created successfully!")