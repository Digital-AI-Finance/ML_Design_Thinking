"""
Create dual pipeline visualization showing ML and Design Thinking pipelines
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
ml_color = '#1f77b4'  # Blue
design_color = '#ff7f0e'  # Orange
connection_color = '#2ca02c'  # Green
bg_color = '#f0f0f0'

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# ML Pipeline stages
ml_stages = ['Data', 'Preprocess', 'Model', 'Evaluate', 'Deploy']
ml_y = 4.5

# Design Pipeline stages  
design_stages = ['Empathize', 'Define', 'Ideate', 'Prototype', 'Test']
design_y = 1.5

# Box dimensions
box_width = 1.4
box_height = 0.6

# Draw ML Pipeline
for i, stage in enumerate(ml_stages):
    x = 1 + i * 1.8
    
    # Draw box
    box = FancyBboxPatch((x - box_width/2, ml_y - box_height/2),
                         box_width, box_height,
                         boxstyle="round,pad=0.05",
                         facecolor=ml_color, alpha=0.7,
                         edgecolor='darkblue', linewidth=2)
    ax.add_patch(box)
    
    # Add text
    ax.text(x, ml_y, stage, ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Draw arrow to next stage
    if i < len(ml_stages) - 1:
        arrow = FancyArrowPatch((x + box_width/2, ml_y),
                               (x + 1.8 - box_width/2, ml_y),
                               arrowstyle='->', lw=2,
                               color='darkblue', alpha=0.8)
        ax.add_patch(arrow)

# Draw Design Pipeline
for i, stage in enumerate(design_stages):
    x = 1 + i * 1.8
    
    # Draw box
    box = FancyBboxPatch((x - box_width/2, design_y - box_height/2),
                         box_width, box_height,
                         boxstyle="round,pad=0.05",
                         facecolor=design_color, alpha=0.7,
                         edgecolor='darkorange', linewidth=2)
    ax.add_patch(box)
    
    # Add text
    ax.text(x, design_y, stage, ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Draw arrow to next stage
    if i < len(design_stages) - 1:
        arrow = FancyArrowPatch((x + box_width/2, design_y),
                               (x + 1.8 - box_width/2, design_y),
                               arrowstyle='->', lw=2,
                               color='darkorange', alpha=0.8)
        ax.add_patch(arrow)

# Draw connections between pipelines
connections = [
    (0, 0, "User data informs empathy"),
    (1, 1, "Patterns define problems"),
    (2, 2, "Models generate ideas"),
    (3, 3, "Evaluation guides prototypes"),
    (4, 4, "Deployment enables testing")
]

for ml_idx, design_idx, label in connections:
    x = 1 + ml_idx * 1.8
    
    # Draw connecting arrow
    arrow = FancyArrowPatch((x, ml_y - box_height/2 - 0.1),
                           (x, design_y + box_height/2 + 0.1),
                           arrowstyle='<->', lw=1.5,
                           color=connection_color, alpha=0.6,
                           linestyle='--')
    ax.add_patch(arrow)
    
    # Add connection label
    ax.text(x + 0.05, (ml_y + design_y) / 2, label,
            fontsize=12, style='italic', color=connection_color,
            rotation=90, va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Add pipeline labels
ax.text(0.2, ml_y, 'ML\nPipeline', fontsize=16, fontweight='bold',
        color=ml_color, ha='center', va='center')
ax.text(0.2, design_y, 'Design\nPipeline', fontsize=16, fontweight='bold',
        color=design_color, ha='center', va='center')

# Add title
ax.text(5, 5.5, 'The Convergence: ML Meets Design Thinking',
        fontsize=20, fontweight='bold', ha='center')

# Add innovation callout
innovation_text = "Each connection point creates innovation opportunities"
ax.text(5, 0.5, innovation_text, fontsize=14, style='italic',
        ha='center', color=connection_color,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# Save the figure
plt.tight_layout()
plt.savefig('dual_pipeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig('dual_pipeline.png', dpi=150, bbox_inches='tight')

print("Dual pipeline visualization created successfully!")