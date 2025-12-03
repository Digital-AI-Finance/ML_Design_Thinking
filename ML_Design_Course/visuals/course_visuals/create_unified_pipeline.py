"""
Create unified pipeline visualization for Conclusion
Shows the integrated ML + Design Thinking pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define ML Pipeline stages (top)
ml_stages = [
    ('Data', 1.5, 7.5, '#2196F3'),
    ('Preprocess', 3.5, 7.5, '#03A9F4'),
    ('Model', 5.5, 7.5, '#00BCD4'),
    ('Evaluate', 7.5, 7.5, '#009688'),
    ('Deploy', 9.5, 7.5, '#4CAF50')
]

# Define Design Pipeline stages (bottom)
design_stages = [
    ('Empathize', 1.5, 2.5, '#FF9800'),
    ('Define', 3.5, 2.5, '#FF5722'),
    ('Ideate', 5.5, 2.5, '#F44336'),
    ('Prototype', 7.5, 2.5, '#E91E63'),
    ('Test', 9.5, 2.5, '#9C27B0')
]

# Draw ML Pipeline
for i, (name, x, y, color) in enumerate(ml_stages):
    # Draw box
    box = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor=color, alpha=0.7,
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Add text
    ax.text(x, y, name, ha='center', va='center',
           fontsize=13, fontweight='bold', color='white')
    
    # Draw arrow to next stage
    if i < len(ml_stages) - 1:
        arrow = FancyArrowPatch((x + 0.6, y), 
                               (ml_stages[i+1][1] - 0.6, y),
                               arrowstyle='->', lw=2,
                               color='black', alpha=0.8)
        ax.add_patch(arrow)

# Draw Design Pipeline
for i, (name, x, y, color) in enumerate(design_stages):
    # Draw box
    box = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor=color, alpha=0.7,
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Add text
    ax.text(x, y, name, ha='center', va='center',
           fontsize=13, fontweight='bold', color='white')
    
    # Draw arrow to next stage
    if i < len(design_stages) - 1:
        arrow = FancyArrowPatch((x + 0.6, y), 
                               (design_stages[i+1][1] - 0.6, y),
                               arrowstyle='->', lw=2,
                               color='black', alpha=0.8)
        ax.add_patch(arrow)

# Draw integration connections (bidirectional arrows)
connections = [
    (1.5, 'Clustering\nSegmentation'),
    (3.5, 'NLP\nContext'),
    (5.5, 'Generation\nVariations'),
    (7.5, 'Validation\nMetrics'),
    (9.5, 'A/B Testing\nEvolution')
]

for x, label in connections:
    # Vertical bidirectional arrow
    arrow_down = FancyArrowPatch((x, 7.1), (x, 2.9),
                                arrowstyle='<->', lw=2.5,
                                color='purple', alpha=0.6)
    ax.add_patch(arrow_down)
    
    # Add integration label in middle
    ax.text(x, 5, label, ha='center', va='center',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', alpha=0.9))

# Add central fusion symbol
fusion_circle = Circle((11.5, 5), 0.8, facecolor='gold', 
                       edgecolor='black', linewidth=3, alpha=0.8)
ax.add_patch(fusion_circle)
ax.text(11.5, 5, 'UNIFIED\nINNOVATION', ha='center', va='center',
       fontsize=12, fontweight='bold')

# Draw convergence arrows
for y in [7.5, 2.5]:
    arrow = FancyArrowPatch((10.1, y), (10.7, 5),
                           arrowstyle='->', lw=2,
                           color='gold', alpha=0.8)
    ax.add_patch(arrow)

# Add labels
ax.text(0.5, 7.5, 'ML\nPipeline', ha='center', va='center',
       fontsize=14, fontweight='bold', color='#2196F3')
ax.text(0.5, 2.5, 'Design\nPipeline', ha='center', va='center',
       fontsize=14, fontweight='bold', color='#FF9800')

# Add week annotations
week_labels = [
    (1.5, 0.5, 'Weeks 1-2'),
    (3.5, 0.5, 'Week 3'),
    (5.5, 0.5, 'Week 6'),
    (7.5, 0.5, 'Weeks 8-9'),
    (9.5, 0.5, 'Week 10')
]

for x, y, week in week_labels:
    ax.text(x, y, week, ha='center', va='center',
           fontsize=11, style='italic', color='gray')

# Add mastery indicators
ax.text(7, 9.2, 'Technical Mastery', fontsize=13, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.9))
ax.text(7, 0.8, 'Human-Centered Mastery', fontsize=13, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# Title
ax.text(7, 9.7, 'The Unified Innovation Pipeline', 
       fontsize=20, fontweight='bold', ha='center')

# Subtitle
ax.text(7, 9.3, 'Where Technology Amplifies Human Creativity', 
       fontsize=14, style='italic', ha='center', color='gray')

plt.tight_layout()

# Save the figure
plt.savefig('unified_pipeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig('unified_pipeline.png', dpi=150, bbox_inches='tight')

print("Unified pipeline visualization created successfully!")