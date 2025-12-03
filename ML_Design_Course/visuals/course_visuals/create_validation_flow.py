"""
Create validation flow visualization for Week 8
Shows validation stages and guardrails
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define validation stages
stages = [
    ('Input', 1.5, 8, '#2196F3'),
    ('Format\nValidation', 4, 8, '#4CAF50'),
    ('Business\nLogic', 7, 8, '#FF9800'),
    ('Safety\nChecks', 10, 8, '#F44336'),
    ('Output', 12.5, 8, '#9C27B0')
]

# Draw main flow
for i, (name, x, y, color) in enumerate(stages):
    # Draw box
    box = FancyBboxPatch((x - 0.8, y - 0.5), 1.6, 1,
                         boxstyle="round,pad=0.05",
                         facecolor=color, alpha=0.7,
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Add text
    ax.text(x, y, name, ha='center', va='center',
           fontsize=13, fontweight='bold', color='white')
    
    # Draw arrow to next stage
    if i < len(stages) - 1:
        arrow = FancyArrowPatch((x + 0.8, y), 
                               (stages[i+1][1] - 0.8, y),
                               arrowstyle='->', lw=2,
                               color='black', alpha=0.8)
        ax.add_patch(arrow)

# Add validation rules for each stage
validation_rules = [
    # Format Validation
    (4, 6.5, [
        '• JSON structure valid',
        '• Required fields present',
        '• Data types correct',
        '• Length constraints met'
    ], '#4CAF50'),
    
    # Business Logic
    (7, 6.5, [
        '• Values in valid ranges',
        '• Dependencies satisfied',
        '• Consistency checks pass',
        '• Domain rules applied'
    ], '#FF9800'),
    
    # Safety Checks
    (10, 6.5, [
        '• No harmful content',
        '• No PII exposed',
        '• No injection attacks',
        '• Rate limits enforced'
    ], '#F44336')
]

for x, y_start, rules, color in validation_rules:
    # Draw box for rules
    rules_box = FancyBboxPatch((x - 1.5, y_start - 2), 3, 2.5,
                               boxstyle="round,pad=0.05",
                               facecolor=color, alpha=0.1,
                               edgecolor=color, linewidth=1)
    ax.add_patch(rules_box)
    
    # Add rules text
    for i, rule in enumerate(rules):
        ax.text(x, y_start - 0.4 - i * 0.4, rule, 
               ha='center', fontsize=11)
    
    # Draw connection line
    ax.plot([x, x], [y_start + 0.3, 7.5], 
           color=color, linestyle='--', alpha=0.5, linewidth=1)

# Add error handling paths
error_y = 3.5
error_box = FancyBboxPatch((5, error_y - 0.5), 4, 1,
                          boxstyle="round,pad=0.05",
                          facecolor='#FFEBEE', edgecolor='#F44336',
                          linewidth=2)
ax.add_patch(error_box)
ax.text(7, error_y, 'Error Handler', ha='center', va='center',
       fontsize=13, fontweight='bold', color='#F44336')

# Draw error paths
for stage_x in [4, 7, 10]:
    arrow = FancyArrowPatch((stage_x, 7.5), (7, error_y + 0.5),
                           arrowstyle='->', lw=1.5,
                           color='#F44336', alpha=0.3,
                           linestyle='--')
    ax.add_patch(arrow)

# Error handling strategies
strategies = [
    '1. Log error details',
    '2. Return helpful message',
    '3. Suggest corrections',
    '4. Retry with defaults'
]

for i, strategy in enumerate(strategies):
    ax.text(7, 2.5 - i * 0.3, strategy, ha='center', fontsize=11)

# Add success rate box
success_box = FancyBboxPatch((0.5, 0.5), 3, 1.5,
                            boxstyle="round,pad=0.05",
                            facecolor='lightgreen', alpha=0.3,
                            edgecolor='green', linewidth=2)
ax.add_patch(success_box)
ax.text(2, 1.5, 'Success Rates', fontsize=13, fontweight='bold', ha='center')
ax.text(2, 1.1, 'Format: 95%', fontsize=11, ha='center')
ax.text(2, 0.8, 'Business: 88%', fontsize=11, ha='center')

# Add title
ax.text(7, 9.5, 'Validation Pipeline: Ensuring Quality at Every Step',
       fontsize=18, fontweight='bold', ha='center')

plt.tight_layout()

# Save the figure
plt.savefig('validation_flow.pdf', dpi=300, bbox_inches='tight')
plt.savefig('validation_flow.png', dpi=150, bbox_inches='tight')

print("Validation flow visualization created successfully!")