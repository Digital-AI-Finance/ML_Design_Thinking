"""
Create template-based generation visualization for Week 8
Shows how templates ensure consistency
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Left plot: Template structure
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Title
ax1.text(5, 9.5, 'Template Structure', fontsize=18, fontweight='bold', ha='center')

# Base template box
template_box = FancyBboxPatch((1, 3), 8, 5.5,
                             boxstyle="round,pad=0.05",
                             facecolor='lightblue', alpha=0.2,
                             edgecolor='blue', linewidth=2)
ax1.add_patch(template_box)

# Template sections
sections = [
    ('{{header}}', 8, 'User-specific greeting'),
    ('{{context}}', 7, 'Relevant background'),
    ('{{main_content}}', 5.5, 'Core information'),
    ('{{recommendations}}', 4, 'AI-generated suggestions'),
    ('{{footer}}', 3.5, 'Standard closing')
]

for placeholder, y, description in sections:
    # Placeholder box
    placeholder_box = FancyBboxPatch((1.5, y - 0.3), 3, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor='yellow', alpha=0.7,
                                    edgecolor='orange', linewidth=1)
    ax1.add_patch(placeholder_box)
    
    # Placeholder text
    ax1.text(3, y, placeholder, ha='center', va='center',
            fontsize=12, fontweight='bold', family='monospace')
    
    # Arrow and description
    ax1.arrow(4.5, y, 0.5, 0, head_width=0.1, head_length=0.1,
             fc='gray', ec='gray', alpha=0.5)
    ax1.text(5.2, y, description, va='center', fontsize=11)

# Template types label
ax1.text(5, 2.5, 'Template Types:', fontsize=13, fontweight='bold', ha='center')
template_types = ['• Email • Report • API Response • Dashboard Card']
ax1.text(5, 2, template_types, ha='center', fontsize=11)

# Right plot: Generation examples
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9.5, 'Template in Action', fontsize=18, fontweight='bold', ha='center')

# Show 3 different outputs from same template
outputs = [
    ('Customer Support', '#4CAF50'),
    ('Sales Report', '#FF9800'),
    ('Product Update', '#2196F3')
]

y_positions = [7.5, 5, 2.5]

for (output_type, color), y_pos in zip(outputs, y_positions):
    # Output box
    output_box = FancyBboxPatch((1, y_pos - 1), 8, 1.8,
                               boxstyle="round,pad=0.05",
                               facecolor=color, alpha=0.1,
                               edgecolor=color, linewidth=2)
    ax2.add_patch(output_box)
    
    # Output type label
    ax2.text(5, y_pos + 0.6, output_type, ha='center',
            fontsize=12, fontweight='bold', color=color)
    
    # Sample content
    if output_type == 'Customer Support':
        content = [
            'Hi Sarah,',
            'Your issue has been resolved...',
            'Recommended: Enable auto-save'
        ]
    elif output_type == 'Sales Report':
        content = [
            'Q4 Summary,',
            'Revenue increased by 23%...',
            'Recommended: Focus on Enterprise'
        ]
    else:
        content = [
            'New Features Available,',
            'Version 2.5 includes...',
            'Recommended: Update immediately'
        ]
    
    for i, line in enumerate(content):
        ax2.text(5, y_pos + 0.2 - i * 0.3, line, ha='center',
                fontsize=10, style='italic' if i > 0 else 'normal')

# Benefits box
benefits_text = (
    'Benefits:\n'
    '✓ Consistent structure\n'
    '✓ Brand compliance\n'
    '✓ Faster generation\n'
    '✓ Quality guaranteed'
)
ax2.text(8.5, 1, benefits_text, fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9))

# Statistics
stats_text = 'Efficiency: 3x faster | Error rate: -75%'
fig.text(0.5, 0.02, stats_text, ha='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# Overall title
fig.suptitle('Template-Based Generation: Consistency at Scale',
            fontsize=20, fontweight='bold', y=0.98)

plt.tight_layout()

# Save the figure
plt.savefig('template_generation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('template_generation.png', dpi=150, bbox_inches='tight')

print("Template generation visualization created successfully!")