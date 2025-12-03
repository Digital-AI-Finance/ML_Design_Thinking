"""
Create temperature control visualization for Week 6
Shows how temperature affects AI creativity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 8))

# Temperature settings
temps = [0.3, 0.7, 1.0]
titles = ['Conservative\n(Temperature = 0.3)', 
          'Balanced\n(Temperature = 0.7)', 
          'Creative\n(Temperature = 1.0)']
colors = ['#2196F3', '#FF9800', '#F44336']  # Blue, Orange, Red

# Prompt example
prompt = "Design a mobile app for..."

# Example outputs for each temperature
outputs = [
    # Temperature 0.3 - Conservative
    [
        "...task management",
        "...expense tracking", 
        "...calendar scheduling",
        "...note taking",
        "...contact management"
    ],
    # Temperature 0.7 - Balanced  
    [
        "...mindful breathing",
        "...plant care reminders",
        "...local food sharing",
        "...skill bartering",
        "...dream journaling"
    ],
    # Temperature 1.0 - Creative
    [
        "...translating pet emotions",
        "...finding lost socks",
        "...rating cloud shapes",
        "...virtual time capsules",
        "...synchronized yawning"
    ]
]

# Characteristics for each temperature
characteristics = [
    ["✓ Predictable", "✓ Safe choices", "✓ Proven concepts", "✓ Low risk"],
    ["✓ Novel yet practical", "✓ Balanced innovation", "✓ Feasible ideas", "✓ Medium risk"],
    ["✓ Unexpected connections", "✓ High creativity", "✓ Breakthrough potential", "✓ High risk"]
]

for idx, (ax, temp, title, color, output_list, chars) in enumerate(zip(axes, temps, titles, colors, outputs, characteristics)):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title with temperature
    ax.text(5, 9.5, title, ha='center', fontsize=16, fontweight='bold', color=color)
    
    # Temperature gauge visual
    gauge_y = 8.5
    gauge_width = 6
    gauge_x = 2
    
    # Draw temperature bar
    rect = FancyBboxPatch((gauge_x, gauge_y - 0.3), gauge_width, 0.6,
                          boxstyle="round,pad=0.02",
                          facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Fill based on temperature
    fill_width = gauge_width * temp
    fill_rect = FancyBboxPatch((gauge_x, gauge_y - 0.3), fill_width, 0.6,
                               boxstyle="round,pad=0.02",
                               facecolor=color, alpha=0.7, edgecolor='none')
    ax.add_patch(fill_rect)
    
    # Prompt box
    ax.text(5, 7.5, 'Prompt:', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 7, f'"{prompt}"', fontsize=13, ha='center', style='italic')
    
    # Output examples box
    box_y = 5.5
    box = FancyBboxPatch((1, box_y - 2), 8, 3.5,
                         boxstyle="round,pad=0.1",
                         facecolor=color, alpha=0.1, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    
    ax.text(5, box_y + 1.2, 'Example Outputs:', fontsize=14, fontweight='bold', ha='center')
    
    # List outputs
    for i, output in enumerate(output_list):
        y_pos = box_y + 0.5 - i * 0.5
        ax.text(5, y_pos, output, fontsize=12, ha='center')
    
    # Characteristics
    char_y = 1.8
    for i, char in enumerate(chars):
        ax.text(5, char_y - i * 0.35, char, fontsize=11, ha='center', color='darkgreen')

# Overall title
fig.suptitle('Temperature Control: Balancing Creativity and Coherence', 
            fontsize=20, fontweight='bold', y=0.98)

# Add bottom guidance
fig.text(0.5, 0.02, 
         'Use Case: Low temp for production | Medium for exploration | High for brainstorming',
         ha='center', fontsize=14, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()

# Save the figure
plt.savefig('temperature_control.pdf', dpi=300, bbox_inches='tight')
plt.savefig('temperature_control.png', dpi=150, bbox_inches='tight')

print("Temperature control visualization created successfully!")