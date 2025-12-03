"""
Create SHAP waterfall chart showing feature impact on user satisfaction
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define features and their SHAP values (impact on user satisfaction)
features = [
    ('Base Value', 0, 6.5, 'neutral'),  # Starting point
    ('Response Time', -0.85, 0, 'negative'),
    ('Ease of Use', 0.92, 0, 'positive'),
    ('Feature Completeness', 0.45, 0, 'positive'),
    ('Customization', 0.38, 0, 'positive'),
    ('Price Value', -0.25, 0, 'negative'),
    ('Documentation', 0.22, 0, 'positive'),
    ('Mobile Experience', -0.42, 0, 'negative'),
    ('Community Support', 0.28, 0, 'positive'),
    ('Update Frequency', 0.15, 0, 'positive'),
    ('Data Security', 0.33, 0, 'positive'),
    ('Integration Options', -0.18, 0, 'negative'),
    ('Performance Stability', 0.41, 0, 'positive'),
]

# Calculate cumulative values
cumulative = [6.5]  # Start with base value
for i in range(1, len(features)):
    cumulative.append(cumulative[-1] + features[i][1])

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Colors for positive/negative/neutral
colors = {
    'positive': '#2ca02c',
    'negative': '#d62728',
    'neutral': '#7f7f7f'
}

# Draw waterfall bars
bar_width = 0.6
x_positions = np.arange(len(features))

for i, (name, value, _, impact_type) in enumerate(features):
    if i == 0:  # Base value
        # Draw the base value bar
        bar = ax.bar(i, cumulative[0], bar_width, 
                    color=colors[impact_type], alpha=0.7,
                    edgecolor='black', linewidth=1)
        # Add value label
        ax.text(i, cumulative[0] + 0.1, f'{cumulative[0]:.1f}',
               ha='center', fontsize=13, fontweight='bold')
    else:
        # Determine bar position and height
        if value >= 0:
            bottom = cumulative[i-1]
            height = value
        else:
            bottom = cumulative[i]
            height = abs(value)
        
        # Draw the bar
        bar = ax.bar(i, height, bar_width, bottom=bottom,
                    color=colors[impact_type], alpha=0.7,
                    edgecolor='black', linewidth=1)
        
        # Add connecting line to previous bar
        ax.plot([i-1 + bar_width/2, i - bar_width/2], 
               [cumulative[i-1], cumulative[i-1]],
               'k--', alpha=0.3, linewidth=1)
        
        # Add value label
        label_y = bottom + height/2
        ax.text(i, label_y, f'{value:+.2f}' if value != 0 else '0',
               ha='center', va='center', fontsize=13, fontweight='bold',
               color='white' if abs(value) > 0.3 else 'black')
        
        # Add cumulative value at top
        if i == len(features) - 1:  # Last bar
            ax.text(i, cumulative[i] + 0.1, f'Final: {cumulative[i]:.1f}',
                   ha='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='yellow', alpha=0.7))

# Customize x-axis
ax.set_xticks(x_positions)
ax.set_xticklabels([f[0] for f in features], rotation=45, ha='right', fontsize=14)

# Labels and title
ax.set_ylabel('User Satisfaction Score', fontsize=15)
ax.set_title('Feature Impact Analysis: What Drives User Satisfaction?', 
            fontsize=18, fontweight='bold', pad=20)

# Add grid
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# Add legend
legend_elements = [
    patches.Patch(color=colors['positive'], label='Positive Impact', alpha=0.7),
    patches.Patch(color=colors['negative'], label='Negative Impact', alpha=0.7),
    patches.Patch(color=colors['neutral'], label='Base Value', alpha=0.7)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=13)

# Add innovation insights
insights_text = (
    "Innovation Priorities:\n"
    "1. Fix: Response Time (biggest negative)\n"
    "2. Enhance: Ease of Use (biggest positive)\n"
    "3. Improve: Mobile Experience\n"
    "ROI: Focus on top 3 features = 60% of impact"
)
ax.text(0.98, 0.5, insights_text, transform=ax.transAxes,
       fontsize=14, ha='right', va='center',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Add threshold line for target satisfaction
target = 8.0
ax.axhline(y=target, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax.text(len(features)-1, target + 0.1, 'Target: 8.0', 
       ha='right', fontsize=13, color='green', fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('shap_waterfall.pdf', dpi=300, bbox_inches='tight')
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')

print("SHAP waterfall chart created successfully!")
print(f"Final satisfaction score: {cumulative[-1]:.2f}")
print(f"Total features analyzed: {len(features)-1}")