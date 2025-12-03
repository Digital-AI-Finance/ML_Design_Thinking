import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define positions for nodes
root = (0.5, 0.9)
level1 = [(0.25, 0.75), (0.75, 0.75)]
level2 = [(0.15, 0.55), (0.35, 0.55), (0.65, 0.55), (0.85, 0.55)]
level3 = [(0.1, 0.35), (0.2, 0.35), (0.3, 0.35), (0.4, 0.35),
          (0.6, 0.35), (0.7, 0.35), (0.8, 0.35), (0.9, 0.35)]
outcomes = [(0.15, 0.15), (0.35, 0.15), (0.65, 0.15), (0.85, 0.15)]

# Colors for different decision paths
color_safe = '#2ecc71'
color_caution = '#f39c12'
color_danger = '#e74c3c'
color_neutral = '#95a5a6'

# Draw root node
root_box = FancyBboxPatch((root[0]-0.12, root[1]-0.04), 0.24, 0.08,
                          boxstyle="round,pad=0.01",
                          facecolor='#3498db', edgecolor='black', linewidth=2)
ax.add_patch(root_box)
ax.text(root[0], root[1], 'AI-Generated Content\nDecision',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Level 1: Purpose Check
purposes = ['Commercial Use', 'Personal/Educational']
colors1 = [color_caution, color_safe]
for i, (pos, purpose, color) in enumerate(zip(level1, purposes, colors1)):
    box = FancyBboxPatch((pos[0]-0.1, pos[1]-0.03), 0.2, 0.06,
                         boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(box)
    ax.text(pos[0], pos[1], purpose, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw connection from root
    ax.plot([root[0], pos[0]], [root[1]-0.04, pos[1]+0.03],
            'k-', linewidth=2, alpha=0.5)

# Level 2: Content Type Check
content_types = ['Brand/Logo', 'Human Likeness', 'Original Art', 'Generic Assets']
colors2 = [color_danger, color_caution, color_caution, color_safe]
positions2 = [level2[0], level2[1], level2[2], level2[3]]

for i, (pos, content, color) in enumerate(zip(positions2, content_types, colors2)):
    box = FancyBboxPatch((pos[0]-0.08, pos[1]-0.03), 0.16, 0.06,
                         boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(box)
    ax.text(pos[0], pos[1], content, ha='center', va='center', fontsize=9)

    # Draw connections
    if i < 2:
        ax.plot([level1[0][0], pos[0]], [level1[0][1]-0.03, pos[1]+0.03],
                'k-', linewidth=1.5, alpha=0.5)
    else:
        ax.plot([level1[1][0], pos[0]], [level1[1][1]-0.03, pos[1]+0.03],
                'k-', linewidth=1.5, alpha=0.5)

# Level 3: Attribution/Disclosure
attributions = ['No Attribution\nNeeded', 'Disclose AI\nGenerated',
                'Credit + Verify\nRights', 'Get Permission\nFirst',
                'Educational\nFair Use', 'Personal Use\nOK',
                'Attribution\nRecommended', 'Free to Use']
colors3 = [color_danger, color_caution, color_caution, color_danger,
          color_safe, color_safe, color_safe, color_safe]

for i, (pos, attr, color) in enumerate(zip(level3, attributions, colors3)):
    box = FancyBboxPatch((pos[0]-0.05, pos[1]-0.03), 0.1, 0.06,
                         boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
    ax.add_patch(box)
    ax.text(pos[0], pos[1], attr, ha='center', va='center', fontsize=8)

    # Draw connections from level 2
    source_idx = i // 2
    ax.plot([positions2[source_idx][0], pos[0]],
            [positions2[source_idx][1]-0.03, pos[1]+0.03],
            'k-', linewidth=1, alpha=0.4)

# Final Outcomes
outcome_texts = ['STOP:\nLegal Risk', 'PROCEED:\nWith Caution',
                'SAFE:\nMinimal Risk', 'GO:\nBest Practice']
outcome_colors = [color_danger, color_caution, color_safe, color_safe]

for pos, text, color in zip(outcomes, outcome_texts, outcome_colors):
    box = FancyBboxPatch((pos[0]-0.08, pos[1]-0.04), 0.16, 0.08,
                         boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(pos[0], pos[1], text, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

# Add legend
legend_elements = [
    patches.Rectangle((0,0),1,1, facecolor=color_safe, label='Safe to Proceed'),
    patches.Rectangle((0,0),1,1, facecolor=color_caution, label='Caution Required'),
    patches.Rectangle((0,0),1,1, facecolor=color_danger, label='High Risk/Stop')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add decision criteria annotations
ax.text(0.05, 0.95, 'Key Questions:', fontsize=10, fontweight='bold')
ax.text(0.05, 0.92, '1. Commercial vs Personal?', fontsize=9)
ax.text(0.05, 0.89, '2. Protected content?', fontsize=9)
ax.text(0.05, 0.86, '3. Attribution needed?', fontsize=9)
ax.text(0.05, 0.83, '4. Legal compliance?', fontsize=9)

# Add best practices box
ax.text(0.95, 0.05, 'Best Practices:\n- Always disclose AI use\n- Verify ownership rights\n- Document generation process\n- Keep prompt records',
        fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', alpha=0.8))

# Title and styling
ax.set_title('Ethical Decision Tree for AI-Generated Content',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Save the figure
plt.tight_layout()
plt.savefig('../charts/ethics_decision_tree.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/ethics_decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created ethics_decision_tree chart")