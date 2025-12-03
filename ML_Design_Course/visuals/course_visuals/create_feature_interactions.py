"""
Create feature interaction effects visualization for Week 7
Shows how features interact to impact outcomes
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Features for interaction
features = ['Speed', 'Usability', 'Price', 'Features', 'Support', 'Design']
n_features = len(features)

# Create interaction matrix (symmetric)
interaction_matrix = np.zeros((n_features, n_features))

# Define strong interactions
strong_interactions = [
    (0, 1, 0.85),  # Speed × Usability
    (1, 3, 0.75),  # Usability × Features
    (2, 3, -0.65), # Price × Features (negative)
    (2, 4, 0.70),  # Price × Support
    (1, 5, 0.80),  # Usability × Design
    (3, 5, 0.60),  # Features × Design
    (0, 4, 0.55),  # Speed × Support
]

# Fill the matrix
for i, j, value in strong_interactions:
    interaction_matrix[i, j] = value
    interaction_matrix[j, i] = value

# Add some medium interactions
np.random.seed(42)
for i in range(n_features):
    for j in range(i+1, n_features):
        if interaction_matrix[i, j] == 0:
            interaction_matrix[i, j] = np.random.uniform(-0.3, 0.3)
            interaction_matrix[j, i] = interaction_matrix[i, j]

# Left plot: Interaction heatmap
sns.heatmap(interaction_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            xticklabels=features, yticklabels=features,
            center=0, vmin=-1, vmax=1,
            cbar_kws={'label': 'Interaction Strength'},
            square=True, linewidths=1, linecolor='gray', ax=ax1)
ax1.set_title('Feature Interaction Matrix', fontsize=18, fontweight='bold')
ax1.set_xlabel('Feature', fontsize=14)
ax1.set_ylabel('Feature', fontsize=14)

# Rotate labels
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=13)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=13)

# Right plot: Impact visualization
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9.5, 'Key Interaction Effects', fontsize=18, fontweight='bold', ha='center')

# Show top interactions with explanations
interactions_explained = [
    ('Speed × Usability', '+0.85', 'Fast but hard to use = Frustration', 'red'),
    ('Usability × Design', '+0.80', 'Good UX + Good design = Delight', 'green'),
    ('Price × Features', '-0.65', 'High price needs more features', 'orange'),
    ('Price × Support', '+0.70', 'Premium price justifies good support', 'blue'),
]

y_start = 8
for i, (interaction, strength, explanation, color) in enumerate(interactions_explained):
    y_pos = y_start - i * 1.8
    
    # Interaction name
    ax2.text(1, y_pos, interaction, fontsize=14, fontweight='bold')
    
    # Strength value
    ax2.text(4, y_pos, strength, fontsize=16, fontweight='bold', 
            color=color, ha='center')
    
    # Arrow
    ax2.arrow(4.5, y_pos, 0.8, 0, head_width=0.15, head_length=0.1,
             fc=color, ec=color, alpha=0.5)
    
    # Explanation
    ax2.text(5.5, y_pos, explanation, fontsize=12, va='center')

# Add insight box
insight_text = (
    "Design Implications:\n"
    "• Don't optimize features in isolation\n"
    "• Strong positive interactions = Focus areas\n"
    "• Negative interactions = Trade-offs to manage\n"
    "• 60% of user satisfaction from interactions"
)
ax2.text(5, 2, insight_text, fontsize=13, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Overall title
fig.suptitle('Feature Interactions: The Hidden Multipliers', 
            fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('feature_interactions.pdf', dpi=300, bbox_inches='tight')
plt.savefig('feature_interactions.png', dpi=150, bbox_inches='tight')

print("Feature interactions visualization created successfully!")