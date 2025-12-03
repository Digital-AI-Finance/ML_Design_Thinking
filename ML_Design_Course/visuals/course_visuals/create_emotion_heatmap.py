"""
Create emotion spectrum heatmap showing emotional analysis of user feedback
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define emotions and product features
emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
features = ['Onboarding', 'Navigation', 'Performance', 'Pricing', 'Support', 
            'Features', 'Design', 'Security', 'Updates', 'Community']

# Generate synthetic emotion scores for each feature (0-100)
# Create realistic patterns
emotion_data = np.zeros((len(emotions), len(features)))

# Onboarding: High joy and anticipation for new users
emotion_data[0, 0] = 75  # Joy
emotion_data[7, 0] = 80  # Anticipation
emotion_data[2, 0] = 20  # Fear (some anxiety)

# Navigation: Mixed emotions
emotion_data[6, 1] = 45  # Anger (confusion)
emotion_data[4, 1] = 30  # Sadness
emotion_data[0, 1] = 25  # Joy

# Performance: Strong emotions
emotion_data[6, 2] = 70  # Anger (when slow)
emotion_data[5, 2] = 40  # Disgust
emotion_data[0, 2] = 15  # Joy

# Pricing: Fear and anger
emotion_data[2, 3] = 60  # Fear (cost concerns)
emotion_data[6, 3] = 50  # Anger
emotion_data[4, 3] = 40  # Sadness

# Support: Trust and joy when good
emotion_data[1, 4] = 70  # Trust
emotion_data[0, 4] = 65  # Joy
emotion_data[3, 4] = 30  # Surprise

# Features: Joy and surprise
emotion_data[0, 5] = 80  # Joy
emotion_data[3, 5] = 70  # Surprise
emotion_data[7, 5] = 60  # Anticipation

# Design: Mostly positive
emotion_data[0, 6] = 85  # Joy
emotion_data[1, 6] = 60  # Trust
emotion_data[3, 6] = 40  # Surprise

# Security: Fear and trust
emotion_data[2, 7] = 75  # Fear
emotion_data[1, 7] = 45  # Trust
emotion_data[6, 7] = 30  # Anger

# Updates: Anticipation and surprise
emotion_data[7, 8] = 85  # Anticipation
emotion_data[3, 8] = 75  # Surprise
emotion_data[0, 8] = 50  # Joy

# Community: Social emotions
emotion_data[0, 9] = 70  # Joy
emotion_data[1, 9] = 80  # Trust
emotion_data[7, 9] = 55  # Anticipation

# Add some noise for realism
emotion_data += np.random.normal(0, 5, emotion_data.shape)
emotion_data = np.clip(emotion_data, 0, 100)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Heatmap
sns.heatmap(emotion_data, annot=True, fmt='.0f', cmap='RdYlGn_r',
            xticklabels=features, yticklabels=emotions,
            cbar_kws={'label': 'Emotion Intensity (0-100)'},
            linewidths=0.5, linecolor='gray', ax=ax1)
ax1.set_title('Emotion Spectrum Across Product Features', fontsize=18, fontweight='bold')
ax1.set_xlabel('Product Features', fontsize=14)
ax1.set_ylabel('Emotions', fontsize=14)

# Rotate x labels for better readability
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Right plot: Innovation opportunities
ax2.axis('off')

# Title
ax2.text(0.5, 0.95, 'Innovation Opportunities from Emotions', 
         fontsize=18, fontweight='bold', ha='center', transform=ax2.transAxes)

# Create opportunity boxes
opportunities = [
    ('High Anger in Performance', 'Optimize speed and responsiveness', 'red', 0.85),
    ('Fear in Security', 'Add transparency and control features', 'orange', 0.70),
    ('Low Joy in Navigation', 'Redesign user flow and information architecture', 'yellow', 0.55),
    ('High Anticipation for Updates', 'Create roadmap transparency', 'lightgreen', 0.40),
    ('Trust in Community', 'Leverage peer support features', 'green', 0.25)
]

for i, (problem, solution, color, y_pos) in enumerate(opportunities):
    # Problem box
    rect = Rectangle((0.05, y_pos - 0.05), 0.4, 0.08, 
                    facecolor=color, alpha=0.3, edgecolor='black')
    ax2.add_patch(rect)
    ax2.text(0.25, y_pos, problem, fontsize=14, fontweight='bold',
            ha='center', va='center')
    
    # Arrow
    ax2.arrow(0.46, y_pos, 0.08, 0, head_width=0.02, head_length=0.02,
             fc='gray', ec='gray')
    
    # Solution box
    rect = Rectangle((0.55, y_pos - 0.05), 0.4, 0.08,
                    facecolor='lightblue', alpha=0.3, edgecolor='black')
    ax2.add_patch(rect)
    ax2.text(0.75, y_pos, solution, fontsize=13, 
            ha='center', va='center', style='italic')

# Add key insight
insight_text = (
    "Key Insight: Negative emotions reveal innovation opportunities\n"
    "87% of breakthrough features address emotional pain points"
)
ax2.text(0.5, 0.1, insight_text, fontsize=14, ha='center',
        transform=ax2.transAxes, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Overall title
fig.suptitle('Understanding Emotions = Finding Innovation Gaps', 
            fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('emotion_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('emotion_heatmap.png', dpi=150, bbox_inches='tight')

print("Emotion heatmap visualization created successfully!")
print(f"Analyzed {len(emotions)} emotions across {len(features)} features")