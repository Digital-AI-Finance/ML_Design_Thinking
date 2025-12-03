"""
Create simpler sentiment polarity visualization for Week 3
Shows positive/neutral/negative with context shifts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left plot: Sentiment Distribution Bars
sentiment_data = {
    'Product Reviews': [65, 20, 15],  # Positive, Neutral, Negative
    'Support Tickets': [25, 35, 40],
    'Social Media': [45, 30, 25],
    'Feature Requests': [55, 25, 20],
    'Bug Reports': [15, 25, 60]
}

categories = list(sentiment_data.keys())
x = np.arange(len(categories))
width = 0.25

# Colors for sentiments
colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Yellow, Red
labels = ['Positive', 'Neutral', 'Negative']

# Create stacked bar chart
bottoms = np.zeros(len(categories))
for i, label in enumerate(labels):
    values = [sentiment_data[cat][i] for cat in categories]
    ax1.bar(x, values, width*3, bottom=bottoms, label=label, 
           color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels
    for j, (val, bot) in enumerate(zip(values, bottoms)):
        if val > 5:  # Only show if > 5%
            ax1.text(j, bot + val/2, f'{val}%', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
    
    bottoms += values

ax1.set_ylabel('Percentage', fontsize=15)
ax1.set_title('Sentiment Distribution Across Channels', fontsize=18, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=13)
ax1.legend(loc='upper right', fontsize=14)
ax1.set_ylim(0, 100)

# Right plot: Context Shifts Sentiment
ax2.axis('off')

# Title
ax2.text(0.5, 0.95, 'How Context Changes Sentiment', 
         fontsize=18, fontweight='bold', ha='center', transform=ax2.transAxes)

# Examples with shifting sentiment
examples = [
    ('Text', 'Without Context', 'With Context', 'Sentiment Shift'),
    ('"Not bad"', '?', 'After trying 10 times', 'Neutral → Positive'),
    ('"Finally works"', '?', 'After 6 month wait', 'Positive → Negative'),
    ('"Interesting choice"', '?', 'In design review', 'Neutral → Negative'),
    ('"It\'s fine"', '?', 'From power user', 'Neutral → Negative'),
    ('"Could be better"', '?', 'From first-timer', 'Negative → Positive')
]

y_start = 0.75
for i, (text, without, with_ctx, shift) in enumerate(examples):
    y_pos = y_start - i * 0.12
    
    if i == 0:  # Header
        ax2.text(0.1, y_pos, text, fontsize=14, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.4, y_pos, without, fontsize=14, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.65, y_pos, with_ctx, fontsize=14, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.9, y_pos, shift, fontsize=14, fontweight='bold', ha='right', transform=ax2.transAxes)
        # Underline
        ax2.plot([0.05, 0.95], [y_pos - 0.02, y_pos - 0.02], 'k-', 
                transform=ax2.transAxes, linewidth=2)
    else:
        # Text examples
        ax2.text(0.1, y_pos, text, fontsize=13, transform=ax2.transAxes)
        
        # Question mark for ambiguous
        ax2.text(0.4, y_pos, without, fontsize=20, ha='center', 
                transform=ax2.transAxes, color='gray')
        
        # Context
        ax2.text(0.65, y_pos, with_ctx, fontsize=12, ha='center',
                transform=ax2.transAxes, style='italic')
        
        # Arrow showing shift
        if 'Positive' in shift.split(' → ')[1]:
            arrow_color = '#4CAF50'
        elif 'Negative' in shift.split(' → ')[1]:
            arrow_color = '#F44336'
        else:
            arrow_color = '#FFC107'
            
        # Draw arrow
        ax2.annotate('', xy=(0.88, y_pos), xytext=(0.82, y_pos),
                    arrowprops=dict(arrowstyle='->', lw=2, color=arrow_color),
                    transform=ax2.transAxes)

# Add key insight box
insight_text = (
    "Key Insight:\n"
    "70% of misclassified sentiments\n"
    "are due to missing context.\n"
    "BERT understands context!"
)
ax2.text(0.5, 0.1, insight_text, fontsize=14, ha='center',
        transform=ax2.transAxes, 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Overall title
fig.suptitle('Sentiment Polarity: Simple but Context-Aware', 
            fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('sentiment_polarity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('sentiment_polarity.png', dpi=150, bbox_inches='tight')

print("Sentiment polarity visualization created successfully!")