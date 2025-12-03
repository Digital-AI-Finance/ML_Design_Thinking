import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Simulate emotion detection on product reviews
# In real BERT, these would come from transformer model outputs
# We'll simulate realistic distributions

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define emotions and colors
emotions = ['Joy', 'Anger', 'Fear', 'Surprise', 'Sadness']
colors = ['#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b']

# ============= Plot 1: Overall Emotion Distribution =============
ax1 = axes[0, 0]

# Simulate emotion scores across 1000 reviews (realistic distribution)
np.random.seed(42)
joy_scores = np.random.beta(5, 3, 1000)  # Skewed positive
anger_scores = np.random.beta(2, 5, 1000)  # Less common
fear_scores = np.random.beta(2, 6, 1000)  # Rare
surprise_scores = np.random.beta(3, 4, 1000)  # Moderate
sadness_scores = np.random.beta(2.5, 4, 1000)  # Moderate-low

# Calculate averages
avg_emotions = [
    np.mean(joy_scores),
    np.mean(anger_scores),
    np.mean(fear_scores),
    np.mean(surprise_scores),
    np.mean(sadness_scores)
]

# Create bar chart
bars = ax1.bar(emotions, avg_emotions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels
for bar, val in zip(bars, avg_emotions):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2%}', ha='center', fontweight='bold', fontsize=11)

ax1.set_ylabel('Average Emotion Intensity', fontsize=12, fontweight='bold')
ax1.set_title('Emotion Distribution Across 1000 Reviews', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 0.8)

# Add insight callout
ax1.text(0.5, 0.7, 'Joy dominates but\nhidden pain exists', 
         fontsize=11, fontweight='bold', color='navy',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# ============= Plot 2: Emotion by Star Rating =============
ax2 = axes[0, 1]

# Simulate emotion by rating
ratings = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
x_pos = np.arange(len(ratings))
width = 0.15

# Emotion intensities by rating (realistic patterns)
emotion_by_rating = {
    'Joy': [0.05, 0.15, 0.35, 0.65, 0.85],
    'Anger': [0.75, 0.55, 0.25, 0.10, 0.05],
    'Fear': [0.35, 0.25, 0.20, 0.10, 0.05],
    'Surprise': [0.45, 0.35, 0.30, 0.35, 0.40],
    'Sadness': [0.65, 0.45, 0.30, 0.15, 0.05]
}

# Plot grouped bars
for i, (emotion, values) in enumerate(emotion_by_rating.items()):
    offset = (i - 2) * width
    ax2.bar(x_pos + offset, values, width, label=emotion, 
           color=colors[i], alpha=0.7)

ax2.set_xlabel('Star Rating', fontsize=12, fontweight='bold')
ax2.set_ylabel('Emotion Intensity', fontsize=12, fontweight='bold')
ax2.set_title('Emotions Vary by Rating Level', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(ratings)
ax2.legend(loc='upper left', fontsize=9)
ax2.set_ylim(0, 1)

# ============= Plot 3: Hidden Emotions in Positive Reviews =============
ax3 = axes[1, 0]

# Focus on 4-5 star reviews with mixed emotions
review_types = ['Pure Joy\n(45%)', 'Joy + Concern\n(25%)', 'Joy + Sadness\n(15%)', 'Joy + Anger\n(15%)']
percentages = [45, 25, 15, 15]
colors_mixed = ['#2ca02c', '#FFA500', '#FF6B6B', '#8B4513']

# Create pie chart
wedges, texts, autotexts = ax3.pie(percentages, labels=review_types, colors=colors_mixed,
                                    autopct='', startangle=90)

# Enhance text
for text in texts:
    text.set_fontsize(10)
    text.set_fontweight('bold')

ax3.set_title('Hidden Emotions in "Positive" Reviews', fontsize=14, fontweight='bold')

# Add key insight
ax3.text(-1.5, -1.3, 'Key Insight: 55% of positive reviews contain negative emotions',
         fontsize=10, fontweight='bold', color='red',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============= Plot 4: Emotion Evolution Over Time =============
ax4 = axes[1, 1]

# Simulate emotion trends over product lifecycle
months = ['Launch', 'Month 1', 'Month 2', 'Month 3', 'Month 6', 'Month 12']
x_months = np.arange(len(months))

# Realistic emotion trajectories
joy_trend = [0.75, 0.65, 0.55, 0.50, 0.55, 0.60]
anger_trend = [0.10, 0.20, 0.30, 0.35, 0.25, 0.20]
fear_trend = [0.40, 0.30, 0.25, 0.20, 0.15, 0.10]

ax4.plot(x_months, joy_trend, 'o-', color='#2ca02c', linewidth=2.5, 
         markersize=8, label='Joy')
ax4.plot(x_months, anger_trend, 's-', color='#d62728', linewidth=2.5, 
         markersize=8, label='Anger')
ax4.plot(x_months, fear_trend, '^-', color='#ff7f0e', linewidth=2.5, 
         markersize=8, label='Fear')

ax4.set_xlabel('Time Period', fontsize=12, fontweight='bold')
ax4.set_ylabel('Emotion Intensity', fontsize=12, fontweight='bold')
ax4.set_title('Emotion Trends Over Product Lifecycle', fontsize=14, fontweight='bold')
ax4.set_xticks(x_months)
ax4.set_xticklabels(months, rotation=45)
ax4.legend(loc='right')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 0.8)

# Add trend annotation
ax4.annotate('Honeymoon\nperiod ends', xy=(1, 0.65), xytext=(0.5, 0.75),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=10, color='gray')

ax4.annotate('Issues\nemerge', xy=(3, 0.35), xytext=(3.5, 0.45),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red')

# Main title
plt.suptitle('Beyond Positive/Negative: The Emotional Spectrum in Reviews', 
            fontsize=16, fontweight='bold', y=1.02)

# Add summary statistics box
summary_text = (
    "Analysis of 1000 Product Reviews:\\n"
    "• 5 core emotions detected\\n"
    "• 55% of 'positive' reviews have concerns\\n"
    "• Anger peaks at month 3 (35%)\\n"
    "• Fear decreases over time (40% → 10%)\\n"
    "• Joy recovery after initial drop"
)

fig.text(0.98, 0.02, summary_text, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('sentiment_analysis_demo.pdf', dpi=300, bbox_inches='tight')
plt.savefig('sentiment_analysis_demo.png', dpi=150, bbox_inches='tight')

print("Chart saved: sentiment_analysis_demo.pdf")
print("\\nEmotion Analysis Summary:")
print(f"Average Joy: {avg_emotions[0]:.1%}")
print(f"Average Anger: {avg_emotions[1]:.1%}") 
print(f"Average Fear: {avg_emotions[2]:.1%}")
print(f"Average Surprise: {avg_emotions[3]:.1%}")
print(f"Average Sadness: {avg_emotions[4]:.1%}")
print(f"\\nKey finding: Mixed emotions in 55% of positive reviews")