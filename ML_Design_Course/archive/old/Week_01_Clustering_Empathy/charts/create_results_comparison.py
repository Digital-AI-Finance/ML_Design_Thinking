import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define colors
traditional_color = '#d62728'
ml_color = '#2ca02c'

# ============= Plot 1: Scale Comparison =============
ax1 = axes[0, 0]

methods = ['Traditional\nEmpathy', 'ML-Powered\nEmpathy']
users_reached = [20, 10000]
colors = [traditional_color, ml_color]

bars = ax1.bar(methods, users_reached, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels
for bar, val in zip(bars, users_reached):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,} users', ha='center', fontweight='bold', fontsize=12)

ax1.set_ylabel('Number of Users Analyzed', fontsize=12, fontweight='bold')
ax1.set_title('Scale: How Many Voices We Hear', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 11000)

# Add improvement callout
ax1.annotate('500x\nmore users!', xy=(1, 10000), xytext=(0.5, 8000),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, fontweight='bold', color='green', ha='center')

# ============= Plot 2: Time Comparison =============
ax2 = axes[0, 1]

time_data = [14*8, 4]  # 14 days * 8 hours, 4 hours
time_labels = ['2 weeks\n(112 hours)', '4 hours']

bars2 = ax2.bar(methods, time_data, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

for bar, val, label in zip(bars2, time_data, time_labels):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             label, ha='center', fontweight='bold', fontsize=11)

ax2.set_ylabel('Time Required', fontsize=12, fontweight='bold')
ax2.set_title('Speed: From Weeks to Hours', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 120)

# Speed improvement
ax2.text(0.5, 60, '96.4%\nfaster', fontsize=14, fontweight='bold', 
        color='green', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# ============= Plot 3: Depth vs Breadth Tradeoff =============
ax3 = axes[1, 0]

# Create scatter plot showing tradeoff
traditional_methods = {
    'Interviews': (9, 20, 50),
    'Focus Groups': (7, 50, 40),
    'Observations': (10, 10, 60),
    'Surveys': (3, 1000, 30)
}

ml_methods = {
    'Clustering': (7, 10000, 100),
    'Sentiment Analysis': (8, 10000, 80),
    'Topic Modeling': (6, 10000, 70),
    'Pattern Discovery': (9, 10000, 90)
}

# Plot traditional methods
for method, (depth, scale, size) in traditional_methods.items():
    ax3.scatter(np.log10(scale), depth, s=size*5, c=traditional_color, 
               alpha=0.6, edgecolors='black', linewidth=1)
    ax3.annotate(method, (np.log10(scale), depth), fontsize=8, ha='center')

# Plot ML methods
for method, (depth, scale, size) in ml_methods.items():
    ax3.scatter(np.log10(scale), depth, s=size*5, c=ml_color, 
               alpha=0.6, edgecolors='black', linewidth=1)
    ax3.annotate(method, (np.log10(scale), depth), fontsize=8, ha='center')

ax3.set_xlabel('Scale (log10 of users)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Depth of Understanding (1-10)', fontsize=12, fontweight='bold')
ax3.set_title('The Empathy Tradeoff: Depth vs Scale', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add legend
ax3.scatter([], [], c=traditional_color, s=100, label='Traditional')
ax3.scatter([], [], c=ml_color, s=100, label='ML-Powered')
ax3.legend(loc='lower left')

# ============= Plot 4: Insights Discovered =============
ax4 = axes[1, 1]

categories = ['User\nSegments', 'Pain\nPoints', 'Hidden\nPatterns', 'Actionable\nInsights']
traditional_insights = [2, 5, 1, 3]
ml_insights = [5, 23, 12, 18]

x = np.arange(len(categories))
width = 0.35

bars_trad = ax4.bar(x - width/2, traditional_insights, width, 
                    label='Traditional', color=traditional_color, alpha=0.7)
bars_ml = ax4.bar(x + width/2, ml_insights, width, 
                 label='ML-Powered', color=ml_color, alpha=0.7)

ax4.set_xlabel('Type of Insight', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number Discovered', fontsize=12, fontweight='bold')
ax4.set_title('Quality: What We Discover', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()

# Add value labels
for bars in [bars_trad, bars_ml]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{int(height)}', ha='center', fontsize=9)

# Main title
plt.suptitle('Traditional vs ML-Powered Empathy: The Numbers', 
            fontsize=16, fontweight='bold', y=1.02)

# Add summary box
summary_text = (
    "Key Advantages of ML-Powered Empathy:\n"
    "• 500x more users analyzed\n"
    "• 28x faster insights\n"
    "• 100x cost reduction\n"
    "• Discovers hidden patterns humans miss"
)

fig.text(0.98, 0.02, summary_text, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('results_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results_comparison.png', dpi=150, bbox_inches='tight')

print("Chart saved: results_comparison.pdf")
print("\nComparison Summary:")
print("Scale: 20 → 10,000 users (500x)")
print("Time: 2 weeks → 4 hours (28x faster)")
print("Cost: $5,000 → $50 (100x cheaper)")
print("Segments found: 2 → 5")
print("Total insights: 11 → 58")