"""Feature Extraction Flow - Supervised learning feature engineering"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

fig, ax = plt.subplots(figsize=(10, 6))

# Data sources (left)
sources = ['Sustainability\nReports', 'News\nArticles', 'Social\nMedia', 'Regulatory\nFilings']
source_y = [4.5, 3.5, 2.5, 1.5]

# Feature extraction (middle)
features = ['Sentiment\nScores', 'ESG\nRatings', 'Risk\nIndicators', 'Trend\nMetrics']
feature_y = [4.5, 3.5, 2.5, 1.5]

# Output (right)
output_label = '100 Features'

# Draw source boxes
for i, (src, y) in enumerate(zip(sources, source_y)):
    rect = mpatches.FancyBboxPatch((0.5, y-0.3), 1.8, 0.6, boxstyle="round,pad=0.05",
                                    facecolor='#3498db', alpha=0.7, edgecolor='#2980b9')
    ax.add_patch(rect)
    ax.text(1.4, y, src, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Draw feature boxes
for i, (feat, y) in enumerate(zip(features, feature_y)):
    rect = mpatches.FancyBboxPatch((4, y-0.3), 1.8, 0.6, boxstyle="round,pad=0.05",
                                    facecolor='#2ecc71', alpha=0.7, edgecolor='#27ae60')
    ax.add_patch(rect)
    ax.text(4.9, y, feat, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Draw output box
rect = mpatches.FancyBboxPatch((7.5, 2.5), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='#9467bd', alpha=0.7, edgecolor='#7d3c98')
ax.add_patch(rect)
ax.text(8.5, 3, output_label, ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# Draw arrows from sources to features
for sy in source_y:
    for fy in feature_y:
        ax.annotate('', xy=(4, fy), xytext=(2.3, sy),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))

# Draw arrows from features to output
for fy in feature_y:
    ax.annotate('', xy=(7.5, 3), xytext=(5.8, fy),
               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))

# Labels
ax.text(1.4, 5.3, 'Data Sources', ha='center', fontsize=10, fontweight='bold', color='#3498db')
ax.text(4.9, 5.3, 'ML Extraction', ha='center', fontsize=10, fontweight='bold', color='#2ecc71')
ax.text(8.5, 3.8, 'Feature Vector', ha='center', fontsize=10, fontweight='bold', color='#9467bd')

ax.set_xlim(0, 10)
ax.set_ylim(0.5, 5.8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Feature Engineering: From Raw Data to ML Features', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("05_feature_extraction created")
