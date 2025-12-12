"""Validation Metrics - Cross-validation and A/B testing results"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

np.random.seed(42)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left: Cross-validation scores
strategies = ['Climate\nLeaders', 'Social\nImpact', 'Governance\nExcellence',
              'ESG\nMomentum', 'Sustainable\nDividend']
cv_means = [0.82, 0.78, 0.75, 0.85, 0.72]
cv_stds = [0.05, 0.07, 0.04, 0.06, 0.08]

colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e74c3c']

bars = ax1.barh(strategies, cv_means, xerr=cv_stds, color=colors, alpha=0.7,
                capsize=4, edgecolor='gray')
ax1.set_xlabel('Cross-Validation Score', fontsize=10)
ax1.set_title('5-Fold CV Performance', fontsize=11, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Target')
ax1.legend(loc='lower right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: A/B test results
metrics = ['Returns', 'Sharpe\nRatio', 'ESG\nScore', 'Volatility']
strategy_a = [0.12, 1.2, 78, 0.15]
strategy_b = [0.09, 0.9, 65, 0.18]
x = np.arange(len(metrics))
width = 0.35

ax2.bar(x - width/2, strategy_a, width, label='ESG Strategy', color='#2ecc71', alpha=0.7)
ax2.bar(x + width/2, strategy_b, width, label='Benchmark', color='#95a5a6', alpha=0.7)

ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.set_ylabel('Value', fontsize=10)
ax2.set_title('A/B Testing: Strategy vs Benchmark', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add significance markers
for i, (a, b) in enumerate(zip(strategy_a, strategy_b)):
    if abs(a - b) / max(a, b) > 0.1:
        ax2.text(i, max(a, b) + 0.05 * max(strategy_a), '*', ha='center', fontsize=14, color='#e74c3c')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("10_validation_metrics created")
