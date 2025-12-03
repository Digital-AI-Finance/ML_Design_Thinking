import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Generate sample data
n_samples = 300
X, y_true = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                       cluster_std=1.0, random_state=42)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Before: Chaos
ax1 = axes[0]
ax1.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=30, edgecolors='none')
ax1.set_title('Before Clustering\n10,000 users = overwhelming', fontsize=12, fontweight='bold')
ax1.set_xlabel('Usage Pattern 1')
ax1.set_ylabel('Usage Pattern 2')
ax1.grid(True, alpha=0.3)
ax1.text(0.5, -0.15, 'Chaos: No clear patterns visible', 
        transform=ax1.transAxes, ha='center', fontsize=10, style='italic')

# During: Process
ax2 = axes[1]
colors_process = ['#ffcccc', '#ccffcc', '#ccccff']
for i in range(3):
    mask = y_true == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors_process[i], 
               alpha=0.3, s=30, edgecolors='gray', linewidth=0.5)
ax2.set_title('Clustering Process\nML finds natural groups', fontsize=12, fontweight='bold')
ax2.set_xlabel('Usage Pattern 1')
ax2.set_ylabel('Usage Pattern 2')
ax2.grid(True, alpha=0.3)
# Add arrows showing grouping
ax2.annotate('Grouping...', xy=(5, 5), xytext=(8, 8),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
            fontsize=10)
ax2.text(0.5, -0.15, 'Processing: Algorithm finding patterns', 
        transform=ax2.transAxes, ha='center', fontsize=10, style='italic')

# After: Clear segments
ax3 = axes[2]
colors_final = ['#e74c3c', '#3498db', '#2ecc71']
labels = ['Power Users', 'Regular Users', 'Casual Users']
for i in range(3):
    mask = y_true == i
    ax3.scatter(X[mask, 0], X[mask, 1], c=colors_final[i], 
               alpha=0.7, s=30, label=labels[i], edgecolors='white', linewidth=0.5)
ax3.set_title('After Clustering\n3 clear user types!', fontsize=12, fontweight='bold')
ax3.set_xlabel('Usage Pattern 1')
ax3.set_ylabel('Usage Pattern 2')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', fontsize=10)
ax3.text(0.5, -0.15, 'Clarity: Distinct user segments identified', 
        transform=ax3.transAxes, ha='center', fontsize=10, style='italic')

plt.suptitle('From Chaos to Clarity Through Clustering', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('chaos_to_clarity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('chaos_to_clarity.png', dpi=150, bbox_inches='tight')
plt.close()

print("Chaos to clarity visualization created!")