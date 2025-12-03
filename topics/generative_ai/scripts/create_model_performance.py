import matplotlib.pyplot as plt
import numpy as np

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Model data (size in billions of parameters, performance score 0-100)
models = {
    'Text Models': {
        'names': ['GPT-2', 'GPT-3', 'Claude', 'GPT-4', 'Gemini', 'Llama-2'],
        'sizes': [1.5, 175, 130, 1760, 540, 70],
        'performance': [65, 85, 88, 95, 92, 82],
        'color': '#3498db',
        'marker': 'o'
    },
    'Image Models': {
        'names': ['DALL-E', 'SD 1.5', 'SD XL', 'Midjourney', 'DALL-E 3', 'Imagen'],
        'sizes': [12, 0.9, 6.6, 20, 40, 35],
        'performance': [70, 75, 85, 90, 93, 88],
        'color': '#e74c3c',
        'marker': 's'
    },
    'Code Models': {
        'names': ['Codex', 'Copilot', 'CodeLlama', 'StarCoder', 'DeepSeek', 'CodeT5'],
        'sizes': [12, 12, 34, 15, 33, 0.77],
        'performance': [78, 80, 83, 76, 85, 72],
        'color': '#2ecc71',
        'marker': '^'
    }
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 9))

# Plot each category
for category, data in models.items():
    # Convert to log scale for better visualization
    sizes_log = np.log10(np.array(data['sizes']) + 0.1)  # Add 0.1 to avoid log(0)

    ax.scatter(sizes_log, data['performance'],
               s=200, alpha=0.7,
               c=data['color'],
               marker=data['marker'],
               label=category,
               edgecolors='black',
               linewidth=1.5)

    # Add model names
    for i, name in enumerate(data['names']):
        ax.annotate(name,
                   (sizes_log[i], data['performance'][i]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9,
                   fontweight='bold')

# Add trend line
all_sizes = []
all_performance = []
for data in models.values():
    all_sizes.extend(np.log10(np.array(data['sizes']) + 0.1))
    all_performance.extend(data['performance'])

z = np.polyfit(all_sizes, all_performance, 2)
p = np.poly1d(z)
x_trend = np.linspace(min(all_sizes), max(all_sizes), 100)
ax.plot(x_trend, p(x_trend), '--', alpha=0.5, linewidth=2, color='gray',
        label='Performance Trend')

# Add efficiency zones
ax.axhspan(85, 100, alpha=0.1, color='green', label='High Performance')
ax.axhspan(70, 85, alpha=0.1, color='yellow')
ax.axhspan(0, 70, alpha=0.1, color='red')

# Add vertical lines for size categories
ax.axvline(x=np.log10(1), color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=np.log10(10), color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=np.log10(100), color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=np.log10(1000), color='gray', linestyle=':', alpha=0.5)

# Add size category labels
y_pos = 60
ax.text(np.log10(0.5), y_pos, 'Small\n<1B', ha='center', fontsize=9, color='#7f8c8d')
ax.text(np.log10(5), y_pos, 'Medium\n1-10B', ha='center', fontsize=9, color='#7f8c8d')
ax.text(np.log10(50), y_pos, 'Large\n10-100B', ha='center', fontsize=9, color='#7f8c8d')
ax.text(np.log10(500), y_pos, 'XL\n>100B', ha='center', fontsize=9, color='#7f8c8d')

# Styling
ax.set_xlabel('Model Size (log scale, billions of parameters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Score (0-100)', fontsize=12, fontweight='bold')
ax.set_title('GenAI Model Size vs Performance Analysis', fontsize=14, fontweight='bold', pad=20)

# Custom x-axis labels
x_ticks = [np.log10(0.1), np.log10(1), np.log10(10), np.log10(100), np.log10(1000)]
x_labels = ['0.1B', '1B', '10B', '100B', '1000B']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)

ax.set_ylim(55, 100)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=10)

# Add annotation for sweet spot
ax.annotate('Sweet Spot:\n10-100B params\n85-90% performance',
            xy=(np.log10(50), 87),
            xytext=(np.log10(200), 75),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#f39c12', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2',
                           color='#e67e22', lw=2))

# Save the figure
plt.tight_layout()
plt.savefig('../charts/model_size_performance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/model_size_performance.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created model_size_performance chart")