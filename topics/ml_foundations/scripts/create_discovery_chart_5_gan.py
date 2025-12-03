"""
Chart 5: The GAN Evolution (Time Series Dashboard)
Shows generator/discriminator training dynamics over epochs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
mlblue = '#1f77b4'
mlorange = '#ff7f0e'
mlgreen = '#2ca02c'
mlred = '#d62728'
mlpurple = '#9467bd'
mlgray = '#7f7f7f'

# Create figure with custom grid
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

# Simulate GAN training dynamics
epochs = np.arange(1, 101)

# Generator loss (high initially, decreases)
gen_loss = 8 * np.exp(-epochs/20) + 2 + np.random.randn(100) * 0.3
gen_loss = np.clip(gen_loss, 1, 10)

# Discriminator loss (starts low, increases, then balances)
disc_loss = 2 + 3 * (1 - np.exp(-epochs/15)) + np.random.randn(100) * 0.3
disc_loss = np.clip(disc_loss, 1, 10)

# Realism score (generator quality)
realism = 100 * (1 - np.exp(-epochs/25))
realism = np.clip(realism, 0, 100)

# Key epochs to show
key_epochs = [1, 10, 50, 100]
realism_at_epochs = [realism[e-1] for e in key_epochs]

# Top row: Generated samples quality progression
sample_texts = ['Noise\nblob', 'Vague\nshape', 'Recognizable\nfeatures', 'Realistic\nimage']

for i, epoch in enumerate(key_epochs):
    ax = fig.add_subplot(gs[0, i])

    # Simulate quality improvement with circles
    if i == 0:  # Epoch 1: Random noise
        np.random.seed(42)
        noise_x = np.random.rand(100) * 10
        noise_y = np.random.rand(100) * 10
        ax.scatter(noise_x, noise_y, c=mlgray, s=20, alpha=0.3)
        quality_color = mlgray
    elif i == 1:  # Epoch 10: Some structure
        theta = np.linspace(0, 2*np.pi, 100) + np.random.randn(100)*0.5
        r = 3 + np.random.randn(100)*0.8
        ax.scatter(5 + r*np.cos(theta), 5 + r*np.sin(theta),
                  c=mlorange, s=30, alpha=0.5)
        quality_color = mlorange
    elif i == 2:  # Epoch 50: Clear shape
        theta = np.linspace(0, 2*np.pi, 100) + np.random.randn(100)*0.2
        r = 3 + np.random.randn(100)*0.3
        ax.scatter(5 + r*np.cos(theta), 5 + r*np.sin(theta),
                  c=mlblue, s=40, alpha=0.7)
        quality_color = mlblue
    else:  # Epoch 100: High quality
        theta = np.linspace(0, 2*np.pi, 100) + np.random.randn(100)*0.05
        r = 3 + np.random.randn(100)*0.1
        ax.scatter(5 + r*np.cos(theta), 5 + r*np.sin(theta),
                  c=mlgreen, s=50, alpha=0.9, edgecolors='black', linewidth=0.5)
        quality_color = mlgreen

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'EPOCH {epoch}', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.02, sample_texts[i], transform=ax.transAxes,
           fontsize=10, ha='center', fontweight='bold')

    # Add realism score
    ax.text(0.5, 0.92, f'Realism: {realism_at_epochs[i]:.0f}%',
           transform=ax.transAxes, fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.6))
    ax.spines['top'].set_color(quality_color)
    ax.spines['right'].set_color(quality_color)
    ax.spines['bottom'].set_color(quality_color)
    ax.spines['left'].set_color(quality_color)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

# Middle row: Loss dynamics
ax_loss = fig.add_subplot(gs[1, :])

ax_loss.plot(epochs, gen_loss, color=mlpurple, linewidth=3,
            label='Generator Loss', alpha=0.8)
ax_loss.plot(epochs, disc_loss, color=mlgreen, linewidth=3,
            label='Discriminator Loss', alpha=0.8)

# Mark equilibrium point
equilibrium_idx = np.argmin(np.abs(gen_loss - disc_loss))
ax_loss.scatter(epochs[equilibrium_idx], gen_loss[equilibrium_idx],
               s=300, c='gold', marker='*', edgecolors='black', linewidth=2,
               zorder=5, label='Equilibrium Point')

ax_loss.axvline(x=epochs[equilibrium_idx], color='gold', linestyle='--',
               linewidth=2, alpha=0.5)

ax_loss.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
ax_loss.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax_loss.set_title('TRAINING DYNAMICS: Generator vs Discriminator',
                 fontsize=13, fontweight='bold', pad=10)
ax_loss.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax_loss.grid(True, alpha=0.3)
ax_loss.set_xlim(0, 100)
ax_loss.set_ylim(0, 10)

# Add phase annotations
ax_loss.text(5, 9, 'Phase 1:\nDiscriminator\nWinning',
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax_loss.text(30, 9, 'Phase 2:\nCompetitive\nImprovement',
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax_loss.text(75, 9, 'Phase 3:\nEquilibrium\nReached',
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Bottom row: Quality metrics
ax_quality = fig.add_subplot(gs[2, :2])

ax_quality.plot(epochs, realism, color=mlblue, linewidth=3)
ax_quality.fill_between(epochs, 0, realism, alpha=0.3, color=mlblue)

ax_quality.set_xlabel('Training Epoch', fontsize=11, fontweight='bold')
ax_quality.set_ylabel('Realism Score (%)', fontsize=11, fontweight='bold')
ax_quality.set_title('Generator Quality Over Time', fontsize=12, fontweight='bold')
ax_quality.grid(True, alpha=0.3)
ax_quality.set_xlim(0, 100)
ax_quality.set_ylim(0, 100)

# Mark key epochs
for epoch in key_epochs:
    ax_quality.scatter(epoch, realism[epoch-1], s=150, c=mlred,
                      edgecolors='black', linewidth=2, zorder=5)
    ax_quality.annotate(f'{realism[epoch-1]:.0f}%',
                       xy=(epoch, realism[epoch-1]),
                       xytext=(epoch, realism[epoch-1]-10),
                       fontsize=9, ha='center', fontweight='bold')

# Bottom right: Nash equilibrium illustration
ax_nash = fig.add_subplot(gs[2, 2:])

# Create success rate curves
gen_skill = np.linspace(0, 1, 100)
disc_skill = 0.5  # Discriminator at equilibrium

gen_success = gen_skill * (1 - disc_skill)
disc_success = disc_skill * (1 - gen_skill)

ax_nash.plot(gen_skill * 100, gen_success * 100, color=mlpurple,
            linewidth=3, label='Generator success rate')
ax_nash.plot(gen_skill * 100, disc_success * 100, color=mlgreen,
            linewidth=3, label='Discriminator success rate')

# Mark equilibrium
ax_nash.scatter(50, 25, s=400, c='gold', marker='*',
               edgecolors='black', linewidth=2, zorder=5)
ax_nash.axvline(x=50, color='gold', linestyle='--', linewidth=2, alpha=0.5)

ax_nash.set_xlabel('Generator Skill (%)', fontsize=11, fontweight='bold')
ax_nash.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax_nash.set_title('Nash Equilibrium: Both at 50%', fontsize=12, fontweight='bold')
ax_nash.legend(loc='upper right', fontsize=10)
ax_nash.grid(True, alpha=0.3)
ax_nash.set_xlim(0, 100)
ax_nash.set_ylim(0, 60)

ax_nash.text(50, 30, 'Equilibrium:\nG=50%, D=50%', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))

# Main title
fig.suptitle('ADVERSARIAL TRAINING: Two Competing Programs Learning Together',
            fontsize=15, fontweight='bold', y=0.98)

# Add explanatory text boxes
# Text box 1: Concept explanation (top of figure)
fig.text(0.5, 0.93, 'TWO PROGRAMS: Generator (creates synthetic data) vs Discriminator (detects fakes)\n' +
        'Both improve through competition - Generator learns to create realistic data by fooling Discriminator',
        ha='center', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

# Text box 2: Loss interpretation (left of middle panel)
fig.text(0.02, 0.45, 'LOSS CURVES:\nHigh Generator loss =\n  Struggling to fool\nHigh Discriminator loss =\n  Struggling to detect\nConvergence =\n  Equilibrium reached',
        ha='left', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='gray'))

# Text box 3: Quality progression (right of top panels)
fig.text(0.98, 0.75, 'QUALITY:\nEpoch 1:\n  Random noise\nEpoch 100:\n  Realistic data',
        ha='right', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='gray'))

# Text box 4: Equilibrium explanation (bottom right)
fig.text(0.98, 0.15, 'EQUILIBRIUM:\nDiscriminator cannot\nreliably detect fakes\n= Generator creates\nperfect imitations',
        ha='right', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7, edgecolor='gray'))

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
plt.savefig('../charts/discovery_chart_5_gan.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/discovery_chart_5_gan.png', dpi=150, bbox_inches='tight')
print("Chart 5 (GAN Evolution) created successfully!")
print(f"Equilibrium reached at epoch {epochs[equilibrium_idx]}")
print(f"Final generator realism: {realism[-1]:.1f}%")

plt.show()
