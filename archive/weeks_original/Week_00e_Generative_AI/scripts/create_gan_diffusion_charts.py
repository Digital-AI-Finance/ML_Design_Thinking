#!/usr/bin/env python3
"""
Week 0e: GAN and Diffusion Charts (Charts 11-20)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Wedge
from mpl_toolkits.mplot3d import Axes3D
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.style.use('seaborn-v0_8-whitegrid')

mlblue = '#1f77b4'
mlorange = '#ff7f0e'
mlgreen = '#2ca02c'
mlred = '#d62728'
mlpurple = '#9467bd'

def save_chart(fig, name):
    fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Created: {name}")

# Chart 11: Two Approaches
def create_two_approaches():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Adversarial approach
    ax1.text(0.5, 0.9, 'Adversarial Training', ha='center', fontsize=16, fontweight='bold', color=mlred)
    ax1.text(0.5, 0.75, 'Two Networks Compete', ha='center', fontsize=12)

    # Generator and Discriminator boxes
    gen_box = FancyBboxPatch((0.15, 0.45), 0.3, 0.15, boxstyle="round,pad=0.02",
                            facecolor=mlblue, alpha=0.3, edgecolor=mlblue, linewidth=2)
    disc_box = FancyBboxPatch((0.55, 0.45), 0.3, 0.15, boxstyle="round,pad=0.02",
                             facecolor=mlred, alpha=0.3, edgecolor=mlred, linewidth=2)
    ax1.add_patch(gen_box)
    ax1.add_patch(disc_box)
    ax1.text(0.3, 0.525, 'Generator', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(0.7, 0.525, 'Discriminator', ha='center', va='center', fontsize=10, fontweight='bold')

    # Competition arrows
    arrow1 = FancyArrowPatch((0.45, 0.525), (0.55, 0.525), arrowstyle='<->',
                            mutation_scale=20, linewidth=3, color='black')
    ax1.add_patch(arrow1)
    ax1.text(0.5, 0.58, 'Compete', ha='center', fontsize=9, fontweight='bold')

    ax1.text(0.5, 0.3, '+ Sharp, realistic outputs', ha='center', fontsize=10, color=mlgreen)
    ax1.text(0.5, 0.2, '- Training instability', ha='center', fontsize=10, color=mlred)
    ax1.text(0.5, 0.05, 'Best for: Image generation', ha='center', fontsize=9, style='italic')

    # Diffusion approach
    ax2.text(0.5, 0.9, 'Diffusion Models', ha='center', fontsize=16, fontweight='bold', color=mlblue)
    ax2.text(0.5, 0.75, 'Iterative Denoising', ha='center', fontsize=12)

    # Steps visualization
    steps = np.linspace(0.1, 0.9, 5)
    for i, x in enumerate(steps):
        alpha = (i+1) / 6
        circle = Circle((x, 0.525), 0.05, facecolor=mlgreen, alpha=alpha,
                       edgecolor='black', linewidth=1.5)
        ax2.add_patch(circle)
        if i < 4:
            arrow = FancyArrowPatch((x+0.05, 0.525), (steps[i+1]-0.05, 0.525),
                                   arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
            ax2.add_patch(arrow)

    ax2.text(0.5, 0.63, 'Noise → Clean (1000 steps)', ha='center', fontsize=9)

    ax2.text(0.5, 0.3, '+ Stable training', ha='center', fontsize=10, color=mlgreen)
    ax2.text(0.5, 0.2, '- Slow sampling', ha='center', fontsize=10, color=mlred)
    ax2.text(0.5, 0.05, 'Best for: Highest quality', ha='center', fontsize=9, style='italic')

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle('Two Revolutionary Approaches to Generation', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig

# Chart 12: Forger Detective Analogy
def create_forger_detective_analogy():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Forger (Generator)
    forger_box = FancyBboxPatch((1, 5), 3, 2, boxstyle="round,pad=0.15",
                               facecolor=mlblue, alpha=0.3, edgecolor=mlblue, linewidth=3)
    ax.add_patch(forger_box)
    ax.text(2.5, 6.5, 'FORGER', ha='center', fontsize=14, fontweight='bold', color=mlblue)
    ax.text(2.5, 6, '(Generator Network)', ha='center', fontsize=10)
    ax.text(2.5, 5.5, 'Creates fake paintings', ha='center', fontsize=9)

    # Detective (Discriminator)
    detective_box = FancyBboxPatch((10, 5), 3, 2, boxstyle="round,pad=0.15",
                                  facecolor=mlred, alpha=0.3, edgecolor=mlred, linewidth=3)
    ax.add_patch(detective_box)
    ax.text(11.5, 6.5, 'DETECTIVE', ha='center', fontsize=14, fontweight='bold', color=mlred)
    ax.text(11.5, 6, '(Discriminator Network)', ha='center', fontsize=10)
    ax.text(11.5, 5.5, 'Spots fakes vs real', ha='center', fontsize=9)

    # Fake painting flow
    fake_box = FancyBboxPatch((5.5, 5.5), 2.5, 1, boxstyle="round,pad=0.1",
                             facecolor=mlpurple, alpha=0.2, edgecolor=mlpurple, linewidth=2)
    ax.add_patch(fake_box)
    ax.text(6.75, 6, 'Fake Painting', ha='center', fontsize=10, fontweight='bold')

    # Real painting
    real_box = FancyBboxPatch((5.5, 3.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                             facecolor=mlgreen, alpha=0.2, edgecolor=mlgreen, linewidth=2)
    ax.add_patch(real_box)
    ax.text(6.75, 3.9, 'Real Painting', ha='center', fontsize=10, fontweight='bold')

    # Arrows
    # Forger creates fake
    arrow1 = FancyArrowPatch((4, 6), (5.5, 6), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color=mlblue)
    ax.add_patch(arrow1)

    # Fake to detective
    arrow2 = FancyArrowPatch((8, 6), (10, 6), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color=mlpurple)
    ax.add_patch(arrow2)

    # Real to detective
    arrow3 = FancyArrowPatch((8, 3.9), (10, 5), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color=mlgreen)
    ax.add_patch(arrow3)

    # Feedback to forger
    arrow4 = FancyArrowPatch((11.5, 5), (4, 5), arrowstyle='->',
                            mutation_scale=25, linewidth=3, color=mlred, linestyle='--')
    ax.add_patch(arrow4)
    ax.text(7.5, 4.6, 'Feedback: "Too obvious!"', ha='center', fontsize=9,
           color=mlred, fontweight='bold', style='italic')

    # Training progress indicator
    ax.text(7, 2, 'Early Training: Detective wins easily', ha='center', fontsize=10, color=mlred)
    ax.text(7, 1.5, 'Late Training: Forger fools detective!', ha='center', fontsize=10, color=mlgreen)
    ax.text(7, 1, 'Equilibrium: 50% accuracy (perfect balance)', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_title('GAN: The Forger vs Detective Game', fontsize=16, fontweight='bold')
    ax.axis('off')

    return fig

# Chart 13: Reverse Corruption Analogy
def create_reverse_corruption_analogy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Forward process
    ax1.text(0.5, 0.95, 'Forward Process: Add Noise', ha='center', fontsize=14,
            fontweight='bold', color=mlred)

    # Show degradation steps
    steps_forward = [(0.15, 0.7, 'Clean\nImage', mlgreen),
                     (0.35, 0.7, 'Slight\nNoise', mlorange),
                     (0.55, 0.7, 'Medium\nNoise', mlred),
                     (0.75, 0.7, 'Heavy\nNoise', mlpurple),
                     (0.90, 0.7, 'Pure\nNoise', '#555555')]

    for i, (x, y, label, color) in enumerate(steps_forward):
        alpha = 1 - (i * 0.18)
        box = FancyBboxPatch((x-0.06, y-0.1), 0.12, 0.15, boxstyle="round,pad=0.02",
                            facecolor=color, alpha=alpha*0.5, edgecolor=color, linewidth=2)
        ax1.add_patch(box)
        ax1.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

        if i < 4:
            arrow = FancyArrowPatch((x+0.07, y), (steps_forward[i+1][0]-0.07, y),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax1.add_patch(arrow)

    ax1.text(0.5, 0.4, 'q(x_t | x_{t-1}) ~ N(sqrt(1-beta_t) x_{t-1}, beta_t I)', ha='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    ax1.text(0.5, 0.2, 'Gradually corrupt clean data', ha='center', fontsize=11)
    ax1.text(0.5, 0.05, '1000 tiny steps', ha='center', fontsize=9, style='italic')

    # Reverse process
    ax2.text(0.5, 0.95, 'Reverse Process: Remove Noise', ha='center', fontsize=14,
            fontweight='bold', color=mlgreen)

    # Show recovery steps
    steps_reverse = [(0.15, 0.7, 'Pure\nNoise', '#555555'),
                     (0.35, 0.7, 'Structure\nEmerges', mlpurple),
                     (0.55, 0.7, 'Details\nForm', mlred),
                     (0.75, 0.7, 'Almost\nClean', mlorange),
                     (0.90, 0.7, 'Final\nImage', mlgreen)]

    for i, (x, y, label, color) in enumerate(steps_reverse):
        alpha = (i+1) * 0.18
        box = FancyBboxPatch((x-0.06, y-0.1), 0.12, 0.15, boxstyle="round,pad=0.02",
                            facecolor=color, alpha=alpha*0.5+0.2, edgecolor=color, linewidth=2)
        ax2.add_patch(box)
        ax2.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

        if i < 4:
            arrow = FancyArrowPatch((x+0.07, y), (steps_reverse[i+1][0]-0.07, y),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax2.add_patch(arrow)

    ax2.text(0.5, 0.4, 'p_theta(x_{t-1} | x_t) - Neural network predicts noise', ha='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    ax2.text(0.5, 0.2, 'Learn to reverse corruption', ha='center', fontsize=11)
    ax2.text(0.5, 0.05, '1000 denoising steps', ha='center', fontsize=9, style='italic')

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle('Diffusion: The Reverse Corruption Process', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Chart 14: GAN Geometric Dynamics
def create_gan_geometric_dynamics():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Real data distribution
    np.random.seed(42)
    real_data = np.random.multivariate_normal([3, 3], [[0.5, 0.2], [0.2, 0.5]], 200)
    ax.scatter(real_data[:, 0], real_data[:, 1], s=40, alpha=0.4, color=mlgreen, label='Real Data Distribution')

    # Generator distribution over training
    gen_early = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    gen_mid = np.random.multivariate_normal([1.5, 1.5], [[0.7, 0.1], [0.1, 0.7]], 100)
    gen_late = np.random.multivariate_normal([2.9, 2.9], [[0.5, 0.2], [0.2, 0.5]], 100)

    ax.scatter(gen_early[:, 0], gen_early[:, 1], s=20, alpha=0.3, color=mlred,
              label='Generator: Early (poor)', marker='x')
    ax.scatter(gen_mid[:, 0], gen_mid[:, 1], s=20, alpha=0.4, color=mlorange,
              label='Generator: Mid (improving)', marker='+')
    ax.scatter(gen_late[:, 0], gen_late[:, 1], s=20, alpha=0.5, color=mlblue,
              label='Generator: Late (converged)', marker='o')

    # Decision boundary
    x_line = np.linspace(-2, 6, 100)
    y_line = x_line + 0.5
    ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label='Decision Boundary (Discriminator)')

    ax.set_xlabel('Feature Dimension 1', fontsize=11)
    ax.set_ylabel('Feature Dimension 2', fontsize=11)
    ax.set_title('GAN Dynamics: Generator Learns to Match Real Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)

    return fig

# Chart 15: GAN Training Walkthrough
def create_gan_training_walkthrough():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves over training
    epochs = np.arange(0, 100)
    d_loss = 1.386 * np.exp(-epochs/50) + 0.693 + np.random.normal(0, 0.05, 100)
    g_loss = 0.693 * (1 + np.exp(-epochs/30)) + np.random.normal(0, 0.05, 100)

    ax1.plot(epochs, d_loss, linewidth=2, color=mlred, label='Discriminator Loss')
    ax1.plot(epochs, g_loss, linewidth=2, color=mlblue, label='Generator Loss')
    ax1.axhline(y=0.693, color='green', linestyle='--', alpha=0.5, label='Equilibrium')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Convergence Over Training', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Discriminator accuracy
    d_acc = 50 + 45 * np.exp(-epochs/30) + np.random.normal(0, 2, 100)
    ax2.plot(epochs, d_acc, linewidth=2, color=mlred)
    ax2.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Equilibrium (50%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Discriminator Accuracy (%)')
    ax2.set_title('Discriminator Performance', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(40, 100)

    # Sample quality over time (FID score)
    fid = 450 * np.exp(-epochs/25) + 8.7 + np.random.normal(0, 5, 100)
    ax3.plot(epochs, fid, linewidth=2, color=mlgreen)
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Photorealistic Threshold (FID<10)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('FID Score (Lower = Better)')
    ax3.set_title('Generation Quality Improvement', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Key metrics table
    metrics_data = [
        ['Epoch', 'D_loss', 'G_loss', 'D_acc', 'FID'],
        ['1', '1.386', '0.693', '95%', '450'],
        ['25', '0.8', '1.2', '65%', '120'],
        ['50', '0.72', '0.85', '55%', '35'],
        ['100', '0.695', '0.698', '51%', '8.7']
    ]

    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=metrics_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header row
    for i in range(5):
        table[(0, i)].set_facecolor(mlpurple)
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax4.set_title('Training Progress Metrics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig

# Chart 16: Diffusion Mathematics
def create_diffusion_mathematics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Forward diffusion schedule
    t = np.linspace(0, 1000, 1000)
    beta_linear = 0.0001 + (0.02 - 0.0001) * (t / 1000)
    beta_cosine = 0.008 + 0.012 * (1 - np.cos(np.pi * t / 1000)) / 2

    ax1.plot(t, beta_linear, linewidth=2, color=mlblue, label='Linear Schedule')
    ax1.plot(t, beta_cosine, linewidth=2, color=mlorange, label='Cosine Schedule')
    ax1.set_xlabel('Timestep t')
    ax1.set_ylabel('Noise Level β_t')
    ax1.set_title('Noise Schedules', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Signal-to-noise ratio over time
    snr_linear = (1 - np.cumsum(beta_linear) / 1000) / (np.cumsum(beta_linear) / 1000 + 1e-10)
    ax2.plot(t, snr_linear, linewidth=2, color=mlgreen)
    ax2.set_xlabel('Timestep t')
    ax2.set_ylabel('Signal-to-Noise Ratio')
    ax2.set_title('SNR Degradation During Forward Process', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Diffusion Mathematical Framework', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Chart 17: Latent Interpolation
def create_latent_interpolation():
    fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5, figsize=(15, 3))

    # Simulate interpolated latent representations
    np.random.seed(42)

    # Generate 5 interpolated "images"
    axes = [ax1, ax2, ax3, ax4, ax5]
    labels = ['Start\nz_1', 't=0.25', 't=0.5\nMidpoint', 't=0.75', 'End\nz_2']

    for i, (ax, label) in enumerate(zip(axes, labels)):
        # Create interpolated random pattern
        t = i / 4
        pattern1 = np.random.rand(20, 20)
        pattern2 = np.random.rand(20, 20)
        interpolated = (1-t) * pattern1 + t * pattern2

        ax.imshow(interpolated, cmap='viridis', interpolation='bilinear')
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Latent Space Interpolation: Smooth Transitions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# Chart 18: Denoising Steps
def create_denoising_steps():
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    np.random.seed(42)

    # Create base pattern
    base = np.zeros((20, 20))
    for i in range(5):
        center_x, center_y = np.random.randint(5, 15, 2)
        y, x = np.ogrid[:20, :20]
        mask = (x - center_x)**2 + (y - center_y)**2 <= 15
        base[mask] = np.random.rand() * 0.5 + 0.5

    timesteps = [1000, 750, 500, 250, 0]

    for idx, t in enumerate(timesteps):
        # Add noise proportional to timestep
        noise_level = t / 1000
        noisy = base + np.random.randn(20, 20) * noise_level

        # Top row: noisy images
        axes[0, idx].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f'T={t}', fontsize=10, fontweight='bold')
        axes[0, idx].axis('off')

        # Bottom row: noise level indicator
        axes[1, idx].bar([0], [noise_level], color=mlred, alpha=0.7)
        axes[1, idx].set_ylim(0, 1)
        axes[1, idx].set_xlim(-0.5, 0.5)
        axes[1, idx].set_ylabel('Noise', fontsize=8)
        axes[1, idx].set_xticks([])

    plt.suptitle('Diffusion Denoising: From Noise to Image in 1000 Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# Chart 19: Adversarial Theory
def create_adversarial_theory():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Minimax game value surface
    from mpl_toolkits.mplot3d import Axes3D
    ax1 = fig.add_subplot(221, projection='3d')

    g_params = np.linspace(-2, 2, 50)
    d_params = np.linspace(-2, 2, 50)
    G, D = np.meshgrid(g_params, d_params)
    V = -(G**2 - D**2)  # Saddle point

    ax1.plot_surface(G, D, V, cmap='RdBu', alpha=0.8)
    ax1.set_xlabel('Generator Params')
    ax1.set_ylabel('Discriminator Params')
    ax1.set_zlabel('Game Value')
    ax1.set_title('Minimax Game Surface', fontsize=11, fontweight='bold')

    # Nash equilibrium convergence
    epochs = np.arange(0, 100)
    gen_quality = 1 - np.exp(-epochs/30) + np.random.normal(0, 0.02, 100)
    gen_quality = np.clip(gen_quality, 0, 1)

    ax2.plot(epochs, gen_quality, linewidth=2, color=mlblue)
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Nash Equilibrium')
    ax2.fill_between(epochs, 0, gen_quality, alpha=0.3, color=mlblue)
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('p_g ≈ p_data (Similarity)')
    ax2.set_title('Convergence to Nash Equilibrium', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # JS Divergence decrease
    js_div = 0.693 * np.exp(-epochs/35) + np.random.normal(0, 0.02, 100)
    js_div = np.clip(js_div, 0, 1)

    ax3.plot(epochs, js_div, linewidth=2, color=mlred)
    ax3.fill_between(epochs, 0, js_div, alpha=0.3, color=mlred)
    ax3.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Perfect Match')
    ax3.set_xlabel('Training Epoch')
    ax3.set_ylabel('JS Divergence')
    ax3.set_title('Jensen-Shannon Divergence Minimization', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mathematical guarantee
    ax4.text(0.5, 0.8, 'Theoretical Guarantee', ha='center', fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.6, 'At equilibrium:', ha='center', fontsize=12)
    ax4.text(0.5, 0.45, 'p_generator = p_data', ha='center', fontsize=14,
            fontweight='bold', color=mlgreen,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=mlgreen, alpha=0.2))
    ax4.text(0.5, 0.25, 'D(x) = 0.5  (50% accuracy)', ha='center', fontsize=12, fontweight='bold')
    ax4.text(0.5, 0.1, 'Discriminator cannot tell real from fake', ha='center',
            fontsize=10, style='italic')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    return fig

# Chart 20: Quality Metrics Over Time
def create_quality_metrics_over_time():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    steps = np.array([0, 10000, 50000, 100000, 250000, 500000, 1000000])
    steps_k = steps / 1000

    # VAE metrics
    vae_is = [1.2, 2.5, 3.8, 4.5, 5.0, 5.1, 5.2]
    vae_fid = [450, 200, 120, 80, 60, 50, 48]

    # GAN metrics
    gan_is = [1.2, 3.5, 6.2, 7.8, 8.5, 8.9, 9.1]
    gan_fid = [450, 180, 65, 28, 15, 10, 8.7]

    # Diffusion metrics
    diff_is = [1.2, 4.2, 7.1, 8.5, 9.0, 9.2, 9.3]
    diff_fid = [450, 120, 45, 18, 8, 4, 3.2]

    # Inception Score
    ax1.plot(steps_k, vae_is, linewidth=2, marker='o', color=mlpurple, label='VAE')
    ax1.plot(steps_k, gan_is, linewidth=2, marker='s', color=mlred, label='GAN')
    ax1.plot(steps_k, diff_is, linewidth=2, marker='^', color=mlblue, label='Diffusion')
    ax1.axhline(y=9.7, color='green', linestyle='--', alpha=0.5, label='Real Data')
    ax1.set_xlabel('Training Steps (thousands)')
    ax1.set_ylabel('Inception Score')
    ax1.set_title('Inception Score Over Training', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 10)

    # FID Score
    ax2.plot(steps_k, vae_fid, linewidth=2, marker='o', color=mlpurple, label='VAE')
    ax2.plot(steps_k, gan_fid, linewidth=2, marker='s', color=mlred, label='GAN')
    ax2.plot(steps_k, diff_fid, linewidth=2, marker='^', color=mlblue, label='Diffusion')
    ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Photorealistic')
    ax2.set_xlabel('Training Steps (thousands)')
    ax2.set_ylabel('FID Score (Lower = Better)')
    ax2.set_title('FID Score Over Training', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Training time comparison
    models = ['VAE', 'GAN', 'Diffusion']
    times = [0.5, 2, 8]  # hours
    final_fid = [48, 8.7, 3.2]

    bars = ax3.bar(models, times, color=[mlpurple, mlred, mlblue], alpha=0.7)
    ax3.set_ylabel('Training Time (hours)')
    ax3.set_title('Time to Convergence', fontsize=12, fontweight='bold')
    for bar, time, fid in zip(bars, times, final_fid):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{time}hr\nFID:{fid}', ha='center', fontsize=9, fontweight='bold')

    # Quality vs Speed tradeoff
    ax4.scatter([48], [0.5], s=500, alpha=0.6, color=mlpurple, label='VAE: Fast but blurry')
    ax4.scatter([8.7], [2], s=500, alpha=0.6, color=mlred, label='GAN: Balanced')
    ax4.scatter([3.2], [8], s=500, alpha=0.6, color=mlblue, label='Diffusion: Slow but best')
    ax4.set_xlabel('Final FID (Lower = Better)')
    ax4.set_ylabel('Training Time (hours)')
    ax4.set_title('Quality-Speed Tradeoff', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 60)
    ax4.set_ylim(0, 10)

    plt.tight_layout()
    return fig

# Main execution
def create_gan_diffusion_charts():
    print("Creating GAN and Diffusion charts (11-20)...")

    save_chart(create_two_approaches(), 'two_approaches')
    save_chart(create_forger_detective_analogy(), 'forger_detective_analogy')
    save_chart(create_reverse_corruption_analogy(), 'reverse_corruption_analogy')
    save_chart(create_gan_geometric_dynamics(), 'gan_geometric_dynamics')
    save_chart(create_gan_training_walkthrough(), 'gan_training_walkthrough')
    save_chart(create_diffusion_mathematics(), 'diffusion_mathematics')
    save_chart(create_latent_interpolation(), 'latent_interpolation')
    save_chart(create_denoising_steps(), 'denoising_steps')
    save_chart(create_adversarial_theory(), 'adversarial_theory')
    save_chart(create_quality_metrics_over_time(), 'quality_metrics_over_time')

    print("\nDone! 10 charts created.")

if __name__ == "__main__":
    create_gan_diffusion_charts()
