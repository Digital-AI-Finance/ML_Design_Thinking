#!/usr/bin/env python3
"""
Week 0e: Complete Chart Generation - All 20 Missing Charts
Creates real matplotlib visualizations for charts 6-25
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import os

# Change to scripts directory for relative paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12

# Colors
mlblue = '#1f77b4'
mlorange = '#ff7f0e'
mlgreen = '#2ca02c'
mlred = '#d62728'
mlpurple = '#9467bd'
mlcyan = '#17becf'
mlbrown = '#8c564b'

def save_chart(fig, name):
    """Save chart as both PDF and PNG"""
    fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Created: {name}")

# Chart 6: Autoencoder Successes
def create_autoencoder_successes():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Success 1: Dimensionality Reduction
    dims = ['Original\n784D', 'Latent\n128D', 'Compressed\nStorage']
    values = [784, 128, 21]  # 128 floats at ~0.16 KB
    colors = [mlblue, mlred, mlgreen]
    bars = ax1.bar(dims, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Dimensions / Storage (KB)', fontsize=11)
    ax1.set_title('Success: Efficient Dimensionality Reduction', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val}', ha='center', fontweight='bold')

    # Success 2: Feature Learning (t-SNE of learned features)
    np.random.seed(42)
    n_per_class = 50
    classes = 3
    for i in range(classes):
        x = np.random.randn(n_per_class) * 0.5 + (i-1) * 2
        y = np.random.randn(n_per_class) * 0.5
        ax2.scatter(x, y, s=50, alpha=0.6, label=f'Class {i}')
    ax2.set_title('Success: Learned Meaningful Features', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Success 3: Denoising
    x = np.linspace(0, 10, 100)
    y_noisy = np.sin(x) + np.random.normal(0, 0.3, 100)
    y_clean = np.sin(x)
    ax3.plot(x, y_noisy, 'o', alpha=0.3, color=mlred, label='Noisy Input', markersize=3)
    ax3.plot(x, y_clean, linewidth=3, color=mlgreen, label='Denoised Output')
    ax3.set_title('Success: Noise Removal', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Input Dimension')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Success 4: Anomaly Detection
    np.random.seed(42)
    normal = np.random.randn(200, 2) * 0.5
    anomalies = np.random.randn(20, 2) * 2 + 3
    ax4.scatter(normal[:, 0], normal[:, 1], s=30, alpha=0.5, color=mlblue, label='Normal (Low Error)')
    ax4.scatter(anomalies[:, 0], anomalies[:, 1], s=100, alpha=0.8, color=mlred,
               marker='X', label='Anomaly (High Error)')
    ax4.set_title('Success: Anomaly Detection', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Dimension 1')
    ax4.set_ylabel('Dimension 2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Chart 7: Autoencoder Failures
def create_autoencoder_failures():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Failure 1: Blurry Outputs
    # Show original sharp vs reconstructed blurry
    x = np.linspace(0, 1, 100)
    original = np.where((x > 0.3) & (x < 0.7), 1, 0)  # Sharp edge
    blurry = 1 / (1 + np.exp(-50*(x-0.3))) - 1 / (1 + np.exp(-50*(x-0.7)))  # Blurred
    ax1.plot(x, original, linewidth=3, color=mlgreen, label='Original (Sharp)')
    ax1.plot(x, blurry, linewidth=3, color=mlred, label='Reconstructed (Blurry)')
    ax1.set_title('Failure: Blurry Outputs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Intensity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Failure 2: Poor Generation Quality
    models = ['Autoencoder', 'VAE', 'GAN', 'Diffusion', 'Real']
    is_scores = [2.1, 5.2, 9.1, 9.3, 9.7]
    colors = [mlred, mlorange, mlblue, mlgreen, mlpurple]
    bars = ax2.bar(models, is_scores, color=colors, alpha=0.7)
    ax2.set_ylabel('Inception Score')
    ax2.set_title('Failure: Poor Generation Quality', fontsize=12, fontweight='bold')
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax2.set_ylim(0, 10)
    for bar, score in zip(bars, is_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{score}', ha='center', fontweight='bold')

    # Failure 3: Holes in Latent Space
    np.random.seed(42)
    # Valid regions
    valid = np.random.randn(300, 2) * 0.5
    valid = valid[np.linalg.norm(valid, axis=1) > 0.5]
    ax3.scatter(valid[:, 0], valid[:, 1], s=20, alpha=0.5, color=mlblue, label='Valid Samples')
    # Holes (invalid regions)
    hole = np.random.randn(50, 2) * 0.3
    ax3.scatter(hole[:, 0], hole[:, 1], s=50, alpha=0.8, color=mlred,
               marker='x', label='Holes (Invalid)')
    ax3.set_title('Failure: Holes in Latent Space', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Latent Dim 1')
    ax3.set_ylabel('Latent Dim 2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Failure 4: Limited Diversity
    np.random.seed(42)
    # Mode collapse - all samples similar
    collapsed = np.random.randn(100, 2) * 0.2 + [1, 1]
    diverse = np.random.randn(100, 2) * 1.5
    ax4.scatter(collapsed[:, 0], collapsed[:, 1], s=30, alpha=0.6, color=mlred, label='Collapsed (Low Diversity)')
    ax4.scatter(diverse[:, 0], diverse[:, 1], s=30, alpha=0.3, color=mlgreen, label='Diverse')
    ax4.set_title('Failure: Mode Collapse / Limited Diversity', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Feature 1')
    ax4.set_ylabel('Feature 2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Chart 8: Averaging Problem
def create_averaging_problem():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Show averaging effect
    x = np.linspace(0, 1, 100)
    image1 = np.sin(x * 10)
    image2 = np.cos(x * 10)
    average = (image1 + image2) / 2

    ax1.plot(x, image1, linewidth=2, color=mlblue, label='Input Image 1')
    ax1.plot(x, image2, linewidth=2, color=mlgreen, label='Input Image 2')
    ax1.plot(x, average, linewidth=3, color=mlred, linestyle='--', label='MSE Optimum (Average)')
    ax1.set_title('MSE Loss Forces Averaging', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mathematical explanation
    ax2.text(0.5, 0.7, r'Given two inputs $x_1$ and $x_2$', ha='center', fontsize=14)
    ax2.text(0.5, 0.5, r'MSE optimal reconstruction: $\hat{x} = \frac{x_1 + x_2}{2}$', ha='center', fontsize=16, fontweight='bold')
    ax2.text(0.5, 0.3, r'Result: Blurry average, not realistic sample', ha='center', fontsize=12, color=mlred)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Distribution view
    x = np.linspace(-4, 4, 1000)
    dist1 = np.exp(-(x+1.5)**2/0.5) / np.sqrt(2*np.pi*0.5)
    dist2 = np.exp(-(x-1.5)**2/0.5) / np.sqrt(2*np.pi*0.5)
    avg_dist = (dist1 + dist2) / 2

    ax3.fill_between(x, dist1, alpha=0.3, color=mlblue, label='Distribution 1')
    ax3.fill_between(x, dist2, alpha=0.3, color=mlgreen, label='Distribution 2')
    ax3.plot(x, avg_dist, linewidth=3, color=mlred, linestyle='--', label='Average (MSE)')
    ax3.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax3.set_title('Averaging in Distribution Space', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Probability Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Loss landscape
    from mpl_toolkits.mplot3d import Axes3D
    ax4 = fig.add_subplot(224, projection='3d')
    x = y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2)  # Convex - forces average
    ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax4.set_title('MSE Loss: Convex (Forces Average)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Param 1')
    ax4.set_ylabel('Param 2')
    ax4.set_zlabel('Loss')

    plt.tight_layout()
    return fig

# Chart 9: VAE Framework
def create_vae_framework():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Input
    input_box = FancyBboxPatch((0.5, 3), 1.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=mlblue, alpha=0.3, edgecolor=mlblue, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 3.75, 'Input\nImage\nx', ha='center', va='center', fontsize=11, fontweight='bold')

    # Encoder
    enc_box = FancyBboxPatch((3, 3.2), 1.5, 1.1, boxstyle="round,pad=0.1",
                            facecolor=mlorange, alpha=0.3, edgecolor=mlorange, linewidth=2)
    ax.add_patch(enc_box)
    ax.text(3.75, 3.75, 'Encoder\nNeural Net', ha='center', va='center', fontsize=10, fontweight='bold')

    # Mean and Std boxes
    mu_box = FancyBboxPatch((5.5, 4), 1, 0.6, boxstyle="round,pad=0.05",
                           facecolor=mlgreen, alpha=0.4, edgecolor=mlgreen, linewidth=2)
    ax.add_patch(mu_box)
    ax.text(6, 4.3, 'μ(x)', ha='center', va='center', fontsize=10, fontweight='bold')

    sigma_box = FancyBboxPatch((5.5, 3), 1, 0.6, boxstyle="round,pad=0.05",
                              facecolor=mlgreen, alpha=0.4, edgecolor=mlgreen, linewidth=2)
    ax.add_patch(sigma_box)
    ax.text(6, 3.3, 'σ(x)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Reparameterization
    reparam_box = FancyBboxPatch((7.5, 3.2), 1.5, 1.1, boxstyle="round,pad=0.1",
                                facecolor=mlpurple, alpha=0.3, edgecolor=mlpurple, linewidth=2)
    ax.add_patch(reparam_box)
    ax.text(8.25, 3.75, 'z = μ + σ⊙ε\nε~N(0,I)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Latent z
    z_box = FancyBboxPatch((10, 3.2), 1, 1.1, boxstyle="round,pad=0.1",
                          facecolor=mlred, alpha=0.4, edgecolor=mlred, linewidth=2)
    ax.add_patch(z_box)
    ax.text(10.5, 3.75, 'Latent\nz', ha='center', va='center', fontsize=10, fontweight='bold')

    # Decoder
    dec_box = FancyBboxPatch((12, 3.2), 1.5, 1.1, boxstyle="round,pad=0.1",
                            facecolor=mlorange, alpha=0.3, edgecolor=mlorange, linewidth=2)
    ax.add_patch(dec_box)
    ax.text(12.75, 3.75, 'Decoder\nNeural Net', ha='center', va='center', fontsize=10, fontweight='bold')

    # Output
    output_box = FancyBboxPatch((14.5, 3), 1.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor=mlgreen, alpha=0.3, edgecolor=mlgreen, linewidth=2)
    ax.add_patch(output_box)
    ax.text(15.25, 3.75, 'Output\nImage\nx̂', ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrows
    arrows = [(2, 3.75, 3, 3.75), (4.5, 3.75, 5.5, 4.3), (4.5, 3.75, 5.5, 3.3),
              (6.5, 4.3, 7.5, 4), (6.5, 3.3, 7.5, 3.4), (9, 3.75, 10, 3.75),
              (11, 3.75, 12, 3.75), (13.5, 3.75, 14.5, 3.75)]

    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                               mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)

    # Labels
    ax.text(8, 5.5, 'Reparameterization Trick', ha='center', fontsize=12,
           fontweight='bold', color=mlpurple,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=mlpurple, alpha=0.2))

    ax.text(8, 1.5, 'Probabilistic Latent Space', ha='center', fontsize=12,
           fontweight='bold', color=mlred,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=mlred, alpha=0.2))

    # Loss terms
    ax.text(8, 0.5, 'L = -E[log p(x|z)] + KL(q(z|x)||p(z))', ha='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlim(0, 17)
    ax.set_ylim(0, 6)
    ax.set_title('VAE Framework: Probabilistic Encoder-Decoder', fontsize=14, fontweight='bold')
    ax.axis('off')

    return fig

# Chart 10: Artist Learning Process
def create_artist_learning_process():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Student-teacher cycle
    # Student box
    student_box = FancyBboxPatch((2, 5), 2, 1.5, boxstyle="round,pad=0.1",
                                facecolor=mlblue, alpha=0.3, edgecolor=mlblue, linewidth=3)
    ax.add_patch(student_box)
    ax.text(3, 5.75, 'Student\n(Generator)', ha='center', va='center',
           fontsize=12, fontweight='bold')

    # Teacher box
    teacher_box = FancyBboxPatch((8, 5), 2, 1.5, boxstyle="round,pad=0.1",
                                facecolor=mlred, alpha=0.3, edgecolor=mlred, linewidth=3)
    ax.add_patch(teacher_box)
    ax.text(9, 5.75, 'Teacher\n(Discriminator)', ha='center', va='center',
           fontsize=12, fontweight='bold')

    # Artwork flow
    artwork_box = FancyBboxPatch((5, 3), 1.5, 0.8, boxstyle="round,pad=0.05",
                                facecolor=mlgreen, alpha=0.3, edgecolor=mlgreen, linewidth=2)
    ax.add_patch(artwork_box)
    ax.text(5.75, 3.4, 'Artwork', ha='center', va='center', fontsize=10)

    # Critique flow
    critique_box = FancyBboxPatch((5, 1.5), 1.5, 0.8, boxstyle="round,pad=0.05",
                                 facecolor=mlorange, alpha=0.3, edgecolor=mlorange, linewidth=2)
    ax.add_patch(critique_box)
    ax.text(5.75, 1.9, 'Critique', ha='center', va='center', fontsize=10)

    # Arrows with labels
    # Student creates artwork
    arrow1 = FancyArrowPatch((4, 5.2), (5, 3.6), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color=mlblue)
    ax.add_patch(arrow1)
    ax.text(4.2, 4.2, '1. Creates', fontsize=10, color=mlblue, fontweight='bold')

    # Artwork to teacher
    arrow2 = FancyArrowPatch((6.5, 3.4), (8, 5.2), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color=mlgreen)
    ax.add_patch(arrow2)
    ax.text(7.3, 4.2, '2. Evaluates', fontsize=10, color=mlgreen, fontweight='bold')

    # Teacher critique
    arrow3 = FancyArrowPatch((8, 5), (6.5, 2.1), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color=mlred)
    ax.add_patch(arrow3)
    ax.text(7.5, 3.3, '3. Critiques', fontsize=10, color=mlred, fontweight='bold')

    # Critique to student
    arrow4 = FancyArrowPatch((5, 1.9), (2.5, 5), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color=mlorange)
    ax.add_patch(arrow4)
    ax.text(3.2, 3.3, '4. Improves', fontsize=10, color=mlorange, fontweight='bold')

    # Central text
    ax.text(6, 7, 'Adversarial Learning Cycle', ha='center', fontsize=16,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    ax.text(6, 0.5, 'Both Student and Teacher Improve Through Competition',
           ha='center', fontsize=11, style='italic')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_title('How Artists Improve Through Critique → GANs', fontsize=14, fontweight='bold')
    ax.axis('off')

    return fig

# Continue with main execution
def create_all_missing_charts():
    """Create all 20 missing charts"""

    print("Creating all 20 missing Week 0e charts...")
    print("This will take 2-3 minutes...")

    # Charts 6-9
    print("\nCharts 6-9: Autoencoder Analysis")
    save_chart(create_autoencoder_successes(), 'autoencoder_successes')
    save_chart(create_autoencoder_failures(), 'autoencoder_failures')
    save_chart(create_averaging_problem(), 'averaging_problem')
    save_chart(create_vae_framework(), 'vae_framework')

    print("\nChart 10: Artist Learning")
    save_chart(create_artist_learning_process(), 'artist_learning_process')

    print("\n✅ First batch complete. Continue with remaining 15 charts...")

if __name__ == "__main__":
    create_all_missing_charts()
