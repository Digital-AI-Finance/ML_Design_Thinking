"""
Create 5 additional charts for Week_00e to reach 25 total charts
Charts: GAN training dynamics, VAE latent space, Diffusion forward/reverse,
        Quality metrics comparison, Ethical considerations framework
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def save_chart(name):
    """Save chart in both PDF and PNG formats"""
    plt.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created {name}")

# Chart 1: GAN Training Dynamics
def create_gan_training_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = np.arange(0, 100)
    # Simulated losses
    d_real = 0.7 + 0.3 * np.exp(-epochs/20) * np.sin(epochs/5)
    d_fake = 0.3 + 0.3 * np.exp(-epochs/20) * np.cos(epochs/5)
    g_loss = 1.5 * np.exp(-epochs/30) + 0.3 * np.sin(epochs/10)

    # Loss curves
    axes[0].plot(epochs, d_real, label='D(real)', linewidth=2)
    axes[0].plot(epochs, d_fake, label='D(fake)', linewidth=2)
    axes[0].plot(epochs, g_loss, label='G loss', linewidth=2)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Nash equilibrium')
    axes[0].set_xlabel('Training Epoch', fontsize=12)
    axes[0].set_ylabel('Loss / Probability', fontsize=12)
    axes[0].set_title('GAN Training Dynamics', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Convergence phases
    phases = ['Initialization\n(D dominates)', 'Oscillation\n(Competition)',
              'Convergence\n(Nash equilibrium)', 'Mode collapse\nrisk']
    phase_starts = [0, 20, 50, 80]
    colors = ['red', 'orange', 'green', 'purple']

    for i, (phase, start, color) in enumerate(zip(phases, phase_starts, colors)):
        axes[1].barh(i, 25 if i < 3 else 20, left=start, color=color, alpha=0.6, edgecolor='black')
        axes[1].text(start + 12, i, phase, ha='center', va='center', fontweight='bold')

    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Training Epoch', fontsize=12)
    axes[1].set_title('Training Phases', fontsize=14, fontweight='bold')
    axes[1].set_yticks([])
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    save_chart('gan_training_dynamics_complete')

# Chart 2: VAE Latent Space Exploration
def create_vae_latent_space():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2D latent space with different digit clusters
    np.random.seed(42)
    n_points = 100

    # Simulate 10 digit classes in 2D latent space
    centers = np.array([[np.cos(i*2*np.pi/10)*3, np.sin(i*2*np.pi/10)*3] for i in range(10)])

    for i in range(10):
        points = np.random.randn(n_points, 2) * 0.5 + centers[i]
        axes[0].scatter(points[:, 0], points[:, 1], label=f'Digit {i}', alpha=0.6, s=30)

    axes[0].set_xlabel('Latent Dimension 1', fontsize=12)
    axes[0].set_ylabel('Latent Dimension 2', fontsize=12)
    axes[0].set_title('VAE Latent Space (MNIST)', fontsize=14, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Interpolation path
    start = centers[3]
    end = centers[7]
    t = np.linspace(0, 1, 10)
    path = np.outer(1-t, start) + np.outer(t, end)

    axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X',
                    edgecolors='black', linewidths=2, label='Class centers', zorder=3)
    axes[1].plot(path[:, 0], path[:, 1], 'b-', linewidth=3, label='Interpolation path')
    axes[1].scatter(path[:, 0], path[:, 1], c='blue', s=100, zorder=2)

    axes[1].annotate('Digit 3', xy=start, xytext=(start[0]-1, start[1]+1),
                    arrowprops=dict(arrowstyle='->', lw=2), fontsize=12, fontweight='bold')
    axes[1].annotate('Digit 7', xy=end, xytext=(end[0]+1, end[1]+1),
                    arrowprops=dict(arrowstyle='->', lw=2), fontsize=12, fontweight='bold')

    axes[1].set_xlabel('Latent Dimension 1', fontsize=12)
    axes[1].set_ylabel('Latent Dimension 2', fontsize=12)
    axes[1].set_title('Smooth Interpolation in Latent Space', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_chart('vae_latent_space_complete')

# Chart 3: Diffusion Process Forward and Reverse
def create_diffusion_process():
    fig, ax = plt.subplots(figsize=(14, 6))

    T = 10  # Number of diffusion steps
    x = np.linspace(0, T, 100)

    # Forward process (adding noise)
    noise_level = 1 - np.exp(-x/3)
    signal_level = np.exp(-x/3)

    ax.fill_between(x, 0, noise_level, alpha=0.3, color='red', label='Noise level')
    ax.fill_between(x, 0, signal_level, alpha=0.3, color='blue', label='Signal level')
    ax.plot(x, noise_level, 'r-', linewidth=2.5, label='Forward diffusion')
    ax.plot(x, signal_level, 'b-', linewidth=2.5, label='Reverse denoising')

    # Add step markers
    steps = np.arange(0, T+1, 2)
    for step in steps:
        noise_val = 1 - np.exp(-step/3)
        signal_val = np.exp(-step/3)
        ax.plot([step, step], [0, 1], 'k--', alpha=0.2)

        # Add sample images representation as boxes
        if step == 0:
            ax.add_patch(plt.Rectangle((step-0.3, 0.85), 0.6, 0.1,
                                      facecolor='blue', edgecolor='black', linewidth=2))
            ax.text(step, 0.75, 'Clean\nImage', ha='center', fontweight='bold')
        elif step == T:
            ax.add_patch(plt.Rectangle((step-0.3, 0.85), 0.6, 0.1,
                                      facecolor='red', edgecolor='black', linewidth=2))
            ax.text(step, 0.75, 'Pure\nNoise', ha='center', fontweight='bold')
        else:
            ax.add_patch(plt.Rectangle((step-0.3, 1-noise_val-0.05), 0.6, 0.1,
                                      facecolor='purple', edgecolor='black', linewidth=1))

    # Add arrows
    ax.annotate('', xy=(T-0.5, 0.5), xytext=(0.5, 0.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(T/2, 0.55, 'Forward Process q(x_t|x_0)', ha='center',
            fontsize=12, fontweight='bold', color='red')

    ax.annotate('', xy=(0.5, 0.3), xytext=(T-0.5, 0.3),
               arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax.text(T/2, 0.25, 'Reverse Process p_θ(x_0|x_T)', ha='center',
            fontsize=12, fontweight='bold', color='blue')

    ax.set_xlabel('Diffusion Step t', fontsize=12)
    ax.set_ylabel('Signal / Noise Ratio', fontsize=12)
    ax.set_title('Diffusion Model: Forward Corruption & Reverse Denoising',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, T+0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_chart('diffusion_forward_reverse_complete')

# Chart 4: Generative Model Quality Metrics
def create_quality_metrics_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = ['VAE', 'GAN', 'Diffusion', 'Flow']
    metrics = {
        'FID': [45, 15, 8, 20],
        'IS': [2.5, 6.8, 8.2, 5.5],
        'Precision': [0.65, 0.85, 0.90, 0.75],
        'Recall': [0.80, 0.60, 0.75, 0.70]
    }

    x = np.arange(len(models))
    width = 0.2

    # FID and IS scores
    ax1_twin = ax1.twinx()
    bars1 = ax1.bar(x - width/2, metrics['FID'], width, label='FID (lower better)', color='coral')
    bars2 = ax1_twin.bar(x + width/2, metrics['IS'], width, label='IS (higher better)', color='skyblue')

    ax1.set_ylabel('FID Score', fontsize=11, color='coral')
    ax1_twin.set_ylabel('Inception Score', fontsize=11, color='skyblue')
    ax1.set_xlabel('Model Type', fontsize=11)
    ax1.set_title('Image Quality Metrics', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.tick_params(axis='y', labelcolor='coral')
    ax1_twin.tick_params(axis='y', labelcolor='skyblue')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Precision-Recall
    bars3 = ax2.bar(x - width/2, metrics['Precision'], width, label='Precision (quality)', color='green', alpha=0.7)
    bars4 = ax2.bar(x + width/2, metrics['Recall'], width, label='Recall (diversity)', color='purple', alpha=0.7)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_xlabel('Model Type', fontsize=11)
    ax2.set_title('Precision-Recall Tradeoff', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_chart('generative_quality_metrics')

# Chart 5: Ethical Considerations Framework
def create_ethical_framework():
    fig, ax = plt.subplots(figsize=(12, 10))

    # Categories and concerns
    categories = {
        'Misinformation': ['Deepfakes', 'Fake news generation', 'Voice cloning', 'Impersonation'],
        'Copyright': ['Training data rights', 'Artist attribution', 'Style mimicry', 'Commercial use'],
        'Bias': ['Training data bias', 'Stereotype amplification', 'Representation gaps', 'Cultural sensitivity'],
        'Privacy': ['Face generation', 'Personal data in training', 'Consent issues', 'Identity theft'],
        'Environment': ['Compute cost (CO2)', 'Energy consumption', 'Hardware waste', 'Sustainability'],
        'Access': ['Digital divide', 'Cost barriers', 'Technical expertise', 'Centralization']
    }

    y_pos = 0
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']

    for (category, concerns), color in zip(categories.items(), colors):
        # Category header
        ax.barh(y_pos, 10, height=0.8, color=color, alpha=0.3, edgecolor='black', linewidth=2)
        ax.text(5, y_pos, category, ha='center', va='center',
               fontsize=13, fontweight='bold', color='black')
        y_pos -= 1

        # Concerns
        for concern in concerns:
            ax.barh(y_pos, 10, height=0.6, color=color, alpha=0.15, edgecolor=color, linewidth=1)
            ax.text(0.5, y_pos, f'• {concern}', va='center', fontsize=10)
            y_pos -= 0.7

        y_pos -= 0.5  # Space between categories

    # Mitigation strategies box
    mitigation_y = y_pos - 1
    ax.barh(mitigation_y, 10, height=2, color='lightgreen', alpha=0.3, edgecolor='darkgreen', linewidth=3)
    ax.text(5, mitigation_y + 0.7, 'Mitigation Strategies', ha='center', va='center',
           fontsize=14, fontweight='bold', color='darkgreen')

    strategies = ['Watermarking', 'Detection tools', 'Regulation', 'Transparency', 'Consent mechanisms']
    for i, strategy in enumerate(strategies):
        ax.text(i*2 + 0.5, mitigation_y - 0.3, strategy, ha='left', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkgreen', linewidth=1.5))

    ax.set_xlim(0, 10)
    ax.set_ylim(mitigation_y - 1.5, 1)
    ax.set_title('Generative AI: Ethical Considerations & Risks',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    save_chart('ethical_considerations_framework')

# Main execution
if __name__ == '__main__':
    print("Creating 5 additional charts for Week_00e...")
    print()

    create_gan_training_dynamics()
    create_vae_latent_space()
    create_diffusion_process()
    create_quality_metrics_comparison()
    create_ethical_framework()

    print()
    print("✓ All 5 charts created successfully!")
    print("Week_00e now has 25 charts total (matching other weeks)")
