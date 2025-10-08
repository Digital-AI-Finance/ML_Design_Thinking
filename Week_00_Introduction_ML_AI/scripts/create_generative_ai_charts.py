"""
Chart Generation for Week 0 Part 5: Generative AI
WCAG AAA Compliant Color Palette
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Ellipse
import numpy as np

# WCAG AAA Compliant Colors
COLORS = {
    'blue': '#1F77B4',
    'orange': '#FF7F0E',
    'green': '#2CA02C',
    'red': '#D62728',
    'purple': '#9467BD',
    'brown': '#8C564B',
    'pink': '#E377C2',
    'gray': '#7F7F7F',
    'olive': '#BCBD22',
    'cyan': '#17BECF'
}

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'


def create_gan_architecture():
    """Create GAN generator-discriminator structure"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    noise_x, noise_y = 2, 10
    ax.text(noise_x, noise_y, 'Random\nNoise z', ha='center', fontsize=11,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['gray'], alpha=0.5))

    gen_box = Rectangle((4, 8), 2.5, 3,
                        facecolor=COLORS['blue'], edgecolor='black',
                        linewidth=2, alpha=0.5)
    ax.add_patch(gen_box)
    ax.text(5.25, 9.5, 'Generator\nG(z)', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')

    ax.annotate('', xy=(4, 9.5), xytext=(2.5, 9.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    fake_x, fake_y = 8, 9.5
    ax.text(fake_x, fake_y, 'Fake\nData', ha='center', fontsize=11,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['orange'], alpha=0.5))

    ax.annotate('', xy=(7.5, 9.5), xytext=(6.5, 9.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    real_x, real_y = 8, 6
    ax.text(real_x, real_y, 'Real\nData', ha='center', fontsize=11,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['green'], alpha=0.5))

    disc_box = Rectangle((9.5, 5), 2.5, 6,
                         facecolor=COLORS['red'], edgecolor='black',
                         linewidth=2, alpha=0.5)
    ax.add_patch(disc_box)
    ax.text(10.75, 8, 'Discriminator\nD(x)', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')

    ax.annotate('', xy=(9.5, 9.5), xytext=(8.5, 9.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.annotate('', xy=(9.5, 6), xytext=(8.5, 6),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    output_x = 12.5
    ax.text(output_x, 9.5, 'Fake\n(0)', ha='center', fontsize=10,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.5))
    ax.text(output_x, 6, 'Real\n(1)', ha='center', fontsize=10,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.5))

    ax.annotate('', xy=(12.2, 9.5), xytext=(12, 9.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(12.2, 6), xytext=(12, 6),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    gradient_path = [(10.75, 4.5), (7, 3), (5.25, 7.5)]
    for i in range(len(gradient_path) - 1):
        ax.plot([gradient_path[i][0], gradient_path[i+1][0]],
               [gradient_path[i][1], gradient_path[i+1][1]],
               'r--', linewidth=2.5, alpha=0.7)

    ax.text(7, 2, 'Backprop Gradient', ha='center', fontsize=10,
           style='italic', color=COLORS['red'])

    ax.text(7, 11.5, 'Generative Adversarial Network (GAN)', ha='center',
           fontsize=16, fontweight='bold')

    loss_text = r'$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$'
    ax.text(7, 0.8, loss_text, ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('../charts/gan_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/gan_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created gan_architecture.pdf")


def create_vae_architecture():
    """Create VAE encoder-decoder with latent space"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    input_x, input_y = 2, 6
    ax.text(input_x, input_y, 'Input\nx', ha='center', fontsize=12,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['blue'], alpha=0.5))

    encoder_box = Rectangle((3.5, 4), 2, 4,
                           facecolor=COLORS['orange'], edgecolor='black',
                           linewidth=2, alpha=0.5)
    ax.add_patch(encoder_box)
    ax.text(4.5, 6, 'Encoder\nq_φ(z|x)', ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')

    ax.annotate('', xy=(3.5, 6), xytext=(2.5, 6),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    mu_x, mu_y = 7, 7
    sigma_x, sigma_y = 7, 5

    ax.text(mu_x, mu_y, 'μ', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor=COLORS['green'], alpha=0.5))
    ax.text(sigma_x, sigma_y, 'σ', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor=COLORS['green'], alpha=0.5))

    ax.annotate('', xy=(6.5, 7), xytext=(5.5, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(6.5, 5), xytext=(5.5, 5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    z_x, z_y = 9, 6
    ellipse = Ellipse((z_x, z_y), 1.5, 2,
                     facecolor=COLORS['purple'], edgecolor='black',
                     linewidth=2, alpha=0.3)
    ax.add_patch(ellipse)
    ax.text(z_x, z_y, 'Latent z\n~ N(μ, σ²)', ha='center', va='center',
           fontsize=11, fontweight='bold')

    ax.annotate('', xy=(8.5, 6), xytext=(7.5, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(8.5, 6), xytext=(7.5, 5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.text(8, 8, 'Reparameterization', ha='center', fontsize=9,
           style='italic', color=COLORS['gray'])

    decoder_box = Rectangle((10.5, 4), 2, 4,
                           facecolor=COLORS['red'], edgecolor='black',
                           linewidth=2, alpha=0.5)
    ax.add_patch(decoder_box)
    ax.text(11.5, 6, 'Decoder\np_θ(x|z)', ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')

    ax.annotate('', xy=(10.5, 6), xytext=(9.75, 6),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    output_x, output_y = 13, 6
    ax.text(output_x, output_y, 'Output\nx̂', ha='center', fontsize=12,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['blue'], alpha=0.5))

    ax.annotate('', xy=(12.5, 6), xytext=(12.5, 6),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    ax.text(7, 10.5, 'Variational Autoencoder (VAE)', ha='center',
           fontsize=16, fontweight='bold')

    loss_text = r'$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$'
    ax.text(7, 2, loss_text, ha='center', fontsize=11, style='italic')

    ax.text(7, 1, 'ELBO: Reconstruction Loss + KL Divergence', ha='center',
           fontsize=10, style='italic', color=COLORS['gray'])

    plt.tight_layout()
    plt.savefig('../charts/vae_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/vae_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created vae_architecture.pdf")


def create_diffusion_process():
    """Create forward and reverse diffusion process"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    forward_stages = [
        ('x₀\nClean', 2, COLORS['blue']),
        ('x₁\nLight\nNoise', 4.5, COLORS['orange']),
        ('x₂\nMedium\nNoise', 7, COLORS['green']),
        ('x₃\nHeavy\nNoise', 9.5, COLORS['red']),
        ('x_T\nPure\nNoise', 12, COLORS['gray'])
    ]

    for label, x, color in forward_stages:
        circle = Circle((x, 7), 0.8, color=color, edgecolor='black',
                       linewidth=2, alpha=0.6)
        ax.add_patch(circle)
        ax.text(x, 7, label, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    for i in range(len(forward_stages) - 1):
        x1 = forward_stages[i][1] + 0.8
        x2 = forward_stages[i + 1][1] - 0.8
        ax.annotate('', xy=(x2, 7), xytext=(x1, 7),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    ax.text(7, 9, 'Forward Diffusion (Add Noise)', ha='center',
           fontsize=13, fontweight='bold', color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.3))

    for i in range(len(forward_stages) - 1, 0, -1):
        x1 = forward_stages[i][1] - 0.8
        x2 = forward_stages[i - 1][1] + 0.8
        ax.annotate('', xy=(x2, 3), xytext=(x1, 3),
                   arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['purple'],
                                 linestyle='--'))

    reverse_stages = [
        ('Pure\nNoise', 12, COLORS['gray']),
        ('Denoise\n1', 9.5, COLORS['red']),
        ('Denoise\n2', 7, COLORS['green']),
        ('Denoise\n3', 4.5, COLORS['orange']),
        ('Generated\nImage', 2, COLORS['blue'])
    ]

    for label, x, color in reverse_stages:
        circle = Circle((x, 3), 0.8, color=color, edgecolor='black',
                       linewidth=2, alpha=0.6)
        ax.add_patch(circle)
        ax.text(x, 3, label, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    ax.text(7, 1, 'Reverse Diffusion (Learned Denoising)', ha='center',
           fontsize=13, fontweight='bold', color=COLORS['purple'],
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.5))

    ax.text(8, 9.5, 'Diffusion Models', ha='center',
           fontsize=18, fontweight='bold')

    formula_text = r'$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$'
    ax.text(8, 0.3, formula_text, ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('../charts/diffusion_process.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/diffusion_process.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created diffusion_process.pdf")


def create_transformer_architecture():
    """Create transformer with attention mechanism"""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    input_y = 1.5
    for i, token in enumerate(['The', 'cat', 'sat']):
        x = 3 + i * 3
        ax.text(x, input_y, token, ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['blue'], alpha=0.5))

    embed_y = 3.5
    for i in range(3):
        x = 3 + i * 3
        rect = Rectangle((x - 0.6, embed_y - 0.3), 1.2, 0.6,
                        facecolor=COLORS['orange'], edgecolor='black',
                        linewidth=1.5, alpha=0.5)
        ax.add_patch(rect)
        ax.text(x, embed_y, f'e_{i+1}', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        ax.annotate('', xy=(x, embed_y - 0.35), xytext=(x, input_y + 0.35),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.text(1, embed_y, 'Embedding', ha='right', fontsize=10,
           style='italic', color=COLORS['gray'])

    attn_box = Rectangle((2, 5), 10, 3,
                         facecolor=COLORS['green'], edgecolor='black',
                         linewidth=2, alpha=0.3)
    ax.add_patch(attn_box)

    qkv_y = 6.5
    for i, label in enumerate(['Q', 'K', 'V']):
        x = 4 + i * 3
        circle = Circle((x, qkv_y), 0.5, color=COLORS['red'],
                       edgecolor='black', linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)
        ax.text(x, qkv_y, label, ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

    ax.text(7, 7.8, 'Multi-Head Attention', ha='center', fontsize=12,
           fontweight='bold')

    formula = r'$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$'
    ax.text(7, 5.3, formula, ha='center', fontsize=10, style='italic')

    output_y = 9
    for i in range(3):
        x = 3 + i * 3
        rect = Rectangle((x - 0.6, output_y - 0.3), 1.2, 0.6,
                        facecolor=COLORS['purple'], edgecolor='black',
                        linewidth=1.5, alpha=0.5)
        ax.add_patch(rect)
        ax.text(x, output_y, f'h_{i+1}', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        ax.annotate('', xy=(x, output_y - 0.35), xytext=(x, 8),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    for i in range(3):
        x = 3 + i * 3
        ax.annotate('', xy=(x, 4.95), xytext=(x, embed_y + 0.35),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ffn_box = Rectangle((2, 10), 10, 1,
                        facecolor=COLORS['brown'], edgecolor='black',
                        linewidth=2, alpha=0.3)
    ax.add_patch(ffn_box)
    ax.text(7, 10.5, 'Feed-Forward Network', ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')

    for i in range(3):
        x = 3 + i * 3
        ax.annotate('', xy=(x, 9.95), xytext=(x, output_y + 0.35),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.text(7, 11.7, 'Transformer Architecture', ha='center',
           fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/transformer_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/transformer_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created transformer_architecture.pdf")


def create_llm_evolution():
    """Create evolution of large language models"""
    fig, ax = plt.subplots(figsize=(14, 9))

    models = ['BERT\n(2018)', 'GPT-2\n(2019)', 'T5\n(2020)', 'GPT-3\n(2020)',
             'PaLM\n(2022)', 'GPT-4\n(2023)', 'Gemini\n(2023)']

    parameters = [0.34, 1.5, 11, 175, 540, 1760, 1560]
    capabilities = [6, 7, 7.5, 8.5, 9, 9.7, 9.5]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                   COLORS['red'], COLORS['purple'], COLORS['brown'], COLORS['pink']]

    bars1 = ax1.bar(x, parameters, width=0.6, color=colors_list,
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Parameters (Billions)', fontsize=13, fontweight='bold')
    ax1.set_title('Model Size Evolution', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')

    for i, (bar, param) in enumerate(zip(bars1, parameters)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param}B', ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(x, capabilities, width=0.6, color=colors_list,
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Capability Score', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax2.set_title('Performance Evolution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Large Language Model Evolution', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../charts/llm_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/llm_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Created llm_evolution.pdf")


if __name__ == '__main__':
    print("Generating Generative AI Charts...")
    create_gan_architecture()
    create_vae_architecture()
    create_diffusion_process()
    create_transformer_architecture()
    create_llm_evolution()
    print("[OK] All generative AI charts created successfully!")