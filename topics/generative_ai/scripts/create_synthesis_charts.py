#!/usr/bin/env python3
"""
Week 0e: Synthesis Charts (Charts 21-25)
Final 5 charts for comparison and summary
"""

import matplotlib.pyplot as plt
import numpy as np
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

# Chart 21: Stable Diffusion API
def create_stable_diffusion_api():
    fig, ax = plt.subplots(figsize=(12, 8))

    # API workflow diagram
    boxes = [
        (2, 6, 'User\nPrompt', mlblue),
        (5, 6, 'API\nRequest', mlpurple),
        (8, 6, 'Diffusion\nModel', mlgreen),
        (11, 6, 'Generated\nImage', mlorange)
    ]

    for x, y, label, color in boxes:
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')

        if x < 11:
            from matplotlib.patches import FancyArrowPatch
            arrow = FancyArrowPatch((x+0.9, y), (x+2.1, y), arrowstyle='->',
                                   mutation_scale=25, linewidth=2.5, color='black')
            ax.add_patch(arrow)

    # Parameters box
    params_box = FancyBboxPatch((1, 3), 5, 2, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', alpha=0.5, edgecolor='orange', linewidth=2)
    ax.add_patch(params_box)
    ax.text(3.5, 4.5, 'Key Parameters:', fontsize=11, fontweight='bold')
    ax.text(3.5, 4, 'cfg_scale: 1-20 (prompt adherence)', fontsize=9)
    ax.text(3.5, 3.6, 'steps: 10-150 (quality vs speed)', fontsize=9)
    ax.text(3.5, 3.2, 'seed: reproducibility', fontsize=9)

    # Cost box
    cost_box = FancyBboxPatch((7.5, 3), 4, 2, boxstyle="round,pad=0.1",
                             facecolor='lightgreen', alpha=0.5, edgecolor='green', linewidth=2)
    ax.add_patch(cost_box)
    ax.text(9.5, 4.5, 'Production APIs:', fontsize=11, fontweight='bold')
    ax.text(9.5, 4, 'DALL-E 3: $0.04-0.12/image', fontsize=9)
    ax.text(9.5, 3.6, 'Midjourney: Subscription', fontsize=9)
    ax.text(9.5, 3.2, 'Stable Diffusion: $0.004/image', fontsize=9, color=mlgreen, fontweight='bold')

    ax.text(6.5, 1.5, 'Example: "A futuristic city at sunset"', ha='center', fontsize=12,
           style='italic', bbox=dict(boxstyle='round,pad=0.5', facecolor=mlblue, alpha=0.2))
    ax.text(6.5, 0.8, '→ High-quality 1024x1024 image in 10-30 seconds', ha='center', fontsize=10)

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.set_title('Stable Diffusion API: Production-Ready Generation', fontsize=14, fontweight='bold')
    ax.axis('off')

    return fig

# Chart 22: Generative Landscape
def create_generative_landscape():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create 2x2 grid of approaches
    approaches = [
        (0.25, 0.75, 'VAEs', mlpurple, ['Probabilistic', 'Smooth latent', 'Blurry outputs', 'Stable training']),
        (0.75, 0.75, 'GANs', mlred, ['Adversarial', 'Sharp outputs', 'Mode collapse risk', 'Training unstable']),
        (0.25, 0.25, 'Diffusion', mlblue, ['Iterative', 'Highest quality', 'Slow sampling', 'Stable']),
        (0.75, 0.25, 'Transformers', mlgreen, ['Sequential', 'Excellent for text', 'Scalable', 'Left-to-right'])
    ]

    for x, y, name, color, features in approaches:
        # Main box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x-0.18, y-0.18), 0.36, 0.36, boxstyle="round,pad=0.02",
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=3)
        ax.add_patch(box)

        ax.text(x, y+0.12, name, ha='center', fontsize=14, fontweight='bold', color=color)

        # Features
        for i, feature in enumerate(features):
            ax.text(x, y+0.05 - i*0.06, f'• {feature}', ha='center', fontsize=7)

    # Central connections
    ax.plot([0.25, 0.75], [0.75, 0.75], 'k--', alpha=0.3, linewidth=1)
    ax.plot([0.25, 0.25], [0.75, 0.25], 'k--', alpha=0.3, linewidth=1)
    ax.plot([0.75, 0.75], [0.75, 0.25], 'k--', alpha=0.3, linewidth=1)
    ax.plot([0.25, 0.75], [0.25, 0.25], 'k--', alpha=0.3, linewidth=1)

    ax.text(0.5, 0.5, 'Modern systems\ncombine approaches', ha='center', fontsize=11,
           fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('The Generative AI Landscape', fontsize=16, fontweight='bold')
    ax.axis('off')

    return fig

# Chart 23: Generative Tradeoffs
def create_generative_tradeoffs():
    fig, ax = plt.subplots(figsize=(10, 8))

    categories = ['Training\nStability', 'Sampling\nSpeed', 'Sample\nQuali                                                                                                                                                                                                                                                                                                                                                                                                                                ty', 'Controllability']
    vae_scores = [9, 9, 4, 6]
    gan_scores = [3, 9, 8, 4]
    diff_scores = [9, 2, 10, 9]
    trans_scores = [7, 6, 7, 8]

    x = np.arange(len(categories))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, vae_scores, width, label='VAE', color=mlpurple, alpha=0.7)
    bars2 = ax.bar(x - 0.5*width, gan_scores, width, label='GAN', color=mlred, alpha=0.7)
    bars3 = ax.bar(x + 0.5*width, diff_scores, width, label='Diffusion', color=mlblue, alpha=0.7)
    bars4 = ax.bar(x + 1.5*width, trans_scores, width, label='Transformer', color=mlgreen, alpha=0.7)

    ax.set_ylabel('Score (Higher = Better)', fontsize=11)
    ax.set_title('Comprehensive Trade-offs Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add score labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                       f'{int(height)}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    return fig

# Chart 24: Modern Applications
def create_modern_applications():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Image generation capabilities
    models_img = ['DALL-E 2', 'DALL-E 3', 'Midjourney v5', 'SD XL', 'Firefly']
    fid_scores = [12.5, 3.2, 5.8, 4.1, 7.3]
    colors_img = [mlblue, mlgreen, mlorange, mlpurple, mlred]

    bars = ax1.barh(models_img, fid_scores, color=colors_img, alpha=0.7)
    ax1.set_xlabel('FID Score (Lower = Better)')
    ax1.set_title('Image Generation: Production Systems', fontsize=12, fontweight='bold')
    ax1.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='Photorealistic')
    ax1.legend()
    for bar, score in zip(bars, fid_scores):
        ax1.text(score + 0.3, bar.get_y() + bar.get_height()/2,
                f'{score}', va='center', fontweight='bold')

    # Text generation perplexity
    models_text = ['GPT-2', 'GPT-3', 'GPT-3.5', 'GPT-4', 'Claude']
    ppl_scores = [35.2, 20.1, 12.3, 5.2, 6.8]
    colors_text = [mlpurple, mlblue, mlgreen, mlorange, mlred]

    bars = ax2.barh(models_text, ppl_scores, color=colors_text, alpha=0.7)
    ax2.set_xlabel('Perplexity (Lower = Better)')
    ax2.set_title('Text Generation: LLM Performance', fontsize=12, fontweight='bold')
    ax2.axvline(x=20, color='green', linestyle='--', alpha=0.5, label='Human-like')
    ax2.legend()
    for bar, score in zip(bars, ppl_scores):
        ax2.text(score + 1, bar.get_y() + bar.get_height()/2,
                f'{score}', va='center', fontweight='bold')

    # Generation speed
    systems = ['VAE', 'GAN', 'Diffusion\n(1000 steps)', 'Diffusion\n(50 steps DDIM)', 'Transformer']
    times = [0.05, 0.08, 25, 2, 0.5]  # seconds
    colors_speed = [mlpurple, mlred, mlblue, mlgreen, mlorange]

    bars = ax3.bar(systems, times, color=colors_speed, alpha=0.7)
    ax3.set_ylabel('Generation Time (seconds)')
    ax3.set_title('Speed Comparison (1024x1024 image)', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    for bar, time in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                f'{time}s', ha='center', fontweight='bold', fontsize=9)

    # Application domains
    domains = ['Text\nGeneration', 'Image\nSynthesis', 'Video\nCreation', 'Audio\nGeneration', '3D\nModeling']
    adoption = [95, 85, 45, 60, 30]  # percentage
    colors_domains = [mlgreen, mlblue, mlorange, mlpurple, mlred]

    bars = ax4.bar(domains, adoption, color=colors_domains, alpha=0.7)
    ax4.set_ylabel('Production Adoption (%)')
    ax4.set_title('Generative AI by Domain (2025)', fontsize=12, fontweight='bold')
    ax4.axhline(y=50, color='black', linestyle='--', alpha=0.3)
    ax4.set_ylim(0, 100)
    for bar, pct in zip(bars, adoption):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct}%', ha='center', fontweight='bold')

    plt.tight_layout()
    return fig

# Chart 25: Ethics Summary
def create_ethics_summary():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Capabilities
    capabilities = ['Image\nGeneration', 'Text\nWriting', 'Code\nGeneration', 'Music\nComposition', 'Drug\nDiscovery']
    cap_scores = [95, 90, 85, 70, 65]
    bars = ax1.bar(capabilities, cap_scores, color=mlgreen, alpha=0.7)
    ax1.set_ylabel('Human-Level Performance (%)')
    ax1.set_title('Current Capabilities (2025)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Human-level')
    ax1.legend()
    for bar, score in zip(bars, cap_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score}%', ha='center', fontweight='bold')

    # Ethical risks
    risks = ['Deepfakes', 'Copyright\nViolation', 'Bias\nAmplification', 'Job\nDisplacement', 'Misinformation']
    risk_severity = [9, 7, 8, 6, 9]
    bars = ax2.bar(risks, risk_severity, color=mlred, alpha=0.7)
    ax2.set_ylabel('Risk Severity (1-10)')
    ax2.set_title('Ethical Challenges', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 10)
    ax2.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='High Risk')
    ax2.legend()
    for bar, sev in zip(bars, risk_severity):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{sev}', ha='center', fontweight='bold')

    # Mitigation strategies
    strategies = ['Watermarking', 'Detection\nSystems', 'Bias\nAuditing', 'Human\nOversight', 'Regulation']
    effectiveness = [7, 6, 7, 8, 5]
    bars = ax3.bar(strategies, effectiveness, color=mlorange, alpha=0.7)
    ax3.set_ylabel('Effectiveness (1-10)')
    ax3.set_title('Mitigation Strategies', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 10)
    for bar, eff in zip(bars, effectiveness):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{eff}', ha='center', fontweight='bold')

    # Future outlook
    years = [2024, 2025, 2026, 2027, 2028]
    capability = [60, 75, 85, 92, 96]
    governance = [20, 35, 50, 65, 75]

    ax4.plot(years, capability, linewidth=3, marker='o', color=mlblue,
            label='Technical Capability', markersize=8)
    ax4.plot(years, governance, linewidth=3, marker='s', color=mlgreen,
            label='Governance Maturity', markersize=8)
    ax4.fill_between(years, capability, alpha=0.2, color=mlblue)
    ax4.fill_between(years, governance, alpha=0.2, color=mlgreen)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Progress (%)')
    ax4.set_title('Future Trajectory', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)

    plt.suptitle('Generative AI: Ethics and Future', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_synthesis_charts():
    print("Creating final synthesis charts (21-25)...")

    save_chart(create_stable_diffusion_api(), 'stable_diffusion_api')
    save_chart(create_generative_landscape(), 'generative_landscape')
    save_chart(create_generative_tradeoffs(), 'generative_tradeoffs')
    save_chart(create_modern_applications(), 'modern_applications')
    save_chart(create_ethics_summary(), 'ethics_summary')

    print("\nAll 5 synthesis charts created!")

if __name__ == "__main__":
    create_synthesis_charts()
