#!/usr/bin/env python3
"""
Create generative AI visualizations for Week 6 presentation.
Generates all charts referenced in the slides.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrow
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme matching the presentation
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f',
    'mlyellow': '#f1c40f',
    'mlcyan': '#17becf'
}

def create_generative_ai_landscape():
    """Create overview of generative AI ecosystem."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define ecosystem components
    components = {
        'Text': {'x': 0.2, 'y': 0.7, 'models': ['GPT-4', 'Claude', 'Gemini', 'LLaMA']},
        'Image': {'x': 0.5, 'y': 0.7, 'models': ['DALL-E', 'Midjourney', 'Stable Diffusion']},
        'Code': {'x': 0.8, 'y': 0.7, 'models': ['Copilot', 'CodeLlama', 'StarCoder']},
        'Audio': {'x': 0.2, 'y': 0.3, 'models': ['Whisper', 'MusicGen', 'Eleven Labs']},
        'Video': {'x': 0.5, 'y': 0.3, 'models': ['Sora', 'RunwayML', 'Pika']},
        '3D': {'x': 0.8, 'y': 0.3, 'models': ['Point-E', 'Shap-E', 'GET3D']}
    }

    # Draw components
    for category, info in components.items():
        # Main circle
        circle = Circle((info['x'], info['y']), 0.12,
                       facecolor=list(colors.values())[list(components.keys()).index(category)],
                       alpha=0.3, edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Category label
        ax.text(info['x'], info['y'], category, fontsize=14, fontweight='bold',
               ha='center', va='center')

        # Model names
        for i, model in enumerate(info['models'][:3]):
            angle = 2 * np.pi * i / 3 + np.pi/6
            x_offset = 0.15 * np.cos(angle)
            y_offset = 0.15 * np.sin(angle)
            ax.text(info['x'] + x_offset, info['y'] + y_offset, model,
                   fontsize=9, ha='center', alpha=0.7)

    # Add title and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('The Generative AI Ecosystem', fontsize=18, fontweight='bold')
    ax.axis('off')

    # Add innovation timeline
    ax.text(0.5, 0.05, '2020: GPT-3 | 2021: DALL-E | 2022: ChatGPT | 2023: GPT-4 | 2024: Multimodal Native',
           fontsize=10, ha='center', alpha=0.7)

    plt.tight_layout()
    plt.savefig('../charts/generative_ai_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/generative_ai_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_innovation_diamond_genai():
    """Create innovation diamond with GenAI integration."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Diamond stages
    stages = [
        {'name': 'Challenge', 'y': 0.9, 'width': 0.1, 'ideas': 1},
        {'name': 'Explore', 'y': 0.7, 'width': 0.3, 'ideas': 10},
        {'name': 'Generate', 'y': 0.5, 'width': 0.6, 'ideas': 1000},
        {'name': 'Peak', 'y': 0.3, 'width': 0.8, 'ideas': 5000},
        {'name': 'Filter', 'y': 0.15, 'width': 0.4, 'ideas': 50},
        {'name': 'Strategy', 'y': 0.05, 'width': 0.15, 'ideas': 5}
    ]

    # Draw diamond shape
    for i, stage in enumerate(stages):
        color = list(colors.values())[i]
        rect = Rectangle((0.5 - stage['width']/2, stage['y']-0.05),
                        stage['width'], 0.08,
                        facecolor=color, alpha=0.6, edgecolor='black')
        ax.add_patch(rect)

        # Stage label
        ax.text(0.5, stage['y'], f"{stage['name']}: {stage['ideas']} ideas",
               fontsize=11, ha='center', va='center', fontweight='bold')

        # AI annotation
        if stage['name'] == 'Generate':
            ax.annotate('AI amplifies 100x', xy=(0.8, stage['y']),
                       xytext=(0.9, stage['y']+0.1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, color='red', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Innovation Diamond: AI-Enhanced Ideation', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../charts/innovation_diamond_genai.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/innovation_diamond_genai.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_gan_architecture():
    """Create GAN architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generator path
    ax.add_patch(Rectangle((0.1, 0.6), 0.15, 0.2,
                          facecolor=colors['mlblue'], alpha=0.5))
    ax.text(0.175, 0.7, 'Noise\nz~p(z)', ha='center', va='center', fontweight='bold')

    ax.add_patch(Rectangle((0.35, 0.6), 0.2, 0.2,
                          facecolor=colors['mlorange'], alpha=0.5))
    ax.text(0.45, 0.7, 'Generator\nG(z)', ha='center', va='center', fontweight='bold')

    # Discriminator path
    ax.add_patch(Rectangle((0.35, 0.2), 0.2, 0.2,
                          facecolor=colors['mlgreen'], alpha=0.5))
    ax.text(0.45, 0.3, 'Real Data\nx~p(data)', ha='center', va='center', fontweight='bold')

    ax.add_patch(Rectangle((0.7, 0.4), 0.2, 0.2,
                          facecolor=colors['mlred'], alpha=0.5))
    ax.text(0.8, 0.5, 'Discriminator\nD(x)', ha='center', va='center', fontweight='bold')

    # Arrows
    ax.arrow(0.25, 0.7, 0.08, 0, head_width=0.02, head_length=0.02, fc='black')
    ax.arrow(0.55, 0.7, 0.13, -0.15, head_width=0.02, head_length=0.02, fc='black')
    ax.arrow(0.55, 0.3, 0.13, 0.15, head_width=0.02, head_length=0.02, fc='black')
    ax.arrow(0.9, 0.5, 0.05, 0, head_width=0.02, head_length=0.02, fc='black')

    # Labels
    ax.text(0.95, 0.5, 'Real/Fake', fontsize=11, va='center')
    ax.text(0.62, 0.65, 'Fake\nSamples', fontsize=9, ha='center')
    ax.text(0.62, 0.35, 'Real\nSamples', fontsize=9, ha='center')

    # Feedback loop
    ax.annotate('', xy=(0.8, 0.38), xytext=(0.45, 0.58),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.3',
                             color='red', lw=2, linestyle='dashed'))
    ax.text(0.55, 0.45, 'Adversarial\nFeedback', fontsize=9, color='red', ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('GAN Architecture: The Creative Duel', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../charts/gan_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/gan_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_vae_latent_space():
    """Create VAE latent space visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    np.random.seed(42)

    # Generate latent space points
    n_points = 500
    # Create clusters for different attributes
    clusters = [
        {'center': [2, 2], 'label': 'Style A', 'color': colors['mlblue']},
        {'center': [-2, 2], 'label': 'Style B', 'color': colors['mlorange']},
        {'center': [2, -2], 'label': 'Style C', 'color': colors['mlgreen']},
        {'center': [-2, -2], 'label': 'Style D', 'color': colors['mlpurple']}
    ]

    for cluster in clusters:
        x = np.random.normal(cluster['center'][0], 0.5, n_points//4)
        y = np.random.normal(cluster['center'][1], 0.5, n_points//4)
        ax.scatter(x, y, alpha=0.3, c=cluster['color'], s=20)
        ax.text(cluster['center'][0], cluster['center'][1], cluster['label'],
               fontsize=12, fontweight='bold', ha='center')

    # Show interpolation path
    path_x = np.linspace(-2, 2, 50)
    path_y = np.sin(path_x * 0.5) * 1.5
    ax.plot(path_x, path_y, 'r--', lw=2, alpha=0.7, label='Interpolation Path')

    # Add encoding/decoding annotations
    ax.annotate('Encode', xy=(-3, 0), xytext=(-4, 0),
               arrowprops=dict(arrowstyle='->', lw=2),
               fontsize=11, fontweight='bold')
    ax.annotate('Decode', xy=(3, 0), xytext=(4, 0),
               arrowprops=dict(arrowstyle='<-', lw=2),
               fontsize=11, fontweight='bold')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.set_title('VAE Latent Space: Smooth Interpolations', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('../charts/vae_latent_space.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/vae_latent_space.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_diffusion_process():
    """Create diffusion model process visualization."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    np.random.seed(42)

    # Simulate diffusion process
    steps = ['Original', 't=250', 't=500', 't=750', 'Pure Noise']
    noise_levels = [0, 0.3, 0.6, 0.9, 1.0]

    # Create synthetic image pattern
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    for idx, (ax, step, noise) in enumerate(zip(axes, steps, noise_levels)):
        # Create base pattern
        Z = np.sin(3*X) * np.cos(3*Y) * (1 - noise)
        # Add noise
        if noise > 0:
            Z += np.random.randn(100, 100) * noise * 2

        im = ax.imshow(Z, cmap='coolwarm', vmin=-3, vmax=3)
        ax.set_title(step, fontsize=10)
        ax.axis('off')

        # Add arrow between frames
        if idx < 4:
            ax.annotate('', xy=(1.15, 0.5), xytext=(1.05, 0.5),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=2))

    # Add reverse process arrow
    fig.text(0.5, 0.05, 'Forward Process (Add Noise) →', ha='center', fontsize=10)
    fig.text(0.5, 0.95, '← Reverse Process (Denoise)', ha='center', fontsize=10, color='red')

    fig.suptitle('Diffusion Models: From Image to Noise and Back', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../charts/diffusion_process.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/diffusion_process.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_transformer_attention():
    """Create transformer attention mechanism visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Attention matrix
    sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n_words = len(sentence)

    # Create attention weights (simulated)
    np.random.seed(42)
    attention = np.random.rand(n_words, n_words)
    attention = attention / attention.sum(axis=1, keepdims=True)

    # Make diagonal stronger (self-attention)
    np.fill_diagonal(attention, attention.diagonal() * 2)
    attention = attention / attention.sum(axis=1, keepdims=True)

    # Plot heatmap
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(range(n_words))
    ax.set_yticks(range(n_words))
    ax.set_xticklabels(sentence)
    ax.set_yticklabels(sentence)
    ax.set_xlabel('Keys', fontsize=12)
    ax.set_ylabel('Queries', fontsize=12)

    # Add values
    for i in range(n_words):
        for j in range(n_words):
            text = ax.text(j, i, f'{attention[i, j]:.2f}',
                         ha='center', va='center', color='black' if attention[i, j] < 0.3 else 'white',
                         fontsize=9)

    ax.set_title('Transformer Self-Attention Weights', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()
    plt.savefig('../charts/transformer_attention.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/transformer_attention.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_api_cost_comparison():
    """Create API cost comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # API pricing data (approximate)
    models = ['GPT-3.5', 'GPT-4', 'Claude Haiku', 'Claude Opus', 'Gemini Flash', 'Gemini Ultra',
              'Llama 2', 'Mistral', 'DALL-E 3', 'Stable Diffusion']
    costs = [0.5, 10, 0.25, 15, 0.35, 7, 0, 0.2, 40, 0.02]  # per 1M tokens or per 1K images
    quality = [7, 10, 6, 9.5, 6.5, 9, 8, 7.5, 9, 8]  # quality score out of 10

    # Create scatter plot
    scatter = ax.scatter(costs, quality, s=[500]*len(models), alpha=0.6,
                        c=range(len(models)), cmap='tab10')

    # Add labels
    for i, model in enumerate(models):
        ax.annotate(model, (costs[i], quality[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)

    # Add quadrant lines
    ax.axhline(y=8, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax.text(1, 9.5, 'High Quality\nLow Cost', fontsize=10, color='green', fontweight='bold')
    ax.text(12, 9.5, 'High Quality\nHigh Cost', fontsize=10, color='orange', fontweight='bold')
    ax.text(1, 6, 'Low Quality\nLow Cost', fontsize=10, color='blue', fontweight='bold')
    ax.text(12, 6, 'Low Quality\nHigh Cost', fontsize=10, color='red', fontweight='bold')

    ax.set_xlabel('Cost ($ per 1M tokens/1K images)', fontsize=12)
    ax.set_ylabel('Quality Score (1-10)', fontsize=12)
    ax.set_title('API Cost vs Quality Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../charts/api_cost_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/api_cost_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_prompt_engineering_tips():
    """Create prompt engineering best practices chart."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Tips organized by category
    categories = {
        'Context': {'y': 0.85, 'tips': ['Set the scene', 'Define the role', 'Provide background']},
        'Specificity': {'y': 0.65, 'tips': ['Be precise', 'Use examples', 'Define format']},
        'Constraints': {'y': 0.45, 'tips': ['Set limits', 'Define scope', 'Specify length']},
        'Iteration': {'y': 0.25, 'tips': ['Refine prompts', 'Chain thoughts', 'Build on results']},
        'Validation': {'y': 0.05, 'tips': ['Check output', 'Verify facts', 'Test edge cases']}
    }

    # Draw categories
    for i, (category, info) in enumerate(categories.items()):
        color = list(colors.values())[i]

        # Category box
        rect = FancyBboxPatch((0.05, info['y']-0.05), 0.2, 0.12,
                             boxstyle="round,pad=0.01",
                             facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.15, info['y']+0.01, category, fontsize=12, fontweight='bold',
               ha='center', va='center')

        # Tips
        for j, tip in enumerate(info['tips']):
            ax.text(0.35 + j*0.2, info['y']+0.01, tip, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color),
                   ha='center', va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Prompt Engineering: The 5 Pillars', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../charts/prompt_engineering_tips.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/prompt_engineering_tips.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_additional_charts():
    """Create remaining charts for the presentation."""

    # Quality vs Speed Tradeoff
    fig, ax = plt.subplots(figsize=(10, 8))

    algorithms = ['GAN', 'VAE', 'Diffusion', 'Transformer', 'Flow']
    quality = [8, 6, 10, 9, 7]
    speed = [8, 10, 3, 6, 4]

    ax.scatter(speed, quality, s=1000, alpha=0.6, c=range(len(algorithms)), cmap='Set2')

    for i, alg in enumerate(algorithms):
        ax.annotate(alg, (speed[i], quality[i]), ha='center', va='center',
                   fontsize=12, fontweight='bold')

    ax.set_xlabel('Generation Speed (1-10)', fontsize=12)
    ax.set_ylabel('Output Quality (1-10)', fontsize=12)
    ax.set_title('Quality vs Speed: Choose Your Algorithm', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)

    plt.tight_layout()
    plt.savefig('../charts/quality_vs_speed_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/quality_vs_speed_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Iteration Workflow
    fig, ax = plt.subplots(figsize=(12, 8))

    # Workflow stages
    stages = [
        {'name': 'Ideate', 'x': 0.1, 'y': 0.7},
        {'name': 'Generate', 'x': 0.3, 'y': 0.7},
        {'name': 'Review', 'x': 0.5, 'y': 0.7},
        {'name': 'Refine', 'x': 0.7, 'y': 0.7},
        {'name': 'Deploy', 'x': 0.9, 'y': 0.7}
    ]

    for i, stage in enumerate(stages):
        circle = Circle((stage['x'], stage['y']), 0.08,
                       facecolor=list(colors.values())[i], alpha=0.6,
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(stage['x'], stage['y'], stage['name'], ha='center', va='center',
               fontsize=11, fontweight='bold')

        # Add arrow to next stage
        if i < len(stages) - 1:
            ax.arrow(stage['x'] + 0.08, stage['y'], 0.12, 0,
                    head_width=0.02, head_length=0.02, fc='black')

    # Add iteration loop
    ax.annotate('', xy=(0.3, 0.6), xytext=(0.7, 0.6),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.3',
                             color='red', lw=2))
    ax.text(0.5, 0.5, 'Iterate Until Perfect', ha='center', fontsize=10,
           color='red', fontweight='bold')

    # Add time annotations
    ax.text(0.1, 0.85, '5 min', ha='center', fontsize=9, color='gray')
    ax.text(0.3, 0.85, '2 min', ha='center', fontsize=9, color='gray')
    ax.text(0.5, 0.85, '5 min', ha='center', fontsize=9, color='gray')
    ax.text(0.7, 0.85, '10 min', ha='center', fontsize=9, color='gray')
    ax.text(0.9, 0.85, '5 min', ha='center', fontsize=9, color='gray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    ax.set_title('Rapid Prototyping Workflow with AI', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../charts/iteration_workflow.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/iteration_workflow.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Human-AI Collaboration
    fig, ax = plt.subplots(figsize=(12, 8))

    # Venn diagram style
    circle1 = Circle((0.35, 0.5), 0.25, facecolor=colors['mlblue'], alpha=0.3,
                    edgecolor=colors['mlblue'], linewidth=2)
    circle2 = Circle((0.65, 0.5), 0.25, facecolor=colors['mlorange'], alpha=0.3,
                    edgecolor=colors['mlorange'], linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Labels
    ax.text(0.25, 0.5, 'Human\nCreativity', ha='center', va='center',
           fontsize=12, fontweight='bold')
    ax.text(0.75, 0.5, 'AI\nCapability', ha='center', va='center',
           fontsize=12, fontweight='bold')
    ax.text(0.5, 0.5, 'Sweet\nSpot', ha='center', va='center',
           fontsize=14, fontweight='bold', color='red')

    # Annotations
    ax.text(0.25, 0.3, '• Vision\n• Ethics\n• Emotion', ha='center', fontsize=9)
    ax.text(0.75, 0.3, '• Speed\n• Scale\n• Pattern', ha='center', fontsize=9)
    ax.text(0.5, 0.3, '• Innovation\n• Quality\n• Impact', ha='center', fontsize=9, color='red')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Human + AI = Superhuman Results', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../charts/human_ai_collaboration.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../charts/human_ai_collaboration.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Generate all charts for Week 6."""
    print("Generating Week 6 charts...")

    # Create all visualizations
    create_generative_ai_landscape()
    print("  [OK] Generative AI landscape")

    create_innovation_diamond_genai()
    print("  [OK] Innovation diamond with GenAI")

    create_gan_architecture()
    print("  [OK] GAN architecture")

    create_vae_latent_space()
    print("  [OK] VAE latent space")

    create_diffusion_process()
    print("  [OK] Diffusion process")

    create_transformer_attention()
    print("  [OK] Transformer attention")

    create_api_cost_comparison()
    print("  [OK] API cost comparison")

    create_prompt_engineering_tips()
    print("  [OK] Prompt engineering tips")

    create_additional_charts()
    print("  [OK] Additional charts")

    # Create placeholder charts for missing ones
    missing_charts = ['rag_architecture', 'production_pipeline']

    for chart_name in missing_charts:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'{chart_name.replace("_", " ").title()}\n[Detailed Diagram]',
               ha='center', va='center', fontsize=20, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(f'../charts/{chart_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'../charts/{chart_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {chart_name} placeholder")

    print("\n[SUCCESS] All charts generated successfully!")
    print(f"Location: Week_06/charts/")

if __name__ == "__main__":
    main()