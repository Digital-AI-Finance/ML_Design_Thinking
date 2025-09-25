"""
Generate final batch of missing NLP charts for Week 3 presentation
Part 3 - Remaining charts for complete coverage
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Wedge
import matplotlib.patheffects as path_effects
import warnings
warnings.filterwarnings('ignore')

# Set style and colors
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define color scheme
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f',
    'mlyellow': '#ffbb78',
    'mlcyan': '#17becf'
}

def create_feature_engineering_nlp():
    """Create NLP feature engineering visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Feature types
    features = [
        ('Lexical', ['Word Count', 'Char Count', 'Avg Word Len'], 0.15, colors['mlblue']),
        ('Syntactic', ['POS Tags', 'Dependency', 'Parse Tree'], 0.35, colors['mlorange']),
        ('Semantic', ['Word2Vec', 'BERT Emb', 'TF-IDF'], 0.55, colors['mlgreen']),
        ('Pragmatic', ['Sentiment', 'Emotion', 'Sarcasm'], 0.75, colors['mlpurple'])
    ]
    
    for feat_type, examples, x, color in features:
        # Main box
        rect = FancyBboxPatch((x - 0.08, 0.5), 0.16, 0.2,
                              boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.65, feat_type, ha='center', va='center',
               fontsize=11, fontweight='bold')
        
        # Examples below
        for i, ex in enumerate(examples):
            ax.text(x, 0.35 - i*0.08, ex, ha='center', fontsize=8, style='italic')
    
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1)
    ax.set_title('NLP Feature Engineering Hierarchy', fontsize=16, fontweight='bold')
    
    return fig

def create_model_selection_flowchart():
    """Create model selection decision flowchart"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Decision tree structure
    decisions = [
        ('Start', 0.5, 0.9, colors['mlgray']),
        ('Large Dataset?\n(>100K)', 0.5, 0.75, colors['mlblue']),
        ('Real-time?\n(<100ms)', 0.3, 0.6, colors['mlorange']),
        ('Context Critical?', 0.7, 0.6, colors['mlorange']),
        ('Rule-Based', 0.15, 0.45, colors['mlred']),
        ('Traditional ML\n(SVM/NB)', 0.35, 0.45, colors['mlgreen']),
        ('BERT/GPT', 0.65, 0.45, colors['mlpurple']),
        ('DistilBERT', 0.85, 0.45, colors['mlcyan'])
    ]
    
    for label, x, y, color in decisions:
        if 'Start' not in label and '?' not in label:
            # Final decisions
            rect = FancyBboxPatch((x - 0.08, y - 0.04), 0.16, 0.08,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, alpha=0.3,
                                  edgecolor=color, linewidth=2)
        else:
            # Decision nodes
            rect = FancyBboxPatch((x - 0.08, y - 0.04), 0.16, 0.08,
                                  boxstyle="round,pad=0.01",
                                  facecolor='white',
                                  edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Connections
    connections = [
        ((0.5, 0.86), (0.5, 0.79), 'Start'),
        ((0.45, 0.71), (0.35, 0.64), 'No'),
        ((0.55, 0.71), (0.65, 0.64), 'Yes'),
        ((0.3, 0.56), (0.2, 0.49), 'Yes'),
        ((0.3, 0.56), (0.35, 0.49), 'No'),
        ((0.7, 0.56), (0.65, 0.49), 'Yes'),
        ((0.7, 0.56), (0.8, 0.49), 'No')
    ]
    
    for start, end, label in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        if label != 'Start':
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, label, fontsize=8, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1)
    ax.set_title('NLP Model Selection Flowchart', fontsize=16, fontweight='bold')
    
    return fig

def create_batch_processing_speed():
    """Create batch processing speed comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Batch size impact
    ax1 = axes[0]
    batch_sizes = [1, 8, 16, 32, 64, 128]
    processing_times = [100, 15, 8, 5, 3, 2.5]
    
    ax1.plot(batch_sizes, processing_times, 'o-', color=colors['mlblue'],
            linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Time per Document (ms)', fontsize=12)
    ax1.set_title('Batch Size vs Processing Speed', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(alpha=0.3)
    ax1.fill_between(batch_sizes, processing_times, alpha=0.3, color=colors['mlblue'])
    
    # Throughput comparison
    ax2 = axes[1]
    models = ['Rule-Based', 'TF-IDF+SVM', 'LSTM', 'BERT-Base', 'BERT-Large']
    throughput = [10000, 5000, 500, 100, 30]  # docs/second
    
    bars = ax2.barh(models, throughput, color=[colors['mlgray'], colors['mlorange'],
                                                colors['mlblue'], colors['mlgreen'],
                                                colors['mlred']], alpha=0.7)
    ax2.set_xlabel('Throughput (docs/second)', fontsize=12)
    ax2.set_title('Model Throughput Comparison', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add values
    for bar, val in zip(bars, throughput):
        width = bar.get_width()
        ax2.text(width * 1.1, bar.get_y() + bar.get_height()/2,
                f'{val:,}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_gpu_cpu_comparison():
    """Create GPU vs CPU performance comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Speed comparison
    ax1 = axes[0]
    tasks = ['Training\nBERT', 'Inference\nBatch=32', 'Fine-tuning', 'Embeddings\nGeneration']
    cpu_time = [480, 120, 360, 60]  # minutes
    gpu_time = [30, 5, 20, 2]  # minutes
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cpu_time, width, label='CPU', color=colors['mlgray'], alpha=0.7)
    bars2 = ax1.bar(x + width/2, gpu_time, width, label='GPU', color=colors['mlgreen'], alpha=0.7)
    
    ax1.set_ylabel('Time (minutes)', fontsize=12)
    ax1.set_title('CPU vs GPU Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Cost analysis
    ax2 = axes[1]
    volume = np.array([100, 1000, 10000, 100000])
    cpu_cost = volume * 0.01
    gpu_cost = 50 + volume * 0.002
    
    ax2.plot(volume, cpu_cost, 'o-', label='CPU', color=colors['mlblue'], linewidth=2)
    ax2.plot(volume, gpu_cost, 's-', label='GPU', color=colors['mlorange'], linewidth=2)
    
    ax2.set_xlabel('Documents Processed', fontsize=12)
    ax2.set_ylabel('Cost ($)', fontsize=12)
    ax2.set_title('Cost Analysis: CPU vs GPU', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Add break-even point
    ax2.axvline(x=6250, color='red', linestyle='--', alpha=0.5)
    ax2.text(6250, 100, 'Break-even', rotation=90, fontsize=9, ha='right')
    
    plt.tight_layout()
    return fig

def create_huggingface_models():
    """Create HuggingFace model ecosystem visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Model categories
    categories = [
        ('Text\nClassification', 0.2, 0.7, colors['mlblue'], 15000),
        ('Token\nClassification', 0.5, 0.7, colors['mlorange'], 8000),
        ('Question\nAnswering', 0.8, 0.7, colors['mlgreen'], 5000),
        ('Text\nGeneration', 0.2, 0.3, colors['mlpurple'], 12000),
        ('Translation', 0.5, 0.3, colors['mlred'], 7000),
        ('Summarization', 0.8, 0.3, colors['mlcyan'], 4000)
    ]
    
    for cat, x, y, color, count in categories:
        # Size based on model count
        size = np.sqrt(count) / 50
        circle = Circle((x, y), size, facecolor=color, alpha=0.3,
                       edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, cat, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x, y - size - 0.05, f'{count:,} models', ha='center', fontsize=8,
               style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('HuggingFace Model Hub Ecosystem', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.1, '50,000+ Pre-trained Models Available', ha='center',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
    
    return fig

def create_deployment_architecture():
    """Create NLP deployment architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Components
    components = [
        ('Load\nBalancer', 0.1, 0.5, colors['mlblue']),
        ('API\nGateway', 0.3, 0.5, colors['mlorange']),
        ('Model\nServers', 0.5, 0.7, colors['mlgreen']),
        ('Cache\nLayer', 0.5, 0.5, colors['mlpurple']),
        ('Database', 0.5, 0.3, colors['mlred']),
        ('Monitoring', 0.7, 0.5, colors['mlcyan']),
        ('CDN', 0.9, 0.5, colors['mlbrown'])
    ]
    
    for comp, x, y, color in components:
        if comp == 'Model\nServers':
            # Multiple servers
            for i in range(3):
                rect = Rectangle((x - 0.12 + i*0.08, y - 0.05 + i*0.02),
                               0.1, 0.1, facecolor=color, alpha=0.3,
                               edgecolor=color, linewidth=2)
                ax.add_patch(rect)
        else:
            rect = FancyBboxPatch((x - 0.06, y - 0.05), 0.12, 0.1,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, alpha=0.3,
                                  edgecolor=color, linewidth=2)
            ax.add_patch(rect)
        
        ax.text(x, y, comp, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Connections
    arrows = [
        ((0.16, 0.5), (0.24, 0.5)),
        ((0.36, 0.5), (0.44, 0.5)),
        ((0.5, 0.45), (0.5, 0.65)),
        ((0.5, 0.45), (0.5, 0.35)),
        ((0.56, 0.5), (0.64, 0.5)),
        ((0.76, 0.5), (0.84, 0.5))
    ]
    
    for start, end in arrows:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.02, head_length=0.01, fc='black', ec='black', alpha=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Production NLP Deployment Architecture', fontsize=16, fontweight='bold')
    
    return fig

def create_confidence_calibration():
    """Create confidence calibration visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calibration plot
    ax1 = axes[0]
    
    # Perfect calibration
    perfect = np.linspace(0, 1, 100)
    ax1.plot(perfect, perfect, 'k--', label='Perfect Calibration', alpha=0.5)
    
    # Model calibrations
    conf_bins = np.linspace(0, 1, 11)
    uncalibrated = conf_bins ** 1.5
    calibrated = conf_bins * 0.95 + np.random.randn(11) * 0.02
    calibrated = np.clip(calibrated, 0, 1)
    
    ax1.plot(conf_bins, uncalibrated, 'o-', color=colors['mlred'],
            label='Before Calibration', linewidth=2, markersize=8)
    ax1.plot(conf_bins, calibrated, 's-', color=colors['mlgreen'],
            label='After Calibration', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax1.set_ylabel('Actual Accuracy', fontsize=12)
    ax1.set_title('Confidence Calibration', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Confidence distribution
    ax2 = axes[1]
    
    # Generate confidence scores
    correct_conf = np.random.beta(8, 2, 500)
    incorrect_conf = np.random.beta(3, 5, 200)
    
    ax2.hist(correct_conf, bins=20, alpha=0.5, color=colors['mlgreen'],
            label='Correct Predictions', density=True)
    ax2.hist(incorrect_conf, bins=20, alpha=0.5, color=colors['mlred'],
            label='Incorrect Predictions', density=True)
    
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_result_interpretation():
    """Create result interpretation visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Sample prediction
    ax.text(0.5, 0.9, 'Input: "The product quality is excellent but the price is too high"',
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
    
    # Prediction components
    components = [
        ('Overall\nSentiment', 0.2, 0.7, 'Mixed\n65% Positive', colors['mlorange']),
        ('Confidence\nScore', 0.4, 0.7, '0.82\nHigh', colors['mlgreen']),
        ('Key Aspects', 0.6, 0.7, 'Quality: +0.9\nPrice: -0.7', colors['mlblue']),
        ('Emotions', 0.8, 0.7, 'Satisfaction\nFrustration', colors['mlpurple'])
    ]
    
    for title, x, y, value, color in components:
        # Box
        rect = FancyBboxPatch((x - 0.08, y - 0.1), 0.16, 0.2,
                              boxstyle="round,pad=0.01",
                              facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.05, title, ha='center', va='center',
               fontsize=9, fontweight='bold')
        ax.text(x, y - 0.05, value, ha='center', va='center',
               fontsize=8)
    
    # Interpretation
    ax.text(0.5, 0.4, 'Interpretation:', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.3,
           'Customer appreciates product quality but finds it overpriced.\n' +
           'Consider pricing strategy or highlight value proposition.',
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlred'], alpha=0.2))
    
    # Action items
    ax.text(0.5, 0.15, 'Recommended Actions:', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.05,
           '1. Review pricing strategy  2. Emphasize quality in marketing  3. Consider promotional offers',
           ha='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('NLP Result Interpretation Framework', fontsize=16, fontweight='bold')
    
    return fig

def create_api_integration_flow():
    """Create API integration flow diagram"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # API flow stages
    stages = [
        ('Client\nApp', 0.1, 0.5, colors['mlblue']),
        ('API Key\nValidation', 0.25, 0.5, colors['mlorange']),
        ('Rate\nLimiter', 0.4, 0.5, colors['mlred']),
        ('Request\nParser', 0.55, 0.5, colors['mlgreen']),
        ('Model\nInference', 0.7, 0.5, colors['mlpurple']),
        ('Response\nFormatter', 0.85, 0.5, colors['mlcyan'])
    ]
    
    for i, (label, x, y, color) in enumerate(stages):
        # Box
        rect = FancyBboxPatch((x - 0.06, y - 0.06), 0.12, 0.12,
                              boxstyle="round,pad=0.01",
                              facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrow to next
        if i < len(stages) - 1:
            ax.arrow(x + 0.06, y, 0.08, 0, head_width=0.02, head_length=0.01,
                    fc='black', ec='black', alpha=0.5)
    
    # Response codes
    codes = [
        ('200 OK', 0.85, 0.7, colors['mlgreen']),
        ('401 Unauthorized', 0.25, 0.3, colors['mlred']),
        ('429 Rate Limited', 0.4, 0.3, colors['mlorange']),
        ('400 Bad Request', 0.55, 0.3, colors['mlyellow'])
    ]
    
    for code, x, y, color in codes:
        ax.text(x, y, code, ha='center', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
        # Connect to stage
        ax.plot([x, x], [y + 0.05, 0.44], 'k--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.set_title('NLP API Integration Flow', fontsize=16, fontweight='bold')
    
    return fig

# Add more chart generation functions for remaining charts...

def create_all_final_charts():
    """Generate all final missing charts"""
    charts_to_create = [
        ('feature_engineering_nlp', create_feature_engineering_nlp),
        ('model_selection_flowchart', create_model_selection_flowchart),
        ('batch_processing_speed', create_batch_processing_speed),
        ('gpu_cpu_comparison', create_gpu_cpu_comparison),
        ('huggingface_models', create_huggingface_models),
        ('deployment_architecture', create_deployment_architecture),
        ('confidence_calibration', create_confidence_calibration),
        ('result_interpretation', create_result_interpretation),
        ('api_integration_flow', create_api_integration_flow),
    ]
    
    print("Generating final batch of missing charts for Week 3...")
    
    import os
    os.makedirs('../charts', exist_ok=True)
    
    for name, func in charts_to_create:
        print(f"Creating {name}...")
        fig = func()
        fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("Final charts batch created successfully!")

if __name__ == "__main__":
    create_all_final_charts()