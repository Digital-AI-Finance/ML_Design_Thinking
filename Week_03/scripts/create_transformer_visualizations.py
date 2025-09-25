"""
Advanced transformer and attention mechanism visualizations for Week 3
Real implementations of attention patterns and BERT-style architectures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as path_effects
import matplotlib.gridspec as gridspec

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

def create_transformer_attention_mechanism():
    """Create detailed transformer attention mechanism visualization"""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main attention visualization
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # Input sentence
    words = ['The', 'product', 'quality', 'is', 'absolutely', 'amazing']
    n_words = len(words)
    
    # Generate realistic attention weights (self-attention)
    # Create patterns where related words attend to each other
    attention_matrix = np.random.rand(n_words, n_words) * 0.3
    
    # Make specific words attend strongly to related words
    attention_matrix[1, 2] = 0.9  # product -> quality
    attention_matrix[2, 1] = 0.8  # quality -> product
    attention_matrix[4, 5] = 0.95  # absolutely -> amazing
    attention_matrix[5, 4] = 0.85  # amazing -> absolutely
    attention_matrix[2, 5] = 0.7  # quality -> amazing
    
    # Add self-attention
    np.fill_diagonal(attention_matrix, 0.6)
    
    # Normalize rows to sum to 1 (attention weights)
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
    
    # Draw words as nodes
    word_positions = []
    for i, word in enumerate(words):
        x = 0.15 + i * 0.14
        y = 0.7
        word_positions.append((x, y))
        
        # Draw word box
        rect = FancyBboxPatch((x - 0.06, y - 0.03), 0.12, 0.06,
                              boxstyle="round,pad=0.01",
                              facecolor=colors['mlblue'], alpha=0.3,
                              edgecolor=colors['mlblue'], linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y, word, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw attention connections
    for i in range(n_words):
        for j in range(n_words):
            if attention_matrix[i, j] > 0.3:  # Only show strong connections
                x1, y1 = word_positions[i]
                x2, y2 = word_positions[j]
                
                # Curve for self-attention
                if i == j:
                    # Draw a loop above the word
                    arc = Circle((x1, y1 + 0.08), 0.03, 
                               fill=False, edgecolor=colors['mlgreen'],
                               linewidth=attention_matrix[i, j] * 5, alpha=0.6)
                    ax1.add_patch(arc)
                else:
                    # Draw curved arrow between words
                    style = "arc3,rad=.3"
                    arrow = FancyArrowPatch((x1, y1 - 0.03), (x2, y2 - 0.03),
                                          connectionstyle=style,
                                          arrowstyle='->,head_width=0.3,head_length=0.2',
                                          linewidth=attention_matrix[i, j] * 5,
                                          color=colors['mlorange'], alpha=0.6)
                    ax1.add_patch(arrow)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Self-Attention Mechanism in Transformers', fontsize=16, fontweight='bold')
    
    # Attention heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    
    sns.heatmap(attention_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=words, yticklabels=words,
                cbar_kws={'label': 'Attention Weight'}, ax=ax2)
    ax2.set_title('Attention Weight Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Keys', fontsize=12)
    ax2.set_ylabel('Queries', fontsize=12)
    
    # Multi-head visualization
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Draw multiple attention heads
    n_heads = 4
    head_colors = [colors['mlblue'], colors['mlgreen'], colors['mlorange'], colors['mlpurple']]
    
    for head in range(n_heads):
        y = 0.8 - head * 0.2
        
        # Head box
        rect = Rectangle((0.1, y - 0.05), 0.3, 0.1,
                        facecolor=head_colors[head], alpha=0.3,
                        edgecolor=head_colors[head], linewidth=2)
        ax3.add_patch(rect)
        ax3.text(0.25, y, f'Head {head + 1}', ha='center', va='center',
                fontsize=10, fontweight='bold')
        
        # Attention pattern description
        patterns = ['Syntactic', 'Semantic', 'Position', 'Long-range']
        ax3.text(0.5, y, patterns[head], ha='left', va='center', fontsize=9)
        
        # Arrow to concatenation
        ax3.arrow(0.4, y, 0.25, 0, head_width=0.02, head_length=0.02,
                 fc=colors['mlgray'], ec=colors['mlgray'], alpha=0.5)
    
    # Concatenation box
    rect = Rectangle((0.7, 0.3), 0.2, 0.4,
                    facecolor=colors['mlred'], alpha=0.3,
                    edgecolor=colors['mlred'], linewidth=2)
    ax3.add_patch(rect)
    ax3.text(0.8, 0.5, 'Concat\n&\nLinear', ha='center', va='center',
            fontsize=10, fontweight='bold')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Multi-Head Attention', fontsize=14, fontweight='bold')
    
    plt.suptitle('Transformer Attention Mechanisms', fontsize=18, fontweight='bold')
    
    return fig

def create_bert_vs_traditional_comparison():
    """Create comparison between BERT and traditional approaches"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    
    # Sample text
    text = "The movie was not bad"
    
    # Traditional approach (left-to-right)
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    words = text.split()
    # Traditional: sequential processing
    for i, word in enumerate(words):
        y = 0.5
        x = 0.2 + i * 0.2
        
        # Word box
        rect = Rectangle((x - 0.08, y - 0.05), 0.16, 0.1,
                        facecolor=colors['mlgray'], alpha=0.3,
                        edgecolor=colors['mlgray'], linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y, word, ha='center', va='center', fontsize=10)
        
        # Arrow to next word
        if i < len(words) - 1:
            ax1.arrow(x + 0.08, y, 0.04, 0, head_width=0.03, head_length=0.02,
                     fc='black', ec='black')
    
    ax1.set_title('Traditional: Sequential Processing', fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.2, 'Processes left-to-right only', ha='center', fontsize=9, style='italic')
    ax1.text(0.5, 0.1, 'Result: "not bad" = negative ❌', ha='center', fontsize=9, color=colors['mlred'])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # BERT approach (bidirectional)
    ax2 = axes[0, 1]
    ax2.axis('off')
    
    # Draw bidirectional connections
    for i, word in enumerate(words):
        y = 0.5
        x = 0.2 + i * 0.2
        
        # Word box
        rect = Rectangle((x - 0.08, y - 0.05), 0.16, 0.1,
                        facecolor=colors['mlblue'], alpha=0.3,
                        edgecolor=colors['mlblue'], linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, y, word, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw connections to all other words
        for j, other_word in enumerate(words):
            if i != j:
                x2 = 0.2 + j * 0.2
                # Draw thin connection lines
                ax2.plot([x, x2], [y - 0.05, y - 0.05], 
                        color=colors['mlorange'], alpha=0.3, linewidth=0.5)
    
    ax2.set_title('BERT: Bidirectional Processing', fontsize=12, fontweight='bold')
    ax2.text(0.5, 0.2, 'Sees full context simultaneously', ha='center', fontsize=9, style='italic')
    ax2.text(0.5, 0.1, 'Result: "not bad" = positive ✓', ha='center', fontsize=9, color=colors['mlgreen'])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Context understanding comparison
    ax3 = axes[0, 2]
    
    contexts = ['Negation', 'Sarcasm', 'Idioms', 'Metaphors', 'Ambiguity']
    traditional_scores = [30, 20, 40, 25, 35]
    bert_scores = [85, 70, 90, 75, 80]
    
    x = np.arange(len(contexts))
    width = 0.35
    
    ax3.bar(x - width/2, traditional_scores, width, label='Traditional',
           color=colors['mlgray'], alpha=0.7)
    ax3.bar(x + width/2, bert_scores, width, label='BERT',
           color=colors['mlblue'], alpha=0.7)
    
    ax3.set_ylabel('Understanding (%)', fontsize=10)
    ax3.set_title('Context Understanding Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(contexts, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Processing speed comparison
    ax4 = axes[1, 0]
    
    sizes = [100, 500, 1000, 5000, 10000]
    traditional_time = [0.1, 0.5, 1.0, 5.0, 10.0]
    bert_time = [0.5, 2.0, 4.0, 18.0, 35.0]
    
    ax4.plot(sizes, traditional_time, 'o-', label='Traditional',
            color=colors['mlgray'], linewidth=2, markersize=8)
    ax4.plot(sizes, bert_time, 's-', label='BERT',
            color=colors['mlblue'], linewidth=2, markersize=8)
    
    ax4.set_xlabel('Number of Documents', fontsize=10)
    ax4.set_ylabel('Processing Time (seconds)', fontsize=10)
    ax4.set_title('Processing Speed Comparison', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Accuracy by task
    ax5 = axes[1, 1]
    
    tasks = ['Sentiment', 'NER', 'QA', 'Summary']
    traditional_acc = [75, 80, 60, 55]
    bert_acc = [94, 96, 89, 82]
    
    x = np.arange(len(tasks))
    ax5.barh(x - width/2, traditional_acc, width, label='Traditional',
            color=colors['mlgray'], alpha=0.7)
    ax5.barh(x + width/2, bert_acc, width, label='BERT',
            color=colors['mlblue'], alpha=0.7)
    
    ax5.set_xlabel('Accuracy (%)', fontsize=10)
    ax5.set_title('Performance by NLP Task', fontsize=12, fontweight='bold')
    ax5.set_yticks(x)
    ax5.set_yticklabels(tasks)
    ax5.legend()
    ax5.grid(axis='x', alpha=0.3)
    
    # Model size and requirements
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Comparison table
    comparisons = [
        ('Model Size', '10 MB', '440 MB'),
        ('Memory', '< 1 GB', '4-8 GB'),
        ('Training Data', 'Small OK', 'Large required'),
        ('Fine-tuning', 'Not needed', 'Recommended'),
        ('Languages', 'One', 'Multiple')
    ]
    
    # Draw table
    for i, (metric, trad, bert) in enumerate(comparisons):
        y = 0.8 - i * 0.15
        
        ax6.text(0.1, y, metric, fontsize=10, fontweight='bold')
        ax6.text(0.4, y, trad, fontsize=9, color=colors['mlgray'])
        ax6.text(0.7, y, bert, fontsize=9, color=colors['mlblue'])
    
    # Headers
    ax6.text(0.1, 0.95, 'Metric', fontsize=11, fontweight='bold')
    ax6.text(0.4, 0.95, 'Traditional', fontsize=11, fontweight='bold', color=colors['mlgray'])
    ax6.text(0.7, 0.95, 'BERT', fontsize=11, fontweight='bold', color=colors['mlblue'])
    
    # Draw separator line
    ax6.plot([0.05, 0.95], [0.9, 0.9], 'k-', linewidth=1)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('Resource Requirements', fontsize=12, fontweight='bold')
    
    plt.suptitle('BERT vs Traditional NLP Approaches', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_emotion_detection_layers():
    """Create multi-layer emotion detection visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Layer 1: Raw text to tokens
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    text = "I'm absolutely thrilled but also nervous!"
    tokens = ['[CLS]', "I'm", 'absolutely', 'thrilled', 'but', 'also', 'nervous', '!', '[SEP]']
    
    # Draw text box
    rect = FancyBboxPatch((0.1, 0.7), 0.8, 0.15,
                          boxstyle="round,pad=0.02",
                          facecolor=colors['mlgray'], alpha=0.2,
                          edgecolor=colors['mlgray'], linewidth=2)
    ax1.add_patch(rect)
    ax1.text(0.5, 0.775, text, ha='center', va='center', fontsize=11)
    
    # Arrow down
    ax1.arrow(0.5, 0.65, 0, -0.1, head_width=0.03, head_length=0.02,
             fc=colors['mlblue'], ec=colors['mlblue'])
    
    # Draw tokens
    for i, token in enumerate(tokens):
        x = 0.05 + i * 0.11
        rect = Rectangle((x, 0.4), 0.1, 0.08,
                        facecolor=colors['mlblue'], alpha=0.3,
                        edgecolor=colors['mlblue'], linewidth=1)
        ax1.add_patch(rect)
        ax1.text(x + 0.05, 0.44, token, ha='center', va='center', fontsize=8)
    
    # Arrow down
    ax1.arrow(0.5, 0.35, 0, -0.1, head_width=0.03, head_length=0.02,
             fc=colors['mlorange'], ec=colors['mlorange'])
    
    # Embeddings visualization
    ax1.text(0.5, 0.15, '768-dimensional embeddings', ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlorange'], alpha=0.3))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Layer 1: Tokenization & Embedding', fontsize=12, fontweight='bold')
    
    # Layer 2: Attention patterns for emotions
    ax2 = axes[0, 1]
    
    # Create attention matrix for emotion words
    emotion_words = ['thrilled', 'nervous']
    attention_scores = {
        'thrilled': {'absolutely': 0.8, 'but': 0.3, '!': 0.6},
        'nervous': {'but': 0.7, 'also': 0.7, '!': 0.5}
    }
    
    # Create small heatmap
    words_subset = ['absolutely', 'thrilled', 'but', 'also', 'nervous', '!']
    matrix = np.random.rand(len(words_subset), len(words_subset)) * 0.3
    
    # Set specific attention values
    matrix[1, 0] = 0.8  # thrilled -> absolutely
    matrix[4, 2] = 0.7  # nervous -> but
    matrix[4, 3] = 0.7  # nervous -> also
    
    sns.heatmap(matrix, xticklabels=words_subset, yticklabels=words_subset,
                cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Attention'})
    ax2.set_title('Layer 2: Emotion Attention Patterns', fontsize=12, fontweight='bold')
    
    # Layer 3: Emotion classification
    ax3 = axes[1, 0]
    
    emotions = ['Joy', 'Fear', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Trust', 'Anticipation']
    probabilities = [0.65, 0.45, 0.35, 0.05, 0.02, 0.01, 0.40, 0.55]
    colors_emotion = [colors['mlgreen'] if p > 0.4 else colors['mlgray'] for p in probabilities]
    
    bars = ax3.barh(emotions, probabilities, color=colors_emotion, alpha=0.7)
    ax3.set_xlabel('Probability', fontsize=10)
    ax3.set_title('Layer 3: Emotion Classification Output', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add threshold line
    ax3.axvline(x=0.4, color=colors['mlred'], linestyle='--', alpha=0.5, label='Threshold')
    ax3.legend()
    
    # Layer 4: Final emotion mix
    ax4 = axes[1, 1]
    
    # Create pie chart of detected emotions
    detected_emotions = ['Joy', 'Fear', 'Anticipation', 'Trust']
    detected_values = [0.65, 0.45, 0.55, 0.40]
    
    # Normalize to percentages
    total = sum(detected_values)
    percentages = [v/total * 100 for v in detected_values]
    
    colors_pie = [colors['mlgreen'], colors['mlorange'], colors['mlblue'], colors['mlpurple']]
    
    wedges, texts, autotexts = ax4.pie(percentages, labels=detected_emotions, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax4.set_title('Layer 4: Emotion Mix Output', fontsize=12, fontweight='bold')
    
    plt.suptitle('Multi-Layer Emotion Detection Pipeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_sentiment_flow_architecture():
    """Create complete sentiment analysis flow architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Define pipeline stages with positions
    stages = [
        {'name': 'Data\nIngestion', 'x': 0.1, 'y': 0.8, 'color': colors['mlblue']},
        {'name': 'Preprocessing', 'x': 0.1, 'y': 0.6, 'color': colors['mlorange']},
        {'name': 'Feature\nExtraction', 'x': 0.1, 'y': 0.4, 'color': colors['mlgreen']},
        {'name': 'Model\nInference', 'x': 0.1, 'y': 0.2, 'color': colors['mlpurple']},
        
        {'name': 'Sentiment\nScoring', 'x': 0.35, 'y': 0.2, 'color': colors['mlred']},
        {'name': 'Aspect\nExtraction', 'x': 0.35, 'y': 0.4, 'color': colors['mlcyan']},
        {'name': 'Emotion\nDetection', 'x': 0.35, 'y': 0.6, 'color': colors['mlyellow']},
        
        {'name': 'Aggregation', 'x': 0.6, 'y': 0.4, 'color': colors['mlbrown']},
        {'name': 'Insights\nGeneration', 'x': 0.85, 'y': 0.4, 'color': colors['mlpink']},
    ]
    
    # Draw stages
    for stage in stages:
        rect = FancyBboxPatch((stage['x'] - 0.06, stage['y'] - 0.05), 0.12, 0.1,
                              boxstyle="round,pad=0.01",
                              facecolor=stage['color'], alpha=0.3,
                              edgecolor=stage['color'], linewidth=2)
        ax.add_patch(rect)
        ax.text(stage['x'], stage['y'], stage['name'], ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3),  # Main pipeline
        (3, 4), (3, 5), (3, 6),  # Split to analysis types
        (4, 7), (5, 7), (6, 7),  # Converge to aggregation
        (7, 8)  # Final output
    ]
    
    for start, end in connections:
        x1, y1 = stages[start]['x'], stages[start]['y']
        x2, y2 = stages[end]['x'], stages[end]['y']
        
        # Calculate arrow direction
        if start <= 3:  # Vertical connections
            arrow = FancyArrowPatch((x1, y1 - 0.05), (x2, y2 + 0.05),
                                  arrowstyle='->,head_width=0.3,head_length=0.2',
                                  linewidth=2, color=colors['mlgray'], alpha=0.5)
        else:  # Horizontal or diagonal connections
            arrow = FancyArrowPatch((x1 + 0.06, y1), (x2 - 0.06, y2),
                                  arrowstyle='->,head_width=0.3,head_length=0.2',
                                  connectionstyle="arc3,rad=.2",
                                  linewidth=2, color=colors['mlgray'], alpha=0.5)
        ax.add_patch(arrow)
    
    # Add data flow annotations
    annotations = [
        {'text': 'Raw Text\n• Reviews\n• Social Media\n• Surveys', 'x': 0.25, 'y': 0.8},
        {'text': 'Clean Text\n• Normalized\n• Tokenized', 'x': 0.25, 'y': 0.6},
        {'text': 'Embeddings\n• BERT\n• TF-IDF', 'x': 0.25, 'y': 0.4},
        {'text': 'Predictions', 'x': 0.25, 'y': 0.2},
        {'text': 'Unified\nInsights', 'x': 0.72, 'y': 0.4},
        {'text': 'Dashboard\nAlerts\nReports', 'x': 0.85, 'y': 0.2}
    ]
    
    for ann in annotations:
        ax.text(ann['x'], ann['y'], ann['text'], fontsize=8, 
               style='italic', alpha=0.7)
    
    # Add performance metrics
    metrics_text = 'Performance Metrics:\n• Latency: <100ms\n• Throughput: 10K docs/min\n• Accuracy: >92%'
    ax.text(0.85, 0.8, metrics_text, fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
    
    # Add title
    ax.text(0.5, 0.95, 'End-to-End Sentiment Analysis Architecture', 
           ha='center', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

def main():
    """Generate all transformer visualizations"""
    print("Generating transformer and advanced NLP visualizations...")
    
    import os
    os.makedirs('charts', exist_ok=True)
    
    charts = [
        ('transformer_attention_mechanism', create_transformer_attention_mechanism),
        ('bert_vs_traditional', create_bert_vs_traditional_comparison),
        ('emotion_detection_layers', create_emotion_detection_layers),
        ('sentiment_flow_architecture', create_sentiment_flow_architecture)
    ]
    
    for name, func in charts:
        print(f"Creating {name}...")
        fig = func()
        fig.savefig(f'charts/{name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'charts/{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("All transformer visualizations created successfully!")

if __name__ == "__main__":
    main()