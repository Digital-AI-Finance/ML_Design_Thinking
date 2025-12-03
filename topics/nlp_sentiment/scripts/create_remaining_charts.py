"""
Generate remaining missing NLP charts for Week 3 presentation
Part 2 - Additional charts referenced in LaTeX files
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
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

def create_tokenization_examples():
    """Create tokenization examples visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Word tokenization
    ax1 = axes[0, 0]
    ax1.axis('off')
    text = "I can't believe it's already 2024!"
    tokens = ["I", "can't", "believe", "it's", "already", "2024", "!"]
    
    ax1.text(0.5, 0.8, "Original:", ha='center', fontsize=10, fontweight='bold')
    ax1.text(0.5, 0.65, text, ha='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
    
    ax1.text(0.5, 0.4, "Word Tokens:", ha='center', fontsize=10, fontweight='bold')
    for i, token in enumerate(tokens):
        x = 0.1 + i * 0.125
        rect = Rectangle((x, 0.15), 0.11, 0.15,
                        facecolor=colors['mlblue'], alpha=0.3,
                        edgecolor=colors['mlblue'], linewidth=1)
        ax1.add_patch(rect)
        ax1.text(x + 0.055, 0.225, token, ha='center', va='center', fontsize=8)
    
    ax1.set_title('Word Tokenization', fontsize=12, fontweight='bold')
    
    # Subword tokenization
    ax2 = axes[0, 1]
    ax2.axis('off')
    word = "unbelievable"
    subwords = ["un", "##believ", "##able"]
    
    ax2.text(0.5, 0.8, "Word:", ha='center', fontsize=10, fontweight='bold')
    ax2.text(0.5, 0.65, word, ha='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
    
    ax2.text(0.5, 0.4, "Subword Tokens (BERT):", ha='center', fontsize=10, fontweight='bold')
    for i, subword in enumerate(subwords):
        x = 0.25 + i * 0.17
        rect = Rectangle((x, 0.15), 0.15, 0.15,
                        facecolor=colors['mlorange'], alpha=0.3,
                        edgecolor=colors['mlorange'], linewidth=1)
        ax2.add_patch(rect)
        ax2.text(x + 0.075, 0.225, subword, ha='center', va='center', fontsize=8)
    
    ax2.set_title('Subword Tokenization', fontsize=12, fontweight='bold')
    
    # Character tokenization
    ax3 = axes[1, 0]
    ax3.axis('off')
    word = "NLP"
    chars = list(word)
    
    ax3.text(0.5, 0.8, "Word:", ha='center', fontsize=10, fontweight='bold')
    ax3.text(0.5, 0.65, word, ha='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
    
    ax3.text(0.5, 0.4, "Character Tokens:", ha='center', fontsize=10, fontweight='bold')
    for i, char in enumerate(chars):
        x = 0.35 + i * 0.1
        rect = Rectangle((x, 0.15), 0.08, 0.15,
                        facecolor=colors['mlgreen'], alpha=0.3,
                        edgecolor=colors['mlgreen'], linewidth=1)
        ax3.add_patch(rect)
        ax3.text(x + 0.04, 0.225, char, ha='center', va='center', fontsize=10)
    
    ax3.set_title('Character Tokenization', fontsize=12, fontweight='bold')
    
    # Sentence piece tokenization
    ax4 = axes[1, 1]
    ax4.axis('off')
    sentence = "Machine learning is amazing"
    pieces = ["▁Machine", "▁learn", "ing", "▁is", "▁amaz", "ing"]
    
    ax4.text(0.5, 0.8, "Sentence:", ha='center', fontsize=10, fontweight='bold')
    ax4.text(0.5, 0.65, sentence, ha='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
    
    ax4.text(0.5, 0.4, "SentencePiece Tokens:", ha='center', fontsize=10, fontweight='bold')
    for i, piece in enumerate(pieces):
        x = 0.05 + i * 0.155
        rect = Rectangle((x, 0.15), 0.14, 0.15,
                        facecolor=colors['mlpurple'], alpha=0.3,
                        edgecolor=colors['mlpurple'], linewidth=1)
        ax4.add_patch(rect)
        ax4.text(x + 0.07, 0.225, piece, ha='center', va='center', fontsize=7)
    
    ax4.set_title('SentencePiece Tokenization', fontsize=12, fontweight='bold')
    
    plt.suptitle('Tokenization Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_word_embedding_evolution():
    """Create word embedding evolution timeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Timeline data
    timeline = [
        (2003, 'Bag of Words', 'One-hot encoding', colors['mlgray']),
        (2008, 'TF-IDF', 'Term frequency weighting', colors['mlbrown']),
        (2013, 'Word2Vec', 'Dense embeddings', colors['mlblue']),
        (2014, 'GloVe', 'Global vectors', colors['mlgreen']),
        (2016, 'FastText', 'Subword embeddings', colors['mlorange']),
        (2018, 'ELMo', 'Contextual embeddings', colors['mlpurple']),
        (2018, 'BERT', 'Bidirectional context', colors['mlred']),
        (2020, 'GPT-3', 'Large-scale models', colors['mlcyan'])
    ]
    
    # Draw timeline
    ax.plot([0.1, 0.9], [0.5, 0.5], 'k-', linewidth=2)
    
    for i, (year, name, desc, color) in enumerate(timeline):
        x = 0.1 + i * 0.1
        y = 0.5
        
        # Year marker
        ax.plot(x, y, 'o', markersize=10, color=color)
        
        # Alternate above/below for readability
        if i % 2 == 0:
            y_text = y + 0.15
            y_desc = y + 0.1
        else:
            y_text = y - 0.15
            y_desc = y - 0.1
        
        # Connect to timeline
        ax.plot([x, x], [y, y_text], 'k--', linewidth=0.5, alpha=0.5)
        
        # Add text
        ax.text(x, y_text, f'{year}\n{name}', ha='center', va='center',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        ax.text(x, y_desc, desc, ha='center', va='center',
               fontsize=7, style='italic', alpha=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Evolution of Word Embeddings', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.9, 'From Sparse to Dense to Contextual', ha='center',
           fontsize=12, style='italic')
    
    return fig

def create_transformer_architecture():
    """Create simplified transformer architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Encoder stack
    for i in range(3):
        y = 0.3 + i * 0.15
        
        # Self-attention
        rect1 = Rectangle((0.15, y), 0.2, 0.08,
                         facecolor=colors['mlorange'], alpha=0.3,
                         edgecolor=colors['mlorange'], linewidth=2)
        ax.add_patch(rect1)
        ax.text(0.25, y + 0.04, 'Self-Attn', ha='center', va='center', fontsize=8)
        
        # Add & Norm
        rect2 = Rectangle((0.37, y), 0.12, 0.08,
                         facecolor=colors['mlgreen'], alpha=0.3,
                         edgecolor=colors['mlgreen'], linewidth=2)
        ax.add_patch(rect2)
        ax.text(0.43, y + 0.04, 'Add&Norm', ha='center', va='center', fontsize=7)
        
        # FFN
        rect3 = Rectangle((0.51, y), 0.2, 0.08,
                         facecolor=colors['mlblue'], alpha=0.3,
                         edgecolor=colors['mlblue'], linewidth=2)
        ax.add_patch(rect3)
        ax.text(0.61, y + 0.04, 'Feed Forward', ha='center', va='center', fontsize=8)
        
        # Add & Norm
        rect4 = Rectangle((0.73, y), 0.12, 0.08,
                         facecolor=colors['mlgreen'], alpha=0.3,
                         edgecolor=colors['mlgreen'], linewidth=2)
        ax.add_patch(rect4)
        ax.text(0.79, y + 0.04, 'Add&Norm', ha='center', va='center', fontsize=7)
    
    # Input/Output
    ax.text(0.5, 0.15, 'Input Embeddings + Positional Encoding', ha='center',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlpurple'], alpha=0.3))
    
    ax.text(0.5, 0.8, 'Output Representations', ha='center',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlred'], alpha=0.3))
    
    # Arrows
    ax.arrow(0.5, 0.2, 0, 0.08, head_width=0.02, head_length=0.01,
            fc='black', ec='black')
    ax.arrow(0.5, 0.7, 0, 0.08, head_width=0.02, head_length=0.01,
            fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Transformer Encoder Architecture', fontsize=16, fontweight='bold')
    ax.text(0.1, 0.5, 'Encoder\nStack', ha='center', va='center',
           fontsize=11, fontweight='bold', rotation=90)
    
    return fig

def create_bert_training_process():
    """Create BERT training process visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Masked Language Modeling
    ax1 = axes[0]
    ax1.axis('off')
    
    # Original sentence
    original = ["The", "cat", "sat", "on", "the", "mat"]
    masked = ["The", "[MASK]", "sat", "on", "the", "[MASK]"]
    predicted = ["The", "cat", "sat", "on", "the", "mat"]
    
    ax1.text(0.5, 0.85, 'Masked Language Modeling', ha='center',
            fontsize=12, fontweight='bold')
    
    # Draw original
    ax1.text(0.1, 0.7, 'Original:', fontsize=10, fontweight='bold')
    for i, word in enumerate(original):
        x = 0.3 + i * 0.1
        ax1.text(x, 0.7, word, ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['mlgreen'], alpha=0.3))
    
    # Draw masked
    ax1.text(0.1, 0.5, 'Training:', fontsize=10, fontweight='bold')
    for i, word in enumerate(masked):
        x = 0.3 + i * 0.1
        color = colors['mlred'] if word == "[MASK]" else colors['mlblue']
        ax1.text(x, 0.5, word, ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
    
    # Draw predicted
    ax1.text(0.1, 0.3, 'Predicted:', fontsize=10, fontweight='bold')
    for i, word in enumerate(predicted):
        x = 0.3 + i * 0.1
        is_predicted = masked[i] == "[MASK]"
        color = colors['mlorange'] if is_predicted else colors['mlgray']
        ax1.text(x, 0.3, word, ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Next Sentence Prediction
    ax2 = axes[1]
    ax2.axis('off')
    
    ax2.text(0.5, 0.85, 'Next Sentence Prediction', ha='center',
            fontsize=12, fontweight='bold')
    
    # Sentence pairs
    pairs = [
        ("I love machine learning.", "It's fascinating.", "IsNext", colors['mlgreen']),
        ("The weather is nice.", "Pizza is delicious.", "NotNext", colors['mlred'])
    ]
    
    y_pos = 0.6
    for sent1, sent2, label, color in pairs:
        ax2.text(0.1, y_pos, 'A:', fontweight='bold', fontsize=9)
        ax2.text(0.15, y_pos, sent1, fontsize=9)
        ax2.text(0.1, y_pos - 0.08, 'B:', fontweight='bold', fontsize=9)
        ax2.text(0.15, y_pos - 0.08, sent2, fontsize=9)
        ax2.text(0.7, y_pos - 0.04, label, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        y_pos -= 0.25
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.suptitle('BERT Pre-training Tasks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_bert_variants_comparison():
    """Create BERT variants comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = ['BERT-Base', 'BERT-Large', 'RoBERTa', 'ALBERT', 'DistilBERT', 'ELECTRA']
    params = [110, 340, 355, 12, 66, 110]  # Million parameters
    speed = [100, 40, 35, 120, 180, 95]  # Relative inference speed
    accuracy = [92, 94, 96, 91, 90, 93]  # Example accuracy scores
    
    x = np.arange(len(models))
    
    # Create bubble chart
    for i, (model, param, spd, acc) in enumerate(zip(models, params, speed, accuracy)):
        # Size based on parameters
        size = param * 3
        # Color based on accuracy
        color = colors['mlgreen'] if acc >= 94 else colors['mlblue'] if acc >= 92 else colors['mlorange']
        
        ax.scatter(spd, acc, s=size, alpha=0.6, color=color, edgecolors='black', linewidth=2)
        ax.annotate(f'{model}\n{param}M', (spd, acc), ha='center', fontsize=8)
    
    ax.set_xlabel('Inference Speed (relative)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('BERT Variants: Speed vs Accuracy vs Model Size', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add legend for size
    ax.text(0.95, 0.95, 'Bubble size = Model parameters',
           transform=ax.transAxes, ha='right', va='top',
           fontsize=9, style='italic')
    
    return fig

def create_sentiment_approaches():
    """Create sentiment analysis approaches comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Rule-based
    ax1 = axes[0, 0]
    ax1.axis('off')
    ax1.text(0.5, 0.9, 'Rule-Based', ha='center', fontsize=12, fontweight='bold')
    
    # Draw word lists
    positive = ['good', 'great', 'excellent']
    negative = ['bad', 'terrible', 'awful']
    
    ax1.text(0.25, 0.7, 'Positive Words', fontsize=10, color=colors['mlgreen'], fontweight='bold')
    for i, word in enumerate(positive):
        ax1.text(0.25, 0.6 - i*0.1, f'+ {word}', fontsize=9, color=colors['mlgreen'])
    
    ax1.text(0.75, 0.7, 'Negative Words', fontsize=10, color=colors['mlred'], fontweight='bold')
    for i, word in enumerate(negative):
        ax1.text(0.75, 0.6 - i*0.1, f'- {word}', fontsize=9, color=colors['mlred'])
    
    ax1.text(0.5, 0.2, 'Count words → Calculate score', ha='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
    
    # Machine Learning
    ax2 = axes[0, 1]
    ax2.axis('off')
    ax2.text(0.5, 0.9, 'Machine Learning', ha='center', fontsize=12, fontweight='bold')
    
    # Draw pipeline
    stages = ['Features\n(TF-IDF)', 'Model\n(SVM/NB)', 'Prediction']
    for i, stage in enumerate(stages):
        x = 0.2 + i * 0.3
        rect = Rectangle((x - 0.1, 0.4), 0.2, 0.2,
                        facecolor=colors['mlblue'], alpha=0.3,
                        edgecolor=colors['mlblue'], linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, 0.5, stage, ha='center', va='center', fontsize=9)
        
        if i < len(stages) - 1:
            ax2.arrow(x + 0.1, 0.5, 0.1, 0, head_width=0.03, head_length=0.02,
                     fc='black', ec='black')
    
    # Deep Learning
    ax3 = axes[1, 0]
    ax3.axis('off')
    ax3.text(0.5, 0.9, 'Deep Learning', ha='center', fontsize=12, fontweight='bold')
    
    # Draw neural network
    layers = [3, 4, 3, 1]
    for layer_idx, n_nodes in enumerate(layers):
        x = 0.2 + layer_idx * 0.2
        for node_idx in range(n_nodes):
            y = 0.5 - (n_nodes - 1) * 0.05 + node_idx * 0.1
            circle = Circle((x, y), 0.03, facecolor=colors['mlpurple'], alpha=0.5)
            ax3.add_patch(circle)
            
            # Connect to next layer
            if layer_idx < len(layers) - 1:
                next_n = layers[layer_idx + 1]
                for next_idx in range(next_n):
                    next_y = 0.5 - (next_n - 1) * 0.05 + next_idx * 0.1
                    ax3.plot([x + 0.03, x + 0.17], [y, next_y], 
                            'k-', alpha=0.2, linewidth=0.5)
    
    # Transformer
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'Transformer (BERT)', ha='center', fontsize=12, fontweight='bold')
    
    # Draw attention mechanism
    words = ['Text', 'Understanding', 'Context']
    for i, word in enumerate(words):
        x = 0.2 + i * 0.3
        rect = FancyBboxPatch((x - 0.08, 0.4), 0.16, 0.15,
                              boxstyle="round,pad=0.01",
                              facecolor=colors['mlred'], alpha=0.3,
                              edgecolor=colors['mlred'], linewidth=2)
        ax4.add_patch(rect)
        ax4.text(x, 0.475, word, ha='center', va='center', fontsize=9)
        
        # Draw attention connections
        for j, other in enumerate(words):
            if i != j:
                x2 = 0.2 + j * 0.3
                ax4.plot([x, x2], [0.4, 0.4], 'r-', alpha=0.2, linewidth=1)
    
    ax4.text(0.5, 0.2, 'Bidirectional Attention', ha='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlred'], alpha=0.2))
    
    plt.suptitle('Sentiment Analysis Approaches', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_cross_validation_text():
    """Create cross-validation for text visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create folds visualization
    n_folds = 5
    fold_height = 0.12
    fold_width = 0.7
    
    ax.text(0.5, 0.9, '5-Fold Cross-Validation for Text Data', ha='center',
           fontsize=14, fontweight='bold')
    
    for fold in range(n_folds):
        y = 0.7 - fold * 0.15
        
        # Draw all data blocks
        for i in range(n_folds):
            x = 0.15 + i * 0.14
            
            if i == fold:
                # Test fold
                color = colors['mlred']
                label = 'Test'
            else:
                # Train fold
                color = colors['mlblue']
                label = 'Train'
            
            rect = Rectangle((x, y), 0.12, fold_height,
                            facecolor=color, alpha=0.3,
                            edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            if i == fold:
                ax.text(x + 0.06, y + fold_height/2, label, ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        
        # Fold label
        ax.text(0.05, y + fold_height/2, f'Fold {fold + 1}', ha='center', va='center',
               fontsize=10, fontweight='bold')
        
        # Score
        score = 0.88 + np.random.randn() * 0.03
        ax.text(0.92, y + fold_height/2, f'{score:.2%}', ha='center', va='center',
               fontsize=10)
    
    # Average score
    ax.text(0.5, 0.1, f'Average Score: 89.2% ± 2.1%', ha='center',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlgreen'], alpha=0.3))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

def create_imbalanced_sentiment():
    """Create imbalanced sentiment data visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Imbalanced distribution
    ax1 = axes[0]
    classes = ['Negative', 'Neutral', 'Positive']
    original = [200, 1500, 300]
    
    bars = ax1.bar(classes, original, color=[colors['mlred'], colors['mlgray'], colors['mlgreen']],
                   alpha=0.7)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Original Imbalanced Dataset', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    total = sum(original)
    for bar, val in zip(bars, original):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{val}\n({val/total*100:.1f}%)', ha='center', fontsize=9)
    
    # Balanced strategies
    ax2 = axes[1]
    
    strategies = ['Original', 'Oversample', 'Undersample', 'SMOTE', 'Weighted']
    neg_counts = [200, 1500, 200, 800, 200]
    neu_counts = [1500, 1500, 200, 800, 1500]
    pos_counts = [300, 1500, 200, 800, 300]
    
    x = np.arange(len(strategies))
    width = 0.25
    
    ax2.bar(x - width, neg_counts, width, label='Negative', color=colors['mlred'], alpha=0.7)
    ax2.bar(x, neu_counts, width, label='Neutral', color=colors['mlgray'], alpha=0.7)
    ax2.bar(x + width, pos_counts, width, label='Positive', color=colors['mlgreen'], alpha=0.7)
    
    ax2.set_xlabel('Balancing Strategy', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Balancing Strategies Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_domain_adaptation():
    """Create domain adaptation visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Source domain
    source_rect = FancyBboxPatch((0.1, 0.5), 0.25, 0.3,
                                 boxstyle="round,pad=0.02",
                                 facecolor=colors['mlblue'], alpha=0.3,
                                 edgecolor=colors['mlblue'], linewidth=2)
    ax.add_patch(source_rect)
    ax.text(0.225, 0.65, 'Source Domain\n(Movie Reviews)', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Target domain
    target_rect = FancyBboxPatch((0.65, 0.5), 0.25, 0.3,
                                 boxstyle="round,pad=0.02",
                                 facecolor=colors['mlgreen'], alpha=0.3,
                                 edgecolor=colors['mlgreen'], linewidth=2)
    ax.add_patch(target_rect)
    ax.text(0.775, 0.65, 'Target Domain\n(Product Reviews)', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Transfer learning arrow
    arrow = FancyArrowPatch((0.35, 0.65), (0.65, 0.65),
                           connectionstyle="arc3,rad=.3",
                           arrowstyle='->,head_width=0.4,head_length=0.2',
                           linewidth=3, color=colors['mlorange'])
    ax.add_patch(arrow)
    ax.text(0.5, 0.75, 'Domain Adaptation', ha='center', fontsize=12,
           fontweight='bold', color=colors['mlorange'])
    
    # Shared features
    ax.text(0.5, 0.4, 'Shared Features:\n• Sentiment patterns\n• Language structure\n• Emotion indicators',
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlpurple'], alpha=0.2))
    
    # Domain-specific features
    ax.text(0.225, 0.4, 'Movie-specific:\n• Plot\n• Acting\n• Cinematography',
           ha='center', va='top', fontsize=8, style='italic')
    ax.text(0.775, 0.4, 'Product-specific:\n• Quality\n• Price\n• Shipping',
           ha='center', va='top', fontsize=8, style='italic')
    
    # Performance comparison
    ax.text(0.5, 0.15, 'Without Adaptation: 72% → With Adaptation: 89%',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlred'], alpha=0.2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Domain Adaptation in NLP', fontsize=16, fontweight='bold')
    
    return fig

def create_all_remaining_charts():
    """Generate all remaining missing charts"""
    charts_to_create = [
        ('tokenization_examples', create_tokenization_examples),
        ('word_embedding_evolution', create_word_embedding_evolution),
        ('transformer_architecture', create_transformer_architecture),
        ('bert_training_process', create_bert_training_process),
        ('bert_variants_comparison', create_bert_variants_comparison),
        ('sentiment_approaches', create_sentiment_approaches),
        ('cross_validation_text', create_cross_validation_text),
        ('imbalanced_sentiment', create_imbalanced_sentiment),
        ('domain_adaptation', create_domain_adaptation),
    ]
    
    print("Generating remaining missing charts for Week 3...")
    
    import os
    os.makedirs('../charts', exist_ok=True)
    
    for name, func in charts_to_create:
        print(f"Creating {name}...")
        fig = func()
        fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("All remaining charts created successfully!")

if __name__ == "__main__":
    create_all_remaining_charts()