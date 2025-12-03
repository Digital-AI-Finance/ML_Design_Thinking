"""
Generate NLP and sentiment analysis visualizations for Week 3
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as path_effects

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

def create_emotion_spectrum_heatmap():
    """Create emotion spectrum heatmap - opening power chart"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create emotion data
    contexts = ['Product Review', 'Customer Support', 'Social Media', 'Survey Response', 'Forum Post']
    emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
    
    # Generate realistic emotion intensities
    data = np.array([
        [0.7, 0.8, 0.1, 0.3, 0.2, 0.1, 0.1, 0.6],  # Product Review
        [0.2, 0.3, 0.5, 0.4, 0.6, 0.5, 0.7, 0.2],  # Customer Support
        [0.6, 0.5, 0.3, 0.7, 0.3, 0.3, 0.4, 0.5],  # Social Media
        [0.5, 0.6, 0.2, 0.3, 0.3, 0.2, 0.2, 0.4],  # Survey Response
        [0.4, 0.4, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4],  # Forum Post
    ])
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(emotions)))
    ax.set_yticks(np.arange(len(contexts)))
    ax.set_xticklabels(emotions, fontsize=12)
    ax.set_yticklabels(contexts, fontsize=12)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Emotion Intensity', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations
    for i in range(len(contexts)):
        for j in range(len(emotions)):
            text = ax.text(j, i, f'{data[i, j]:.1f}', ha="center", va="center",
                         color="white" if data[i, j] > 0.5 else "black", fontsize=10)
    
    ax.set_title('Emotion Spectrum Across Different Contexts', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_text_preprocessing_pipeline():
    """Create text preprocessing pipeline visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Pipeline stages
    stages = [
        ('Raw Text', 'HTML, emojis,\nspecial chars'),
        ('Clean', 'Remove noise\nNormalize'),
        ('Tokenize', 'Split into\nwords/subwords'),
        ('Process', 'Lowercase\nStem/Lemma'),
        ('Vectorize', 'Convert to\nnumbers'),
        ('Model Ready', 'Input to\nBERT/ML')
    ]
    
    # Draw pipeline
    y_pos = 0.5
    for i, (stage, desc) in enumerate(stages):
        x_pos = i / (len(stages) - 1)
        
        # Draw box
        if i == 0:
            color = colors['mlred']
        elif i == len(stages) - 1:
            color = colors['mlgreen']
        else:
            color = colors['mlblue']
            
        rect = FancyBboxPatch((x_pos - 0.08, y_pos - 0.15), 0.16, 0.3,
                              boxstyle="round,pad=0.01", 
                              facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x_pos, y_pos + 0.05, stage, ha='center', va='center',
               fontsize=12, fontweight='bold')
        ax.text(x_pos, y_pos - 0.05, desc, ha='center', va='center',
               fontsize=9, style='italic')
        
        # Draw arrow
        if i < len(stages) - 1:
            ax.arrow(x_pos + 0.09, y_pos, 0.06, 0, 
                    head_width=0.03, head_length=0.02,
                    fc=colors['mlgray'], ec=colors['mlgray'])
    
    # Add example text transformation
    examples = [
        "<p>I LOVE this!!! 😊</p>",
        "I LOVE this!!!",
        "['I', 'LOVE', 'this']",
        "['i', 'love', 'this']",
        "[0.2, 0.8, 0.3]",
        "Tensor([...])"
    ]
    
    for i, example in enumerate(examples):
        x_pos = i / (len(stages) - 1)
        ax.text(x_pos, 0.05, example, ha='center', va='center',
               fontsize=8, style='italic', color=colors['mlgray'])
    
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 0.85)
    ax.set_title('Text Preprocessing Pipeline', fontsize=16, fontweight='bold')
    
    return fig

def create_word_embedding_space():
    """Create 3D word embedding visualization"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create word clusters
    categories = {
        'Positive': ['excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'love', 'perfect'],
        'Negative': ['terrible', 'awful', 'horrible', 'bad', 'hate', 'worst', 'disappointing'],
        'Product': ['quality', 'feature', 'design', 'price', 'value', 'performance', 'size'],
        'Service': ['support', 'help', 'response', 'team', 'customer', 'service', 'staff']
    }
    
    # Generate embeddings (simulated)
    for cat_idx, (category, words) in enumerate(categories.items()):
        # Create cluster center
        if category == 'Positive':
            center = np.array([5, 5, 5])
            color = colors['mlgreen']
        elif category == 'Negative':
            center = np.array([-5, -5, -5])
            color = colors['mlred']
        elif category == 'Product':
            center = np.array([5, -5, 0])
            color = colors['mlblue']
        else:
            center = np.array([-5, 5, 0])
            color = colors['mlorange']
        
        # Plot points
        points = center + np.random.randn(len(words), 3) * 1.5
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=color, s=100, alpha=0.6, label=category)
        
        # Add word labels
        for word, point in zip(words, points):
            ax.text(point[0], point[1], point[2], word,
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Dimension 1', fontsize=10)
    ax.set_ylabel('Dimension 2', fontsize=10)
    ax.set_zlabel('Dimension 3', fontsize=10)
    ax.set_title('Word Embeddings in 3D Space', fontsize=16, fontweight='bold')
    ax.legend()
    
    return fig

def create_sentiment_distribution():
    """Create sentiment distribution chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Simple sentiment distribution
    ax1 = axes[0]
    sentiments = ['Negative', 'Neutral', 'Positive']
    counts = [3500, 2000, 4500]
    colors_list = [colors['mlred'], colors['mlgray'], colors['mlgreen']]
    
    bars = ax1.bar(sentiments, counts, color=colors_list, alpha=0.7)
    ax1.set_ylabel('Number of Reviews', fontsize=12)
    ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count/total*100:.1f}%', ha='center', fontsize=10)
    
    # Sentiment over time
    ax2 = axes[1]
    days = np.arange(30)
    positive = 50 + 10 * np.sin(days/5) + np.random.randn(30) * 3
    negative = 30 - 5 * np.sin(days/5) + np.random.randn(30) * 3
    neutral = 20 + np.random.randn(30) * 2
    
    ax2.plot(days, positive, color=colors['mlgreen'], linewidth=2, label='Positive')
    ax2.plot(days, negative, color=colors['mlred'], linewidth=2, label='Negative')
    ax2.plot(days, neutral, color=colors['mlgray'], linewidth=2, label='Neutral')
    
    ax2.set_xlabel('Days', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_attention_visualization():
    """Create attention weights heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample sentence
    words = ['The', 'product', 'quality', 'is', 'excellent', 'but', 'the', 'price', 'is', 'high']
    
    # Generate attention weights (simulated)
    n_words = len(words)
    attention_weights = np.random.rand(n_words, n_words)
    
    # Make it more realistic (diagonal and semantic relationships stronger)
    for i in range(n_words):
        attention_weights[i, i] += 0.5  # Self-attention
    
    # Semantic relationships
    attention_weights[1, 2] += 0.4  # product -> quality
    attention_weights[2, 1] += 0.4  # quality -> product
    attention_weights[2, 4] += 0.3  # quality -> excellent
    attention_weights[7, 9] += 0.3  # price -> high
    
    # Normalize
    attention_weights = attention_weights / attention_weights.max()
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='equal')
    
    # Set labels
    ax.set_xticks(np.arange(n_words))
    ax.set_yticks(np.arange(n_words))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_yticklabels(words)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Add title
    ax.set_title('Self-Attention Weights in BERT', fontsize=16, fontweight='bold')
    ax.set_xlabel('Keys/Values', fontsize=12)
    ax.set_ylabel('Queries', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_model_comparison():
    """Create model performance comparison chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    ax1 = axes[0]
    models = ['Rule-Based', 'Naive Bayes', 'SVM', 'LSTM', 'BERT', 'RoBERTa']
    accuracy = [68, 75, 82, 87, 92, 94]
    speed = [1000, 500, 200, 50, 20, 15]  # texts/second
    
    colors_list = [colors['mlgray'], colors['mlorange'], colors['mlblue'], 
                   colors['mlpurple'], colors['mlgreen'], colors['mlred']]
    
    bars = ax1.barh(models, accuracy, color=colors_list, alpha=0.7)
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add values
    for bar, acc in zip(bars, accuracy):
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc}%', ha='left', va='center', fontsize=10)
    
    # Speed vs Accuracy tradeoff
    ax2 = axes[1]
    ax2.scatter(speed, accuracy, s=200, c=colors_list, alpha=0.7)
    
    for i, model in enumerate(models):
        ax2.annotate(model, (speed[i], accuracy[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)
    
    ax2.set_xlabel('Processing Speed (texts/second)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Speed vs Accuracy Tradeoff', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix():
    """Create confusion matrix for sentiment classification"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Confusion matrix data
    labels = ['Negative', 'Neutral', 'Positive']
    cm = np.array([[850, 100, 50],
                   [150, 700, 150],
                   [30, 70, 900]])
    
    # Create heatmap
    im = ax.imshow(cm, cmap='Blues', aspect='equal')
    
    # Set labels
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, str(cm[i, j]),
                          ha="center", va="center",
                          color="white" if cm[i, j] > 500 else "black",
                          fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Sentiment Classification Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig

def create_emotional_journey_map():
    """Create emotional journey map"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Journey stages
    stages = ['Discover', 'Research', 'Purchase', 'Onboard', 'Use', 'Support', 'Advocate']
    x = np.arange(len(stages))
    
    # Emotion scores for different user segments
    segment1 = [0.6, 0.5, 0.7, 0.3, 0.8, 0.2, 0.9]  # Happy path
    segment2 = [0.4, 0.3, 0.5, -0.2, 0.2, -0.5, 0.1]  # Struggled
    segment3 = [0.5, 0.6, 0.6, 0.4, 0.5, 0.3, 0.6]  # Average
    
    # Plot lines
    ax.plot(x, segment1, marker='o', markersize=10, linewidth=3, 
           color=colors['mlgreen'], label='Delighted Users', alpha=0.8)
    ax.plot(x, segment2, marker='s', markersize=10, linewidth=3,
           color=colors['mlred'], label='Frustrated Users', alpha=0.8)
    ax.plot(x, segment3, marker='^', markersize=10, linewidth=3,
           color=colors['mlblue'], label='Average Users', alpha=0.8)
    
    # Add emotion zones
    ax.axhspan(0.5, 1.0, alpha=0.1, color='green', label='Positive')
    ax.axhspan(-0.5, 0.5, alpha=0.1, color='gray', label='Neutral')
    ax.axhspan(-1.0, -0.5, alpha=0.1, color='red', label='Negative')
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=12)
    ax.set_ylabel('Emotion Score', fontsize=12)
    ax.set_ylim(-1, 1)
    ax.set_title('Emotional Journey Map Across User Segments', fontsize=16, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_priority_matrix():
    """Create priority matrix for issues"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Issue data
    issues = [
        ('Login Problems', 85, -0.8, 500),
        ('Slow Performance', 70, -0.7, 400),
        ('Missing Features', 60, -0.5, 300),
        ('UI Confusion', 45, -0.6, 350),
        ('Documentation', 30, -0.4, 200),
        ('Payment Issues', 95, -0.9, 150),
        ('Data Export', 20, -0.3, 100),
        ('Mobile App', 55, -0.6, 450),
        ('Notifications', 35, -0.4, 250),
        ('Search Function', 50, -0.5, 320)
    ]
    
    # Extract data
    names, frequency, sentiment, size = zip(*issues)
    
    # Create scatter plot
    scatter = ax.scatter(frequency, [-s for s in sentiment], s=size, 
                        c=size, cmap='YlOrRd', alpha=0.6, edgecolors='black')
    
    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name, (frequency[i], -sentiment[i]),
                   fontsize=9, ha='center')
    
    # Add quadrant lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(75, 0.85, 'HIGH PRIORITY', fontsize=12, fontweight='bold',
           color=colors['mlred'], ha='center')
    ax.text(25, 0.85, 'MONITOR', fontsize=12, fontweight='bold',
           color=colors['mlorange'], ha='center')
    ax.text(75, 0.15, 'QUICK WINS', fontsize=12, fontweight='bold',
           color=colors['mlgreen'], ha='center')
    ax.text(25, 0.15, 'LOW PRIORITY', fontsize=12, fontweight='bold',
           color=colors['mlgray'], ha='center')
    
    # Labels and title
    ax.set_xlabel('Frequency (mentions per day)', fontsize=12)
    ax.set_ylabel('Negative Sentiment Intensity', fontsize=12)
    ax.set_title('Issue Priority Matrix', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Impact Score', rotation=270, labelpad=20)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# Generate all charts
if __name__ == "__main__":
    import os
    os.makedirs('../charts', exist_ok=True)
    
    print("Generating NLP visualization charts...")
    
    # Generate each chart
    charts = [
        ('emotion_spectrum_heatmap', create_emotion_spectrum_heatmap),
        ('text_preprocessing_pipeline', create_text_preprocessing_pipeline),
        ('word_embedding_space', create_word_embedding_space),
        ('sentiment_distribution', create_sentiment_distribution),
        ('attention_visualization', create_attention_visualization),
        ('model_comparison', create_model_comparison),
        ('confusion_matrix', create_confusion_matrix),
        ('emotional_journey_map', create_emotional_journey_map),
        ('priority_matrix', create_priority_matrix)
    ]
    
    for name, func in charts:
        print(f"Creating {name}...")
        fig = func()
        fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("All charts created successfully!")
    plt.close('all')