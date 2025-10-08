"""
Create all visualization charts for Week 3: NLP for Emotional Context
Generates all charts referenced in the presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define color palette matching the presentation
colors = {
    'mlblue': '#0066CC',
    'mlpurple': '#3333B2',
    'mllavender': '#ADADE0',
    'mllavender2': '#C1C1E8',
    'mllavender3': '#CCCCEB',
    'mllavender4': '#D6D6EF',
    'mlorange': '#FF7F0E',
    'mlgreen': '#2CA02C',
    'mlred': '#D62728',
    'mlgray': '#7F7F7F',
    'mlyellow': '#FFCE54'
}

def save_figure(name):
    """Save figure in both PDF and PNG formats"""
    plt.savefig(f'{name}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{name}.png', dpi=150, bbox_inches='tight', format='png')
    plt.close()

# ==================== PART 1: THE CHALLENGE ====================

def create_review_volume_growth():
    """Chart showing exponential growth of review volume"""
    fig, ax = plt.subplots(figsize=(8, 5))

    days = np.arange(0, 30)
    reviews_cumulative = 50000 * (1 - np.exp(-days/10))
    new_daily = 50000/10 * np.exp(-days/10) * 100

    ax.plot(days, reviews_cumulative, color=colors['mlpurple'], linewidth=3, label='Total Reviews')
    ax.bar(days, new_daily, color=colors['mlorange'], alpha=0.6, label='New Daily')

    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Number of Reviews', fontsize=12)
    ax.set_title('Review Volume: Exponential Growth Problem', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Already behind!', xy=(7, 30000), xytext=(15, 35000),
                arrowprops=dict(arrowstyle='->', color=colors['mlred'], lw=2))

    save_figure('review_volume_growth')

def create_context_dependency():
    """Show how context changes word meaning"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Word "cold" in different contexts
    contexts = [
        ('Restaurant', 'Soup was cold', -0.8, colors['mlred']),
        ('Service', 'Support was cold', -0.6, colors['mlorange']),
        ('Beer', 'Nice and cold', 0.9, colors['mlgreen']),
        ('Logic', 'Cold hard facts', 0.0, colors['mlgray'])
    ]

    for ax, (context, phrase, sentiment, color) in zip(axes.flat, contexts):
        ax.barh([0], [sentiment], color=color, height=0.5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(f'{context}: "{phrase}"', fontsize=11, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=10)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Same Word "Cold" - Different Meanings by Context', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('context_dependency')

def create_complexity_explosion():
    """Visualize the multiplication of complexity"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create tree structure showing multiplication
    levels = [
        ('Words', 10000, 0, colors['mlblue']),
        ('× Emotions', 50000, 1, colors['mlgreen']),
        ('× Contexts', 150000, 2, colors['mlorange']),
        ('× Cultures', 600000, 3, colors['mlred'])
    ]

    for i, (label, value, level, color) in enumerate(levels):
        width = value / 600000 * 8
        rect = FancyBboxPatch((5 - width/2, 3 - level), width, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=color, alpha=0.6,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(5, 3.3 - level, f'{label}: {value:,}',
               ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title('Complexity Explosion: 600,000 Combinations', fontsize=14, fontweight='bold')

    save_figure('complexity_explosion')

def create_information_loss():
    """Show information theory perspective on compression"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Information content breakdown
    components = ['Words', 'Position', 'Context', 'Emotion', 'Sarcasm']
    bits = [14, 7, 10, 3, 1]
    colors_list = [colors['mlblue'], colors['mlgreen'], colors['mlpurple'],
                   colors['mlorange'], colors['mlred']]

    ax1.pie(bits, labels=components, colors=colors_list, autopct='%1.1f%%',
            startangle=90)
    ax1.set_title('Information per Word (35 bits)', fontsize=12, fontweight='bold')

    # Right: Compression loss
    methods = ['Original\n(175M bits)', 'Keywords\n(1.7M bits)', 'Lost\n(173.3M bits)']
    values = [175, 1.7, 173.3]
    colors_bars = [colors['mlgreen'], colors['mlblue'], colors['mlred']]

    ax2.bar(methods, values, color=colors_bars, alpha=0.7)
    ax2.set_ylabel('Information (Million bits)', fontsize=11)
    ax2.set_title('99% Information Lost in Compression', fontsize=12, fontweight='bold')

    # Add percentage annotation
    ax2.text(2, 173.3/2, '99%\nLOST', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')

    plt.tight_layout()
    save_figure('information_loss')

# ==================== PART 2: FIRST SOLUTION & LIMITS ====================

def create_sentiment_word_lists():
    """Visualize positive/negative word lists"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Positive words
    pos_words = ['excellent', 'amazing', 'love', 'great', 'wonderful',
                 'fantastic', 'perfect', 'best', 'awesome', 'brilliant']
    pos_scores = np.random.uniform(0.7, 1.0, len(pos_words))

    ax1.barh(range(len(pos_words)), pos_scores, color=colors['mlgreen'])
    ax1.set_yticks(range(len(pos_words)))
    ax1.set_yticklabels(pos_words)
    ax1.set_xlabel('Positive Score', fontsize=11)
    ax1.set_title('Positive Word List', fontsize=12, fontweight='bold', color=colors['mlgreen'])
    ax1.set_xlim(0, 1)

    # Negative words
    neg_words = ['terrible', 'awful', 'hate', 'horrible', 'worst',
                 'bad', 'poor', 'useless', 'broken', 'disappointed']
    neg_scores = np.random.uniform(0.7, 1.0, len(neg_words))

    ax2.barh(range(len(neg_words)), neg_scores, color=colors['mlred'])
    ax2.set_yticks(range(len(neg_words)))
    ax2.set_yticklabels(neg_words)
    ax2.set_xlabel('Negative Score', fontsize=11)
    ax2.set_title('Negative Word List', fontsize=12, fontweight='bold', color=colors['mlred'])
    ax2.set_xlim(0, 1)

    plt.suptitle('Traditional Sentiment Analysis: Word Lists', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('sentiment_word_lists')

def create_bow_visualization():
    """Visualize Bag of Words process"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Original text
    ax = axes[0, 0]
    ax.text(0.5, 0.5, 'Original:\n"The product quality is\nexcellent excellent excellent\nbut service terrible terrible"',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mllavender4']))
    ax.set_title('1. Original Review', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Tokenized
    ax = axes[0, 1]
    words = ['The', 'product', 'quality', 'is', 'excellent', 'excellent',
             'excellent', 'but', 'service', 'terrible', 'terrible']
    for i, word in enumerate(words):
        color = colors['mlgreen'] if 'excellent' in word else colors['mlred'] if 'terrible' in word else colors['mlgray']
        ax.text((i % 4) * 0.25 + 0.1, 0.8 - (i // 4) * 0.3, word,
               fontsize=10, color=color, weight='bold' if color != colors['mlgray'] else 'normal')
    ax.set_title('2. Tokenized Words', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Word counts
    ax = axes[1, 0]
    words_unique = ['excellent', 'terrible', 'product', 'quality', 'service']
    counts = [3, 2, 1, 1, 1]
    colors_bars = [colors['mlgreen'], colors['mlred'], colors['mlgray'], colors['mlgray'], colors['mlgray']]
    ax.bar(words_unique, counts, color=colors_bars)
    ax.set_title('3. Word Counts', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency')

    # Vector representation
    ax = axes[1, 1]
    vector = [0, 1, 1, 1, 3, 0, 0, 1, 2, 0]
    ax.bar(range(len(vector)), vector, color=colors['mlpurple'])
    ax.set_title('4. Vector (10,000 dimensions)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Word Index')
    ax.set_ylabel('Count')
    ax.text(0.5, 0.9, 'Mostly zeros (sparse)', transform=ax.transAxes,
           fontsize=10, style='italic', ha='center')

    plt.tight_layout()
    save_figure('bow_visualization')

def create_performance_degradation():
    """Show performance drop with complexity"""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Simple\nDirect', 'Mixed\nSentiment', 'Sarcasm', 'Context\nDependent',
                  'Subtle\nEmotion', 'Negation']
    traditional = [95, 67, 23, 31, 44, 28]

    x = np.arange(len(categories))
    bars = ax.bar(x, traditional, color=[colors['mlgreen'] if v > 60 else colors['mlorange']
                                         if v > 40 else colors['mlred'] for v in traditional])

    # Add value labels on bars
    for bar, val in zip(bars, traditional):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{val}%', ha='center', va='bottom', fontweight='bold')

    ax.axhline(50, color=colors['mlgray'], linestyle='--', alpha=0.5, label='Random Guess (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Traditional NLP Performance Degradation', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure('performance_degradation')

def create_information_preserved():
    """Pie chart showing what information is kept vs lost"""
    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [18, 82]
    labels = ['Kept (18%)\nWord counts', 'Lost (82%)\nOrder, context,\nrelationships']
    colors_pie = [colors['mlgreen'], colors['mlred']]
    explode = (0.1, 0)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
           autopct='%1.0f%%', shadow=True, startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})

    ax.set_title('Information Preserved in Bag of Words', fontsize=14, fontweight='bold')

    save_figure('information_preserved')

# ==================== PART 3: THE BREAKTHROUGH ====================

def create_human_attention_process():
    """Visualize how humans selectively attend"""
    fig, ax = plt.subplots(figsize=(10, 6))

    words = ['I', 'went', 'to', 'the', 'bank', '.', 'The', 'water', 'was', 'nice', '.']
    positions = np.arange(len(words))

    # Draw words
    for i, word in enumerate(words):
        size = 20 if word in ['bank', 'water'] else 12
        weight = 'bold' if word in ['bank', 'water'] else 'normal'
        color = colors['mlpurple'] if word == 'bank' else colors['mlblue'] if word == 'water' else colors['mlgray']
        ax.text(i, 0.5, word, fontsize=size, weight=weight, ha='center', color=color)

    # Draw attention arrow
    arrow = FancyArrowPatch((7, 0.3), (4, 0.3),
                           connectionstyle="arc3,rad=-.3",
                           arrowstyle='->', lw=3,
                           color=colors['mlgreen'])
    ax.add_patch(arrow)
    ax.text(5.5, 0.1, '55% attention', fontsize=11, color=colors['mlgreen'], weight='bold')

    ax.set_xlim(-0.5, len(words) - 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Human Attention: "water" helps understand "bank"', fontsize=14, fontweight='bold')

    save_figure('human_attention_process')

def create_word_vectors_2d():
    """2D visualization of word vectors"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Word positions (manually placed for clarity)
    words = {
        'river': (3, 4),
        'water': (2.5, 3.5),
        'stream': (3.5, 3.8),
        'bank': (0, 0),
        'money': (-3, -2),
        'finance': (-2.5, -2.5),
        'loan': (-3.5, -1.5)
    }

    # Draw vectors
    for word, (x, y) in words.items():
        color = colors['mlblue'] if word in ['river', 'water', 'stream'] else colors['mlgreen']
        ax.arrow(0, 0, x, y, head_width=0.2, head_length=0.2,
                fc=color, ec=color, alpha=0.6, width=0.05)
        ax.text(x*1.1, y*1.1, word, fontsize=11, weight='bold')

    # Draw angle arc for similarity
    angle1 = np.arctan2(4, 3)
    angle2 = np.arctan2(3.5, 2.5)
    theta = np.linspace(angle2, angle1, 20)
    r = 1.5
    ax.plot(r*np.cos(theta), r*np.sin(theta), color=colors['mlpurple'], linewidth=2)
    ax.text(1.8, 1.8, 'Similar\ndirection', fontsize=10, color=colors['mlpurple'])

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.set_title('Word Vectors in 2D Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    save_figure('word_vectors_2d')

def create_attention_three_steps():
    """Visualize the three steps of attention"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    steps = [
        ('Step 1: SCORE\nFind Relevance', colors['mlblue']),
        ('Step 2: NORMALIZE\nCreate Percentages', colors['mlorange']),
        ('Step 3: AGGREGATE\nBlend Information', colors['mlgreen'])
    ]

    for ax, (title, color) in zip(axes, steps):
        # Draw box for step
        rect = FancyBboxPatch((0.1, 0.3), 0.8, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, 0.5, title, ha='center', va='center',
               fontsize=11, weight='bold')

        # Draw arrow to next step
        if ax != axes[-1]:
            ax.arrow(0.92, 0.5, 0.06, 0, head_width=0.05,
                    head_length=0.02, fc='black', ec='black')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle('Attention Mechanism: Three Essential Steps', fontsize=14, fontweight='bold')
    save_figure('attention_three_steps')

def create_bert_bidirectional():
    """Show bidirectional vs unidirectional reading"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    sentence = ['The', 'bank', 'charges', 'are', 'too', 'high']

    # Unidirectional (top)
    for i, word in enumerate(sentence):
        color = colors['mlpurple'] if word == 'bank' else colors['mlgray']
        if i <= 1:  # Can see
            alpha = 1.0
            box_color = colors['mllavender4']
        else:  # Cannot see
            alpha = 0.3
            box_color = 'white'

        ax1.text(i, 0.5, word, fontsize=12, ha='center', alpha=alpha,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color))

    ax1.arrow(0, 0.3, 1.5, 0, head_width=0.05, head_length=0.1,
             fc=colors['mlred'], ec=colors['mlred'])
    ax1.text(0.75, 0.15, 'Can only see left', fontsize=10, color=colors['mlred'])
    ax1.set_xlim(-0.5, len(sentence) - 0.5)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Traditional (Left-to-Right): 50% chance of error',
                  fontsize=12, fontweight='bold')

    # Bidirectional (bottom)
    for i, word in enumerate(sentence):
        color = colors['mlpurple'] if word == 'bank' else colors['mlgray']
        ax2.text(i, 0.5, word, fontsize=12, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mllavender4']))

    # Arrows in both directions
    ax2.arrow(1, 0.3, -0.8, 0, head_width=0.05, head_length=0.1,
             fc=colors['mlgreen'], ec=colors['mlgreen'])
    ax2.arrow(1, 0.3, 3, 0, head_width=0.05, head_length=0.1,
             fc=colors['mlgreen'], ec=colors['mlgreen'])
    ax2.text(2.5, 0.15, 'Can see both directions', fontsize=10, color=colors['mlgreen'])

    ax2.set_xlim(-0.5, len(sentence) - 0.5)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('BERT (Bidirectional): 99% accurate',
                  fontsize=12, fontweight='bold')

    plt.suptitle('Bidirectional Understanding Changes Everything',
                fontsize=14, fontweight='bold')
    save_figure('bert_bidirectional')

def create_pretraining_scale():
    """Visualize the scale of pre-training data"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Data sources pie chart
    sizes = [2500, 800]
    labels = ['Wikipedia\n2.5B words', 'BookCorpus\n0.8B words']
    colors_pie = [colors['mlblue'], colors['mlgreen']]

    ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('3.3 Billion Words of Training Data', fontsize=12, fontweight='bold')

    # Equivalent scale
    items = ['Books\n(16,500)', 'Years to read\n(104)', 'Training time\n(4 days)', 'Cost\n($50-100K)']
    values = [16500, 104, 4, 75]
    colors_bars = [colors['mlpurple'], colors['mlorange'], colors['mlgreen'], colors['mlred']]

    bars = ax2.bar(items, values, color=colors_bars, alpha=0.7)
    ax2.set_ylabel('Scale', fontsize=11)
    ax2.set_title('Pre-training Investment', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{val:,}' if val > 10 else f'{val}', ha='center', fontweight='bold')

    plt.tight_layout()
    save_figure('pretraining_scale')

def create_finetuning_process():
    """Show the fine-tuning process"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw process flow
    steps = [
        (1, 3, 'Pre-trained\nBERT\n(3.3B words)', colors['mlblue']),
        (3, 3, '→', 'black'),
        (4, 3, 'Your Data\n(1000 reviews)', colors['mlorange']),
        (6, 3, '→', 'black'),
        (7.5, 3, 'Fine-tuned\nModel', colors['mlgreen'])
    ]

    for x, y, text, color in steps:
        if text == '→':
            ax.arrow(x, y, 0.5, 0, head_width=0.2, head_length=0.1,
                    fc=color, ec=color)
        else:
            rect = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, alpha=0.3,
                                  edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center',
                   fontsize=11, weight='bold')

    # Add details
    details = [
        (1, 2, '• General language\n• World knowledge\n• No domain specific'),
        (4, 2, '• Your products\n• Your users\n• Your language'),
        (7.5, 2, '• Both general AND\n  specific knowledge')
    ]

    for x, y, text in details:
        ax.text(x, y, text, ha='center', va='top', fontsize=9)

    ax.set_xlim(0, 9)
    ax.set_ylim(1, 4)
    ax.axis('off')
    ax.set_title('Fine-tuning: Teaching BERT Your Specific Task',
                fontsize=14, fontweight='bold')

    save_figure('finetuning_process')

def create_performance_comparison_final():
    """Final performance comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 7))

    categories = ['Simple\nReviews', 'Sarcasm', 'Context\nDependent', 'Overall']
    bow_scores = [95, 23, 34, 51]
    bert_scores = [98, 89, 94, 93]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, bow_scores, width, label='Bag of Words',
                   color=colors['mlred'], alpha=0.7)
    bars2 = ax.bar(x + width/2, bert_scores, width, label='BERT',
                   color=colors['mlgreen'], alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}%', ha='center', va='bottom', fontweight='bold')

    # Add improvement arrows
    for i in range(len(categories)):
        if bert_scores[i] > bow_scores[i]:
            improvement = bert_scores[i] - bow_scores[i]
            ax.annotate(f'+{improvement}%',
                       xy=(x[i], bow_scores[i] + 5),
                       fontsize=10, color=colors['mlpurple'],
                       fontweight='bold', ha='center')

    ax.axhline(79, color=colors['mlgray'], linestyle='--', alpha=0.5,
              label='Human Agreement (79%)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('The Transformer Breakthrough: Context Problem Solved',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    save_figure('performance_comparison_final')

# ==================== PART 4: DESIGN SYNTHESIS ====================

def create_emotion_taxonomy():
    """Create emotion taxonomy hierarchy"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Main emotions with percentages
    emotions = [
        ('Frustration', 28, colors['mlred'], 0.2, 0.7),
        ('Delight', 22, colors['mlgreen'], 0.5, 0.8),
        ('Confusion', 18, colors['mlorange'], 0.8, 0.7),
        ('Trust', 15, colors['mlblue'], 0.2, 0.4),
        ('Disappointment', 12, colors['mlpurple'], 0.5, 0.3),
        ('Satisfaction', 5, colors['mlgray'], 0.8, 0.4)
    ]

    # Draw circles for each emotion
    for emotion, percent, color, x, y in emotions:
        circle = Circle((x, y), percent/100, color=color, alpha=0.6)
        ax.add_patch(circle)
        ax.text(x, y, f'{emotion}\n{percent}%', ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Emotion Taxonomy from 50,000 Reviews', fontsize=14, fontweight='bold')

    save_figure('emotion_taxonomy')

def create_user_journey_emotions():
    """Create emotional journey timeline"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Timeline data
    stages = ['Day 1', 'Week 1', 'Week 2', 'Week 4', 'Month 2', 'Month 3']
    x_pos = np.arange(len(stages))

    # Emotion levels for different aspects
    excitement = [90, 70, 50, 40, 30, 25]
    confusion = [60, 80, 50, 30, 20, 20]
    mastery = [10, 20, 40, 60, 70, 70]
    satisfaction = [40, 30, 50, 70, 60, 50]

    # Plot lines
    ax.plot(x_pos, excitement, 'o-', color=colors['mlorange'], linewidth=2, label='Excitement')
    ax.plot(x_pos, confusion, 's-', color=colors['mlred'], linewidth=2, label='Confusion')
    ax.plot(x_pos, mastery, '^-', color=colors['mlgreen'], linewidth=2, label='Mastery')
    ax.plot(x_pos, satisfaction, 'd-', color=colors['mlblue'], linewidth=2, label='Satisfaction')

    # Add critical points
    ax.axvspan(0.5, 1.5, alpha=0.2, color=colors['mlred'], label='Critical Period')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages)
    ax.set_ylabel('Emotion Level (%)', fontsize=12)
    ax.set_title('Emotional User Journey Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)

    save_figure('user_journey_emotions')

def create_churn_priority_matrix():
    """Create priority matrix for churn factors"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Factors with impact and frequency
    factors = [
        ('Frustration +\nConfusion', 72, 35, colors['mlred']),
        ('Disappointment +\nDistrust', 64, 25, colors['mlorange']),
        ('Boredom', 45, 40, colors['mlyellow']),
        ('Minor Issues', 8, 60, colors['mlgreen']),
        ('Price Concerns', 30, 20, colors['mlblue'])
    ]

    for name, impact, frequency, color in factors:
        size = impact * 20  # Size based on churn impact
        ax.scatter(frequency, impact, s=size, color=color, alpha=0.6, edgecolors='black', linewidth=2)
        ax.text(frequency, impact, name, ha='center', va='center', fontsize=10, fontweight='bold')

    # Add quadrant lines
    ax.axhline(50, color=colors['mlgray'], linestyle='--', alpha=0.5)
    ax.axvline(30, color=colors['mlgray'], linestyle='--', alpha=0.5)

    # Quadrant labels
    ax.text(15, 85, 'High Impact\nLow Frequency', fontsize=9, alpha=0.5, ha='center')
    ax.text(45, 85, 'CRITICAL\nHigh Impact\nHigh Frequency', fontsize=9, alpha=0.5, ha='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlred'], alpha=0.2))
    ax.text(15, 15, 'Low Priority', fontsize=9, alpha=0.5, ha='center')
    ax.text(45, 15, 'Quick Wins', fontsize=9, alpha=0.5, ha='center')

    ax.set_xlabel('Frequency (%)', fontsize=12)
    ax.set_ylabel('Churn Impact (%)', fontsize=12)
    ax.set_title('Emotion-Based Churn Priority Matrix', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    save_figure('churn_priority_matrix')

def create_empathy_scale_pyramid():
    """Create empathy at scale visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Pyramid levels
    levels = [
        ('1M Users', 1, colors['mlpurple'], 6),
        ('10K Segments', 0.8, colors['mlblue'], 5),
        ('100 Personas', 0.6, colors['mlgreen'], 4),
        ('10 Key Insights', 0.4, colors['mlorange'], 3),
        ('1 Strategy', 0.2, colors['mlred'], 2)
    ]

    for label, width, color, y in levels:
        rect = FancyBboxPatch((0.5 - width/2, y - 0.4), width, 0.8,
                              boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.6,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y, label, ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

    # Add processing time annotations
    times = ['24 hours', '4 hours', '1 hour', '10 minutes', '1 minute']
    for time, y in zip(times, [6, 5, 4, 3, 2]):
        ax.text(1.1, y, time, fontsize=10, style='italic')

    ax.set_xlim(0, 1.5)
    ax.set_ylim(1, 7)
    ax.axis('off')
    ax.set_title('Empathy at Scale: From Millions to One', fontsize=14, fontweight='bold')

    save_figure('empathy_scale_pyramid')

def create_spotify_wrapped_emotions():
    """Spotify Wrapped emotion distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Emotion words used
    emotions = ['Rebellious', 'Nostalgic', 'Adventurous', 'Focused', 'Chill', 'Energetic']
    percentages = [18, 25, 15, 12, 20, 10]

    ax1.barh(emotions, percentages, color=[colors['mlpurple'], colors['mlblue'],
                                           colors['mlgreen'], colors['mlorange'],
                                           colors['mlgray'], colors['mlred']])
    ax1.set_xlabel('% of Users', fontsize=11)
    ax1.set_title('Emotion Words That Drove Sharing', fontsize=12, fontweight='bold')

    # Engagement metrics
    metrics = ['Users\nEngaged', 'Social\nShares', 'App Usage\nIncrease', 'New\nSubs']
    values = [120, 60, 40, 21]
    colors_bars = [colors['mlgreen'], colors['mlblue'], colors['mlorange'], colors['mlpurple']]

    bars = ax2.bar(metrics, values, color=colors_bars, alpha=0.7)
    ax2.set_ylabel('Millions / Percentage', fontsize=11)
    ax2.set_title('Spotify Wrapped Results', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        suffix = 'M' if val > 50 else '%'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}{suffix}', ha='center', fontweight='bold')

    plt.suptitle('Spotify Wrapped: Emotion-Driven Engagement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('spotify_wrapped_emotions')

# Generate all charts
if __name__ == '__main__':
    print("Generating Week 3 NLP charts...")

    # Part 1: The Challenge
    print("Creating Part 1 charts...")
    create_review_volume_growth()
    create_context_dependency()
    create_complexity_explosion()
    create_information_loss()

    # Part 2: First Solution
    print("Creating Part 2 charts...")
    create_sentiment_word_lists()
    create_bow_visualization()
    create_performance_degradation()
    create_information_preserved()

    # Part 3: The Breakthrough
    print("Creating Part 3 charts...")
    create_human_attention_process()
    create_word_vectors_2d()
    create_attention_three_steps()
    create_bert_bidirectional()
    create_pretraining_scale()
    create_finetuning_process()
    create_performance_comparison_final()

    # Part 4: Design Synthesis
    print("Creating Part 4 charts...")
    create_emotion_taxonomy()
    create_user_journey_emotions()
    create_churn_priority_matrix()
    create_empathy_scale_pyramid()
    create_spotify_wrapped_emotions()

    print("\nAll Week 3 charts generated successfully!")
    print("Total: 21 charts created (PDF and PNG versions)")