import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with multiple attention visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Example sentences with their tokens
examples = [
    {
        'text': "The product is not bad at all",
        'tokens': ['The', 'product', 'is', 'not', 'bad', 'at', 'all'],
        'sentiment': 'POSITIVE (0.82)',
        'key_pattern': 'double_negative'
    },
    {
        'text': "Absolutely terrible experience",
        'tokens': ['Absolutely', 'terrible', 'experience'],
        'sentiment': 'NEGATIVE (0.95)',
        'key_pattern': 'intensifier'
    },
    {
        'text': "Great product if you like disappointment",
        'tokens': ['Great', 'product', 'if', 'you', 'like', 'disappointment'],
        'sentiment': 'NEGATIVE (0.88) - Sarcasm detected',
        'key_pattern': 'sarcasm'
    },
    {
        'text': "Works perfectly despite the cheap price",
        'tokens': ['Works', 'perfectly', 'despite', 'the', 'cheap', 'price'],
        'sentiment': 'POSITIVE (0.75)',
        'key_pattern': 'contrast'
    }
]

# Generate attention patterns for each example
for idx, (ax, example) in enumerate(zip(axes.flat, examples)):
    tokens = example['tokens']
    n_tokens = len(tokens)
    
    # Create attention matrix based on pattern type
    attention = np.zeros((n_tokens, n_tokens))
    
    if example['key_pattern'] == 'double_negative':
        # Strong attention between "not" and "bad"
        not_idx, bad_idx = 3, 4
        attention[not_idx, bad_idx] = 0.9
        attention[bad_idx, not_idx] = 0.9
        attention[not_idx, not_idx] = 0.8
        attention[bad_idx, bad_idx] = 0.7
        # Some attention to "at all" (intensifier)
        attention[5, 4] = 0.5
        attention[6, 4] = 0.5
        
    elif example['key_pattern'] == 'intensifier':
        # "Absolutely" intensifies "terrible"
        attention[0, 1] = 0.95
        attention[1, 0] = 0.85
        attention[1, 1] = 0.9
        attention[1, 2] = 0.7
        attention[2, 1] = 0.6
        
    elif example['key_pattern'] == 'sarcasm':
        # Contradiction detection: "Great" vs "disappointment"
        attention[0, 5] = 0.95  # Great -> disappointment
        attention[5, 0] = 0.95  # disappointment -> Great
        attention[2, 5] = 0.6   # if -> disappointment
        attention[4, 5] = 0.7   # like -> disappointment
        
    elif example['key_pattern'] == 'contrast':
        # "despite" creates contrast attention
        despite_idx = 2
        attention[0, 2] = 0.7  # Works -> despite
        attention[1, 2] = 0.8  # perfectly -> despite
        attention[2, 4] = 0.85  # despite -> cheap
        attention[2, 5] = 0.75  # despite -> price
        attention[4, 1] = 0.6   # cheap -> perfectly
    
    # Add self-attention (diagonal) and noise
    np.fill_diagonal(attention, np.random.uniform(0.3, 0.6, n_tokens))
    attention += np.random.uniform(0, 0.2, (n_tokens, n_tokens))
    
    # Normalize to [0, 1]
    attention = np.clip(attention, 0, 1)
    
    # Create heatmap
    sns.heatmap(attention, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='YlOrRd',
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': 'Attention Weight'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    # Rotate labels for readability
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens, rotation=0)
    
    # Add title with sentiment
    ax.set_title(f'"{example["text"]}"\\n{example["sentiment"]}', 
                fontsize=11, fontweight='bold', pad=10)
    
    # Highlight the key attention areas
    if example['key_pattern'] == 'double_negative':
        ax.add_patch(plt.Rectangle((3, 4), 1, 1, fill=False, 
                                  edgecolor='blue', lw=3))
        ax.add_patch(plt.Rectangle((4, 3), 1, 1, fill=False, 
                                  edgecolor='blue', lw=3))
    elif example['key_pattern'] == 'sarcasm':
        ax.add_patch(plt.Rectangle((0, 5), 1, 1, fill=False, 
                                  edgecolor='red', lw=3))
        ax.add_patch(plt.Rectangle((5, 0), 1, 1, fill=False, 
                                  edgecolor='red', lw=3))

# Main title
plt.suptitle('BERT Attention Patterns: What the Model Focuses On', 
            fontsize=16, fontweight='bold', y=1.02)

# Add explanation box
explanation_text = (
    "How to read these heatmaps:\\n"
    "• Darker cells = stronger attention between words\\n"
    "• Blue boxes = key relationships for sentiment\\n"
    "• Red boxes = sarcasm/contradiction detection\\n"
    "• BERT uses these patterns to understand context"
)

fig.text(0.98, 0.02, explanation_text, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Add insights box
insights_text = (
    "Key Insights:\\n"
    "• 'not bad' → focuses on negation\\n"
    "• Intensifiers strengthen sentiment\\n"
    "• Contradictions signal sarcasm\\n"
    "• Context words like 'despite' matter"
)

fig.text(0.02, 0.02, insights_text, fontsize=10, ha='left',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('bert_attention_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('bert_attention_heatmap.png', dpi=150, bbox_inches='tight')

print("Chart saved: bert_attention_heatmap.pdf")
print("\\nAttention Pattern Analysis:")
print("Example 1: Double negative detection (not bad)")
print("Example 2: Intensifier recognition (absolutely terrible)")
print("Example 3: Sarcasm through contradiction (great...disappointment)")
print("Example 4: Contrast understanding (perfectly despite cheap)")