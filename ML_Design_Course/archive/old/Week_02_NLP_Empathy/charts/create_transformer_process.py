import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig = plt.figure(figsize=(14, 10))

# Create main architecture diagram
ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)

# Define components and positions
components = {
    'input': {'pos': (1, 1), 'size': (1.5, 0.5), 'color': '#e8f4f8'},
    'embedding': {'pos': (1, 2), 'size': (1.5, 0.5), 'color': '#d4e6f1'},
    'attention': {'pos': (1, 3.5), 'size': (1.5, 1), 'color': '#aed6f1'},
    'feedforward': {'pos': (1, 5.5), 'size': (1.5, 0.5), 'color': '#85c1e2'},
    'output': {'pos': (1, 7), 'size': (1.5, 0.5), 'color': '#5dade2'}
}

# Draw the main transformer block
for name, props in components.items():
    box = FancyBboxPatch(
        props['pos'], props['size'][0], props['size'][1],
        boxstyle="round,pad=0.05",
        facecolor=props['color'],
        edgecolor='black',
        linewidth=2
    )
    ax_main.add_patch(box)
    
    # Add labels
    x, y = props['pos']
    if name == 'input':
        ax_main.text(x + props['size'][0]/2, y + props['size'][1]/2,
                    'Input Text\\n"Not bad at all"', 
                    ha='center', va='center', fontweight='bold', fontsize=11)
    elif name == 'embedding':
        ax_main.text(x + props['size'][0]/2, y + props['size'][1]/2,
                    'Word Embeddings\\n768 dimensions', 
                    ha='center', va='center', fontweight='bold', fontsize=11)
    elif name == 'attention':
        ax_main.text(x + props['size'][0]/2, y + props['size'][1]/2,
                    'Multi-Head\\nAttention\\n(12 heads)', 
                    ha='center', va='center', fontweight='bold', fontsize=11)
    elif name == 'feedforward':
        ax_main.text(x + props['size'][0]/2, y + props['size'][1]/2,
                    'Feed Forward\\nNetwork', 
                    ha='center', va='center', fontweight='bold', fontsize=11)
    elif name == 'output':
        ax_main.text(x + props['size'][0]/2, y + props['size'][1]/2,
                    'Sentiment Output\\nPOSITIVE: 0.82', 
                    ha='center', va='center', fontweight='bold', fontsize=11)

# Add arrows between components
arrow_props = dict(arrowstyle='->', lw=2, color='black')
ax_main.annotate('', xy=(1.75, 2), xytext=(1.75, 1.5), arrowprops=arrow_props)
ax_main.annotate('', xy=(1.75, 3.5), xytext=(1.75, 2.5), arrowprops=arrow_props)
ax_main.annotate('', xy=(1.75, 5.5), xytext=(1.75, 4.5), arrowprops=arrow_props)
ax_main.annotate('', xy=(1.75, 7), xytext=(1.75, 6), arrowprops=arrow_props)

# Add positional encoding
pos_enc_box = FancyBboxPatch(
    (3, 2), 1.2, 0.5,
    boxstyle="round,pad=0.05",
    facecolor='#fdebd0',
    edgecolor='orange',
    linewidth=2
)
ax_main.add_patch(pos_enc_box)
ax_main.text(3.6, 2.25, 'Positional\\nEncoding', 
            ha='center', va='center', fontweight='bold', fontsize=10)
ax_main.annotate('', xy=(2.5, 2.25), xytext=(3, 2.25), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))

# Add layer normalization indicators
for y_pos in [3, 5, 6.5]:
    norm_box = FancyBboxPatch(
        (3, y_pos), 0.8, 0.3,
        boxstyle="round,pad=0.02",
        facecolor='#e8f8f5',
        edgecolor='green',
        linewidth=1
    )
    ax_main.add_patch(norm_box)
    ax_main.text(3.4, y_pos + 0.15, 'Norm', 
                ha='center', va='center', fontsize=9)

# Add residual connections
residual_color = '#ec7063'
ax_main.plot([0.5, 0.5], [2.5, 4], '--', color=residual_color, lw=1.5)
ax_main.plot([0.5, 1], [4, 4], '--', color=residual_color, lw=1.5)
ax_main.text(0.3, 3.25, 'Residual', rotation=90, 
            color=residual_color, fontsize=9)

# BERT specific annotations
ax_main.text(5, 6.5, 'BERT Stack:\\n12 Layers', 
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

ax_main.text(5, 4.5, 'Bidirectional:\\nSees all words\\nat once', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

ax_main.text(5, 2.5, '110M\\nparameters', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral'))

# Set main plot limits and remove axes
ax_main.set_xlim(0, 7)
ax_main.set_ylim(0.5, 8)
ax_main.axis('off')
ax_main.set_title('Transformer Architecture (Simplified)', 
                 fontsize=16, fontweight='bold', pad=20)

# ============= Attention Detail (Bottom Left) =============
ax_att = plt.subplot2grid((3, 3), (2, 0), colspan=1)

# Simple attention visualization
words = ['Not', 'bad', 'at', 'all']
n_words = len(words)
attention_weights = np.array([
    [0.7, 0.9, 0.2, 0.1],  # Not attends to itself and bad
    [0.8, 0.6, 0.1, 0.1],  # bad attends to Not
    [0.1, 0.2, 0.5, 0.4],  # at
    [0.1, 0.3, 0.3, 0.6]   # all
])

im = ax_att.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax_att.set_xticks(range(n_words))
ax_att.set_yticks(range(n_words))
ax_att.set_xticklabels(words)
ax_att.set_yticklabels(words)
ax_att.set_xlabel('Attending to', fontsize=10)
ax_att.set_ylabel('Query word', fontsize=10)
ax_att.set_title('Attention Weights', fontsize=11, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax_att, fraction=0.046, pad=0.04)
cbar.set_label('Weight', fontsize=9)

# ============= Embedding Space (Bottom Middle) =============
ax_emb = plt.subplot2grid((3, 3), (2, 1), colspan=1)

# Simulate word embeddings in 2D
np.random.seed(42)
positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor']
neutral_words = ['okay', 'fine', 'average', 'normal', 'regular']

# Generate clustered positions
pos_x = np.random.normal(0.7, 0.1, len(positive_words))
pos_y = np.random.normal(0.7, 0.1, len(positive_words))
neg_x = np.random.normal(0.3, 0.1, len(negative_words))
neg_y = np.random.normal(0.3, 0.1, len(negative_words))
neu_x = np.random.normal(0.5, 0.08, len(neutral_words))
neu_y = np.random.normal(0.5, 0.08, len(neutral_words))

ax_emb.scatter(pos_x, pos_y, c='green', s=50, alpha=0.6, label='Positive')
ax_emb.scatter(neg_x, neg_y, c='red', s=50, alpha=0.6, label='Negative')
ax_emb.scatter(neu_x, neu_y, c='gray', s=50, alpha=0.6, label='Neutral')

# Annotate a few words
ax_emb.annotate('good', (pos_x[0], pos_y[0]), fontsize=8)
ax_emb.annotate('bad', (neg_x[0], neg_y[0]), fontsize=8)
ax_emb.annotate('okay', (neu_x[0], neu_y[0]), fontsize=8)

# Special case: "not bad"
ax_emb.scatter([0.6], [0.65], c='blue', s=100, marker='*')
ax_emb.annotate('not bad', (0.6, 0.65), xytext=(0.55, 0.72),
               arrowprops=dict(arrowstyle='->', color='blue', lw=1),
               fontsize=9, fontweight='bold', color='blue')

ax_emb.set_xlim(0, 1)
ax_emb.set_ylim(0, 1)
ax_emb.set_xlabel('Dimension 1', fontsize=10)
ax_emb.set_ylabel('Dimension 2', fontsize=10)
ax_emb.set_title('Word Embedding Space', fontsize=11, fontweight='bold')
ax_emb.legend(loc='upper left', fontsize=8)

# ============= Processing Steps (Bottom Right) =============
ax_steps = plt.subplot2grid((3, 3), (2, 2), colspan=1)

steps = [
    '1. Tokenize text',
    '2. Convert to embeddings',
    '3. Add position info',
    '4. Apply attention',
    '5. Transform features',
    '6. Classify sentiment'
]

colors_steps = ['#e8f4f8', '#d4e6f1', '#aed6f1', '#85c1e2', '#5dade2', '#3498db']

for i, (step, color) in enumerate(zip(steps, colors_steps)):
    y_pos = 5 - i * 0.8
    box = FancyBboxPatch(
        (0.1, y_pos), 2.3, 0.6,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=1
    )
    ax_steps.add_patch(box)
    ax_steps.text(1.25, y_pos + 0.3, step, 
                 ha='center', va='center', fontsize=10)
    
    if i < len(steps) - 1:
        ax_steps.annotate('', xy=(1.25, y_pos - 0.05), 
                         xytext=(1.25, y_pos - 0.15),
                         arrowprops=dict(arrowstyle='->', lw=1.5))

ax_steps.set_xlim(0, 2.5)
ax_steps.set_ylim(-0.5, 6)
ax_steps.axis('off')
ax_steps.set_title('Processing Pipeline', fontsize=11, fontweight='bold')

# Add main title and description
fig.suptitle('How Transformers Process Text for Sentiment Analysis', 
            fontsize=16, fontweight='bold', y=0.98)

# Add key facts box
facts_text = (
    "BERT Key Facts:\\n"
    "• Pre-trained on 3.3B words\\n"
    "• 12 attention heads per layer\\n"
    "• 768-dimensional embeddings\\n"
    "• Processes all words simultaneously\\n"
    "• Fine-tunable for specific tasks"
)

fig.text(0.02, 0.02, facts_text, fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('transformer_process.pdf', dpi=300, bbox_inches='tight')
plt.savefig('transformer_process.png', dpi=150, bbox_inches='tight')

print("Chart saved: transformer_process.pdf")
print("\\nTransformer Architecture Summary:")
print("Input: Text sequence")
print("Embedding: 768 dimensions per token")
print("Attention: 12 heads, bidirectional")
print("Layers: 12 transformer blocks")
print("Parameters: 110 million (BERT-base)")
print("Output: Sentiment classification")