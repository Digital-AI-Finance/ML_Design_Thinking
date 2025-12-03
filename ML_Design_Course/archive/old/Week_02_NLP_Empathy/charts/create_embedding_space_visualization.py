"""
Create embedding space visualization for Week 2 - BERT theory section
Shows how words cluster in semantic space with t-SNE projection
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Define word categories and their semantic relationships (reduced for speed)
word_categories = {
    'Emotions': ['happy', 'sad', 'angry', 'frustrated', 'delighted'],
    'Products': ['app', 'interface', 'feature', 'design'],
    'Quality': ['good', 'bad', 'excellent', 'terrible'],
    'Speed': ['fast', 'slow', 'responsive', 'laggy'],
    'Difficulty': ['easy', 'hard', 'simple', 'complex']
}

# Create synthetic embeddings that cluster by category
n_dims = 50  # Reduced for faster computation (BERT is 768)
embeddings = []
words = []
categories = []
colors = []

# Color palette for categories
color_map = {
    'Emotions': '#FF6B6B',     # Red
    'Products': '#4ECDC4',      # Teal
    'Actions': '#45B7D1',       # Blue
    'Quality': '#96CEB4',       # Green
    'Speed': '#FFEAA7',         # Yellow
    'Difficulty': '#DDA0DD'     # Purple
}

# Generate embeddings with category clustering
for category, word_list in word_categories.items():
    # Create a category center in high-dimensional space
    category_center = np.random.randn(n_dims) * 10
    
    for word in word_list:
        # Add noise around category center to create natural clustering
        noise = np.random.randn(n_dims) * 2
        embedding = category_center + noise
        
        # Add some cross-category relationships for realism
        if word in ['good', 'excellent', 'amazing']:
            # Positive words slightly closer to happy emotions
            embedding += np.random.randn(n_dims) * 0.5
        elif word in ['bad', 'terrible', 'awful']:
            # Negative words slightly closer to sad/angry emotions
            embedding += np.random.randn(n_dims) * 0.5
            
        embeddings.append(embedding)
        words.append(word)
        categories.append(category)
        colors.append(color_map[category])

# Convert to numpy array
embeddings = np.array(embeddings)

# Apply t-SNE for 2D visualization
print("Applying t-SNE projection...")
tsne = TSNE(n_components=2, perplexity=10, random_state=42, max_iter=500)
embeddings_2d = tsne.fit_transform(embeddings)

# Create the visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Word embeddings with labels
ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=100, alpha=0.6, edgecolors='gray', linewidth=0.5)

# Add word labels
for i, word in enumerate(words):
    ax1.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                fontsize=8, ha='center', va='center')

ax1.set_title('Word Embeddings in Semantic Space', fontsize=14, fontweight='bold')
ax1.set_xlabel('Dimension 1 (t-SNE)', fontsize=11)
ax1.set_ylabel('Dimension 2 (t-SNE)', fontsize=11)
ax1.grid(True, alpha=0.3)

# Add legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=cat, alpha=0.6) 
                   for cat, color in color_map.items()]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Right plot: Semantic relationships with arrows
ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=80, alpha=0.3)

# Show specific semantic relationships
relationships = [
    ('good', 'bad', 'Quality Spectrum'),
    ('fast', 'slow', 'Speed Spectrum'),
    ('easy', 'hard', 'Difficulty Spectrum'),
    ('happy', 'sad', 'Emotion Spectrum')
]

for word1, word2, label in relationships:
    idx1 = words.index(word1)
    idx2 = words.index(word2)
    
    # Draw arrow showing relationship
    ax2.annotate('', xy=embeddings_2d[idx2], xytext=embeddings_2d[idx1],
                arrowprops=dict(arrowstyle='->', lw=2, alpha=0.7, color='darkgray'))
    
    # Add labels at endpoints
    ax2.annotate(word1, embeddings_2d[idx1], fontsize=10, fontweight='bold', 
                ha='center', va='bottom')
    ax2.annotate(word2, embeddings_2d[idx2], fontsize=10, fontweight='bold',
                ha='center', va='top')
    
    # Add relationship label at midpoint
    mid_point = (embeddings_2d[idx1] + embeddings_2d[idx2]) / 2
    ax2.annotate(label, mid_point, fontsize=9, style='italic',
                ha='center', va='bottom', color='darkred')

ax2.set_title('Semantic Relationships in Embedding Space', fontsize=14, fontweight='bold')
ax2.set_xlabel('Dimension 1 (t-SNE)', fontsize=11)
ax2.set_ylabel('Dimension 2 (t-SNE)', fontsize=11)
ax2.grid(True, alpha=0.3)

# Add context-dependent example
context_box = """Context Changes Meaning:
"bank" near "river" vs "bank" near "money"
"fast" near "delivery" vs "fast" near "battery drain"
BERT captures these distinctions!"""

ax2.text(0.02, 0.98, context_box, transform=ax2.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Overall title
fig.suptitle('BERT Embedding Space: How Words Cluster by Meaning', 
             fontsize=16, fontweight='bold', y=1.02)

# Add explanatory text
explanation = ("Words with similar meanings cluster together in high-dimensional space (768D for BERT).\n"
               "t-SNE projection shows 2D representation while preserving relative distances.")
fig.text(0.5, -0.02, explanation, ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()

# Save the figure
plt.savefig('embedding_space_visualization.pdf', dpi=300, bbox_inches='tight')
plt.savefig('embedding_space_visualization.png', dpi=150, bbox_inches='tight')

print("Embedding space visualization created successfully!")
print(f"Total words visualized: {len(words)}")
print(f"Categories: {', '.join(word_categories.keys())}")