"""Generate NLP method selection decision tree"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.style.use('seaborn-v0_8-whitegrid')

def save_chart(name):
    plt.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(7, 9.5, 'When to Use Which NLP Method: Decision Framework',
        ha='center', fontsize=16, fontweight='bold')

root_box = FancyBboxPatch((5.5, 8.2), 3, 0.8, boxstyle="round,pad=0.1",
                          facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(root_box)
ax.text(7, 8.6, 'What\'s your priority?', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Branch 1: Speed/Simplicity -> Rule-based
ax.arrow(6.5, 8.2, -2, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
simple_box = FancyBboxPatch((2.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#9E9E9E', edgecolor='black', linewidth=2, alpha=0.7)
ax.add_patch(simple_box)
ax.text(4, 6.2, 'SPEED & SIMPLICITY', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(4, 5.4, 'Known keywords\nSimple patterns\nReal-time needed\nNo training data',
        ha='center', va='top', fontsize=9)

ax.arrow(4, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
result_rule = FancyBboxPatch((2.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                           facecolor='#BDBDBD', edgecolor='black', linewidth=2)
ax.add_patch(result_rule)
ax.text(4, 4.5, 'RULE-BASED', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(4, 4.1, 'Regex/keywords\nInstant results\n<70% accuracy\nNo training needed',
        ha='center', va='center', fontsize=8)

# Branch 2: Good enough -> Traditional ML
ax.arrow(7, 8.2, 0, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
balanced_box = FancyBboxPatch((5.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                           facecolor='#FF9800', edgecolor='black', linewidth=2, alpha=0.7)
ax.add_patch(balanced_box)
ax.text(7, 6.2, 'BALANCED PERFORMANCE', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(7, 5.4, 'Simple language\nLimited context\nModerate data\nFast inference',
        ha='center', va='top', fontsize=9)

ax.arrow(7, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
result_traditional = FancyBboxPatch((5.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                  facecolor='#FFB74D', edgecolor='black', linewidth=2)
ax.add_patch(result_traditional)
ax.text(7, 4.5, 'TRADITIONAL ML', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(7, 4.1, 'BoW/TF-IDF + ML\n75-85% accuracy\nFast training\nLow cost',
        ha='center', va='center', fontsize=8)

# Branch 3: Best performance -> Transformers
ax.arrow(7.5, 8.2, 2, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
best_box = FancyBboxPatch((8.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                        facecolor='#4CAF50', edgecolor='black', linewidth=2, alpha=0.7)
ax.add_patch(best_box)
ax.text(10, 6.2, 'MAXIMUM ACCURACY', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(10, 5.4, 'Complex context\nSarcasm/nuance\nLarge dataset\nProduction critical',
        ha='center', va='top', fontsize=9)

ax.arrow(10, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
result_transformer = FancyBboxPatch((8.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                   facecolor='#81C784', edgecolor='black', linewidth=2)
ax.add_patch(result_transformer)
ax.text(10, 4.5, 'TRANSFORMERS', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(10, 4.1, 'BERT/RoBERTa\n90-98% accuracy\nSlower/expensive\nPre-trained models',
        ha='center', va='center', fontsize=8)

# Additional considerations
consider_box = FancyBboxPatch((0.5, 0.5), 13, 2.5, boxstyle="round,pad=0.1",
                              facecolor='#F0F0F0', edgecolor='black', linewidth=2)
ax.add_patch(consider_box)
ax.text(7, 2.7, 'Additional Considerations', ha='center', va='center',
        fontsize=12, fontweight='bold')

considerations_text = """
Data Volume: <1K samples → Rule-based or few-shot; 1K-10K → Traditional ML; >10K → Transformers viable
Languages: Multi-lingual needs → Multilingual BERT (mBERT, XLM-RoBERTa); English only → simpler models
Domain: Medical/Legal → Fine-tune domain-specific transformer; General → Use pre-trained as-is
Latency: Real-time (<100ms) → Rule-based or cached ML; Batch processing → Transformers acceptable
Budget: Limited → Traditional ML (10-100x cheaper); Enterprise → Transformers for best results
Explainability: High need → Rule-based (transparent) or LIME/SHAP on ML; Black box OK → Transformers
    """
ax.text(7, 1.5, considerations_text, ha='center', va='center', fontsize=8, family='monospace')

ax.text(7, 0.2, 'Principle: Start simple (rules/traditional ML), upgrade to transformers only when context/nuance critical',
        ha='center', va='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
save_chart('nlp_method_decision')
print('NLP method decision tree created successfully!')
