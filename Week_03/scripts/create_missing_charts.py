"""
Generate missing NLP charts for Week 3 presentation
Creates all charts referenced in LaTeX but not yet generated
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
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

def create_amazon_data_pipeline():
    """Create Amazon data processing pipeline visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Pipeline stages
    stages = [
        ('Ingestion\n2M+ Reviews/Day', 0.1, 0.5, colors['mlblue']),
        ('Cleaning\nSpam Filter', 0.3, 0.5, colors['mlorange']),
        ('NLP\nProcessing', 0.5, 0.5, colors['mlgreen']),
        ('Insights\nExtraction', 0.7, 0.5, colors['mlpurple']),
        ('Action\nDeployment', 0.9, 0.5, colors['mlred'])
    ]
    
    for i, (label, x, y, color) in enumerate(stages):
        # Draw box
        rect = FancyBboxPatch((x - 0.08, y - 0.1), 0.16, 0.2,
                              boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw arrow to next stage
        if i < len(stages) - 1:
            ax.arrow(x + 0.08, y, 0.14, 0, head_width=0.03, head_length=0.02,
                    fc=colors['mlgray'], ec=colors['mlgray'], alpha=0.5)
    
    # Add metrics below each stage
    metrics = [
        '500K/sec', '95% accuracy', '< 100ms', 'Real-time', 'Automated'
    ]
    for (_, x, _, _), metric in zip(stages, metrics):
        ax.text(x, 0.25, metric, ha='center', fontsize=9, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Amazon Review Processing Pipeline', fontsize=16, fontweight='bold')
    
    return fig

def create_sentiment_vs_stars():
    """Create sentiment score vs star rating analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate realistic data
    stars = np.repeat([1, 2, 3, 4, 5], 200)
    # Add noise to sentiment scores based on star ratings
    sentiment_scores = []
    for star in stars:
        if star == 1:
            score = np.random.beta(2, 8) * 0.4
        elif star == 2:
            score = 0.2 + np.random.beta(3, 7) * 0.3
        elif star == 3:
            score = 0.35 + np.random.beta(5, 5) * 0.3
        elif star == 4:
            score = 0.55 + np.random.beta(7, 3) * 0.3
        else:  # 5 stars
            score = 0.7 + np.random.beta(8, 2) * 0.3
        sentiment_scores.append(score)
    
    # Plot 1: Scatter plot with trend
    ax1 = axes[0]
    ax1.scatter(stars + np.random.normal(0, 0.1, len(stars)), sentiment_scores,
               alpha=0.3, s=20, c=stars, cmap='RdYlGn')
    
    # Add trend line
    z = np.polyfit(stars, sentiment_scores, 1)
    p = np.poly1d(z)
    ax1.plot([1, 5], p([1, 5]), "r-", linewidth=2, alpha=0.7, label='Trend')
    
    ax1.set_xlabel('Star Rating', fontsize=12)
    ax1.set_ylabel('Sentiment Score', fontsize=12)
    ax1.set_title('Sentiment vs Star Correlation', fontsize=14, fontweight='bold')
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Plot 2: Box plot by star rating
    ax2 = axes[1]
    data_by_star = [np.array(sentiment_scores)[stars == s] for s in [1, 2, 3, 4, 5]]
    bp = ax2.boxplot(data_by_star, labels=['1⭐', '2⭐', '3⭐', '4⭐', '5⭐'],
                     patch_artist=True)
    
    # Color boxes
    for patch, star in zip(bp['boxes'], [1, 2, 3, 4, 5]):
        if star <= 2:
            color = colors['mlred']
        elif star == 3:
            color = colors['mlorange']
        else:
            color = colors['mlgreen']
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_xlabel('Star Rating', fontsize=12)
    ax2.set_ylabel('Sentiment Score Distribution', fontsize=12)
    ax2.set_title('Sentiment Distribution by Rating', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_aspect_sentiment_matrix():
    """Create aspect-based sentiment matrix"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Product aspects and categories
    aspects = ['Battery', 'Screen', 'Camera', 'Performance', 'Design', 'Price', 'Build', 'Software']
    categories = ['Phones', 'Laptops', 'Tablets', 'Watches', 'Earbuds']
    
    # Generate sentiment scores for each aspect-category pair
    data = np.random.rand(len(categories), len(aspects))
    # Make some patterns
    data[0, [0, 2, 3]] = [0.85, 0.9, 0.88]  # Phones good at battery, camera, performance
    data[1, [3, 4, 6]] = [0.92, 0.87, 0.89]  # Laptops good at performance, design, build
    data[2, [1, 4]] = [0.91, 0.86]  # Tablets good at screen, design
    data[3, [0, 4, 6]] = [0.88, 0.9, 0.85]  # Watches good at battery, design, build
    data[4, [5, 7]] = [0.82, 0.84]  # Earbuds good at price, software
    
    # Create heatmap
    sns.heatmap(data, xticklabels=aspects, yticklabels=categories,
                annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Sentiment Score'}, ax=ax)
    
    ax.set_title('Aspect-Based Sentiment by Product Category', fontsize=16, fontweight='bold')
    ax.set_xlabel('Product Aspects', fontsize=12)
    ax.set_ylabel('Product Categories', fontsize=12)
    
    return fig

def create_insights_to_actions():
    """Create insights to actions flow"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Define insights and actions
    insights_actions = [
        ('Battery complaints\n35% negative', 'Increase battery\ncapacity', colors['mlred']),
        ('Camera praised\n89% positive', 'Marketing focus\non camera', colors['mlgreen']),
        ('Price concerns\n42% mention', 'Consider price\nadjustment', colors['mlorange']),
        ('Software bugs\n28% report', 'Priority bug\nfixes', colors['mlpurple'])
    ]
    
    y_positions = [0.8, 0.6, 0.4, 0.2]
    
    for (insight, action, color), y_pos in zip(insights_actions, y_positions):
        # Insight box
        rect1 = FancyBboxPatch((0.1, y_pos - 0.06), 0.25, 0.12,
                               boxstyle="round,pad=0.01",
                               facecolor=color, alpha=0.3,
                               edgecolor=color, linewidth=2)
        ax.add_patch(rect1)
        ax.text(0.225, y_pos, insight, ha='center', va='center', fontsize=9)
        
        # Arrow
        ax.arrow(0.35, y_pos, 0.25, 0, head_width=0.03, head_length=0.03,
                fc=colors['mlgray'], ec=colors['mlgray'], alpha=0.5)
        
        # Action box
        rect2 = FancyBboxPatch((0.65, y_pos - 0.06), 0.25, 0.12,
                               boxstyle="round,pad=0.01",
                               facecolor=colors['mlblue'], alpha=0.3,
                               edgecolor=colors['mlblue'], linewidth=2)
        ax.add_patch(rect2)
        ax.text(0.775, y_pos, action, ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('From Insights to Actions', fontsize=16, fontweight='bold')
    ax.text(0.225, 0.95, 'NLP Insights', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.775, 0.95, 'Business Actions', ha='center', fontsize=12, fontweight='bold')
    
    return fig

def create_improvement_metrics():
    """Create improvement metrics dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Before/After metrics
    ax1 = axes[0, 0]
    metrics = ['Response\nTime', 'Resolution\nRate', 'Customer\nSatisfaction', 'Retention']
    before = [24, 65, 72, 78]
    after = [8, 85, 88, 91]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, before, width, label='Before NLP', color=colors['mlgray'], alpha=0.7)
    bars2 = ax1.bar(x + width/2, after, width, label='After NLP', color=colors['mlgreen'], alpha=0.7)
    
    ax1.set_ylabel('Score/Hours', fontsize=10)
    ax1.set_title('Key Metrics Improvement', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # ROI timeline
    ax2 = axes[0, 1]
    months = np.arange(12)
    roi = -100 + 50 * months + 2 * months**2
    
    ax2.plot(months, roi, color=colors['mlblue'], linewidth=2, marker='o')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(months, roi, where=(roi > 0), alpha=0.3, color=colors['mlgreen'])
    ax2.fill_between(months, roi, where=(roi <= 0), alpha=0.3, color=colors['mlred'])
    
    ax2.set_xlabel('Months', fontsize=10)
    ax2.set_ylabel('ROI (%)', fontsize=10)
    ax2.set_title('ROI Timeline', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Cost savings breakdown
    ax3 = axes[1, 0]
    savings = ['Labor', 'Support', 'Returns', 'Marketing', 'Other']
    values = [35, 28, 20, 12, 5]
    colors_pie = [colors['mlblue'], colors['mlorange'], colors['mlgreen'], 
                  colors['mlpurple'], colors['mlgray']]
    
    ax3.pie(values, labels=savings, colors=colors_pie, autopct='%1.1f%%',
           startangle=90)
    ax3.set_title('Cost Savings Breakdown', fontsize=12, fontweight='bold')
    
    # Efficiency gains
    ax4 = axes[1, 1]
    processes = ['Ticket\nRouting', 'Sentiment\nAnalysis', 'Report\nGeneration', 'Alert\nSystem']
    efficiency = [450, 380, 520, 290]
    
    bars = ax4.bar(processes, efficiency, color=[colors['mlcyan'], colors['mlyellow'],
                                                  colors['mlpink'], colors['mlbrown']], alpha=0.7)
    ax4.set_ylabel('Efficiency Gain (%)', fontsize=10)
    ax4.set_title('Process Efficiency Improvements', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, efficiency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'+{val}%', ha='center', fontsize=9)
    
    plt.suptitle('NLP Implementation Impact Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_text_cleaning_pipeline():
    """Create text preprocessing pipeline visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Sample text transformation
    stages = [
        ("Raw Input", "OMG!!! This product is AMAZINGGG 😍😍😍 #BestEver http://link.co"),
        ("Lower & Strip", "omg!!! this product is amazinggg 😍😍😍 #bestever"),
        ("Remove Emoji/URL", "omg!!! this product is amazinggg #bestever"),
        ("Remove Punctuation", "omg this product is amazinggg bestever"),
        ("Fix Spelling", "omg this product is amazing best ever"),
        ("Lemmatize", "omg product amazing best"),
        ("Final Output", "product amazing best")
    ]
    
    for i, (stage_name, text) in enumerate(stages):
        y = 0.9 - i * 0.13
        
        # Stage label
        rect = Rectangle((0.05, y - 0.04), 0.2, 0.08,
                        facecolor=colors['mlblue'], alpha=0.3,
                        edgecolor=colors['mlblue'], linewidth=2)
        ax.add_patch(rect)
        ax.text(0.15, y, stage_name, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow
        if i < len(stages) - 1:
            ax.arrow(0.27, y, 0.03, 0, head_width=0.02, head_length=0.01,
                    fc=colors['mlgray'], ec=colors['mlgray'])
        
        # Text sample
        ax.text(0.35, y, text, ha='left', va='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Text Cleaning Pipeline', fontsize=16, fontweight='bold')
    
    return fig

def create_roc_curves():
    """Create ROC curves for multiple classifiers"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate sample data for 3-class problem
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Binarize labels for ROC curve
    y_bin = label_binarize(y, classes=[0, 1, 2])
    
    # Train different classifiers and plot ROC
    classifiers = [
        ('Naive Bayes', MultinomialNB(), colors['mlorange']),
        ('SVM', SVC(probability=True, random_state=42), colors['mlblue']),
        ('Random Forest', RandomForestClassifier(random_state=42), colors['mlgreen'])
    ]
    
    for name, clf, color in classifiers:
        # Handle negative values for MultinomialNB
        if name == 'Naive Bayes':
            X_adj = X - X.min() + 1
        else:
            X_adj = X
        
        # Train classifier
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_adj, y, test_size=0.3, random_state=42)
        clf.fit(X_train, y_train)
        
        # Get probabilities
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)
        else:
            y_score = clf.decision_function(X_test)
        
        # Calculate ROC curve for first class
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_score[:, 0])
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(fpr, tpr, color=color, linewidth=2,
               label=f'{name} (AUC = {roc_auc:.2f})', alpha=0.8)
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Sentiment Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig

def create_bert_bidirectional():
    """Create BERT bidirectional processing visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Sample sentence
    words = ['The', 'movie', 'was', 'not', 'bad', 'at', 'all']
    n_words = len(words)
    
    # Draw words
    for i, word in enumerate(words):
        x = 0.15 + i * 0.12
        y = 0.5
        
        # Word box
        rect = FancyBboxPatch((x - 0.05, y - 0.04), 0.1, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=colors['mlblue'], alpha=0.3,
                              edgecolor=colors['mlblue'], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, word, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Bidirectional arrows to all other words
        for j, other_word in enumerate(words):
            if i != j:
                x2 = 0.15 + j * 0.12
                # Forward connections
                if j > i:
                    arrow = FancyArrowPatch((x + 0.04, y + 0.02), (x2 - 0.04, y + 0.02),
                                          connectionstyle="arc3,rad=.3",
                                          arrowstyle='->',
                                          linewidth=0.5, color=colors['mlgreen'], alpha=0.3)
                    ax.add_patch(arrow)
                # Backward connections
                else:
                    arrow = FancyArrowPatch((x - 0.04, y - 0.02), (x2 + 0.04, y - 0.02),
                                          connectionstyle="arc3,rad=-.3",
                                          arrowstyle='->',
                                          linewidth=0.5, color=colors['mlorange'], alpha=0.3)
                    ax.add_patch(arrow)
    
    # Add labels
    ax.text(0.5, 0.8, 'BERT: Bidirectional Understanding', ha='center',
           fontsize=16, fontweight='bold')
    ax.text(0.5, 0.7, 'Each word sees all other words simultaneously',
           ha='center', fontsize=12, style='italic')
    
    # Result
    ax.text(0.5, 0.2, 'Result: "not bad" understood as POSITIVE in context',
           ha='center', fontsize=12, color=colors['mlgreen'], fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['mlgreen'], alpha=0.2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

# Add more missing chart functions...

def create_all_missing_charts():
    """Generate all missing charts"""
    charts_to_create = [
        ('amazon_data_pipeline', create_amazon_data_pipeline),
        ('sentiment_vs_stars', create_sentiment_vs_stars),
        ('aspect_sentiment_matrix', create_aspect_sentiment_matrix),
        ('insights_to_actions', create_insights_to_actions),
        ('improvement_metrics', create_improvement_metrics),
        ('text_cleaning_pipeline', create_text_cleaning_pipeline),
        ('roc_curves', create_roc_curves),
        ('bert_bidirectional', create_bert_bidirectional),
    ]
    
    print("Generating missing charts for Week 3...")
    
    import os
    os.makedirs('../charts', exist_ok=True)
    
    for name, func in charts_to_create:
        print(f"Creating {name}...")
        fig = func()
        fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("Missing charts created successfully!")

if __name__ == "__main__":
    create_all_missing_charts()