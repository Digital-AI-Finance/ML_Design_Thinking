#!/usr/bin/env python3
"""Update all topic markdown files with chart embeds."""

from pathlib import Path

CONTENT_PATH = Path(__file__).parent.parent / 'content' / 'topics'
GITHUB_BASE = 'https://github.com/Digital-AI-Finance/ML_Design_Thinking_16/blob/main/tools/generate_web_charts.py'

# Chart configurations per topic
CHARTS = {
    'ml-foundations': [
        ('learning_paradigms', 'Learning Paradigms', 'Supervised, unsupervised, and reinforcement learning with industry adoption rates'),
        ('ml_vs_traditional', 'ML vs Traditional Programming', 'Traditional programming uses rules to produce output; ML learns rules from data'),
        ('performance_scaling', 'Performance Scaling', 'How different approaches scale with more data'),
    ],
    'supervised-learning': [
        ('regression_vs_classification', 'Regression vs Classification', 'Regression predicts continuous values; classification predicts categories'),
        ('bias_variance', 'Bias-Variance Tradeoff', 'Finding the optimal model complexity to minimize total error'),
        ('train_test_split', 'Train/Test Split', 'Standard 80/20 split for model evaluation'),
    ],
    'unsupervised-learning': [
        ('clustering_example', 'Clustering Example', 'K-means finds natural groups in unlabeled data'),
        ('dimensionality_reduction', 'Dimensionality Reduction', 'PCA reduces high-dimensional data to 2D for visualization'),
        ('supervised_vs_unsupervised', 'Supervised vs Unsupervised', 'Key difference: labeled data requirement'),
    ],
    'neural-networks': [
        ('network_architecture', 'Network Architecture', 'Layers of interconnected neurons processing information'),
        ('activation_functions', 'Activation Functions', 'Sigmoid, ReLU, and Tanh introduce non-linearity'),
        ('training_loss', 'Training Progress', 'Loss decreases over epochs with early stopping to prevent overfitting'),
    ],
    'clustering': [
        ('kmeans_steps', 'K-Means Algorithm', 'Initialize centroids, assign points, update centroids, repeat'),
        ('elbow_method', 'Elbow Method', 'Find optimal k where inertia reduction slows'),
        ('silhouette_scores', 'Silhouette Scores', 'Measure cluster quality (higher is better)'),
    ],
    'nlp-sentiment': [
        ('sentiment_distribution', 'Sentiment Distribution', 'Distribution of positive, neutral, and negative sentiments in customer reviews'),
        ('word_embeddings', 'Word Embeddings', 'Similar words cluster together in vector space'),
        ('preprocessing_pipeline', 'Text Preprocessing', 'Steps from raw text to numerical vectors'),
    ],
    'classification': [
        ('decision_boundary', 'Decision Boundary', 'The line (or surface) that separates classes'),
        ('confusion_matrix', 'Confusion Matrix', 'True vs predicted labels showing model performance'),
        ('decision_tree', 'Decision Tree', 'Rule-based classification through sequential questions'),
    ],
    'generative-ai': [
        ('llm_capabilities', 'LLM Capabilities', 'Modern language models excel at diverse tasks'),
        ('prompt_flow', 'Prompt Engineering', 'From user input to model output'),
        ('scaling_laws', 'Scaling Laws', 'Larger models achieve better performance'),
    ],
    'topic-modeling': [
        ('topic_words', 'Topic Word Distribution', 'Each topic is a distribution over words'),
        ('document_topics', 'Document-Topic Mix', 'Each document contains multiple topics'),
        ('coherence_score', 'Finding Optimal Topics', 'Coherence score peaks at the best number of topics'),
    ],
    'responsible-ai': [
        ('fairness_metrics', 'Fairness Metrics', 'Checking for disparate impact across groups'),
        ('shap_importance', 'SHAP Feature Importance', 'Understanding which features drive predictions'),
        ('ethics_radar', 'Ethics Assessment', 'Evaluating AI systems across ethical dimensions'),
    ],
    'structured-output': [
        ('json_success_rate', 'JSON Generation Success', 'Structured output methods dramatically improve reliability'),
        ('validation_pipeline', 'Validation Pipeline', 'Steps to ensure clean, valid data'),
        ('error_types', 'Common Errors', 'Types of structured output failures'),
    ],
    'validation-metrics': [
        ('roc_curve', 'ROC Curve', 'Comparing classifier performance at all thresholds'),
        ('precision_recall', 'Precision-Recall Tradeoff', 'Adjusting threshold affects both metrics'),
        ('cross_validation', 'Cross-Validation', 'K-fold validation reduces overfitting to test set'),
    ],
    'ab-testing': [
        ('conversion_rates', 'Conversion Rates', 'Comparing control and variant performance'),
        ('statistical_significance', 'Statistical Significance', 'P-value drops below 0.05 when result is significant'),
        ('sample_size', 'Sample Size Planning', 'Smaller effects require larger samples to detect'),
    ],
    'finance-applications': [
        ('risk_metrics', 'Risk Metrics', 'Portfolio performance with Value at Risk (VaR)'),
        ('portfolio_allocation', 'Portfolio Allocation', 'ML-optimized asset distribution'),
        ('prediction_accuracy', 'Prediction Models', 'Comparing forecasting approaches'),
    ],
}


def generate_chart_section(topic):
    """Generate the chart section HTML for a topic (3-column grid)."""
    charts = CHARTS.get(topic, [])
    if not charts:
        return ''

    lines = ['\n## Visual Guides\n']
    lines.append('<div class="chart-grid">')

    for filename, title, description in charts:
        img_path = f'/ML_Design_Thinking_16/images/topics/{topic}/{filename}.png'
        lines.append('<div class="chart-item">')
        lines.append(f'<a href="{GITHUB_BASE}"><img src="{img_path}" alt="{title}"></a>')
        lines.append(f'<div class="chart-caption">{title}</div>')
        lines.append('</div>')

    lines.append('</div>\n')

    return '\n'.join(lines)


def update_topic_file(topic):
    """Update a single topic markdown file."""
    filepath = CONTENT_PATH / f'{topic}.md'
    if not filepath.exists():
        print(f'  Skipped: {topic}.md not found')
        return

    content = filepath.read_text(encoding='utf-8')

    # Remove existing Visual Guides section if present
    if '## Visual Guides' in content:
        parts = content.split('## Visual Guides')
        # Keep everything before Visual Guides
        before = parts[0].rstrip()
        # Find next ## section after Visual Guides
        after_visual = parts[1]
        next_section_idx = after_visual.find('\n## ')
        if next_section_idx != -1:
            after = after_visual[next_section_idx:]
        else:
            after = ''
        content = before + after

    # Find insertion point (before ## When to Use or at end)
    insertion_markers = ['## When to Use', '## Common Pitfalls', '## Prerequisites']
    insert_pos = len(content)

    for marker in insertion_markers:
        pos = content.find(marker)
        if pos != -1 and pos < insert_pos:
            insert_pos = pos

    # Generate and insert chart section
    chart_section = generate_chart_section(topic)
    new_content = content[:insert_pos].rstrip() + '\n' + chart_section + '\n' + content[insert_pos:]

    filepath.write_text(new_content, encoding='utf-8')
    print(f'  Updated: {topic}.md')


def main():
    print('Updating topic pages with charts...\n')
    for topic in CHARTS.keys():
        update_topic_file(topic)
    print('\nAll topic pages updated!')


if __name__ == '__main__':
    main()
