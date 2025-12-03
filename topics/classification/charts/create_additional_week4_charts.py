"""
Create additional visualization charts for Week 4: Classification & Definition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color palette matching the presentation
colors = {
    'mlpurple': '#3333B2',
    'mllavender': '#ADADE0',
    'mllavender2': '#C1C1E8',
    'mllavender3': '#CCCBEB',
    'mllavender4': '#D6D6EF',
    'mlblue': '#0066CC',
    'mlorange': '#FF7F0E',
    'mlgreen': '#2CA02C',
    'mlred': '#D62728',
    'mlgray': '#7F7F7F',
    'mlyellow': '#FFCE54',
    'mlcyan': '#17BECF'
}

def create_decision_cost_matrix():
    """Create decision cost matrix visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix style for costs
    costs = np.array([[0, -10], [-100, 0]])  # False positive vs False negative costs

    sns.heatmap(costs, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=['Predict Fail', 'Predict Success'],
                yticklabels=['Actually Fail', 'Actually Success'],
                cbar_kws={'label': 'Cost ($M)'}, ax=ax1)
    ax1.set_title('Cost Matrix: Type I vs Type II Errors', fontsize=14, fontweight='bold')

    # Impact visualization
    categories = ['False Rejection\n(Missed Unicorn)', 'False Acceptance\n(Failed Investment)']
    costs = [100, 10]  # Relative costs
    bars = ax2.bar(categories, costs, color=[colors['mlred'], colors['mlorange']])
    ax2.set_ylabel('Relative Cost Impact', fontsize=12)
    ax2.set_title('Asymmetric Error Costs', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost}x', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('decision_cost_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('decision_cost_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: decision_cost_matrix")

def create_algorithm_complexity_tradeoff():
    """Create algorithm complexity vs accuracy tradeoff chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Algorithm data
    algorithms = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest', 'SVM', 'Neural\nNetwork']
    accuracy = [76, 78, 89, 85, 92]
    complexity = [1, 2, 4, 5, 8]
    interpretability = [10, 9, 4, 2, 1]
    training_time = [0.1, 0.5, 2, 5, 10]

    # Create scatter plot with bubble size representing training time
    scatter = ax.scatter(complexity, accuracy, s=np.array(training_time)*100,
                        c=interpretability, cmap='RdYlGn',
                        alpha=0.7, edgecolors='black', linewidth=2)

    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        ax.annotate(alg, (complexity[i], accuracy[i]),
                   ha='center', va='center', fontweight='bold')

    ax.set_xlabel('Model Complexity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Trade-offs: Complexity vs Performance', fontsize=14, fontweight='bold')

    # Add colorbar for interpretability
    cbar = plt.colorbar(scatter)
    cbar.set_label('Interpretability Score', fontsize=11)

    # Add legend for bubble size
    legend_elements = [plt.scatter([], [], s=100, c='gray', alpha=0.7, label='Fast (<1s)'),
                      plt.scatter([], [], s=500, c='gray', alpha=0.7, label='Medium (1-5s)'),
                      plt.scatter([], [], s=1000, c='gray', alpha=0.7, label='Slow (>5s)')]
    ax.legend(handles=legend_elements, loc='lower right', title='Training Time')

    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 95)

    plt.tight_layout()
    plt.savefig('algorithm_complexity_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('algorithm_complexity_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: algorithm_complexity_tradeoff")

def create_confidence_distribution():
    """Create confidence score distribution chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate sample confidence scores
    np.random.seed(42)
    n_samples = 1000

    # Binary classification confidence
    confident_correct = np.random.beta(8, 2, n_samples // 2)
    uncertain = np.random.beta(2, 2, n_samples // 4)
    confident_wrong = np.random.beta(2, 8, n_samples // 4)

    ax1.hist(confident_correct, bins=30, alpha=0.7, label='Correct & Confident',
             color=colors['mlgreen'], edgecolor='black')
    ax1.hist(uncertain, bins=30, alpha=0.7, label='Uncertain',
             color=colors['mlyellow'], edgecolor='black')
    ax1.hist(confident_wrong, bins=30, alpha=0.7, label='Wrong & Confident',
             color=colors['mlred'], edgecolor='black')

    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Number of Predictions', fontsize=12)
    ax1.set_title('Confidence Distribution Analysis', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')

    # Calibration plot
    expected = np.linspace(0, 1, 10)
    actual_perfect = expected
    actual_overconfident = expected ** 1.5
    actual_underconfident = expected ** 0.7

    ax2.plot(expected, actual_perfect, 'k--', label='Perfect Calibration', linewidth=2)
    ax2.plot(expected, actual_overconfident, color=colors['mlred'],
             label='Overconfident Model', linewidth=2)
    ax2.plot(expected, actual_underconfident, color=colors['mlblue'],
             label='Underconfident Model', linewidth=2)

    ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax2.set_ylabel('Fraction of Positives', fontsize=12)
    ax2.set_title('Calibration Plot', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('confidence_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: confidence_distribution")

def create_feature_importance_comparison():
    """Create feature importance comparison across algorithms"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sample feature names
    features = ['Novelty Score', 'Market Size', 'Team Experience',
                'Dev Time', 'User Testing', 'Competition', 'Funding',
                'Tech Stack', 'Location', 'Industry']

    # Different importance scores for different algorithms
    np.random.seed(42)

    # Logistic Regression (more balanced)
    log_importance = np.random.uniform(0.05, 0.15, len(features))
    log_importance = log_importance / log_importance.sum()

    # Tree-based (more selective)
    tree_importance = np.random.exponential(0.1, len(features))
    tree_importance = tree_importance / tree_importance.sum()

    # Random Forest (ensemble view)
    rf_importance = np.random.beta(2, 5, len(features))
    rf_importance = rf_importance / rf_importance.sum()

    # Neural Network (complex interactions)
    nn_importance = np.random.gamma(2, 0.05, len(features))
    nn_importance = nn_importance / nn_importance.sum()

    # Plot each
    algorithms = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network']
    importances = [log_importance, tree_importance, rf_importance, nn_importance]

    for ax, alg, imp in zip(axes.flat, algorithms, importances):
        sorted_idx = np.argsort(imp)
        pos = np.arange(len(features))

        bars = ax.barh(pos, imp[sorted_idx], color=colors['mllavender2'])
        ax.set_yticks(pos)
        ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title(alg, fontsize=12, fontweight='bold')

        # Highlight top 3
        top3_idx = np.argsort(imp)[-3:]
        for idx in top3_idx:
            bar_idx = np.where(sorted_idx == idx)[0][0]
            bars[bar_idx].set_color(colors['mlpurple'])

    plt.suptitle('Feature Importance Across Algorithms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: feature_importance_comparison")

def create_real_world_performance():
    """Create real-world performance metrics visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Performance over data size
    data_sizes = [100, 500, 1000, 5000, 10000]
    log_acc = [65, 70, 73, 75, 76]
    rf_acc = [72, 80, 85, 88, 89]
    nn_acc = [60, 75, 83, 90, 92]

    ax1.plot(data_sizes, log_acc, 'o-', label='Logistic', color=colors['mlblue'], linewidth=2)
    ax1.plot(data_sizes, rf_acc, 's-', label='Random Forest', color=colors['mlgreen'], linewidth=2)
    ax1.plot(data_sizes, nn_acc, '^-', label='Neural Net', color=colors['mlpurple'], linewidth=2)
    ax1.set_xlabel('Training Set Size', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Learning Curves', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confusion matrix for best model
    conf_matrix = np.array([[850, 150], [100, 400]])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Predicted Fail', 'Predicted Success'],
                yticklabels=['Actual Fail', 'Actual Success'])
    ax2.set_title('Confusion Matrix (Random Forest)', fontsize=12, fontweight='bold')

    # Processing speed comparison
    algorithms = ['Logistic', 'Tree', 'RF', 'SVM', 'NN']
    train_times = [0.1, 0.5, 2, 5, 10]
    predict_times = [0.001, 0.001, 0.01, 0.005, 0.01]

    x = np.arange(len(algorithms))
    width = 0.35

    bars1 = ax3.bar(x - width/2, train_times, width, label='Training', color=colors['mlorange'])
    bars2 = ax3.bar(x + width/2, predict_times, width, label='Prediction', color=colors['mlcyan'])

    ax3.set_xlabel('Algorithm', fontsize=11)
    ax3.set_ylabel('Time (seconds)', fontsize=11)
    ax3.set_title('Speed Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms)
    ax3.legend()
    ax3.set_yscale('log')

    # Business metrics impact
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROI']
    before = [62, 58, 45, 51, 100]
    after = [89, 85, 82, 83, 340]

    x = np.arange(len(metrics))
    bars = ax4.bar(x, after, color=colors['mlgreen'], alpha=0.7, label='With ML')
    bars_before = ax4.bar(x, before, color=colors['mlgray'], alpha=0.7, label='Without ML')

    ax4.set_xlabel('Metric', fontsize=11)
    ax4.set_ylabel('Score / Percentage', fontsize=11)
    ax4.set_title('Business Impact', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()

    # Add improvement percentages
    for i, (b, a) in enumerate(zip(before, after)):
        improvement = ((a - b) / b) * 100
        ax4.text(i, a + 5, f'+{improvement:.0f}%', ha='center', fontweight='bold', color=colors['mlgreen'])

    plt.suptitle('Real-World Performance Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('real_world_performance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('real_world_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: real_world_performance")

def create_error_analysis_matrix():
    """Create error analysis matrix showing common failure patterns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error types and their frequencies
    error_types = ['Edge Cases', 'Noisy Labels', 'Feature Drift',
                   'Class Imbalance', 'Outliers', 'Missing Data']
    frequencies = [25, 15, 20, 30, 5, 5]

    # Create pie chart
    wedges, texts, autotexts = ax1.pie(frequencies, labels=error_types,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=[colors[c] for c in ['mlred', 'mlorange',
                                               'mlyellow', 'mlgreen', 'mlblue', 'mlpurple']])
    ax1.set_title('Common Error Sources', fontsize=14, fontweight='bold')

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    # Error impact heatmap
    algorithms = ['Logistic', 'Tree', 'RF', 'SVM', 'NN']
    impact_matrix = np.random.rand(len(error_types), len(algorithms)) * 100

    sns.heatmap(impact_matrix, xticklabels=algorithms, yticklabels=error_types,
                annot=True, fmt='.0f', cmap='YlOrRd', ax=ax2,
                cbar_kws={'label': 'Impact on Accuracy (%)'})
    ax2.set_title('Error Impact by Algorithm', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Algorithm', fontsize=11)
    ax2.set_ylabel('Error Type', fontsize=11)

    plt.tight_layout()
    plt.savefig('error_analysis_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('error_analysis_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: error_analysis_matrix")

def create_deployment_pipeline():
    """Create deployment pipeline visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pipeline stages
    stages = [
        ('Data\nIngestion', 0, 0, colors['mlblue']),
        ('Feature\nEngineering', 1, 0, colors['mlcyan']),
        ('Model\nTraining', 2, 0, colors['mlgreen']),
        ('Validation', 3, 0, colors['mlyellow']),
        ('Deployment', 4, 0, colors['mlorange']),
        ('Monitoring', 5, 0, colors['mlred']),
        ('Feedback', 5, -1.5, colors['mlpurple']),
        ('Retraining', 3, -1.5, colors['mlpurple'])
    ]

    # Draw stages
    for stage, x, y, color in stages:
        if 'Feedback' in stage or 'Retraining' in stage:
            rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6,
                                fill=True, facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=2, linestyle='--')
        else:
            rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6,
                                fill=True, facecolor=color, alpha=0.7,
                                edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, stage, ha='center', va='center',
               fontweight='bold', fontsize=10)

    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    # Main pipeline
    for i in range(5):
        ax.annotate('', xy=(i+1-0.4, .25), xytext=(i+0.4, 0),
                   arrowprops=arrow_props)

    # Feedback loop
    ax.annotate('', xy=(4.6, -1.2), xytext=(5, -0.3),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['mlpurple']))
    ax.annotate('', xy=(3.4, -1.2), xytext=(4.6, -1.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['mlpurple']))
    ax.annotate('', xy=(3, -0.3), xytext=(3, -1.2),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['mlpurple']))

    # Add timing annotations
    timings = ['1 min', '5 min', '10 min', '2 min', '1 min', 'Real-time']
    for i, timing in enumerate(timings):
        if i < 6:
            ax.text(i, 0.5, timing, ha='center', fontsize=9, style='italic')

    # Add title and labels
    ax.set_xlim(-1, 6.5)
    ax.set_ylim(-2.5, 1.5)
    ax.set_title('ML Classification: Production Pipeline', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors['mlgreen'], alpha=0.7, label='Core Pipeline'),
        plt.Rectangle((0,0),1,1, facecolor=colors['mlpurple'], alpha=0.3, label='Feedback Loop')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('deployment_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('deployment_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: deployment_pipeline")

def create_user_impact_metrics():
    """Create user impact metrics dashboard visualization"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # User satisfaction over time
    ax1 = fig.add_subplot(gs[0, :2])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    without_ml = [65, 66, 64, 65, 63, 64]
    with_ml = [65, 72, 78, 82, 85, 87]

    ax1.plot(months, without_ml, 'o--', label='Without ML', color=colors['mlgray'], linewidth=2)
    ax1.plot(months, with_ml, 'o-', label='With ML Classification', color=colors['mlgreen'], linewidth=2)
    ax1.fill_between(range(len(months)), without_ml, with_ml, alpha=0.3, color=colors['mlgreen'])
    ax1.set_ylabel('User Satisfaction (%)', fontsize=11)
    ax1.set_title('User Satisfaction Improvement', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # KPI improvements
    ax2 = fig.add_subplot(gs[0, 2])
    kpis = ['CTR', 'Conv', 'Retention', 'LTV']
    improvements = [34, 28, 42, 51]
    bars = ax2.barh(kpis, improvements, color=colors['mlpurple'])
    ax2.set_xlabel('Improvement (%)', fontsize=11)
    ax2.set_title('KPI Uplift', fontsize=12, fontweight='bold')
    for bar, imp in zip(bars, improvements):
        ax2.text(imp + 1, bar.get_y() + bar.get_height()/2,
                f'+{imp}%', va='center', fontweight='bold')

    # Response time distribution
    ax3 = fig.add_subplot(gs[1, 0])
    response_times = np.random.gamma(2, 10, 1000)
    ax3.hist(response_times, bins=30, color=colors['mlcyan'], alpha=0.7, edgecolor='black')
    ax3.axvline(x=50, color='red', linestyle='--', label='Target: 50ms')
    ax3.set_xlabel('Response Time (ms)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Prediction Speed', fontsize=12, fontweight='bold')
    ax3.legend()

    # Personalization effectiveness
    ax4 = fig.add_subplot(gs[1, 1:])
    segments = ['New Users', 'Regular', 'Power Users', 'Churning']
    generic = [45, 60, 70, 30]
    personalized = [65, 75, 85, 55]

    x = np.arange(len(segments))
    width = 0.35
    bars1 = ax4.bar(x - width/2, generic, width, label='Generic', color=colors['mlgray'])
    bars2 = ax4.bar(x + width/2, personalized, width, label='ML Personalized', color=colors['mlorange'])

    ax4.set_ylabel('Engagement Score', fontsize=11)
    ax4.set_title('Personalization Impact by User Segment', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(segments)
    ax4.legend()

    # A/B test results
    ax5 = fig.add_subplot(gs[2, :])
    tests = ['Test 1:\nRecommendations', 'Test 2:\nPricing', 'Test 3:\nUI Layout',
             'Test 4:\nContent', 'Test 5:\nNotifications']
    control = [100, 100, 100, 100, 100]
    treatment = [118, 107, 123, 115, 109]

    x = np.arange(len(tests))
    bars_control = ax5.bar(x - 0.2, control, 0.4, label='Control', color=colors['mlgray'], alpha=0.7)
    bars_treatment = ax5.bar(x + 0.2, treatment, 0.4, label='ML-Driven', color=colors['mlgreen'], alpha=0.7)

    ax5.set_ylabel('Relative Performance', fontsize=11)
    ax5.set_title('A/B Test Results: ML-Driven Decisions', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(tests)
    ax5.legend()
    ax5.axhline(y=100, color='black', linestyle='--', alpha=0.5)

    # Add significance markers
    for i, (c, t) in enumerate(zip(control, treatment)):
        if t - c > 10:
            ax5.text(i, t + 2, '***', ha='center', fontweight='bold', color=colors['mlgreen'])
        elif t - c > 5:
            ax5.text(i, t + 2, '**', ha='center', fontweight='bold', color=colors['mlorange'])

    plt.suptitle('User Impact Dashboard: Classification in Production', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('user_impact_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('user_impact_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: user_impact_metrics")

def create_risk_dashboard_mockup():
    """Create risk assessment dashboard mockup"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Overall risk gauge
    ax1 = fig.add_subplot(gs[0, 0])
    angles = np.linspace(0.75 * np.pi, 2.25 * np.pi, 100)
    values = np.ones(100)
    risk_level = 73  # Current risk score

    # Draw gauge segments
    segments = [(0, 30, colors['mlgreen']), (30, 70, colors['mlyellow']), (70, 100, colors['mlred'])]
    for start, end, color in segments:
        segment_angles = angles[(angles >= 0.75 * np.pi + (start/100) * 1.5 * np.pi) &
                               (angles <= 0.75 * np.pi + (end/100) * 1.5 * np.pi)]
        ax1.fill_between(segment_angles, 0.8, 1.0, color=color, alpha=0.5)

    # Draw needle
    needle_angle = 0.75 * np.pi + (risk_level/100) * 1.5 * np.pi
    ax1.plot([0, np.cos(needle_angle)], [0, np.sin(needle_angle)],
            'k-', linewidth=3)
    ax1.scatter([0], [0], color='black', s=100, zorder=5)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.text(0, -0.3, f'Risk Score: {risk_level}', ha='center', fontsize=14, fontweight='bold')
    ax1.set_title('Overall Risk Level', fontsize=12, fontweight='bold')

    # Risk factors
    ax2 = fig.add_subplot(gs[0, 1:])
    factors = ['Market Risk', 'Team Risk', 'Tech Risk', 'Competition', 'Timing']
    scores = [85, 45, 60, 70, 55]
    colors_list = [colors['mlred'] if s > 70 else colors['mlyellow'] if s > 40 else colors['mlgreen']
                   for s in scores]

    bars = ax2.barh(factors, scores, color=colors_list)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Risk Score', fontsize=11)
    ax2.set_title('Key Risk Drivers', fontsize=12, fontweight='bold')

    for bar, score in zip(bars, scores):
        ax2.text(score + 2, bar.get_y() + bar.get_height()/2,
                f'{score}%', va='center', fontweight='bold')

    # Historical trend
    ax3 = fig.add_subplot(gs[1, :2])
    days = list(range(30))
    risk_history = 50 + np.cumsum(np.random.randn(30) * 2)
    ax3.plot(days, risk_history, color=colors['mlpurple'], linewidth=2)
    ax3.fill_between(days, risk_history, 50, where=(risk_history >= 50),
                     color=colors['mlred'], alpha=0.3, label='Above baseline')
    ax3.fill_between(days, risk_history, 50, where=(risk_history < 50),
                     color=colors['mlgreen'], alpha=0.3, label='Below baseline')
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax3.set_xlabel('Days', fontsize=11)
    ax3.set_ylabel('Risk Score', fontsize=11)
    ax3.set_title('30-Day Risk Trend', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Confidence meter
    ax4 = fig.add_subplot(gs[1, 2])
    confidence = 82
    wedges, texts = ax4.pie([confidence, 100-confidence],
                           colors=[colors['mlgreen'], 'lightgray'],
                           startangle=90, counterclock=False)
    ax4.text(0, 0, f'{confidence}%\nConfidence', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax4.set_title('Model Confidence', fontsize=12, fontweight='bold')

    # Recommendations
    ax5 = fig.add_subplot(gs[2, :])
    recommendations = [
        ('1. Reduce market risk', 'Expand target market analysis', 'High', colors['mlred']),
        ('2. Strengthen team', 'Add senior technical advisor', 'Medium', colors['mlyellow']),
        ('3. Accelerate timeline', 'Move launch date up 2 months', 'Medium', colors['mlyellow']),
        ('4. Monitor competition', 'Weekly competitor analysis', 'Low', colors['mlgreen'])
    ]

    ax5.axis('off')
    ax5.set_title('AI-Generated Recommendations', fontsize=12, fontweight='bold', pad=20)

    for i, (title, desc, priority, color) in enumerate(recommendations):
        y_pos = 0.8 - i * 0.25
        ax5.text(0.05, y_pos, title, fontsize=11, fontweight='bold')
        ax5.text(0.05, y_pos - 0.08, desc, fontsize=9, style='italic')
        ax5.add_patch(plt.Rectangle((0.7, y_pos - 0.05), 0.15, 0.08,
                                   facecolor=color, alpha=0.5))
        ax5.text(0.775, y_pos - 0.01, priority, ha='center', va='center',
                fontsize=9, fontweight='bold')

    plt.suptitle('Risk Assessment Dashboard - Powered by ML Classification',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('risk_dashboard_mockup.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('risk_dashboard_mockup.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: risk_dashboard_mockup")

def main():
    """Generate all additional charts"""
    print("\n" + "="*50)
    print("Generating Additional Week 4 Classification Charts")
    print("="*50 + "\n")

    # Create all charts
    create_decision_cost_matrix()
    create_algorithm_complexity_tradeoff()
    create_confidence_distribution()
    create_feature_importance_comparison()
    create_real_world_performance()
    create_error_analysis_matrix()
    create_deployment_pipeline()
    create_user_impact_metrics()
    create_risk_dashboard_mockup()

    print("\n" + "="*50)
    print("All additional charts created successfully!")
    print("Total: 9 new charts (PDF and PNG versions)")
    print("="*50)

if __name__ == "__main__":
    main()