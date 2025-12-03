import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import seaborn as sns
from datetime import datetime

# Nature Professional Color Palette
ForestGreen = '#14532D'
Teal = '#0D9488'
Amber = '#F59E0B'
Slate = '#475569'
MintCream = '#F0FDF4'
LightGreen = '#86EFAC'
DarkTeal = '#0F766E'
LightAmber = '#FCD34D'

plt.style.use('seaborn-v0_8-whitegrid')

def save_chart(name):
    plt.tight_layout()
    plt.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created {name}")

# Chart 1: Ethics Timeline
def create_ethics_timeline():
    fig, ax = plt.subplots(figsize=(12, 8))

    years = np.arange(2020, 2025)
    incidents = [45, 78, 103, 149, 233]

    ax.plot(years, incidents, 'o-', linewidth=3, markersize=12,
            color=ForestGreen, label='AI Incidents Reported')
    ax.fill_between(years, 0, incidents, alpha=0.2, color=Teal)

    # Annotations
    annotations = [
        (2020, 45, "Detroit\nFR Arrest #1"),
        (2021, 78, "Healthcare\nAI Bias"),
        (2022, 103, "Hiring\nAlgorithms"),
        (2023, 149, "EU AI Act\nProposed"),
        (2024, 233, "56% Increase\nPolicy Reforms")
    ]

    for year, incidents_val, label in annotations:
        ax.annotate(label, xy=(year, incidents_val), xytext=(year, incidents_val + 25),
                   ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=Amber, alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color=Slate, lw=2))

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Incidents', fontsize=12, fontweight='bold')
    ax.set_title('AI Ethics Incidents Timeline (2020-2024)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    save_chart('ethics_timeline')

# Chart 2: Bias Sources
def create_bias_sources():
    fig, ax = plt.subplots(figsize=(12, 8))

    stages = ['Data\nCollection', 'Feature\nEngineering', 'Model\nTraining',
              'Evaluation', 'Deployment']
    bias_types = ['Historical', 'Representation', 'Measurement', 'Evaluation', 'Feedback']
    severity = [0.9, 0.7, 0.8, 0.6, 0.75]

    x = np.arange(len(stages))
    bars = ax.bar(x, severity, color=[ForestGreen, Teal, Amber, DarkTeal, Slate], alpha=0.8)

    for i, (bar, sev, bias_type) in enumerate(zip(bars, severity, bias_types)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{sev:.0%}\n{bias_type}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel('Bias Risk Level', fontsize=12, fontweight='bold')
    ax.set_title('Where Bias Enters the ML Pipeline', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Risk Threshold')
    ax.legend(loc='upper right')

    save_chart('bias_sources')

# Chart 3: Fairness Metrics Comparison
def create_fairness_metrics_comparison():
    fig, ax = plt.subplots(figsize=(12, 8))

    metrics = ['Demographic\nParity', 'Equal\nOpportunity', 'Equalized\nOdds',
               'Calibration', 'Individual\nFairness']
    ease_implement = [0.9, 0.7, 0.5, 0.6, 0.3]
    fairness_strength = [0.5, 0.7, 0.9, 0.7, 0.95]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, ease_implement, width, label='Ease of Implementation',
                   color=Teal, alpha=0.8)
    bars2 = ax.bar(x + width/2, fairness_strength, width, label='Fairness Strength',
                   color=Amber, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Fairness Metrics: Implementation vs Strength', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    save_chart('fairness_metrics_comparison')

# Chart 4: Demographic Parity Visual
def create_demographic_parity():
    fig, ax = plt.subplots(figsize=(12, 8))

    groups = ['Group A\n(Privileged)', 'Group B\n(Unprivileged)']
    approved = [50, 50]  # Both 50% - demographic parity
    denied = [50, 50]

    x = np.arange(len(groups))
    width = 0.5

    bars1 = ax.bar(x, approved, width, label='Approved', color=LightGreen, alpha=0.8)
    bars2 = ax.bar(x, denied, width, bottom=approved, label='Denied', color='#fee08b', alpha=0.8)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Demographic Parity: Equal Approval Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 110)

    # Add equality line
    ax.axhline(y=50, color=ForestGreen, linestyle='--', linewidth=2, label='Parity Line')
    ax.text(0.5, 55, 'P(Approve|A) = P(Approve|B) = 50%', ha='center',
            fontsize=12, fontweight='bold', color=ForestGreen)

    save_chart('demographic_parity')

# Chart 5: Equal Opportunity Visual
def create_equal_opportunity():
    fig, ax = plt.subplots(figsize=(12, 8))

    groups = ['Group A', 'Group B']
    tpr = [0.85, 0.85]  # Equal TPR
    tnr = [0.70, 0.60]  # Different TNR allowed

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax.bar(x - width/2, tpr, width, label='True Positive Rate (Equal)',
                   color=Teal, alpha=0.8)
    bars2 = ax.bar(x + width/2, tnr, width, label='True Negative Rate (Can Differ)',
                   color=Slate, alpha=0.8)

    ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
    ax.set_title('Equal Opportunity: Equal TPR, TNR Can Differ', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.85, color=ForestGreen, linestyle='--', alpha=0.7)

    save_chart('equal_opportunity')

# Chart 6: Bias Detection Workflow
def create_bias_detection_workflow():
    fig, ax = plt.subplots(figsize=(14, 10))

    steps = [
        ('1. Identify\nProtected\nGroups', 0.1, 0.8),
        ('2. Choose\nFairness\nMetrics', 0.3, 0.8),
        ('3. Measure\nDisparities', 0.5, 0.8),
        ('4. Investigate\nRoot Causes', 0.7, 0.8),
        ('5. Apply\nMitigation', 0.9, 0.8),
        ('6. Re-test &\nMonitor', 0.5, 0.5)
    ]

    for i, (label, x, y) in enumerate(steps):
        color = [ForestGreen, Teal, Amber, DarkTeal, Slate, ForestGreen][i]
        box = FancyBboxPatch((x-0.08, y-0.08), 0.16, 0.16,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=3, color=Slate)
    for i in range(len(steps)-2):
        ax.annotate('', xy=(steps[i+1][1]-0.08, steps[i+1][2]),
                   xytext=(steps[i][1]+0.08, steps[i][2]),
                   arrowprops=arrow_props)

    # Last step arrows
    ax.annotate('', xy=(steps[5][1], steps[5][2]+0.08),
               xytext=(steps[4][1], steps[4][2]-0.08),
               arrowprops=arrow_props)
    ax.annotate('', xy=(steps[0][1], steps[0][2]-0.08),
               xytext=(steps[5][1]-0.15, steps[5][2]),
               arrowprops=dict(arrowstyle='->', lw=2, color=Slate, linestyle='dashed'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1)
    ax.axis('off')
    ax.set_title('Bias Detection Workflow', fontsize=16, fontweight='bold', pad=20)

    save_chart('bias_detection_workflow')

# Chart 7: Fairness Toolkit Comparison
def create_fairness_toolkit_comparison():
    fig, ax = plt.subplots(figsize=(12, 8))

    toolkits = ['IBM\nAIF360', 'Microsoft\nFairlearn', 'Google\nWhat-If', 'LinkedIn\nFaireseq', 'Aequitas']
    metrics = [70, 50, 30, 40, 60]
    algorithms = [10, 8, 5, 6, 7]
    ease_of_use = [6, 9, 8, 7, 7]

    x = np.arange(len(toolkits))
    width = 0.25

    bars1 = ax.bar(x - width, np.array(metrics)/10, width, label='Metrics (scaled /10)',
                   color=ForestGreen, alpha=0.8)
    bars2 = ax.bar(x, algorithms, width, label='Algorithms',
                   color=Teal, alpha=0.8)
    bars3 = ax.bar(x + width, ease_of_use, width, label='Ease of Use (1-10)',
                   color=Amber, alpha=0.8)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Fairness Toolkit Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(toolkits, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    save_chart('fairness_toolkit_comparison')

# Chart 8: SHAP Explanation Example
def create_shap_explanation_example():
    fig, ax = plt.subplots(figsize=(12, 8))

    features = ['Income', 'Credit\nHistory', 'Debt\nRatio', 'Employment', 'Age', 'Location']
    shap_values = [0.35, 0.28, -0.22, 0.15, 0.08, -0.05]
    colors_list = [ForestGreen if v > 0 else Amber for v in shap_values]

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, shap_values, color=colors_list, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
    ax.set_title('SHAP Feature Importance Example: Loan Approval', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    positive_patch = patches.Patch(color=ForestGreen, label='Positive Impact')
    negative_patch = patches.Patch(color=Amber, label='Negative Impact')
    ax.legend(handles=[positive_patch, negative_patch], loc='lower right', fontsize=10)

    save_chart('shap_explanation_example')

# Chart 9: Model Card Template (Simplified visual)
def create_model_card_template():
    fig, ax = plt.subplots(figsize=(12, 10))

    sections = [
        ('Model Details', 0.1, 0.9, ForestGreen),
        ('Intended Use', 0.1, 0.75, Teal),
        ('Factors & Metrics', 0.1, 0.6, Amber),
        ('Training Data', 0.6, 0.9, DarkTeal),
        ('Evaluation Data', 0.6, 0.75, Slate),
        ('Ethical Considerations', 0.6, 0.6, ForestGreen),
        ('Caveats & Recommendations', 0.35, 0.4, Amber)
    ]

    for label, x, y, color in sections:
        width = 0.4 if y == 0.4 else 0.35
        box = FancyBboxPatch((x, y-0.08), width, 0.12,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(x + width/2, y, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.axis('off')
    ax.set_title('Model Card Template Structure', fontsize=16, fontweight='bold', pad=20)

    save_chart('model_card_template')

# Chart 10: Privacy Techniques
def create_privacy_techniques():
    fig, ax = plt.subplots(figsize=(12, 8))

    techniques = ['Differential\nPrivacy', 'Federated\nLearning', 'Secure\nMPC',
                  'Homomorphic\nEncryption', 'Differential\nPrivacy + FL']
    privacy_level = [0.9, 0.75, 0.95, 0.98, 0.92]
    utility_preserved = [0.70, 0.85, 0.65, 0.50, 0.75]
    complexity = [0.6, 0.7, 0.9, 0.95, 0.85]

    x = np.arange(len(techniques))
    width = 0.25

    bars1 = ax.bar(x - width, privacy_level, width, label='Privacy Level',
                   color=ForestGreen, alpha=0.8)
    bars2 = ax.bar(x, utility_preserved, width, label='Utility Preserved',
                   color=Teal, alpha=0.8)
    bars3 = ax.bar(x + width, complexity, width, label='Implementation Complexity',
                   color=Amber, alpha=0.8)

    ax.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Privacy-Preserving ML Techniques Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    save_chart('privacy_techniques')

# Continue with remaining charts...
print("Generating all charts...")

create_ethics_timeline()
create_bias_sources()
create_fairness_metrics_comparison()
create_demographic_parity()
create_equal_opportunity()
create_bias_detection_workflow()
create_fairness_toolkit_comparison()
create_shap_explanation_example()
create_model_card_template()
create_privacy_techniques()

# Chart 11: Carbon Footprint ML
def create_carbon_footprint_ml():
    fig, ax = plt.subplots(figsize=(12, 8))

    models = ['BERT\nBase', 'BERT\nLarge', 'GPT-2', 'GPT-3', 'Transformer\nNAS']
    co2_tons = [0.65, 1.51, 2.8, 552, 284]

    bars = ax.barh(models, co2_tons, color=[ForestGreen, Teal, Amber, '#e74c3c', DarkTeal], alpha=0.8)

    ax.set_xlabel('CO2 Emissions (metric tons)', fontsize=12, fontweight='bold')
    ax.set_title('Carbon Footprint of Training Large ML Models', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')

    # Add comparison annotations
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.8, 4.3, 'Average\nCar/Year\n(4.6 tons)', ha='right', fontsize=9, color=Slate)
    ax.axvline(x=626, color='red', linestyle='--', alpha=0.5)
    ax.text(550, 4.3, '5 Cars\nLifetime', ha='right', fontsize=9, color='#e74c3c', fontweight='bold')

    save_chart('carbon_footprint_ml')

# Chart 12: Inclusive Design Principles
def create_inclusive_design_principles():
    fig, ax = plt.subplots(figsize=(12, 8))

    principles = [
        ('Recognize\nDiversity', 0.2, 0.7),
        ('Provide\nChoice', 0.5, 0.7),
        ('Be Flexible', 0.8, 0.7),
        ('Enable\nParticipation', 0.35, 0.4),
        ('Distribute\nPower', 0.65, 0.4)
    ]

    colors = [ForestGreen, Teal, Amber, DarkTeal, Slate]

    for (label, x, y), color in zip(principles, colors):
        circle = plt.Circle((x, y), 0.12, color=color, alpha=0.8, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Add connecting lines
    for i in range(len(principles)-2):
        ax.plot([principles[i][1], principles[i+1][1]],
                [principles[i][2], principles[i+1][2]],
                'k-', linewidth=2, alpha=0.3)

    ax.plot([principles[2][1], principles[3][1]], [principles[2][2], principles[3][2]],
            'k-', linewidth=2, alpha=0.3)
    ax.plot([principles[2][1], principles[4][1]], [principles[2][2], principles[4][2]],
            'k-', linewidth=2, alpha=0.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.9)
    ax.axis('off')
    ax.set_title('Inclusive Design Principles', fontsize=16, fontweight='bold', pad=20)

    save_chart('inclusive_design_principles')

# Chart 13: Case Study Timeline
def create_case_study_timeline():
    fig, ax = plt.subplots(figsize=(14, 8))

    events = [
        (2020, 'Detroit FR Arrest #1', ForestGreen),
        (2020.5, 'Amazon Rekognition Bias', Teal),
        (2021, 'More FR Arrests', Amber),
        (2022, 'Healthcare AI Discrimination', DarkTeal),
        (2023, 'Hiring Algorithm Lawsuits', Slate),
        (2024, 'UK Facewatch Case', ForestGreen),
        (2024.3, 'Uber Eats FR Failure', Teal),
        (2024.5, 'Detroit Settlement', Amber)
    ]

    for year, event, color in events:
        ax.scatter(year, 1, s=500, color=color, alpha=0.8, edgecolor='black', linewidth=2, zorder=3)
        ax.text(year, 1.15, event, ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.6))

    ax.axhline(y=1, color='black', linewidth=3, zorder=1)
    ax.set_xlim(2019.5, 2024.8)
    ax.set_ylim(0.8, 1.5)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_title('Major AI Ethics Incidents Timeline (2020-2024)', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')

    save_chart('case_study_timeline')

# Chart 14: Ethical Framework Decision Tree
def create_ethical_framework_decision_tree():
    fig, ax = plt.subplots(figsize=(14, 10))

    nodes = [
        ('Should we\nbuild this?', 0.5, 0.9, ForestGreen),
        ('Alternatives\nbetter?', 0.25, 0.7, Teal),
        ('Can build\nresponsibly?', 0.75, 0.7, Teal),
        ('STOP\nNo Build', 0.1, 0.5, '#e74c3c'),
        ('Explore\nAlternatives', 0.35, 0.5, Amber),
        ('Define\nSafeguards', 0.65, 0.5, DarkTeal),
        ('Can mitigate\nharms?', 0.9, 0.5, Slate),
        ('Implement\nwith Monitoring', 0.5, 0.3, ForestGreen),
        ('STOP\nRe-evaluate', 0.9, 0.3, '#e74c3c')
    ]

    for label, x, y, color in nodes:
        size = 0.08 if 'STOP' in label else 0.1
        box = FancyBboxPatch((x-size, y-size), size*2, size*2,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Add decision arrows
    arrow_props = dict(arrowstyle='->', lw=2, color=Slate)
    connections = [
        (0, 1, 'Yes'), (0, 2, 'No'),
        (1, 3, 'Yes'), (1, 4, 'No'),
        (2, 5, 'Yes'), (2, 6, 'No'),
        (5, 7, ''), (6, 7, 'Yes'), (6, 8, 'No'),
        (4, 7, '')
    ]

    for start, end, label_text in connections:
        if label_text:
            mid_x = (nodes[start][1] + nodes[end][1]) / 2
            mid_y = (nodes[start][2] + nodes[end][2]) / 2
            ax.text(mid_x, mid_y + 0.02, label_text, ha='center',
                   fontsize=9, fontweight='bold', color=Slate)
        ax.annotate('', xy=(nodes[end][1], nodes[end][2]+0.1),
                   xytext=(nodes[start][1], nodes[start][2]-0.1),
                   arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 1)
    ax.axis('off')
    ax.set_title('Ethical Decision Framework for AI Projects', fontsize=16, fontweight='bold', pad=20)

    save_chart('ethical_framework_decision_tree')

# Chart 15: Regulatory Landscape
def create_regulatory_landscape():
    fig, ax = plt.subplots(figsize=(12, 8))

    regions = ['EU', 'USA', 'China', 'UK', 'Canada', 'Australia']
    comprehensiveness = [0.95, 0.40, 0.75, 0.70, 0.55, 0.50]
    enforcement = [0.85, 0.35, 0.90, 0.65, 0.45, 0.40]
    penalties = [0.90, 0.30, 0.80, 0.70, 0.40, 0.35]

    x = np.arange(len(regions))
    width = 0.25

    bars1 = ax.bar(x - width, comprehensiveness, width, label='Comprehensiveness',
                   color=ForestGreen, alpha=0.8)
    bars2 = ax.bar(x, enforcement, width, label='Enforcement Strength',
                   color=Teal, alpha=0.8)
    bars3 = ax.bar(x + width, penalties, width, label='Penalty Severity',
                   color=Amber, alpha=0.8)

    ax.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Global AI Regulation Landscape (2024)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add note
    ax.text(0.5, -0.25, 'EU AI Act (2024) sets global gold standard',
            ha='center', fontsize=10, fontweight='bold',
            transform=ax.transAxes, color=ForestGreen)

    save_chart('regulatory_landscape')

print("\nFirst 10 charts created successfully!")
print("Creating remaining 5 charts...")

create_carbon_footprint_ml()
create_inclusive_design_principles()
create_case_study_timeline()
create_ethical_framework_decision_tree()
create_regulatory_landscape()

print("\nAll 15 charts created successfully!")