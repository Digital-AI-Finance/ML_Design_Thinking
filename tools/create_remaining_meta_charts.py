"""Generate meta-knowledge decision charts for Weeks 4, 5, 7"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.style.use('seaborn-v0_8-whitegrid')

def create_decision_tree(title, root_question, branches, considerations, principle, output_path):
    """Generic decision tree creator"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, title, ha='center', fontsize=16, fontweight='bold')

    root_box = FancyBboxPatch((5.5, 8.2), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(root_box)
    ax.text(7, 8.6, root_question, ha='center', va='center', fontsize=12, fontweight='bold')

    positions = [(4, -2, -1.5), (7, 0, -1.5), (10, 2, -1.5)]

    for i, (branch, (x_base, dx, dy)) in enumerate(zip(branches, positions)):
        ax.arrow(7 - dx/2, 8.2, dx, dy, head_width=0.15, head_length=0.1, fc='black', ec='black')

        branch_box = FancyBboxPatch((x_base - 1.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                                   facecolor=branch['color'], edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(branch_box)
        ax.text(x_base, 6.2, branch['title'], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(x_base, 5.4, branch['when'], ha='center', va='top', fontsize=9)

        ax.arrow(x_base, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')

        result_box = FancyBboxPatch((x_base - 1.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                   facecolor=branch['result_color'], edgecolor='black', linewidth=2)
        ax.add_patch(result_box)
        ax.text(x_base, 4.5, branch['method'], ha='center', va='center',
                fontsize=11, fontweight='bold')
        ax.text(x_base, 4.1, branch['details'], ha='center', va='center', fontsize=8)

    consider_box = FancyBboxPatch((0.5, 0.5), 13, 2.5, boxstyle="round,pad=0.1",
                                  facecolor='#F0F0F0', edgecolor='black', linewidth=2)
    ax.add_patch(consider_box)
    ax.text(7, 2.7, 'Additional Considerations', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(7, 1.5, considerations, ha='center', va='center',
            fontsize=8, family='monospace')

    ax.text(7, 0.2, principle, ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path + '.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Created {os.path.basename(output_path)}.pdf/png')

# Week 4: Classification Algorithm Selection
week4_considerations = """
Class Balance: Severe imbalance (>95:5) → Use SMOTE/class weights or ensemble methods
Feature Count: <20 features → Logistic sufficient; 20-100 → Random Forest; >100 → Neural nets
Linearity: Linear separable → Logistic/SVM; Complex boundaries → Trees/Neural nets
Training Time: Real-time updates → Online logistic regression; Batch → Any method viable
Deployment: Edge devices → Small models (logistic, small trees); Cloud → Large ensembles OK
Multi-class: 2 classes → All methods; >10 classes → Neural nets or hierarchical classification
"""

create_decision_tree(
    'When to Use Which Classification Algorithm: Decision Framework',
    'What is your priority?',
    [
        {'title': 'INTERPRETABLE', 'color': '#2196F3', 'when': 'Need explainability\nRegulated domain\nStakeholder trust\nSimple relationships',
         'method': 'LOGISTIC/LINEAR', 'result_color': '#64B5F6',
         'details': 'Coefficients interpretable\nFast training\n75-85% accuracy\nGood baseline'},
        {'title': 'BALANCED PERFORMANCE', 'color': '#4CAF50', 'when': 'Non-linear patterns\nModerate data\nFeature interactions\nRobust to outliers',
         'method': 'RANDOM FOREST', 'result_color': '#81C784',
         'details': 'Handles non-linearity\n85-92% accuracy\nFeature importance\nParallel training'},
        {'title': 'MAXIMUM ACCURACY', 'color': '#FF9800', 'when': 'Large dataset\nComplex patterns\nProduction critical\nGPU available',
         'method': 'NEURAL NETWORKS', 'result_color': '#FFB74D',
         'details': 'Deep learning\n90-98% accuracy\nSlower training\nNeeds tuning'}
    ],
    week4_considerations,
    'Principle: Start interpretable (logistic), add complexity (trees/SVM) only when accuracy demands it',
    'Week_04/charts/classification_algorithm_decision'
)

# Week 5: Topic Modeling Selection
week5_considerations = """
Dataset Size: <1K docs → LSA/NMF simple; 1K-100K → LDA optimal; >100K → BERTopic scalable
Topic Count: Know K → LDA/NMF; Discover K → Hierarchical or BERTopic clustering
Languages: Multilingual → BERTopic with mBERT; Single language → Any method works
Real-time: Streaming topics → Online LDA; Batch analysis → Any method suitable
Coherence: Need coherent topics → BERTopic (best); LDA with tuning; NMF varies
Computation: Limited resources → NMF (fastest); GPU available → BERTopic; Medium → LDA
"""

create_decision_tree(
    'When to Use Which Topic Modeling Method: Decision Framework',
    'What is your priority?',
    [
        {'title': 'INTERPRETABLE TOPICS', 'color': '#9C27B0', 'when': 'Need clear themes\nStatistical foundation\nExplainable results\nMedium dataset',
         'method': 'LDA', 'result_color': '#BA68C8',
         'details': 'Probabilistic model\nInterpretable topics\nWell-established\nRequires tuning'},
        {'title': 'SPEED & SIMPLICITY', 'color': '#FF5722', 'when': 'Fast results needed\nSimple requirements\nSmaller dataset\nLinear algebra',
         'method': 'NMF', 'result_color': '#FF8A65',
         'details': 'Matrix factorization\nFastest method\nSparse topics\nEasy to implement'},
        {'title': 'BEST SEMANTICS', 'color': '#00BCD4', 'when': 'Rich embeddings\nContext matters\nLarge dataset\nCutting-edge results',
         'method': 'BERTOPIC', 'result_color': '#4DD0E1',
         'details': 'Transformer-based\nSemantic similarity\nDynamic topics\nBest coherence'}
    ],
    week5_considerations,
    'Principle: LDA for interpretable topics, NMF for speed, BERTopic for best coherence and modern semantics',
    'Week_05/charts/topic_modeling_decision'
)

# Week 7: Fairness Intervention Selection
week7_considerations = """
Intervention Stage: Have data access → Pre-processing; Model training → In-processing; Deployed model → Post-processing
Accuracy Trade-off: Minimize loss → Pre-processing (best); Balance → In-processing; Accept loss → Post-processing
Fairness Definition: Demographic parity → Pre-processing; Equal opportunity → In-processing; Equalized odds → Post-processing
Computational Cost: Limited compute → Pre-processing (once); GPU available → In-processing; Inference only → Post-processing
Transparency: Need audit trail → Pre-processing (data changes visible); Black box OK → In/post-processing
Stakeholders: Data scientists → Pre/in-processing; ML ops/deployment → Post-processing
"""

create_decision_tree(
    'When to Use Which Fairness Intervention: Decision Framework',
    'Where can you intervene?',
    [
        {'title': 'FIX THE DATA', 'color': '#3F51B5', 'when': 'Biased training data\nHistorical disparities\nEarly intervention\nData access',
         'method': 'PRE-PROCESSING', 'result_color': '#7986CB',
         'details': 'Reweighting samples\nSynthetic oversampling\nFairness constraints\nLeast accuracy loss'},
        {'title': 'FIX THE TRAINING', 'color': '#E91E63', 'when': 'Model training control\nCustom loss function\nAlgorithm modification\nGPU available',
         'method': 'IN-PROCESSING', 'result_color': '#F06292',
         'details': 'Fairness constraints\nRegularization terms\nAdversarial debiasing\nMedium accuracy loss'},
        {'title': 'FIX PREDICTIONS', 'color': '#FFC107', 'when': 'Deployed model\nNo retraining\nQuick fix needed\nThreshold tuning',
         'method': 'POST-PROCESSING', 'result_color': '#FFD54F',
         'details': 'Threshold optimization\nCalibration adjustment\nFastest to deploy\nMost accuracy loss'}
    ],
    week7_considerations,
    'Principle: Fix bias at the earliest stage possible - pre-processing preferred, post-processing as last resort',
    'Week_07/charts/fairness_intervention_decision'
)

print('\nAll meta-knowledge charts created successfully!')
print('Week 4: classification_algorithm_decision.pdf/png')
print('Week 5: topic_modeling_decision.pdf/png')
print('Week 7: fairness_intervention_decision.pdf/png')
