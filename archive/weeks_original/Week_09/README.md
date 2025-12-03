# Week 9: Multi-Metric Validation & Model Selection

## Overview

Week 9 moves beyond single-metric evaluation to comprehensive model validation. Students learn why accuracy alone is dangerous, master the precision-recall trade-off, and build production-ready validation pipelines. The focus is on rigorous statistical testing, business-aligned metrics, and confident deployment decisions.

**Core Message**: "Accuracy can be 95% and completely useless. Validate comprehensively, test statistically, align with business, communicate clearly, deploy confidently."

## Learning Objectives

By the end of Week 9, students will be able to:

1. Explain why accuracy fails for imbalanced datasets and cost-asymmetric problems
2. Calculate and interpret precision, recall, F1, F-beta, ROC-AUC, and PR-AUC
3. Build confusion matrices and extract all derived metrics (sensitivity, specificity, NPV, PPV)
4. Compare models using multiple metrics simultaneously
5. Perform cross-validation with confidence intervals
6. Test statistical significance of model differences (McNemar test)
7. Optimize decision thresholds for business constraints
8. Translate ML metrics into business impact ("95% precision" → "$2M saved annually")
9. Create stakeholder-appropriate performance dashboards
10. Make defensible deployment decisions with comprehensive validation

## Modular Structure

**Total Slides**: 51 (across 5 parts) - V1.1 updated for pedagogical framework compliance
**Charts**: 16 visualizations
**Handouts**: 3 skill-level guides
**Workshop**: Credit Risk Model Validation (60 minutes)

### Part 1: Foundation - Beyond Accuracy (10 slides)
**File**: `part1_foundation.tex`
**Purpose**: Establish why single metrics fail and introduce validation pyramid

**Content**:
1. The accuracy trap: 95% accurate spam filter that catches 0 spam
2. Real-world failure examples: Healthcare, fraud detection, hiring algorithms
3. The validation pyramid: Business metrics → ML metrics → Statistical tests → Error analysis
4. Confusion matrix anatomy: TP, TN, FP, FN definitions with visual diagram
5. Precision vs Recall: Fundamental inverse relationship explained
6. When to prioritize each metric: Spam (precision), cancer (recall), balanced (F1)
7. Class imbalance problems: Why 99% accuracy means nothing with 1% positive class
8. Cost-asymmetric decisions: FN costs $50K vs FP costs $5K in credit risk
9. Multi-metric mindset: Every metric has trade-offs, none tells full story
10. Learning objectives: What students will master by end of week

**Key Visualizations**:
- accuracy_trap.pdf: Pie chart showing 95% class imbalance
- validation_pyramid.pdf: 4-level hierarchy from business to technical
- confusion_matrix_anatomy.pdf: 2×2 grid with all derived metrics

### Part 2: Algorithms - Comprehensive Metrics (11 slides)
**File**: `part2_algorithms.tex`
**Purpose**: Deep dive into validation metrics with mathematical foundations

**Content**:
1. Confusion matrix mechanics: How to calculate all 4 quadrants
2. Precision formula: TP / (TP + FP) - "Of predicted positives, how many correct?"
3. Recall (Sensitivity) formula: TP / (TP + FN) - "Of actual positives, how many caught?"
4. Specificity formula: TN / (TN + FP) - "Of actual negatives, how many correct?"
5. F-beta family: F1 (balanced), F2 (favor recall), F0.5 (favor precision)
6. ROC curves: TPR vs FPR across all thresholds, threshold-independent evaluation
7. AUC interpretation: 0.5 (random), 0.7 (fair), 0.8 (good), 0.9 (excellent)
8. Precision-Recall curves: Better than ROC for imbalanced datasets
9. Multi-class metrics: Macro (equal class weight), micro (aggregate), weighted (by frequency)
10. Regression metrics: MSE, MAE, R², adjusted R² - when classification isn't appropriate
11. Custom business metrics: Expected profit, cost-weighted accuracy, regulatory compliance

**Key Visualizations**:
- precision_recall_tradeoff.pdf: Interactive slider showing inverse relationship
- f_beta_family.pdf: Contour plots of F0.5, F1, F2 in precision-recall space
- roc_curve_explained.pdf: Multiple models overlaid with AUC values
- pr_vs_roc.pdf: Same model on both curves showing imbalance impact

### Part 3: Implementation - Building Validation Systems (10 slides)
**File**: `part3_implementation.tex`
**Purpose**: Hands-on sklearn.metrics API and production pipelines

**Content**:
1. sklearn.metrics API tour: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
2. Cross-validation strategies: K-fold, stratified K-fold, time-series split, leave-one-out
3. Confusion matrix visualization: Seaborn heatmap with annotations
4. ROC curve plotting: Multi-model comparison with matplotlib
5. Statistical significance testing: McNemar test for paired model comparison
6. Model comparison framework: Pandas DataFrame with all metrics in one table
7. Hyperparameter impact: Threshold tuning from 0.1 to 0.9, business cost optimization
8. Validation pipeline architecture: Train/val/test split → CV → statistical tests → final eval
9. Common pitfalls: Data leakage, test set overfitting, wrong metric selection
10. Production-ready checklist: Before training, during training, before deployment

**Key Visualizations**:
- cross_validation_strategies.pdf: K-fold, stratified, time-series splits illustrated
- metric_correlation_heatmap.pdf: Which metrics move together vs independently
- threshold_optimization.pdf: Cost vs threshold curve showing optimal point
- validation_pipeline.pdf: Flowchart from raw data to deployment decision

### Part 4: Design - Communicating Model Performance (11 slides)
**File**: `part4_design.tex`
**Purpose**: Translate technical metrics for non-technical stakeholders

**Content**:
1. Stakeholder-specific dashboards: Executives (ROI), PMs (features vs metrics), engineers (debugging)
2. Confidence intervals: Bootstrap resampling for metric uncertainty quantification
3. Model comparison tables: Clear decision matrices for model selection
4. Error analysis: Which samples fail? Systematic patterns? Subgroup disparities?
5. Business impact translation: "95% precision" → "$2M saved annually from reduced false alarms"
6. A/B test readiness: Pre-deployment validation gates
7. Model cards: Performance section best practices
8. Explaining precision-recall to PMs: Trade-off communication
9. Interactive validation tools: Plotly Dash dashboards
10. **NEW (V1.1)**: When to use multi-metric validation - Judgment criteria for validation depth selection
11. Design framework summary: Core principles and communication checklist

**Key Visualizations**:
- model_comparison_dashboard.pdf: Executive summary with 5 models × 10 metrics
- business_metric_alignment.pdf: ML metrics mapped to business KPIs
- validation_checklist.pdf: Visual checklist of 20 pre-deployment requirements
- validation_depth_decision.pdf: **NEW (V1.1)** - Decision tree for choosing validation rigor

### Part 5: Practice - Credit Risk Workshop (9 slides)
**File**: `part5_practice.tex`
**Purpose**: Hands-on comprehensive validation exercise (NO industry case studies)

**Content**:
1. Workshop introduction: Compare 5 ML models for credit risk prediction using 10+ metrics
2. Dataset description: 10,000 loans, 5% default rate, 5 features (income, credit_score, employment_years, debt_to_income, loan_amount)
3. Business context: FN costs $50K (missed default), FP costs $5K (rejected good customer)
4. Step 1: Baseline evaluation - All 5 models × 10 metrics at default threshold 0.5
5. Step 2: Confusion matrix analysis - Which model has highest TP? Lowest FN? Systematic errors?
6. Step 3: Threshold optimization - Sweep 0.1 to 0.9, find minimum cost threshold per model
7. Step 4: Cross-validation - 5-fold CV with confidence intervals, stability assessment
8. Step 5: Statistical comparison - McNemar test: Is XGBoost significantly better than Random Forest?
9. Key takeaways: Accuracy dangerous, context determines metrics, validate comprehensively, deploy confidently

**Models to Compare**:
1. Logistic Regression (baseline)
2. Random Forest (ensemble stability)
3. XGBoost (gradient boosting power)
4. SVM with RBF kernel (non-linear boundaries)
5. Neural Network (2 hidden layers)

**Deliverable**: Jupyter notebook with:
- 5 models trained and evaluated
- Comprehensive metric table (5 models × 10+ metrics)
- Confusion matrices for all models
- Threshold optimization curves
- Cross-validation results with confidence intervals
- Statistical significance tests
- Final recommendation with business justification

## Key Visualizations (16 Charts)

All charts generated programmatically via `scripts/generate_all_charts.py`:

1. **accuracy_trap.pdf**: Pie chart showing 95% class imbalance - why accuracy is misleading
2. **confusion_matrix_anatomy.pdf**: 2×2 grid with TP/TN/FP/FN and all derived metrics
3. **precision_recall_tradeoff.pdf**: Interactive visualization of inverse relationship
4. **f_beta_family.pdf**: Contour plots showing F0.5, F1, F2 in precision-recall space
5. **roc_curve_explained.pdf**: Multiple models overlaid showing TPR vs FPR
6. **auc_interpretation.pdf**: Visual guide to AUC values (0.5 to 1.0)
7. **pr_vs_roc.pdf**: Same model on both curves - imbalance impact comparison
8. **multi_class_strategies.pdf**: One-vs-rest, macro, micro, weighted averaging illustrated
9. **cross_validation_strategies.pdf**: K-fold, stratified, time-series splits
10. **metric_correlation_heatmap.pdf**: Which metrics are redundant vs complementary
11. **threshold_optimization.pdf**: Business cost vs threshold curve
12. **model_comparison_dashboard.pdf**: 5 models × 10 metrics in executive summary
13. **validation_pipeline.pdf**: Data split → training → validation → testing → deployment
14. **business_metric_alignment.pdf**: ML metrics (precision, recall) → business KPIs (profit, risk)
15. **validation_checklist.pdf**: 20-item pre-deployment verification
16. **validation_pyramid.pdf**: 4-level hierarchy from business to technical validation
17. **validation_depth_decision.pdf** **NEW (V1.1)**: Decision tree for matching validation rigor to problem stakes

**Chart Generation**:
```bash
cd Week_09/scripts
python generate_all_charts.py
```

All charts saved as:
- PDF (300 dpi) for LaTeX inclusion
- PNG (150 dpi) for documentation/presentations

## Handouts (3 Skill Levels)

### Handout 1: Basic Metrics Guide (~200 lines)
**File**: `handouts/handout_1_basic_metrics.md`
**Audience**: Beginners with no ML background
**Content**:
- Plain English definitions: Accuracy, precision, recall, F1
- Real-world analogies: Spam filter, medical diagnosis, fraud detection
- When to use each metric (decision tree)
- Common mistakes and how to avoid them
- Python code snippets for sklearn.metrics
- Practice exercises with solutions

### Handout 2: Intermediate Validation (~400 lines)
**File**: `handouts/handout_2_intermediate_validation.md`
**Audience**: Students with basic ML knowledge
**Content**:
- Confusion matrix deep dive with derived metrics
- ROC and PR curves: When to use which
- Cross-validation strategies for different data types
- Threshold optimization techniques
- Statistical significance testing (McNemar, paired t-test)
- Multi-class and multi-label extensions
- Case studies: Credit risk, healthcare, recommender systems
- End-to-end validation pipeline code

### Handout 3: Advanced Model Selection (~500 lines)
**File**: `handouts/handout_3_advanced_selection.md`
**Audience**: Advanced students/practitioners
**Content**:
- Bayesian optimization for metric combinations
- Pareto frontier analysis for multi-objective selection
- Calibration curves and Brier scores
- Cost-sensitive learning theory
- Fairness metrics (demographic parity, equalized odds)
- Production monitoring: Metric drift detection
- A/B testing for model deployment
- Advanced topics: Conformal prediction, adversarial validation
- Research frontiers: AutoML, neural architecture search

## Compilation

### Using compile.py (Recommended)
```bash
cd Week_09
python compile.py
```

**Features**:
- Auto-detects latest main.tex file
- Runs pdflatex twice (for references)
- Archives auxiliary files to `archive/aux_YYYYMMDD_HHMM/`
- Opens PDF automatically on Windows
- Shows full output path

### Manual Compilation
```bash
cd Week_09
pdflatex 20250927_1021_main.tex
pdflatex 20250927_1021_main.tex  # Second pass for references
```

**Output**: `20250927_1021_main.pdf` (52 pages, ~600KB)

## Workshop Exercise: Credit Risk Model Validation

**Duration**: 60 minutes
**Format**: Jupyter notebook (individual or pairs)
**Dataset**: Synthetic but realistic 10,000 loan applications

### Dataset Specification
```python
# Features (numerical)
- income: Annual income ($20K-$200K)
- credit_score: FICO score (300-850)
- employment_years: Years at current job (0-40)
- debt_to_income: Debt-to-income ratio (0-1)
- loan_amount: Requested loan ($1K-$50K)

# Target
- default: 0 (repaid) or 1 (defaulted)
- Class imbalance: 5% default rate (realistic)

# Splits
- Train: 7,000 (70%)
- Validation: 1,500 (15%)
- Test: 1,500 (15%)
```

### Business Constraints
```
Cost Matrix (per loan):
- True Negative (approve good customer): +$2,000 profit
- False Positive (reject good customer): -$5,000 (lost business)
- False Negative (approve bad customer): -$50,000 (loan default)
- True Positive (reject bad customer): $0 (no transaction)

Key Insight: FN costs 10× more than FP → optimize for recall, but not at expense of precision
```

### Workshop Steps

**Step 1: Baseline Evaluation (15 min)**
- Train all 5 models on training set
- Evaluate on validation set with default threshold (0.5)
- Calculate 10 metrics: Accuracy, Precision, Recall, F1, F2, Specificity, NPV, ROC-AUC, PR-AUC, Expected Cost
- Create comparison table

**Step 2: Confusion Matrix Analysis (10 min)**
- Plot confusion matrices for all 5 models
- Identify which model has:
  - Highest TP (best at catching defaults)
  - Lowest FN (fewest missed defaults)
  - Lowest FP (fewest false alarms)
- Analyze error patterns: Are mistakes systematic?

**Step 3: Threshold Optimization (15 min)**
- For each model, sweep threshold from 0.1 to 0.9
- Calculate expected cost at each threshold
- Plot cost vs threshold curves
- Find optimal threshold for each model
- Compare optimized vs default performance

**Step 4: Cross-Validation (10 min)**
- Perform 5-fold stratified cross-validation
- Calculate mean and standard deviation for F1 score
- Compute 95% confidence intervals
- Identify most stable model (lowest variance)

**Step 5: Statistical Testing (10 min)**
- Apply McNemar test to top 2 models
- Calculate p-value
- Determine if performance difference is statistically significant
- Make final model recommendation with justification

### Expected Outcomes

**Typical Results**:
- Logistic Regression: High precision, low recall, conservative
- Random Forest: Balanced performance, very stable (low CV variance)
- XGBoost: Highest F1 and recall, moderate stability
- SVM: Good performance but sensitive to hyperparameters
- Neural Network: High variance, requires careful tuning

**Key Insight**: XGBoost and Random Forest likely tie on validation set. Statistical test shows no significant difference. Choose based on:
- XGBoost if prioritizing mean performance
- Random Forest if prioritizing stability and interpretability

**Optimal Thresholds**: Typically 0.2-0.4 (not default 0.5) due to 10:1 cost ratio favoring recall.

### Deliverable Template

Students submit Jupyter notebook with:

```markdown
# Credit Risk Model Validation Report

## 1. Executive Summary
- Recommendation: [Model name]
- Expected annual profit: [Dollar amount]
- Key trade-offs: [1-2 sentences]

## 2. Baseline Comparison
[Table: 5 models × 10 metrics]

## 3. Confusion Matrix Analysis
[5 confusion matrix heatmaps]
[Analysis of error patterns]

## 4. Threshold Optimization
[5 cost vs threshold curves]
[Table: Optimal threshold per model]

## 5. Cross-Validation Results
[Table: Mean ± Std with 95% CI for each model]
[Bar plot with error bars]

## 6. Statistical Significance
[McNemar test results for top 2 models]
[Interpretation: Significant or not?]

## 7. Final Recommendation
[Chosen model with justification addressing:]
- Performance metrics
- Stability
- Business alignment
- Deployment considerations
```

## Learning Outcomes

After completing Week 9, students will have mastered:

**Technical Skills**:
1. Implementing comprehensive validation pipelines with sklearn
2. Calculating and visualizing all classification metrics
3. Performing cross-validation with confidence intervals
4. Conducting statistical significance tests (McNemar)
5. Optimizing decision thresholds for business constraints
6. Building multi-metric model comparison frameworks

**Strategic Skills**:
1. Recognizing when accuracy is misleading
2. Selecting appropriate metrics for problem context
3. Balancing precision-recall trade-offs
4. Translating ML metrics to business impact
5. Making defensible deployment decisions
6. Communicating performance to non-technical stakeholders

**Professional Skills**:
1. Systematic validation methodology
2. Statistical rigor in model selection
3. Documentation and reproducibility
4. Stakeholder communication
5. Production readiness assessment

**Mindset Shifts**:
- From "highest accuracy wins" to "context determines metric"
- From "single metric" to "comprehensive validation"
- From "point estimate" to "confidence intervals"
- From "better on test set" to "statistically significant improvement"
- From "technical metric" to "business impact"

## Prerequisites

**Required**:
- Python 3.8+
- Week 4 (Classification fundamentals)
- Understanding of binary classification
- Basic statistics (mean, variance, confidence intervals)

**Python Packages**:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy statsmodels
```

**Recommended**:
- Week 1-3 (ML fundamentals)
- Jupyter notebook proficiency
- Basic knowledge of logistic regression, decision trees, ensemble methods

## Tools and Technologies

- **sklearn.metrics**: All classification metrics
- **sklearn.model_selection**: Cross-validation strategies
- **matplotlib/seaborn**: Visualization
- **pandas**: Model comparison tables
- **scipy.stats**: Statistical tests
- **statsmodels**: McNemar test

## Connection to Course Theme

Week 9 bridges ML validation with Design Thinking's "Test" phase:

**Design Thinking Parallel**:
- User testing requires multiple success metrics (usability, satisfaction, task completion)
- Similarly, ML models need multi-metric validation (precision, recall, F1, AUC)
- Both require iterative refinement based on comprehensive feedback
- Both must align with business objectives, not just technical perfection

**Innovation Diamond Context**:
- Validation is the final filter before the 5 strategic solutions
- Ensures only statistically sound, business-aligned models reach production
- Prevents false confidence from single-metric optimization

## Next Steps

**Week 10 Preview**: A/B Testing and Iterative Model Improvement
- Deploying models to production with gradual rollout
- Designing experiments to measure real-world impact
- Handling concept drift and model decay
- Continuous validation and retraining strategies

## Additional Resources

**Documentation**:
- sklearn.metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- Statistical tests: https://www.statsmodels.org/

**Academic Papers**:
- Powers, D. (2011). "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation"
- Davis, J. & Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves"

**Practical Guides**:
- Google ML Crash Course: Classification Metrics
- Towards Data Science: Comprehensive Guide to Classification Metrics

## Support

For questions about Week 9 content:
- Review handouts in `handouts/` directory
- Examine chart generation code in `scripts/`
- Consult CLAUDE.md for technical setup

## File Structure

```
Week_09/
├── 20250927_1021_main.tex              # Master presentation file
├── part1_foundation.tex                # 10 slides: Beyond Accuracy
├── part2_algorithms.tex                # 11 slides: Comprehensive Metrics
├── part3_implementation.tex            # 10 slides: Production Validation
├── part4_design.tex                    # 10 slides: Communicating Performance
├── part5_practice.tex                  # 9 slides: Credit Risk Workshop
├── compile.py                          # Automated LaTeX compilation
├── README.md                           # This file
├── charts/                             # 15 PDF + PNG visualizations
│   ├── accuracy_trap.pdf
│   ├── confusion_matrix_anatomy.pdf
│   ├── precision_recall_tradeoff.pdf
│   ├── f_beta_family.pdf
│   ├── roc_curve_explained.pdf
│   ├── auc_interpretation.pdf
│   ├── pr_vs_roc.pdf
│   ├── multi_class_strategies.pdf
│   ├── cross_validation_strategies.pdf
│   ├── metric_correlation_heatmap.pdf
│   ├── threshold_optimization.pdf
│   ├── model_comparison_dashboard.pdf
│   ├── validation_pipeline.pdf
│   ├── business_metric_alignment.pdf
│   └── validation_checklist.pdf
├── scripts/                            # Chart generation
│   └── generate_all_charts.py
├── handouts/                           # Skill-level guides
│   ├── handout_1_basic_metrics.md
│   ├── handout_2_intermediate_validation.md
│   └── handout_3_advanced_selection.md
└── archive/                            # Auxiliary files (auto-generated)
    └── aux_20250927_1021/
```

## Version History

- **2025-10-03 V1.1**: Pedagogical framework compliance upgrade
  - Added "When to Use Multi-Metric Validation" judgment criteria slide to Part 4
  - New chart: validation_depth_decision.pdf (decision tree for validation rigor)
  - Total slides: 50 → 51
  - Total charts: 15 → 16
  - Satisfies pedagogical_framework_Template.md requirements (Anti-Pattern #5)
  - Matches Week 8 V2.1 meta-knowledge standard

- **2025-09-27 10:21**: Initial release
  - 50 slides across 5 modular parts
  - 15 visualization charts generated
  - 3 skill-level handouts
  - Credit risk workshop exercise
  - Compilation tested successfully