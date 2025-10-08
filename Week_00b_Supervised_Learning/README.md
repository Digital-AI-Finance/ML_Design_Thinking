# Week 00b: Supervised Learning - The Prediction Challenge

## Overview
**Duration**: 90 minutes
**Format**: 4-act dramatic narrative
**Slides**: 27 total
**Charts**: 25 visualizations
**Structure**: Challenge → Solutions → Integration

## Learning Objectives
By the end of this session, students will be able to:
- Understand the prediction challenge (real estate pricing, spam detection)
- Recognize when linear models succeed vs fail
- Explain regularization and its importance
- Apply decision trees and ensemble methods
- Choose appropriate algorithms for supervised tasks
- Navigate the interpretability-accuracy tradeoff

## Prerequisites
- Week 00a (ML Foundations) recommended but not required
- Basic understanding of what "learning from data" means
- No calculus or linear algebra required

## Files
```
Week_00b_Supervised_Learning/
├── 20250928_1900_main.tex              # Main presentation file
├── act1_challenge.tex                  # The Challenge (5 slides)
├── act2_linear_regularization.tex      # Linear & Regularization (6 slides)
├── act3_nonlinear_methods.tex          # Nonlinear Methods (10 slides)
├── act4_synthesis.tex                  # Synthesis (6 slides)
├── charts/                             # 25 visualizations
│   ├── algorithm_comparison_table.pdf
│   ├── algorithm_landscape.pdf
│   ├── cart_algorithm_steps.pdf
│   ├── curse_dimensionality.pdf
│   ├── decision_boundaries_comparison.pdf
│   ├── ensemble_methods_performance.pdf
│   ├── feature_combinations.pdf
│   ├── human_decision_process.pdf
│   ├── interpretability_accuracy_tradeoff.pdf
│   ├── linear_failure_cases.pdf
│   ├── linear_regression_fit.pdf
│   ├── linear_success_cases.pdf
│   ├── linear_vs_nonlinear_boundaries.pdf
│   ├── nonlinear_methods_overview.pdf
│   ├── ols_example.pdf
│   ├── piecewise_approximation.pdf
│   ├── production_ml_pipeline.pdf
│   ├── real_estate_scatter.pdf
│   ├── regression_vs_classification.pdf
│   ├── regularization_tradeoff.pdf
│   ├── ridge_lasso_comparison.pdf
│   ├── supervised_to_unsupervised.pdf
│   ├── tree_2d_boundaries.pdf
│   ├── tree_building_example.pdf
│   └── twenty_questions_tree.pdf
├── scripts/                            # Chart generation
│   └── create_week0b_charts.py
└── archive/                            # Version control
```

## Compilation

### Standard Compilation
```powershell
cd Week_00b_Supervised_Learning
pdflatex 20250928_1900_main.tex
pdflatex 20250928_1900_main.tex  # Run twice for TOC
```

### Generate Charts
```powershell
cd scripts
python create_week0b_charts.py
```

## 4-Act Structure

### Act 1: The Prediction Challenge (5 slides)
**Scenario**: Real estate pricing - predict house prices from features
**Data**: Square footage, bedrooms, location, age
**Human Approach**: How do YOU estimate house value?
**ML Approach**: Learn from 10,000 historical sales
**Forward Question**: Can we do better than human intuition?

### Act 2: Linear Models & Regularization (6 slides)
**Success**: Linear regression works for simple, monotonic relationships
**Tool**: OLS (Ordinary Least Squares) - minimizes squared error
**Failure**: Overfitting with many features (curse of dimensionality)
**Solution**: Regularization (Ridge/Lasso) - penalize complexity
**Result**: Improved generalization on test data

### Act 3: Nonlinear Methods (10 slides)
**Human Insight**: "How do YOU play 20 questions?"
**Hypothesis**: Decision trees mimic human reasoning
**Mechanism**: Recursive binary splits (CART algorithm)
**Piecewise Approximation**: Trees create step-functions
**Ensemble Power**: Random forests average many trees
**Validation**: 15% accuracy improvement over linear models

### Act 4: Synthesis & Integration (6 slides)
**Modern Applications**:
- Credit scoring (logistic regression + trees)
- Customer churn prediction (ensemble methods)
- Medical diagnosis (interpretable models preferred)

**Meta-Knowledge**:
- When to use linear vs nonlinear
- Interpretability-accuracy tradeoff
- Production deployment considerations

**Common Pitfalls**:
- Overfitting small datasets
- Ignoring feature scaling
- Using wrong metric (accuracy vs AUC)

**Bridge**: Preview unsupervised learning (Week 00c)

## Key Algorithms Covered

### Linear Methods
1. **Linear Regression** - Continuous prediction
2. **Logistic Regression** - Binary classification
3. **Ridge Regression** - L2 regularization
4. **Lasso Regression** - L1 regularization (feature selection)

### Nonlinear Methods
5. **Decision Trees** - Human-interpretable splits
6. **Random Forests** - Ensemble of trees
7. **Gradient Boosting** - Sequential improvement
8. **SVM** - Maximum margin classification

## Pedagogical Approach

### From Human to Machine
Each algorithm starts with: "How would YOU solve this?"
- Linear regression → "Draw best-fit line through points"
- Decision trees → "Play 20 questions to classify"
- Ensembles → "Wisdom of crowds voting"

### Concrete Before Abstract
- Real estate example before mathematical formulation
- Visual decision boundaries before equations
- Actual performance numbers (not synthetic)

### Systematic Comparison
Every algorithm gets:
- Visual representation
- Strengths/weaknesses
- When to use / when NOT to use
- Real-world application example

## Connection to Innovation Framework

**Design Thinking Parallel**:
- **Define** → Supervised learning frames prediction problems
- **Data** → Historical examples replace user interviews
- **Model** → Algorithm is the designed solution
- **Validate** → Test set performance = user testing

**Innovation Application**:
- Predict innovation success from features
- Classify ideas into categories
- Forecast market adoption
- Score opportunity potential

## Status
✅ **Production Ready** (October 2025)
- All 25 charts verified
- Zero overflow issues (100% clean)
- Unicode compliant (no special characters)
- Bottom notes on all slides
- Pedagogical framework complete

## Teaching Notes

**Timing Breakdown**:
- Act 1 (Challenge): 12 min
- Act 2 (Linear): 18 min
- Act 3 (Nonlinear): 40 min
- Act 4 (Synthesis): 15 min
- Q&A: 5 min

**Live Demos** (if time):
- sklearn LinearRegression on real estate data
- Decision tree visualization with graphviz
- Feature importance from random forest

**Interactive Moments**:
- Slide 5: Estimate house price before revealing model
- Slide 12: Predict which features matter most
- Slide 18: Vote on decision tree split points

**Assessment Questions**:
1. When would you prefer a linear model over a tree?
2. How does regularization prevent overfitting?
3. What's the cost of ensemble methods?

## Dependencies
```powershell
pip install scikit-learn numpy pandas matplotlib seaborn
```
