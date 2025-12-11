---
title: "Classification"
weight: 7
description: "Categorizing data into predefined classes with decision trees"
difficulty: "Intermediate"
duration: "90 minutes"
pdf_url: "downloads/classification.pdf"
---

# Classification

Categorizing data into predefined classes using tree-based methods.

## Learning Outcomes

By completing this topic, you will:
- Build and interpret decision trees
- Apply random forests for robust predictions
- Handle imbalanced classification problems
- Evaluate classifiers with appropriate metrics

## Visual Guides

<div class="chart-grid">
<div class="chart-item">
<a href="/ML_Design_Thinking/images/topics/classification/decision_boundary.png" target="_blank"><img src="/ML_Design_Thinking/images/topics/classification/decision_boundary.png" alt="Decision Boundary"></a>
<div class="chart-caption">Decision Boundary</div>
</div>
<div class="chart-item">
<a href="/ML_Design_Thinking/images/topics/classification/confusion_matrix.png" target="_blank"><img src="/ML_Design_Thinking/images/topics/classification/confusion_matrix.png" alt="Confusion Matrix"></a>
<div class="chart-caption">Confusion Matrix</div>
</div>
<div class="chart-item">
<a href="/ML_Design_Thinking/images/topics/classification/decision_tree.png" target="_blank"><img src="/ML_Design_Thinking/images/topics/classification/decision_tree.png" alt="Decision Tree"></a>
<div class="chart-caption">Decision Tree</div>
</div>
</div>

## Prerequisites

- Supervised Learning concepts
- Understanding of entropy and information gain
- Basic probability concepts

## Key Concepts

### Decision Trees
Interpretable rule-based classifiers:
- Split data based on feature thresholds
- Information gain guides split selection
- Pruning prevents overfitting

### Random Forests
Ensemble of decision trees:
- Bootstrap aggregating (bagging)
- Random feature selection
- Reduced variance, improved accuracy

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / predicted positives
- **Recall**: True positives / actual positives
- **F1-score**: Harmonic mean of precision and recall
- **ROC-AUC**: Discrimination ability

## When to Use

| Method | Best For |
|--------|----------|
| Decision Tree | Interpretability, simple rules |
| Random Forest | Accuracy, feature importance |
| Gradient Boosting | Maximum performance |

## Common Pitfalls

- Overfitting with deep trees
- Ignoring class imbalance
- Using accuracy on skewed datasets
- Not tuning hyperparameters
- Forgetting to scale features for some algorithms
