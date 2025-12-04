---
title: "Validation & Metrics"
weight: 12
description: "Evaluating and measuring model performance"
difficulty: "Intermediate"
duration: "75 minutes"
pdf_url: "downloads/validation-metrics.pdf"
---

# Validation & Metrics

Evaluating and measuring model performance systematically.

## Learning Outcomes

By completing this topic, you will:
- Implement cross-validation strategies
- Choose appropriate metrics for your problem
- Avoid common evaluation mistakes
- Design validation for production systems

## Prerequisites

- Supervised Learning concepts
- Classification and Regression basics
- Understanding of overfitting

## Key Concepts

### Cross-Validation
Robust model evaluation:
- **K-fold**: Split data into K parts, rotate test set
- **Stratified**: Preserve class distribution in folds
- **Time series**: Respect temporal ordering

### Classification Metrics
| Metric | Best For |
|--------|----------|
| Accuracy | Balanced classes |
| Precision | Cost of false positives high |
| Recall | Cost of false negatives high |
| F1-score | Balance precision and recall |
| ROC-AUC | Overall discrimination |

### Regression Metrics
- **MSE/RMSE**: Penalizes large errors
- **MAE**: Robust to outliers
- **R-squared**: Explained variance

## When to Use

Validation depth depends on:
- Model complexity and risk
- Data size and variability
- Deployment requirements
- Regulatory constraints

## Common Pitfalls

- Using accuracy on imbalanced data
- Data leakage during preprocessing
- Optimizing wrong metric for business
- Not holding out final test set
- Ignoring variance across folds
