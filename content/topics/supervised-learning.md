---
title: "Supervised Learning"
weight: 2
description: "Learning from labeled examples to make predictions"
difficulty: "Beginner"
duration: "60 minutes"
pdf_url: "downloads/supervised-learning.pdf"
---

# Supervised Learning

Learning from labeled examples to make predictions on new data.

## Learning Outcomes

By completing this topic, you will:
- Distinguish between regression and classification tasks
- Implement linear regression using OLS
- Evaluate model performance with appropriate metrics
- Understand the prediction workflow

## Visual Guides

<div class="chart-grid">
<div class="chart-item">
<a href="https://github.com/Digital-AI-Finance/ML_Design_Thinking/blob/main/tools/generate_web_charts.py"><img src="/ML_Design_Thinking/images/topics/supervised-learning/regression_vs_classification.png" alt="Regression vs Classification"></a>
<div class="chart-caption">Regression vs Classification</div>
</div>
<div class="chart-item">
<a href="https://github.com/Digital-AI-Finance/ML_Design_Thinking/blob/main/tools/generate_web_charts.py"><img src="/ML_Design_Thinking/images/topics/supervised-learning/bias_variance.png" alt="Bias-Variance Tradeoff"></a>
<div class="chart-caption">Bias-Variance Tradeoff</div>
</div>
<div class="chart-item">
<a href="https://github.com/Digital-AI-Finance/ML_Design_Thinking/blob/main/tools/generate_web_charts.py"><img src="/ML_Design_Thinking/images/topics/supervised-learning/train_test_split.png" alt="Train/Test Split"></a>
<div class="chart-caption">Train/Test Split</div>
</div>
</div>

## Prerequisites

- ML Foundations concepts
- Basic linear algebra (vectors, matrices)
- Understanding of mean squared error

## Key Concepts

### Regression vs Classification
- **Regression**: Predict continuous values (price, temperature)
- **Classification**: Predict discrete categories (spam/not spam, species)

### Linear Regression
The foundation of predictive modeling:
- Ordinary Least Squares (OLS) minimizes squared errors
- Coefficients show feature importance
- R-squared measures explained variance

### Model Evaluation
- **Regression**: MSE, RMSE, MAE, R-squared
- **Classification**: Accuracy, Precision, Recall, F1-score

## When to Use

Supervised learning works when:
- You have labeled historical data
- The relationship between inputs and outputs is learnable
- Future data resembles training data
- Predictions drive decisions

## Common Pitfalls

- Confusing correlation with causation
- Using accuracy on imbalanced datasets
- Ignoring feature scaling for distance-based methods
- Not validating on held-out data
