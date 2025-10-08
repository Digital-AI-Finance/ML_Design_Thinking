# Week 00b Basic Handout: Supervised Learning - Predicting from Data

## Overview
Learn prediction algorithms without advanced math. Focus on intuition and practical use.

## Key Concepts

### What is Supervised Learning?
Learning to predict outputs from inputs when you have labeled examples.

**Examples**:
- Predict house prices from features (regression)
- Classify emails as spam/not spam (classification)
- Diagnose diseases from symptoms (classification)

### When to Use
- Have input-output pairs
- Want to predict new cases
- Pattern is consistent

## Algorithms at a Glance

### 1. Linear Regression
**Use**: Predict continuous values (price, temperature)
**Pro**: Simple, interpretable, fast
**Con**: Only captures linear relationships

### 2. Logistic Regression  
**Use**: Binary classification (yes/no decisions)
**Pro**: Probability outputs, interpretable
**Con**: Linear decision boundary

### 3. Decision Trees
**Use**: Both regression and classification
**Pro**: Human-readable, handles non-linear
**Con**: Overfits easily

### 4. Random Forest
**Use**: Most tasks, especially tabular data
**Pro**: Robust, handles overfitting, accurate
**Con**: Slower, less interpretable

### 5. Gradient Boosting (XGBoost)
**Use**: Kaggle competitions, production systems
**Pro**: State-of-art accuracy
**Con**: Requires tuning, slow training

## Decision Guide

**Start with**: Random Forest (most forgiving)
**Need speed**: Logistic/Linear Regression
**Need interpretability**: Decision Tree
**Need max accuracy**: XGBoost
**Non-linear + small data**: SVM with kernel

## Common Pitfalls
- Using accuracy on imbalanced data
- Ignoring feature scaling
- No train/test split
- Overfitting to training set

## Next Steps
- Week 00c: Unsupervised Learning
- Try: Kaggle Titanic competition
