# Week 4: Classification Practice Exercises

## Overview
These exercises are designed to reinforce classification concepts at three skill levels. Choose exercises appropriate to your experience and comfort level.

---

## Level 1: Beginner Exercises (No coding required)

### Exercise 1.1: Manual Classification
**Time: 15 minutes**

You are evaluating 10 startup ideas. Classify each as "Likely to Succeed" or "Likely to Fail" based on the given features.

| Startup | Innovation Score (0-10) | Team Experience (years) | Market Size ($M) | Your Prediction |
|---------|-------------------------|-------------------------|------------------|-----------------|
| A | 8 | 12 | 500 | ? |
| B | 3 | 2 | 50 | ? |
| C | 7 | 8 | 200 | ? |
| D | 9 | 15 | 1000 | ? |
| E | 4 | 5 | 100 | ? |
| F | 6 | 10 | 300 | ? |
| G | 2 | 1 | 20 | ? |
| H | 8 | 3 | 400 | ? |
| I | 5 | 7 | 150 | ? |
| J | 9 | 20 | 800 | ? |

**Questions:**
1. What pattern did you use to make decisions?
2. Which feature influenced your decisions most?
3. Were any startups difficult to classify? Why?

### Exercise 1.2: Algorithm Matching
**Time: 10 minutes**

Match each algorithm with its best use case:

**Algorithms:**
- A. Logistic Regression
- B. Decision Tree
- C. Random Forest
- D. Neural Network
- E. K-Nearest Neighbors

**Use Cases:**
1. Need to explain every decision to stakeholders
2. Have millions of samples and complex patterns
3. Want a simple, fast baseline model
4. Have a small dataset and want to use similar examples
5. Need high accuracy and can sacrifice some explainability

### Exercise 1.3: Metric Interpretation
**Time: 10 minutes**

A classifier for detecting innovative products has these results:
- Accuracy: 85%
- 100 products tested
- 70 actually innovative
- 30 actually not innovative

**Questions:**
1. How many products did the classifier get wrong?
2. If it predicted 65 as innovative, how many were false positives?
3. Why might accuracy alone be misleading here?

### Exercise 1.4: Feature Importance
**Time: 10 minutes**

Rank these features from most to least important for predicting startup success:
- Location (Silicon Valley vs elsewhere)
- Founder's previous exits
- Current revenue
- Social media followers
- Years since founding
- Number of employees
- Patent count
- Customer reviews

**Explain your reasoning for the top 3 choices.**

---

## Level 2: Intermediate Exercises (Basic coding)

### Exercise 2.1: Build Your First Classifier
**Time: 30 minutes**

```python
# Complete this code to build a classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('innovation_dataset.csv')

# TODO: Select features (choose at least 3)
X = data[[_____, _____, _____]]

# TODO: Select target variable
y = data[_____]

# TODO: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    _____, _____, test_size=_____, random_state=42
)

# TODO: Create and train model
model = _____
model.fit(_____, _____)

# TODO: Make predictions
predictions = model._____

# TODO: Calculate accuracy
accuracy = _____
print(f"Model accuracy: {accuracy:.2%}")
```

**Bonus:** Try different features and see how accuracy changes.

### Exercise 2.2: Algorithm Comparison
**Time: 30 minutes**

Compare three algorithms on the same dataset:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Dictionary to store results
results = {}

# TODO: Train and evaluate each algorithm
algorithms = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, algorithm in algorithms.items():
    # Train model
    # Test model
    # Store accuracy
    pass  # Replace with your code

# TODO: Create a bar chart comparing accuracies
```

**Questions:**
1. Which algorithm performed best?
2. Which was fastest to train?
3. What trade-offs do you observe?

### Exercise 2.3: Handle Imbalanced Data
**Time: 25 minutes**

```python
# Create an imbalanced dataset
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    random_state=42
)

# TODO: Check class distribution
# TODO: Train a classifier without balancing
# TODO: Train a classifier with class_weight='balanced'
# TODO: Compare the results

# Hint: Look at both accuracy and recall for the minority class
```

### Exercise 2.4: Feature Engineering
**Time: 20 minutes**

Create new features to improve classification:

```python
# Original features
data = pd.DataFrame({
    'age': [25, 35, 45, 22, 38],
    'income': [30000, 60000, 80000, 25000, 70000],
    'education_years': [12, 16, 18, 14, 16],
    'purchased': [0, 1, 1, 0, 1]
})

# TODO: Create these new features:
# 1. income_per_year_education
# 2. age_income_ratio
# 3. high_education (1 if education_years > 15, else 0)

# TODO: Train models with and without new features
# TODO: Compare performance
```

---

## Level 3: Advanced Exercises (Full implementation)

### Exercise 3.1: Cross-Validation Pipeline
**Time: 45 minutes**

Implement a complete cross-validation pipeline with hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# TODO: Create a pipeline with:
# 1. StandardScaler
# 2. RandomForestClassifier

# TODO: Define parameter grid
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20, 30],
    'randomforestclassifier__min_samples_split': [2, 5, 10]
}

# TODO: Implement GridSearchCV with 5-fold cross-validation
# TODO: Find best parameters
# TODO: Evaluate on test set
# TODO: Plot learning curves
```

### Exercise 3.2: Custom Evaluation Metrics
**Time: 40 minutes**

Implement custom evaluation for innovation classification:

```python
def innovation_score(y_true, y_pred, cost_matrix):
    """
    Calculate custom score considering:
    - Cost of missing an innovative idea (false negative): -1000
    - Cost of pursuing non-innovative idea (false positive): -100
    - Reward for correct innovative prediction: +500
    - Reward for correct non-innovative prediction: +10
    """
    # TODO: Implement custom scoring
    pass

# TODO: Use this metric to evaluate different algorithms
# TODO: Compare with standard accuracy
# TODO: Discuss which metric is more appropriate
```

### Exercise 3.3: Ensemble Model
**Time: 50 minutes**

Build a voting ensemble combining multiple algorithms:

```python
from sklearn.ensemble import VotingClassifier

# TODO: Create 5 different base models
# TODO: Combine them in a VotingClassifier
# TODO: Compare soft vs hard voting
# TODO: Implement stacking with a meta-classifier
# TODO: Evaluate ensemble vs individual models
```

### Exercise 3.4: Production Deployment
**Time: 60 minutes**

Deploy a classifier as a REST API:

```python
# model_service.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# TODO: Load pre-trained model
# TODO: Implement preprocessing function
# TODO: Create prediction endpoint
# TODO: Add confidence scores
# TODO: Implement error handling
# TODO: Add logging
# TODO: Create Dockerfile

@app.route('/predict', methods=['POST'])
def predict():
    # Your implementation here
    pass

if __name__ == '__main__':
    app.run(debug=True)
```

**Deliverables:**
1. Working API
2. Test script
3. Documentation
4. Performance metrics

---

## Challenge Problems

### Challenge 1: Multi-class Classification
**Difficulty: Medium**

Extend binary classification to predict startup outcome in 5 categories:
- Unicorn (valuation > $1B)
- Success (profitable)
- Moderate (breaking even)
- Struggling (losing money)
- Failed (closed)

### Challenge 2: Real-time Classification
**Difficulty: Hard**

Build a system that:
1. Accepts streaming data
2. Updates predictions in real-time
3. Retrains periodically
4. Monitors for drift

### Challenge 3: Explainable AI
**Difficulty: Hard**

Implement SHAP or LIME to explain predictions:
1. Install SHAP library
2. Train a complex model
3. Generate explanations for individual predictions
4. Create visualizations
5. Build user interface for explanations

---

## Solutions Guide

### Exercise 1.1 Solution Hints
- High innovation score + high experience + large market = Success
- Consider creating a simple scoring rule
- Startups H (high innovation, low experience) are edge cases

### Exercise 1.2 Solutions
1. B (Decision Tree - explainable)
2. D (Neural Network - complex patterns)
3. A (Logistic Regression - simple baseline)
4. E (KNN - small dataset)
5. C (Random Forest - high accuracy)

### Exercise 2.1 Solution Framework
```python
X = data[['novelty_score', 'market_size', 'team_experience']]
y = data['success']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

---

## Rubric for Grading

### Beginner Level (Exercises 1.1-1.4)
- Completion: 40%
- Reasoning quality: 30%
- Understanding demonstrated: 30%

### Intermediate Level (Exercises 2.1-2.4)
- Code functionality: 40%
- Correct implementation: 30%
- Analysis and interpretation: 30%

### Advanced Level (Exercises 3.1-3.4)
- Technical implementation: 35%
- Code quality and style: 25%
- Documentation: 20%
- Innovation/creativity: 20%

---

## Additional Resources

### Datasets for Practice
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Toy Datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)

### Tutorials
- [Scikit-learn Classification Tutorial](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)
- [Python ML Tutorial](https://www.python-course.eu/machine_learning_classification.php)

### Reading
- "Pattern Recognition and Machine Learning" by Bishop (Chapter 4)
- "The Elements of Statistical Learning" (Chapters 2-4)
- "Hands-On Machine Learning" by Géron (Chapters 3-4)

---

## Tips for Success

1. **Start Simple**: Begin with beginner exercises even if you're experienced
2. **Experiment**: Try different parameters and observe changes
3. **Visualize**: Plot decision boundaries and confusion matrices
4. **Document**: Write comments explaining your reasoning
5. **Collaborate**: Discuss approaches with classmates
6. **Ask Questions**: No question is too basic
7. **Practice Daily**: Spend 30 minutes daily on exercises
8. **Build Portfolio**: Save your best solutions for your portfolio

---

*Remember: The goal is not just to get high accuracy, but to understand WHY your classifier makes certain decisions and HOW to improve it.*