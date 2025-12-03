# Week 4: Classification & Definition - Instructor Guide

## Overview
This guide provides comprehensive teaching notes for Week 4, which introduces classification algorithms for innovation success prediction. The materials are designed with dual versions to accommodate different audience levels.

## Course Context
- **Week Position**: 4 of 10
- **Prerequisites**: Weeks 1-3 (Clustering, NLP basics)
- **Duration**: 90-120 minutes lecture + 60 minutes lab
- **Difficulty Levels**: Beginner and Advanced versions available

## Learning Objectives

### Primary Objectives
1. Understand classification as pattern recognition for decision-making
2. Compare multiple classification algorithms and their trade-offs
3. Implement classifiers using scikit-learn
4. Evaluate model performance with appropriate metrics
5. Connect ML predictions to design thinking processes

### Secondary Objectives
- Understand overfitting and validation strategies
- Apply feature engineering techniques
- Deploy models to production systems
- Design user-friendly prediction interfaces

## Material Structure

### Available Versions
1. **Advanced Version** (`20250923_2140_main.tex`)
   - Mathematical foundations included
   - Technical implementation details
   - Complex visualizations
   - 45 slides total

2. **Beginner Version** (`20250923_2245_main_beginner.tex`)
   - No mathematical notation
   - Everyday analogies
   - Simplified visualizations
   - 48 slides total

### Choosing the Right Version
- **Use Beginner Version if**:
  - Students have no ML background
  - Mixed ability classroom
  - Focus is on concepts over implementation
  - Time is limited

- **Use Advanced Version if**:
  - Students have programming experience
  - Mathematical foundations are important
  - Deep technical understanding required
  - Graduate-level course

## Teaching Flow

### Part 1: Foundation - Why Classification? (25 minutes)

#### Key Concepts
- Problem: Scaling human judgment
- Evolution from manual to automated decisions
- Pattern discovery in innovation data
- Real-world applications

#### Teaching Points
1. **Start with Email Spam** (Slide 2-3)
   - Everyone understands spam filtering
   - Introduce binary classification naturally
   - Show how rules become algorithms

2. **Innovation Context** (Slide 4-5)
   - Connect to startup success prediction
   - Discuss human biases in evaluation
   - Show need for systematic approach

3. **The Diamond Metaphor** (Slide 6)
   - Recall Week 1's innovation diamond
   - Classification helps filter 5000→5 ideas
   - Bridge clustering to classification

#### Interactive Elements
- **Quick Poll**: "How do you decide if an email is spam?"
- **Discussion**: "What makes a startup likely to succeed?"
- **Activity**: Manual classification of 10 items

#### Common Student Questions
- Q: "Why not just use rules?"
  - A: Rules don't scale and miss complex patterns
- Q: "How is this different from clustering?"
  - A: Clustering finds groups; classification predicts labels

### Part 2: Algorithms - The Technical Core (30 minutes)

#### Algorithm Teaching Order (Beginner Version)
1. **Simple Scoring** (Logistic Regression)
   - "Judge giving points"
   - Linear combination of features
   - Show decision boundary

2. **20 Questions** (Decision Trees)
   - "Playing guessing game"
   - Visual tree structure
   - Easy to explain decisions

3. **100 Experts** (Random Forest)
   - "Wisdom of crowds"
   - Multiple trees voting
   - Reduces overfitting

4. **Brain-like** (Neural Networks)
   - "Mimicking neurons"
   - Hidden patterns
   - Black box nature

5. **Learn & Improve** (Gradient Boosting)
   - "Learning from mistakes"
   - Iterative improvement
   - Often best performance

#### Visual Demonstrations
- **Algorithm Showdown** (Slide 20-21)
  - Side-by-side comparison
  - Same data, different boundaries
  - Performance metrics table

#### Hands-On Demo
```python
# Live coding suggestion (5 minutes)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.1%}")
```

### Part 3: Implementation - Making It Work (25 minutes)

#### Pipeline Components
1. **Data Preparation**
   - Cleaning and preprocessing
   - Train/test split importance
   - Feature scaling

2. **Model Training**
   - Fitting process
   - Hyperparameters
   - Validation strategies

3. **Evaluation**
   - Accuracy vs other metrics
   - Confusion matrices
   - ROC curves (advanced only)

#### Key Teaching Points
- **Overfitting Prevention**
  - Use cooking analogy: "tasting while cooking"
  - Show overfitting visually
  - Introduce cross-validation

- **Feature Engineering**
  - Creating new features
  - Importance ranking
  - Dimensionality reduction

#### Common Pitfalls to Address
1. Testing on training data
2. Ignoring class imbalance
3. Not scaling features
4. Choosing wrong evaluation metric

### Part 4: Design Integration (15 minutes)

#### Connecting to Design Thinking
1. **Empathy**: Understanding user needs for predictions
2. **Define**: What decisions need automation?
3. **Ideate**: Which algorithms fit the context?
4. **Prototype**: Quick model iterations
5. **Test**: User feedback on predictions

#### Interface Design
- Show prediction confidence
- Explain decisions when possible
- Handle errors gracefully
- Provide override options

#### Ethics Discussion
- Bias in training data
- Fairness across groups
- Transparency requirements
- Human oversight needs

## Laboratory Session (60 minutes)

### Setup (10 minutes)
1. Ensure Jupyter notebooks are accessible
2. Load `Week04_Part1_Data_Exploration.ipynb`
3. Verify all libraries installed
4. Quick environment check

### Guided Exploration (20 minutes)
Work through notebook cells 1-10:
- Load innovation dataset
- Explore features
- Visualize distributions
- Check class balance

### Independent Practice (20 minutes)
Students complete:
- Feature importance analysis
- Data preparation pipeline
- First classifier implementation
- Performance evaluation

### Wrap-up (10 minutes)
- Share results
- Discuss challenges
- Preview next week

## Assessment Strategies

### Formative Assessment
- **Quick Checks**: After each algorithm, ask for comparison
- **Peer Explanation**: Students explain algorithms to each other
- **Notebook Completion**: Monitor progress through exercises

### Summative Assessment Options

#### Beginner Level
1. Classify 20 items manually, compare with algorithm (20%)
2. Run provided code, interpret results (40%)
3. Explain one algorithm in own words (40%)

#### Intermediate Level
1. Build classifier for provided dataset (30%)
2. Compare 3 algorithms, write report (40%)
3. Implement cross-validation (30%)

#### Advanced Level
1. Feature engineering challenge (25%)
2. Handle imbalanced dataset (25%)
3. Deploy model to web service (25%)
4. Write technical documentation (25%)

## Differentiation Strategies

### For Struggling Students
- Pair with stronger students
- Provide pre-written code templates
- Focus on conceptual understanding
- Use beginner handout exclusively
- Extra office hours

### For Advanced Students
- Challenge with real Kaggle dataset
- Implement from scratch
- Research papers to read
- Mentor struggling students
- Advanced handout problems

## Time Management

### 90-Minute Lecture
- 5 min: Review Week 3
- 25 min: Part 1 (Foundation)
- 30 min: Part 2 (Algorithms)
- 20 min: Part 3 (Implementation)
- 10 min: Part 4 (Design) & Wrap-up

### 120-Minute Lecture
- 5 min: Review Week 3
- 30 min: Part 1 (Foundation)
- 35 min: Part 2 (Algorithms) with demos
- 30 min: Part 3 (Implementation)
- 15 min: Part 4 (Design)
- 5 min: Preview Week 5

## Technical Requirements

### Software
- Python 3.8+
- Jupyter Notebook/Lab
- scikit-learn 1.0+
- pandas, numpy, matplotlib
- Optional: seaborn, plotly

### Hardware
- 4GB RAM minimum
- Internet for data downloads
- Projector for demonstrations

### Backup Plans
- Google Colab for system issues
- Pre-run notebook outputs
- Printed handouts available
- Recorded demonstrations

## Common Technical Issues

### Issue: Import errors
```python
# Solution: Install missing packages
!pip install scikit-learn pandas matplotlib
```

### Issue: Memory errors with large datasets
```python
# Solution: Reduce sample size
df_sample = df.sample(n=1000, random_state=42)
```

### Issue: Jupyter kernel dying
- Restart kernel
- Clear outputs
- Reduce visualization size
- Use subset of data

## Resources and References

### Required Reading
- Handout 1: Basic Classification (all students)
- Slides: Parts 1-4 appropriate to level

### Supplementary Materials
- Handout 2: Intermediate Classification
- Handout 3: Advanced Classification
- Original research papers (advanced)

### Online Resources
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Visual Introduction to ML](http://www.r2d3.us/)

## Week 4 to Week 5 Transition

### Concepts to Emphasize
- Classification finds patterns in labeled data
- Next week: Finding topics in text (unsupervised)
- Both help in innovation ideation process

### Homework Assignment
1. Complete notebook exercises
2. Apply classifier to own dataset
3. Read Week 5 preview materials
4. Think about text data sources

## Teaching Tips

### Do's
✓ Start with relatable examples (spam, recommendations)
✓ Use visual demonstrations liberally
✓ Emphasize practical applications
✓ Allow time for questions
✓ Provide immediate feedback in lab
✓ Connect to previous weeks' content
✓ Use appropriate version for audience

### Don'ts
✗ Don't rush through algorithms
✗ Don't assume mathematical knowledge (beginner version)
✗ Don't skip the "why" for the "how"
✗ Don't let one algorithm dominate discussion
✗ Don't forget ethical considerations
✗ Don't use only toy datasets

## Reflection Questions for Instructors

After teaching:
1. Which algorithms did students grasp best?
2. Were visualizations effective?
3. Did the pace work for all students?
4. What questions were most common?
5. How can the lab be improved?
6. Should more time be spent on any topic?

## Support and Contact

For questions or issues:
- Course Coordinator: innovation-ml@university.edu
- Technical Support: ml-support@university.edu
- Office Hours: Tuesday/Thursday 2-4pm
- Slack Channel: #ml-design-thinking

## Version Notes

- **v3** (Sept 23, 2025): Added beginner version, removed heavy math
- **v2** (Sept 23, 2025): Restructured to 4 parts
- **v1** (Sept 20, 2025): Initial version

---

*Remember: The goal is not just to teach algorithms, but to empower students to solve real problems with classification. Keep it practical, visual, and connected to innovation!*