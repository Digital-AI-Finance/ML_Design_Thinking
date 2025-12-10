---
title: "A/B Testing"
weight: 13
description: "Statistical experimentation for data-driven decisions"
difficulty: "Intermediate"
duration: "75 minutes"
pdf_url: "downloads/ab-testing.pdf"
---

# A/B Testing

Statistical experimentation for data-driven decisions.

## Learning Outcomes

By completing this topic, you will:
- Design valid A/B experiments
- Calculate required sample sizes
- Analyze results with statistical rigor
- Avoid common experimentation pitfalls

## Visual Guides

### Conversion Rates

[![Conversion Rates](/ML_Design_Thinking_16/images/topics/ab-testing/conversion_rates.png)](https://github.com/Digital-AI-Finance/ML_Design_Thinking_16/blob/main/tools/generate_web_charts.py)

*Comparing control and variant performance*

### Statistical Significance

[![Statistical Significance](/ML_Design_Thinking_16/images/topics/ab-testing/statistical_significance.png)](https://github.com/Digital-AI-Finance/ML_Design_Thinking_16/blob/main/tools/generate_web_charts.py)

*P-value drops below 0.05 when result is significant*

### Sample Size Planning

[![Sample Size Planning](/ML_Design_Thinking_16/images/topics/ab-testing/sample_size.png)](https://github.com/Digital-AI-Finance/ML_Design_Thinking_16/blob/main/tools/generate_web_charts.py)

*Smaller effects require larger samples to detect*

## Prerequisites

- Basic statistics (mean, variance)
- Hypothesis testing concepts
- Understanding of p-values and confidence intervals

## Key Concepts

### Experiment Design
1. Define hypothesis and metrics
2. Calculate sample size for power
3. Randomize assignment
4. Run experiment for planned duration
5. Analyze and interpret results

### Statistical Analysis
- **Null hypothesis**: No difference between variants
- **p-value**: Probability of result under null
- **Confidence interval**: Range of plausible effects
- **Effect size**: Magnitude of difference

### Sample Size Calculation
Depends on:
- Minimum detectable effect (MDE)
- Statistical power (typically 80%)
- Significance level (typically 5%)
- Baseline conversion rate

## When to Use

A/B testing is appropriate when:
- Changes can be randomized fairly
- Sufficient traffic for statistical power
- Metric is measurable and relevant
- Time allows for proper experiment

## Common Pitfalls

- Stopping experiments early (peeking)
- Running multiple tests without correction
- Ignoring network effects
- Small sample sizes
- Wrong randomization unit
