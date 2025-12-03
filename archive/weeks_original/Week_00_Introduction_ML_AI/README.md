# Week 00: Introduction to Machine Learning & AI

## Overview
**Duration**: 90 minutes
**Format**: Comprehensive ML survey
**Slides**: 45 (41 content + 4 appendix)
**Structure**: 5-part modular presentation

## Learning Objectives
- Understand ML/AI landscape and key paradigms
- Recognize supervised, unsupervised, and reinforcement learning
- Apply fundamental algorithms to real problems
- Master neural networks and deep learning basics
- Explore generative AI and modern applications

## Structure
```
Week_00_Introduction_ML_AI/
├── 20250928_1539_main.tex              # Main controller
├── part1_foundations.tex               # ML paradigms
├── part2_supervised_learning.tex       # Prediction algorithms
├── part3_unsupervised_learning.tex     # Clustering & dimensionality reduction
├── part4_neural_networks.tex           # Deep learning
├── part5_generative_ai.tex             # VAE, GAN, Diffusion
├── appendix_mathematics.tex            # Math reference
├── charts/                             # 42 visualizations
├── handouts/                           # Discovery-based worksheets
│   ├── 20251007_2300_discovery_handout_v2.tex
│   ├── 20251008_0800_discovery_solutions_v2.tex
│   ├── QUICK_START.md
│   └── README.md
└── scripts/                            # Chart generators + discovery charts
    ├── create_*_charts.py
    └── create_discovery_chart_*.py
```

## Compilation
```powershell
cd Week_00_Introduction_ML_AI
pdflatex 20250928_1539_main.tex  # Run twice for TOC
```

## Part Summaries

### Part 1: Foundations (8 slides)
- Learning paradigms comparison
- Supervised vs unsupervised vs reinforcement
- ML pipeline overview
- When to use ML

### Part 2: Supervised Learning (9 slides)
- Linear/logistic regression
- Decision trees and ensembles
- SVM and kernel methods
- Performance metrics

### Part 3: Unsupervised Learning (8 slides)
- K-means, DBSCAN, hierarchical clustering
- PCA and t-SNE
- Dimensionality reduction
- Cluster validation

### Part 4: Neural Networks (10 slides)
- Perceptron to deep learning
- CNNs for vision
- RNNs for sequences
- Transformers and attention

### Part 5: Generative AI (10 slides)
- Autoencoders and VAEs
- GANs (adversarial training)
- Diffusion models
- LLMs and foundation models

## Discovery-Based Handout System

**Pre-Lecture Worksheet** (45-55 minutes):
- 6 chart-driven discovery activities
- Zero prerequisites - pattern discovery before formalization
- Each activity: Chart → Observation → Conceptual tasks → Summary

**Charts**:
1. Overfitting demonstration (true overfitting: Train=0.0, Test>>Train)
2. K-means iterations (centroid movement)
3. Linear vs nonlinear boundaries
4. Gradient descent optimization
5. GAN adversarial dynamics
6. PCA dimensionality reduction

**Generation**:
```powershell
cd scripts
python create_discovery_chart_1_overfitting.py
python create_discovery_chart_2_kmeans.py
# ... (6 total)
```

## Status
✅ **Production Ready** (October 2025)
- 41 bottom notes added (all parts)
- Mathematically verified
- 42 charts rendering correctly
- Discovery handout complete with solutions
- Zero Unicode violations

## Use Cases

**Choose this version when**:
- Time-constrained (single 90-min session)
- General BSc audience (broad overview needed)
- Prerequisites for Weeks 1-10
- Quick introduction before project work

**Alternatives**:
- **Week 0a-0e series**: Deep pedagogical approach (5 × 90 min)
- **Week_00_Finance_Theory**: Finance-focused (quant students)

## Dependencies
```powershell
pip install scikit-learn numpy pandas matplotlib seaborn
```

## Teaching Notes
- Use discovery handout 1 week before lecture
- Emphasize breadth over depth (survey format)
- Connect each part to innovation applications
- Preview Weeks 1-10 specialization opportunities
