# Innovation Diamond: ML-Powered Innovation

Capstone presentation integrating all 14 course topics through the Innovation Diamond framework.

## Overview

**Title**: ML-Powered Innovation: From Challenge to Strategy

**Duration**: 30-45 minutes (~25 slides)

**Running Example**: ESG/Sustainable Investing

**Key Message**: Balance expansion (divergent) and focus (convergent) for successful innovation

## The Innovation Diamond

```
                    DIVERGENT PHASE
                  (Exploring the Possible)
                         /\
                        /  \
                       /    \
        1 Challenge   /      \   10 dimensions
                     /        \
       100 features /          \  1,000 ideas
                   /            \
                  /   5,000      \
                 /     IDEAS      \
                 \     (PEAK)     /
                  \              /
       2,000      \            /   500 patterns
       filtered    \          /
                    \        /
        50 insights  \      /   5 strategies
                      \    /
                       \  /
                        \/
                   CONVERGENT PHASE
                 (Finding the Optimal)
```

## Topic Mapping

### Divergent Phase
| Stage | ML Technique | Course Topics |
|-------|--------------|---------------|
| Challenge (1) | Problem Framing | ML Foundations |
| Exploration (10) | Data Mining | Unsupervised Learning |
| Discovery (100) | Feature Engineering | Supervised Learning |
| Generation (1000) | Generative Algorithms | Generative AI, Topic Modeling |
| Peak (5000) | NLP Analysis | NLP & Sentiment |

### Convergent Phase
| Stage | ML Technique | Course Topics |
|-------|--------------|---------------|
| Extraction (2000) | Clustering | Clustering |
| Patterns (500) | Classification | Classification |
| Insights (50) | Optimization | Validation & Metrics, A/B Testing |
| Strategy (5) | Decision Support | Responsible AI, Finance Applications |

## ESG Case Study

**Challenge**: "How can we create an investment portfolio that maximizes returns while ensuring genuine environmental and social impact?"

The presentation follows this challenge through all diamond stages, demonstrating how ML techniques enable both creative expansion (discovering 5000 potential investment criteria) and strategic focus (converging to 5 actionable portfolio strategies).

## Folder Structure

```
innovation_diamond/
├── slides/
│   └── YYYYMMDD_HHMM_main.tex
├── charts/
│   ├── 01_diamond_overview/
│   ├── 02_divergent_process/
│   ├── 03_convergent_process/
│   └── ... (15 chart folders)
├── handouts/
│   ├── basic.md
│   ├── intermediate.md
│   └── advanced.md
├── compile.py
└── README.md
```

## Compilation

```bash
cd innovation_diamond
python compile.py
```

## Color Scheme

Uses diamond stage colors throughout:
- Purple: Challenge
- Blue: Exploration
- Green: Generation
- Yellow: Peak
- Orange: Filtering
- Red: Strategy
