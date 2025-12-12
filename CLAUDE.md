# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

"Machine Learning for Smarter Innovation" - BSc-level course bridging ML/AI with design thinking. Content organized by ML topic with 14 modules covering foundations through advanced applications.

**Core Concept**: The Innovation Diamond - expansion from 1 challenge to 5000 possibilities, then ML-driven convergence to 5 strategic solutions.

## Architecture

```
ML_Design_Thinking/
├── topics/                    # Active course content (14 topics)
│   ├── ml_foundations/        # ML/AI overview, learning paradigms
│   ├── supervised_learning/   # Regression, classification, ensembles
│   ├── unsupervised_learning/ # Clustering theory
│   ├── clustering/            # Applied clustering, personas, design thinking
│   ├── nlp_sentiment/         # Text analysis, BERT, sentiment
│   ├── classification/        # Decision trees, random forests
│   ├── topic_modeling/        # LDA, document analysis
│   ├── generative_ai/         # LLMs, prompt engineering, GANs
│   ├── neural_networks/       # Deep learning architectures
│   ├── responsible_ai/        # Ethics, fairness, SHAP
│   ├── structured_output/     # JSON output, reliability
│   ├── validation_metrics/    # Model evaluation
│   ├── ab_testing/            # Experiments, statistics
│   └── finance_applications/  # Quantitative finance, risk
├── innovation_diamond/        # Innovation Diamond standalone topic
├── archive/weeks_original/    # Read-only: original Week_00-10 folders
├── docs/                      # Status reports, planning documents
├── tools/                     # Python utilities
└── template_beamer_final.tex  # Standard Beamer template
```

### Topic Folder Structure

```
topics/{topic_name}/
├── compile.py        # LaTeX compilation script (run from topic root)
├── slides/           # .tex and .pdf files
├── charts/           # Visualizations (.pdf, .png) + generation scripts
├── scripts/          # Additional chart generation scripts (create_*.py)
├── handouts/         # basic.md, intermediate.md, advanced.md
└── README.md         # Topic-specific documentation
```

## Quick Start

```powershell
# COMPILE SLIDES (run from topic root, compile.py auto-finds .tex in slides/)
cd topics/clustering
python compile.py                       # Auto-detects latest .tex file in slides/

# GENERATE CHARTS
cd topics/clustering/charts
python create_kmeans_animation.py       # Creates .pdf and .png

# BATCH COMPILE
foreach ($t in @("clustering","nlp_sentiment","classification")) {
    cd "D:\Joerg\Research\slides\ML_Design_Thinking_16\topics\$t"
    python compile.py
}
```

## compile.py Behavior

- Auto-detects `main.tex` or latest timestamped file (YYYYMMDD_HHMM_*.tex)
- Runs pdflatex twice for references
- Archives PDF to `archive/builds/` with timestamp
- Moves aux files (.aux, .log, .nav, .snm, .toc) to `archive/aux/`

**Note:** compile.py exists in 10 of 14 topics (clustering, nlp_sentiment, classification, topic_modeling, generative_ai, responsible_ai, structured_output, validation_metrics, ab_testing, finance_applications). For topics without compile.py, compile manually:

```powershell
cd topics/{topic_name}/slides
pdflatex main.tex && pdflatex main.tex
```

## LaTeX/Beamer Requirements

### Document Setup
```latex
\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
\usepackage{graphicx,booktabs,adjustbox,multicol,amsmath}
```

### Required Colors
```latex
% Standard palette
\definecolor{mlblue}{RGB}{0,102,204}
\definecolor{mlpurple}{RGB}{51,51,178}
\definecolor{mllavender}{RGB}{173,173,224}
\definecolor{mllavender2}{RGB}{193,193,232}   % Lighter variant
\definecolor{mllavender3}{RGB}{204,204,235}   % Frame title bg
\definecolor{mllavender4}{RGB}{214,214,239}   % Block body bg
\definecolor{mlorange}{RGB}{255,127,14}
\definecolor{mlgreen}{RGB}{44,160,44}
\definecolor{mlred}{RGB}{214,39,40}
\definecolor{mlgray}{RGB}{127,127,127}

% Innovation stages (clustering topic)
\definecolor{challenge}{RGB}{148,103,189}
\definecolor{explore}{RGB}{52,152,219}
\definecolor{generate}{RGB}{46,204,113}
\definecolor{peak}{RGB}{241,196,15}
\definecolor{filter}{RGB}{230,126,34}
\definecolor{refine}{RGB}{231,76,60}
\definecolor{strategy}{RGB}{192,57,43}
```

### Template Commands
```latex
\bottomnote{text}              % Lavender annotation at slide bottom
\compactlist                   % Use inside itemize for tight spacing
\chartplaceholder[height]{desc}% Placeholder box for charts (default 5cm)
```

### Critical Rules

| Rule | Correct | Wrong |
|------|---------|-------|
| Font sizes | `\Large`, `\normalsize`, `\small`, `\footnotesize` | Other sizes |
| File naming | `20250928_1539_main.tex` | `main.tex`, `slides_v2.tex` |
| Characters | ASCII only | Unicode, emojis |
| Code blocks | `\begin{frame}[fragile,t]` | `\begin{frame}[t]` with lstlisting |
| Charts | Python-generated PDF | TikZ |
| Column widths | 0.48/0.48, 0.55/0.43 | Arbitrary splits |

### Bottom Notes

Every slide requires a `\bottomnote{}` with:
- Present tense, active voice
- Universal principles (no company names, dates, attributions)
- No instructional language ("You'll build...")

## Chart Generation Standards

```python
# Standard chart template
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

# Color palette
MLPURPLE = '#3333B2'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Save both formats
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
```

## Jupyter Notebooks

Located in `notebooks/` folder with Colab links on the website:

| Notebook | Description |
|----------|-------------|
| `01_random_forest_simple.ipynb` | RF classification (rating, price -> returned) |
| `02_neural_network_simple.ipynb` | NN classification with StandardScaler |
| `03_embeddings_similarity.ipynb` | HuggingFace embeddings + PCA + K-means |
| `04_kmeans_clustering.ipynb` | Customer segmentation with K-Means |
| `05_dbscan_clustering.ipynb` | Density-based clustering for outlier detection |
| `06_data_exploration.ipynb` | Descriptive analytics and visualization |
| `07_supervised_learning.ipynb` | Complete supervised learning workflow |
| `08_single_agent_api.ipynb` | Single LLM agent API call |
| `09_multi_agent_system.ipynb` | Writer/Critic/Editor agents |
| `10_agent_csv_analysis.ipynb` | Agent reads CSV, ranks 10 topics |

Dataset: `Dataset_Machine_Learning.csv` (rating, price, returned columns)

## Hugo Website

```powershell
# Build site
hugo --source "D:\Joerg\Research\slides\ML_Design_Thinking_16"

# Serve locally
hugo server --source "D:\Joerg\Research\slides\ML_Design_Thinking_16"

# Deploy (auto via GitHub Actions on push to main)
git add . && git commit -m "Update" && git push
```

Site: https://digital-ai-finance.github.io/ML_Design_Thinking/

Content files: `content/_index.md`, `content/resources.md`, `content/about.md`

Theme: `themes/course-theme/` (custom theme with sidebar, topic cards, download buttons)

### Website Content Structure

```
content/
├── _index.md           # Homepage content
├── resources.md        # Downloads, notebooks, reading list
├── about.md            # Course information
└── topics/
    ├── _index.md       # Topics listing page
    └── {topic}.md      # 14 topic pages with charts and descriptions

static/
├── downloads/          # PDF lecture files
└── images/
    └── topics/         # Chart PNGs for each topic (3 per topic)
```

## Git Remotes

| Remote | URL | Purpose |
|--------|-----|---------|
| origin | github.com/josterri/ML_Design_Thinking | Personal repo |
| digital-finance | github.com/Digital-AI-Finance/ML_Design_Thinking | Organization repo (public site) |

Push to both: `git push origin main && git push digital-finance main`

## Static Downloads

PDFs stored in `static/downloads/` and copied to `public/downloads/` during Hugo build:
- Individual topic PDFs (14 files, ~1 MB each)
- `all-lectures.zip` - ZIP of all 14 PDFs (8 MB)
- `all-lectures.pdf` - Merged single PDF (12 MB)

Regenerate with: `python tools/create_all_downloads.py`

## Tools

| Script | Purpose |
|--------|---------|
| `tools/create_all_downloads.py` | Generate all-lectures.zip and all-lectures.pdf |
| `tools/generate_web_charts.py` | Generate PNG charts for Hugo website |
| `tools/copy_pdfs_to_downloads.py` | Copy topic PDFs to static/downloads |
| `tools/check_links.py` | Validate links in content files |
| `tools/create_topic_readmes.py` | Generate README.md for each topic |
| `tools/update_topic_pages.py` | Update Hugo topic pages |
| `tools/check_beamer_compliance.py` | Check PDF slides against template rules (requires PyMuPDF) |

## Python Dependencies

```powershell
# Core (all topics)
pip install scikit-learn numpy pandas scipy matplotlib seaborn

# Topic-specific
pip install graphviz                    # supervised_learning (tree viz)
pip install textblob transformers nltk  # nlp_sentiment
pip install gensim pyLDAvis             # topic_modeling
pip install imblearn                    # classification
pip install statsmodels                 # ab_testing

# Notebooks (Generative AI)
pip install anthropic                   # Claude API
```

## Pedagogical Framework (Week 0 Series)

The ml_foundations, supervised_learning, unsupervised_learning, neural_networks, and generative_ai topics follow a 4-act narrative structure:

1. **Act 1: Challenge** - Build tension, show problem
2. **Act 2: First Solution** - Success THEN failure pattern
3. **Act 3: Breakthrough** - Human introspection, numerical walkthrough
4. **Act 4: Synthesis** - Applications, "When to Use/Not Use", common pitfalls

Required elements: success-before-failure, root cause diagnosis, zero-jargon explanation, experimental validation.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| PDF locked | Close PDF viewer, use SumatraPDF |
| Undefined color | Add color definitions to preamble |
| Unicode error | Replace with ASCII (`->` not arrows) |
| lstlisting error | Add `[fragile,t]` to frame |
| Missing chart | Run `python create_*.py` in charts/ folder |
| Aux file clutter | `compile.py` auto-archives them |

## Beamer Compliance Checking

```powershell
# Check all PDFs in static/downloads/
python tools/check_beamer_compliance.py

# Check specific topic
python tools/check_beamer_compliance.py --pdf clustering

# Save report to file
python tools/check_beamer_compliance.py --output docs/compliance_report.txt
```

Checks performed:
- Aspect ratio (16:9)
- Margins (5mm minimum)
- Bullet count (max 5 single-column, max 10 two-column)
- Bottomnote presence on content slides
- Chart sizing (50-70% of page width)
- Python code detection (should be avoided on slides)

## Innovation Diamond Materials

The capstone `innovation_diamond/` folder contains:
- Full presentation (51 slides)
- 10-slide summaries: Technical (with formulas) and Practical (plain English)
- Student worksheet (5 pages with LLM prompts for applying framework to own challenge)

Downloads in `static/downloads/`:
- `innovation-diamond.pdf` - Full presentation
- `innovation-diamond-summary.pdf` - Technical summary
- `innovation-diamond-practical.pdf` - Practical summary
- `innovation-diamond-worksheet.pdf` - Student worksheet

## Key Documentation

- `EDUCATIONAL_PRESENTATION_FRAMEWORK.md` - Complete pedagogical methodology
- `WEEK_0_SERIES_README.md` - Week 0a-0e narrative structure overview
- `docs/GAP_ANALYSIS_REPORT.md` - Course completion tracking
- `template_beamer_final.tex` - 28 professional slide layouts

## Topic-to-Week Mapping

| Topic | Original Source |
|-------|-----------------|
| ml_foundations | Week_00_Introduction, Week_00a |
| supervised_learning | Week_00b |
| unsupervised_learning | Week_00c |
| neural_networks | Week_00d |
| generative_ai | Week_00e, Week_06 |
| finance_applications | Week_00_Finance_Theory |
| clustering | Week_01, Week_02 |
| nlp_sentiment | Week_03 |
| classification | Week_04 |
| topic_modeling | Week_05 |
| responsible_ai | Week_07 |
| structured_output | Week_08 |
| validation_metrics | Week_09 |
| ab_testing | Week_10 |
