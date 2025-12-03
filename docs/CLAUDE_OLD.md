# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

"Machine Learning for Smarter Innovation" - BSc-level course bridging ML/AI with design thinking across 10 weeks (310 slides). The course demonstrates how machine learning augments human-centered design processes through practical applications and real algorithm implementations.

**Core Innovation Concept**: The Innovation Diamond - a visual metaphor showing how ideas expand from 1 challenge through divergent thinking to 5000 possibilities, then converge through ML filtering to 5 strategic solutions.

## High-Level Architecture

### Modular LaTeX Structure
Each week uses a **4-file modular architecture**:
```
Week_##/
├── YYYYMMDD_HHMM_week##_modular.tex    # Main file with \input commands
├── part1_foundation.tex                 # Slides 1-10: Problem & context
├── part2_technical.tex                  # Slides 11-20: ML algorithms
├── part3_design.tex                     # Slides 21-27: Human applications
└── part4_summary.tex                    # Slides 28-31: Practice & appendix
```

The main `.tex` file coordinates all sections through `\input{}` commands, enabling parallel development and consistent structure across all 10 weeks.

### Jupyter Notebook Pipeline
**Function-First Architecture**: All notebooks follow this structure:
```
Cells 0-1:   Headers and imports
Cells 2-4:   Complete Function Library (all functions defined here)
Cells 5+:    Function calls and demonstrations only
```

This architecture enables:
- Modular testing of individual functions
- Cross-notebook function reuse
- Clear separation of implementation and demonstration

### Visualization Generation System
Charts follow a **two-format pipeline** with consistent styling:
```python
# Standard template for all 50+ charts
plt.style.use('seaborn-v0_8-whitegrid')
plt.savefig('charts/chart_name.pdf', dpi=300, bbox_inches='tight')  # Print quality
plt.savefig('charts/chart_name.png', dpi=150, bbox_inches='tight')  # Preview
```

**Unified Color Palette** (used across all visualizations):
```python
colors = {
    'challenge': '#9467bd',  # Purple - starting point
    'explore': '#3498db',    # Blue - exploration phase
    'generate': '#2ecc71',   # Green - generation phase
    'peak': '#f1c40f',       # Yellow - maximum ideas
    'filter': '#e67e22',     # Orange - filtering phase
    'refine': '#e74c3c',     # Red - refinement phase
    'strategy': '#c0392b'    # Dark Red - final strategy
}
```

Key requirements:
- Real sklearn algorithms only (no synthetic data)
- Consistent color progression through innovation stages
- Self-explanatory charts without accompanying text
- Standard figure size: `figsize=(14, 10)` for consistency

## Common Commands

### LaTeX Compilation
```bash
# Standard compilation (Windows)
pdflatex filename.tex
pdflatex filename.tex  # Run twice for TOC

# When PDF is locked
pdflatex -jobname=filename_v2 filename.tex

# Clean auxiliary files (PowerShell)
powershell -Command "New-Item -ItemType Directory -Force -Path temp; Move-Item *.aux,*.log,*.nav,*.snm,*.toc,*.vrb,*.out -Destination temp -Force -ErrorAction SilentlyContinue"
```

### Jupyter Notebook Operations
```bash
# Execute all cells in a notebook
jupyter nbconvert --to notebook --execute Week_01/Week01_Part1_Setup_Foundation.ipynb --output executed.ipynb

# Batch execute all three parts
for %i in (1 2 3) do jupyter nbconvert --to notebook --execute Week_01/Week01_Part%i_*.ipynb --output executed_part%i.ipynb
```

### Chart Generation
```bash
# Generate Week 1 visualizations
cd Week_01/charts
python create_improved_charts.py
python create_part3_design_charts.py

# Batch generate all charts
for %f in (*.py) do python %f
```

## LaTeX/Beamer Configuration

### Required Setup
```latex
\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]

% Color definitions (mandatory)
\definecolor{mlblue}{RGB}{31, 119, 180}
\definecolor{mlorange}{RGB}{255, 127, 14}
\definecolor{mlgreen}{RGB}{44, 160, 44}
\definecolor{mlred}{RGB}{214, 39, 40}
\definecolor{mlpurple}{RGB}{148, 103, 189}
```

### Strict Formatting Rules
- **Font sizes**: Only 3 allowed (`\Large`, `\normalsize`, `\small`)
- **Column widths**: `0.48/0.48` (equal), `0.55/0.43` (unequal), `0.32/0.32/0.32` (three)
- **Chart widths**: `0.95` (chart-only), `0.85` (full), `0.75` (medium), `0.65` (sidebar)
- **Plain frames**: Use `\begin{frame}[plain]` for chart-only slides
- **No Unicode**: ASCII only, use `` `` for quotes, not " "

## Weekly Structure (31 Slides)

```
Slide 1:      Opening Power Chart (compelling visualization)
Slides 2-10:  Part 1 Foundation (problem statement & context)
              - Include section divider
              - Problem before solution
Slides 11-20: Part 2 Technical (ML algorithms & implementation)
              - Include section divider
              - Real sklearn implementations
Slides 21-27: Part 3 Design (human-centered applications)
              - Include section divider
              - Bridge to user impact
Slides 28-31: Part 4 Summary/Appendix (practice & math details)
```

## Critical Implementation Patterns

### From Week 1 Success
1. **Section dividers** between all 4 parts
2. **Transition slides** bridging major topics
3. **Problem-first approach** before each methodology
4. **"The [X] Challenge" slides** framing each concept
5. **Chart-only slides** using `[plain]` frame option
6. **Real algorithms only** (KMeans, DBSCAN, GMM from sklearn)

### Common Chart Generation Pattern
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# Always use real ML
np.random.seed(42)  # Consistent reproducibility
X, y = make_blobs(n_samples=1000, centers=3)
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 10))  # Standard size

# Innovation stage colors (for diamond visualizations)
colors = {
    'challenge': '#9467bd', 'explore': '#3498db',
    'generate': '#2ecc71', 'peak': '#f1c40f',
    'filter': '#e67e22', 'refine': '#e74c3c',
    'strategy': '#c0392b'
}

# Add visual enhancements
text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
```

## File Naming Convention

```
YYYYMMDD_HHMM_description.tex    # LaTeX files with timestamp
Week_##/charts/chart_name.pdf    # Generated visualizations
Week_##/notebooks/Week##_Part#_*.ipynb    # Jupyter notebooks
Week_##/previous/                # Version archive
```

## Quality Checks

### Before Compilation
1. Font sizes: Exactly 3 (`\Large`, `\normalsize`, `\small`)
2. Column widths: Consistent (0.48/0.48 or 0.55/0.43)
3. Chart widths: Standard sizes (0.95, 0.85, 0.75, 0.65)
4. Section dividers: Present between all 4 parts
5. Real ML algorithms: No synthetic/fake data

### Common Issues
| Issue | Solution |
|-------|----------|
| PDF locked | Use `-jobname=filename_v2` |
| Overfull hbox | Adjust chart width down by 0.05 |
| Font inconsistency | Limit to 3 sizes only |
| Missing sklearn | All charts must use real algorithms |

## Current Status

| Week | Topic | Primary Algorithm | Status |
|------|-------|------------------|--------|
| 1 | Clustering & Empathy | KMeans, DBSCAN | ✅ Complete (70 pages) |
| 2 | Advanced Clustering | Hierarchical, GMM | 🔄 Structure ready |
| 3-10 | Various | See full outline | 📋 Planned |

### Week 1 Key Visualizations
**Innovation Diamond Series** (5 complementary views):
- `innovation_diamond.pdf` - Complete diamond showing expansion & convergence
- `innovation_expansion.pdf` - Divergent phase detail (1→5000 ideas)
- `innovation_convergence_updated.pdf` - Convergent phase detail (5000→5 strategies)
- `innovation_flow_3d.pdf` - 3D perspective of the complete process
- `innovation_stages_detail.pdf` - Linear stage-by-stage breakdown

**Clustering Demonstrations** (20+ charts):
- Real KMeans, DBSCAN, Hierarchical, GMM implementations
- Interactive parameter exploration visualizations
- Failure case demonstrations
- 3-panel convergence flow animations

Week 1 establishes the visual language and technical foundation for the entire course.