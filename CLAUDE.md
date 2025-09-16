# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains "Machine Learning for Smarter Innovation" - a comprehensive BSc-level course demonstrating how ML/AI augments design thinking processes. The course bridges technical ML knowledge with human-centered design methodology across 10 weeks (310 slides total).

## Current Directory Structure
```
ML_Design_Thinking_16/
├── Week_01/                         # COMPLETED - Full implementation with latest improvements
│   ├── 20250913_2133_week01_modular.tex   # Modular main file with \input commands
│   ├── part1_foundation.tex        # Part 1: Foundation & Problem Statement
│   ├── part2_technical.tex         # Part 2: Technical ML Content  
│   ├── part3_design.tex            # Part 3: Design Integration
│   ├── part4_summary.tex           # Part 4: Summary & Practice
│   ├── appendix_technical.tex      # Technical appendix
│   ├── week01_modular_updated.pdf  # Latest compiled version (87 pages)
│   ├── week01_discovery_worksheet.tex      # Expanded 9-page worksheet
│   ├── charts/                      # 50+ generated visualizations
│   ├── scripts/                     # Chart generation scripts  
│   ├── archive/                     # Old versions (20+ files)
│   └── temp/                        # Compilation auxiliary files
├── Week_02-10/                      # PLANNED - Structure ready, content pending
├── ML_Design_Course/
│   ├── 20250912_0848_course_overview_10week.tex  # Overview presentation
│   └── full_course_content_toc.md  # Complete 310-slide outline
├── check_font_sizes.py             # Font consistency validator
├── fix_overfull_charts.py          # Auto-fix overfull warnings
└── compile_slides.py                # Automated compilation with cleanup
```

## Course Architecture

### Core Structure
- **10 Weeks**: Each aligned with design thinking stages (Empathize → Define → Ideate → Prototype → Test)
- **31 Slides per Week**: Consistent 4-part structure plus appendix
- **310 Total Slides**: Complete journey from ML basics to advanced applications

### Weekly Slide Structure (Mandatory)
```
1. Opening Power Chart      # Compelling visualization hook
2-4. Part 1: Foundation     # Problem statement & context (with section divider)
5-14. Part 2: Technical     # ML algorithms & implementation (with section divider)
15-22. Part 3: Design       # Human-centered applications (with section divider)
23-27. Part 4: Summary      # Case study & practice (with section divider)
28-31. Appendix            # Mathematical details (optional)
```

### Critical Improvements (Week 1 Pattern)
1. **Section Dividers**: Clear visual breaks between 4 parts
2. **Transition Slides**: Bridges between major topics
3. **Problem-First**: Problem statement before each methodology
4. **Narrative Flow**: Smooth progression from problem → solution → application

## Building and Compilation

### LaTeX/PDF Compilation (Windows)
```bash
# Compile with automatic cleanup (recommended)
python compile_slides.py Week_01/20250913_2133_week01_modular.tex

# Manual compilation
pdflatex filename.tex
pdflatex filename.tex  # Run twice for TOC/references

# If PDF is locked by viewer
pdflatex -jobname=filename_v2 filename.tex
# Or use updated name
pdflatex -jobname=week01_modular_updated 20250913_2133_week01_modular.tex

# Clean auxiliary files after compilation (PowerShell)
powershell -Command "Move-Item '*.aux', '*.log', '*.nav', '*.snm', '*.toc', '*.vrb', '*.out' -Destination 'temp\' -Force -ErrorAction SilentlyContinue"
```

### Quality Checks
```bash
# Check font size consistency (max 3 sizes allowed)
python check_font_sizes.py filename.tex

# Fix overfull boxes automatically
python fix_overfull_charts.py filename.tex

# Verify slide count
grep -c "begin{frame}" filename.tex  # Should be ~48-49 with improvements (31 base + enhancements)
```

## Visualization Development

### Chart Requirements
- **Real ML algorithms** (no fake data)
- **Sample size**: 1000-10000 points
- **Consistent styling** across all charts
- **Save both formats**: PDF (print) and PNG (preview)

### Standard Chart Template
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # Use real ML

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)
fig, ax = plt.subplots(figsize=(14, 10))  # Standard size

# Color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e', 
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd'
}

# Save outputs
plt.savefig('chart_name.pdf', dpi=300, bbox_inches='tight')
plt.savefig('chart_name.png', dpi=150, bbox_inches='tight')
```

## LaTeX/Beamer Standards

### Document Configuration
```latex
\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
```

### Critical LaTeX Rules
- **Font sizes**: Exactly 3 (`\Large`, `\normalsize`, `\small`)
- **Columns**: `0.48/0.48` (equal), `0.55/0.43` (unequal), or `0.32/0.32/0.32` (three columns)
- **Chart widths**: `0.95` (charts-only), `0.85` (full), `0.75` (medium), `0.65` (sidebar)
- **No Unicode**: ASCII only (no emojis, special symbols)
- **Quotes**: Use `` `` not " "
- **Lists**: `\begin{itemize}` not HTML tags
- **Plain frames**: Use `\begin{frame}[plain]` for charts-only slides

## Current Week Development Status

| Week | Topic | Status | Key Files |
|------|-------|--------|-----------|
| 1 | Clustering & Empathy | ✅ Complete | `Week_01/20250913_2133_week01_modular.tex` |
| 2 | Advanced Clustering | 🔄 Planned | Focus: Dynamic segmentation |
| 3 | NLP & Sentiment | 🔄 Planned | Focus: Context understanding |
| 4 | Classification | 🔄 Planned | Focus: Problem patterns |
| 5 | Topic Modeling | 🔄 Planned | Focus: Hidden themes |
| 6 | Generative AI | 🔄 Planned | Focus: Creative exploration |
| 7 | SHAP Analysis | 🔄 Planned | Focus: Feature importance |
| 8 | Structured Output | 🔄 Planned | Focus: Consistent generation |
| 9 | Multi-Metric Validation | 🔄 Planned | Focus: Beyond accuracy |
| 10 | A/B Testing | 🔄 Planned | Focus: Statistical validation |

## File Naming Convention
```
YYYYMMDD_HHMM_description.tex  # Timestamp versioning
Week_##/scripts/create_*.py    # Visualization generators
Week_##/charts/*.pdf           # Generated charts
Week_##/previous/              # Version history
```

## Automation Tools

### `compile_slides.py`
Compiles LaTeX with automatic cleanup and PDF opening

### `fix_overfull_charts.py`
Auto-adjusts chart sizes to fix overfull warnings
- Strategy: >100pt: 20%, 50-100pt: 12%, 20-50pt: 7%, <20pt: 5%

### `check_font_sizes.py`
Verifies font consistency with auto-fix option

## Development Workflow

1. **Create timestamp**: `YYYYMMDD_HHMM` format
2. **Copy template**: From successful Week 1 modular structure
3. **Add improvements**: Section dividers, transitions, problem slides
4. **Generate charts**: Use real ML algorithms (50+ charts in Week 1)
5. **Check quality**: Font sizes, overfull boxes
6. **Compile PDF**: With cleanup (use `-jobname` if locked)
7. **Version control**: Move old to `archive/` folder

## Content Guidelines

### BSc-Level Requirements
- Visual explanations over mathematics
- Complex math → appendix only
- Industry examples must be verified
- Show concepts, not code on slides
- Focus on practical application

### Slide Content Rules
- One main concept per slide
- Maximum 5 bullet points
- Charts speak for themselves
- Problem before solution
- Real data, real algorithms

## Critical Success Patterns (from Week 1)

1. **Opening Hook**: Power visualization that shows end result
2. **Transition**: "But first, let's understand the problem..."
3. **Problem Statement**: Clear pain points before each method
4. **Technical Depth**: Progressive complexity with visual aids
5. **Design Bridge**: "What this means for users..."
6. **Real Application**: Industry case study (e.g., Spotify)
7. **Call to Action**: Practice exercise with clear tasks

## Common Issues and Solutions

| Issue | Solution | Prevention |
|-------|----------|------------|
| Overfull hbox | Run `fix_overfull_charts.py` | Use standard chart widths |
| Font inconsistency | Run `check_font_sizes.py` | Stick to 3 sizes only |
| PDF locked | Use `-jobname` parameter | Close PDF viewer first |
| Missing transitions | Add bridge slides | Use Week 1 as template |
| No problem context | Add "Challenge" slides | Problem before solution |

## Version Control Strategy

1. **Timestamp all files**: YYYYMMDD_HHMM format
2. **Keep previous versions**: Move to `previous/` folder
3. **Track improvements**: Use descriptive suffixes (_improved, _final)
4. **Document changes**: Update changelog.md after major edits

## Discovery Materials

### Current Working Files
- **week01_discovery_worksheet.tex** - Expanded 9-page academic worksheet
  - Exercise 1-2: Pattern recognition and innovation sorting
  - Exercise 3: ML Pipeline (Data → Preprocess → Model → Evaluate → Deploy)
  - Exercise 4: DT Pipeline (Empathize → Define → Ideate → Prototype → Test)
  - Exercise 5: Pipeline convergence analysis
  - Reflection: Methodological synthesis with 6 academic questions

### Jupyter Notebooks (NEW)
- **Week01_Part1_Setup_Foundation.ipynb** - Complete setup and foundation
  - Section 0: All 21 helper functions (visualization, data generation, clustering, innovation analysis)
  - Section 1: Pre-Discovery exploration (3 subsections with exercises)
  - Section 2: Foundation & Context (5 subsections including dual pipeline)
- **Week01_Part2_Technical_Design.ipynb** - Technical algorithms and design integration
  - Section 3: All clustering algorithms (K-Means, DBSCAN, Hierarchical, GMM)
  - Section 4: Design applications (archetypes, taxonomy, opportunity analysis, ecosystem)
- **Week01_Part3_Practice_Advanced.ipynb** - Practice and advanced topics
  - Section 5: Spotify case study and innovation challenge
  - Section 6: Advanced visualizations (PCA/t-SNE, 3D, Dashboard)
  - Section 7: Hands-on exercises with solutions
  - Section 8: Quick reference guide

## Recent Slide Modifications (Latest Session)

### Chart Size Adjustments
- Innovation Discovery: 0.85 → 0.75
- Your Innovation Journey: 0.85 → 0.95  
- Current Reality: 0.85 → 0.75
- From Data Points to Innovation: 0.85 → 0.75

### New Charts-Only Slides Added
- Algorithm Visual Comparison (plain frame)
- Gaussian Mixture Models (plain frame)
- Common Mistakes (plain frame)
- Data Preprocessing Pipeline duplicate with examples

### Content Reorganization
- K-Means slides: Charts moved to top, text to bottom with smaller font
- Hierarchical Clustering: Chart enlarged, text at bottom
- AI-Generated Archetypes: Split into two separate slides

## Development Best Practices

### From Week 1 Improvements
- **Always add section dividers** between 4 main parts
- **Include transition slides** to bridge major topic changes
- **Start with problems** before presenting solutions
- **Use real ML algorithms** (KMeans, DBSCAN, etc.) not fake data
- **Include "The [X] Challenge" slides** to frame each methodology
- **Build narrative flow** from opening hook to final practice

### Quality Assurance Checklist
- [ ] Font sizes: Exactly 3 (\Large, \normalsize, \small)
- [ ] Column widths: Consistent (0.48/0.48 or 0.55/0.43)
- [ ] Chart widths: Standardized (0.95, 0.85, 0.75, or 0.65)
- [ ] Section dividers: All 4 parts clearly marked
- [ ] Transition slides: Smooth flow between topics
- [ ] Problem statements: Before each methodology
- [ ] Real data: Actual ML algorithms used
- [ ] Slide count: ~50-55 with recent enhancements (includes charts-only slides)