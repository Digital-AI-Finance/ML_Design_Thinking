# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

"Machine Learning for Smarter Innovation" - BSc-level course bridging ML/AI with design thinking. The course demonstrates how machine learning augments human-centered design processes through practical applications and real algorithm implementations.

**Core Concept**: The Innovation Diamond - visual metaphor showing expansion from 1 challenge through divergent thinking to 5000 possibilities, then convergence through ML filtering to 5 strategic solutions.

## High-Level Architecture

### Current Structure Evolution
- **Week 0 Series**: 5 standalone presentations following 4-act dramatic structure
- **Week 1**: Traditional single file + compile.py (47 slides)
- **Week 2-3**: Enhanced modular architecture (5-6 parts)
- **Week 4+**: Dual-version system (advanced + beginner-friendly)

### Standard Week Structure
```
Week_##/
├── YYYYMMDD_HHMM_main.tex              # Master controller with \input{} commands
├── YYYYMMDD_HHMM_main_beginner.tex     # Beginner version (Week 4+)
├── part1_*.tex                         # Modular content parts (varies by week)
├── part2_*.tex
├── part3_*.tex
├── part4_*.tex
├── part5_*.tex (optional)
├── appendix_*.tex (optional)
├── compile.py                          # Automated compilation with cleanup
├── charts/                             # PDF (300dpi) + PNG (150dpi) visualizations
├── scripts/                            # Chart generation Python scripts
├── handouts/                           # 3-level skill-targeted guides (.md)
└── archive/                            # Version control & cleanup
    ├── aux/                            # Auxiliary files auto-moved here
    ├── builds/                         # Timestamped PDF archives
    └── previous/                       # Version history
```

## Common Commands

### Primary Compilation Workflow
```bash
# Standard compilation with automatic cleanup (RECOMMENDED)
cd Week_##
python compile.py                       # Auto-detects latest main.tex
python compile.py YYYYMMDD_HHMM_main.tex   # Specific file

# Manual compilation if needed
pdflatex YYYYMMDD_HHMM_main.tex
pdflatex YYYYMMDD_HHMM_main.tex        # Run twice for references

# When PDF is locked (Windows)
pdflatex -jobname=filename_temp filename.tex
```

### Chart Generation
```bash
cd Week_##/scripts
python create_*.py                      # Generates both PDF and PNG
```

### Clean Auxiliary Files (Manual)
```powershell
New-Item -ItemType Directory -Force -Path archive/aux
Move-Item *.aux,*.log,*.nav,*.snm,*.toc,*.vrb,*.out -Destination archive/aux -Force
```

## Critical LaTeX/Beamer Requirements

### Document Setup
```latex
\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
```

### Mandatory Color Definitions
```latex
\definecolor{mlblue}{RGB}{31, 119, 180}
\definecolor{mlorange}{RGB}{255, 127, 14}
\definecolor{mlgreen}{RGB}{44, 160, 44}
\definecolor{mlred}{RGB}{214, 39, 40}
\definecolor{mlpurple}{RGB}{148, 103, 189}
```

### Critical Rules
- **Font sizes**: Only `\Large`, `\normalsize`, `\small` allowed
- **Column widths**: 0.48/0.48, 0.55/0.43, or 0.32/0.32/0.32
- **No Unicode**: ASCII only (no emojis, special characters)
- **File naming**: YYYYMMDD_HHMM_ prefix mandatory for all .tex files
- **Charts**: Python-generated only, NO TikZ
- **Week 0 Finance**: NO Python code on slides

## Pedagogical Framework (4-Act Structure)

All Week 0 series presentations follow this proven template:

1. **Act 1: Challenge** (5 slides) - Build tension from zero knowledge
2. **Act 2: First Solution** (5-6 slides) - Initial success → failure pattern
3. **Act 3: Breakthrough** (9-10 slides) - Key insight → validation
4. **Act 4: Synthesis** (4 slides) - Modern applications + lessons

### Critical Pedagogical Beats
- Success slide BEFORE failure (builds hope)
- Quantified failure with data table
- "How do YOU...?" human introspection
- Zero-jargon explanation before technical terms
- Full numerical walkthrough with actual numbers
- Before/after experimental validation

## Week-Specific Notes

### Week 0 Series (5 Presentations)
- **0a**: ML Foundations (26 slides) - Learning journey metaphor
- **0b**: Supervised Learning (25 slides) - Prediction challenge
- **0c**: Unsupervised Learning (26 slides) - Discovery without labels
- **0d**: Neural Networks (25 slides) - Depth challenge
- **0e**: Generative AI (25 slides) - Creation challenge

### Weeks with Special Features
- **Week 1**: Innovation Diamond visualization series (18 charts)
- **Week 3**: NLP with 75 charts, Twitter sentiment workshop
- **Week 4**: Dual versions (advanced + beginner), first to use v2/v3 structure
- **Week 7**: Custom Nature Professional color theme
- **Week 10**: Complete with verified industry statistics

## compile.py Features

The `compile.py` script (present in all weeks) provides:
- Auto-detection of latest main.tex file
- Dual pdflatex pass for complete references
- Automatic archiving to `archive/builds/` with timestamps
- Auxiliary file cleanup to `archive/aux/`
- 60-second timeout per compilation pass
- Clear progress indicators and file size reporting

## Handout System

All weeks include 3 skill-level handouts:

1. **Basic** (~150-200 lines): No math/code, checklists, plain English
2. **Intermediate** (~300-400 lines): Python implementation guides, case studies
3. **Advanced** (~400-500 lines): Mathematical proofs, production considerations

## Development Workflow

1. **New content**: Copy Week_04 structure as template (most complete)
2. **File creation**: Use timestamp prefix YYYYMMDD_HHMM_
3. **Charts**: Generate via Python scripts → save as PDF + PNG
4. **Compilation**: Always use `compile.py` for consistency
5. **Archiving**: Previous versions auto-moved to `archive/previous/`
6. **Cleanup**: Auxiliary files auto-moved to `archive/aux/`

## Python Dependencies

```python
# Core
scikit-learn, numpy, pandas, scipy, matplotlib, seaborn

# NLP (Week 3+)
textblob, transformers, nltk, wordcloud

# Topic Modeling (Week 5)
gensim, pyLDAvis

# Imbalanced Data (Week 4)
imblearn

# Statistics (Week 10)
statsmodels
```

## Quality Checklist Before Compilation

1. Font sizes limited to 3 allowed sizes
2. No Unicode characters (ASCII only)
3. Verify `\input{}` paths in main.tex
4. Color definitions included in preamble
5. Charts use real sklearn data (no synthetic)
6. Previous folder exists for archiving

## Common Issues & Solutions

### LaTeX Errors
- **Undefined color**: Add color definitions to preamble
- **PDF locked**: Use `-jobname` parameter or close viewer
- **Unicode error**: Replace with ASCII equivalents

### Python Script Issues
- **Import error**: Check dependencies list
- **Chart not rendering**: Verify matplotlib backend
- **Memory error**: Reduce algorithm sample size

## Statistics Verification Protocol

When including industry statistics:
1. Verify all claims via web search
2. Cite sources when available
3. Use conservative estimates when sources conflict
4. Examples of verified stats kept in Week 10 documentation