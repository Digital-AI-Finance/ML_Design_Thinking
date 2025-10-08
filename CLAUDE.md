# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

"Machine Learning for Smarter Innovation" - BSc-level course bridging ML/AI with design thinking. The course demonstrates how machine learning augments human-centered design processes through practical applications and real algorithm implementations.

**Core Concept**: The Innovation Diamond - visual metaphor showing expansion from 1 challenge through divergent thinking to 5000 possibilities, then convergence through ML filtering to 5 strategic solutions.

## High-Level Architecture

### Course Structure Evolution

**Week 0 Options** (Choose based on course needs):
- **Week_00_Introduction_ML_AI**: Comprehensive ML survey (45 slides, 90-min session) - broad overview
  - **Status (Oct 2025):** ✅ Complete with 41 bottom notes, mathematically verified, production-ready
- **Week 0a-0e Series**: 5 standalone narrative presentations (127 slides total, 5 × 90 min) - deep pedagogical approach
  - Week_00a_ML_Foundations (32 slides) - "Learning Journey" metaphor - ✅ Math errors fixed
  - Week_00b_Supervised_Learning (27 slides) - "Prediction Challenge" - ✅ Zero overflows
  - Week_00c_Unsupervised_Learning (26 slides) - "Discovery Challenge" - ✅ Unicode compliant
  - Week_00d_Neural_Networks (27 slides) - "Depth Challenge" - ✅ Unicode compliant
  - Week_00e_Generative_AI (29 slides) - "Creation Challenge" - ✅ Major overflow reduction
- **Week_00_Finance_Theory**: Advanced finance theory (45+ slides, 10 parts, quant track)
  - **Status (Oct 2025):** ✅ Expanded from 124 to 1181 lines (9.5x growth), 15 new comprehensive slides added
  - **Note:** Renamed from Week_00b_ML_Finance_Applications to avoid confusion with Week_00b_Supervised_Learning

**Main Course** (Weeks 1-10):
- **Week 1-10**: Complete weekly modules (47-59 slides each) covering full innovation cycle
- **Weeks 11-12**: Reserved for future expansion

**Build System**: compile.py available in Weeks 1-10 and Week_00_Finance_Theory

### Standard Week Structure

**Traditional 5-Part Structure** (Weeks 1-7, 9-10):
```
Week_##/
├── YYYYMMDD_HHMM_main.tex              # Master controller with \input{} commands
├── YYYYMMDD_HHMM_main_beginner.tex     # Beginner version (Week 4+)
├── part1_foundation.tex                # Part 1: Foundation
├── part2_technical.tex                 # Part 2: Technical/Algorithms
├── part3_implementation.tex            # Part 3: Implementation
├── part4_design.tex                    # Part 4: Design/UX
├── part5_practice.tex                  # Part 5: Workshop/Practice
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

**4-Act Dramatic Structure** (Week 0a-0e):
```
Week_##/
├── YYYYMMDD_HHMM_main.tex              # Master controller with 4 act inputs
├── act1_*_challenge.tex                # Act 1: The Challenge (5 slides)
├── act2_*_solution.tex                 # Act 2: First Solution (5-6 slides)
├── act3_*_breakthrough.tex             # Act 3: The Breakthrough (9-10 slides)
├── act4_*_synthesis.tex                # Act 4: Synthesis (4 slides)
├── charts/                             # Visualizations
├── scripts/                            # Chart generation
└── archive/                            # Version control
```

**Note**: Week 0 series presentations compile directly with pdflatex (simpler structure). Week 8 uses the 5-part structure above despite having 4-act narrative content.

## Quick Start Cheat Sheet

```powershell
# COMPILE SLIDES
cd Week_01                  # Navigate to any week
python compile.py           # Auto-detects and compiles latest .tex
start *.pdf                 # View generated PDF (Windows)

# GENERATE CHARTS
cd scripts
python create_*.py          # Generates both PDF (300dpi) and PNG (150dpi)

# CHECK STATUS
git status                  # See modified files
```

## Quick Troubleshooting Guide

**PDF won't compile?**
→ Close PDF viewer → Try again
→ Still fails? Run: `git status` to check for Unicode characters
→ Check `archive/aux/*.log` for detailed errors

**Chart missing in slides?**
→ Run: `cd Week_##/scripts && python create_*.py`
→ Verify: `ls Week_##/charts/*.pdf`

**Git shows hundreds of .aux files?**
→ Normal after fresh clone
→ Run: `python compile.py` in each week (auto-archives them)
→ Or manually: `cd Week_## && mkdir -p archive/aux && mv *.aux *.log *.nav *.out *.snm *.toc archive/aux/`

**Week_00b confusion?**
→ `Week_00b_Supervised_Learning` = Part of 0a-0e narrative series
→ `Week_00_Finance_Theory` = Finance track (formerly Week_00b_ML_Finance_Applications)

## Common Commands

### Primary Compilation Workflow
```powershell
# Standard compilation with automatic cleanup (RECOMMENDED)
cd Week_##
python compile.py                          # Auto-detects latest main.tex
python compile.py YYYYMMDD_HHMM_main.tex   # Compile specific file

# What compile.py does:
# 1. Runs pdflatex twice (for references/TOC)
# 2. Archives PDF to archive/builds/ with timestamp
# 3. Moves aux files to archive/aux/
# 4. Reports file size and success/errors

# Manual compilation if needed
pdflatex YYYYMMDD_HHMM_main.tex
pdflatex YYYYMMDD_HHMM_main.tex           # Run twice for references

# When PDF is locked by viewer (Windows-specific)
pdflatex -jobname=filename_temp filename.tex
# Alternative: Close PDF viewer before recompiling
# Recommended: Use SumatraPDF (lightweight, doesn't lock files)
```

### Chart Generation
```powershell
cd Week_##/scripts
python create_*.py                         # Generates both PDF and PNG

# Chart output format:
# - PDF: 300 dpi for print quality
# - PNG: 150 dpi for web/presentations
# Both saved to ../charts/ directory
```

### View Generated PDF
```powershell
# Windows
start YYYYMMDD_HHMM_main.pdf              # Default viewer
start .                                    # Open folder in Explorer

# Alternative viewers
# SumatraPDF: Lightweight, doesn't lock files
# Adobe Acrobat: Full features but locks files
```

### Clean Auxiliary Files (Manual)
```powershell
# Create archive structure
New-Item -ItemType Directory -Force -Path archive\aux,archive\builds,archive\previous

# Move auxiliary files
Move-Item *.aux,*.log,*.nav,*.snm,*.toc,*.vrb,*.out -Destination archive\aux -Force

# Note: compile.py does this automatically
```

## Critical LaTeX/Beamer Requirements

### Document Setup
```latex
\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}

% Required packages (all presentations)
\usepackage{graphicx}      % Images
\usepackage{booktabs}      % Professional tables
\usepackage{adjustbox}     % Box adjustments
\usepackage{multicol}      % Multi-column layouts
\usepackage{amsmath}       % Mathematical typesetting
```

### Mandatory Color Definitions

**Standard Colors** (template_beamer_final.tex):
```latex
\definecolor{mlblue}{RGB}{0,102,204}
\definecolor{mlpurple}{RGB}{51,51,178}
\definecolor{mllavender}{RGB}{173,173,224}
\definecolor{mllavender2}{RGB}{193,193,232}
\definecolor{mllavender3}{RGB}{204,204,235}
\definecolor{mllavender4}{RGB}{214,214,239}
\definecolor{mlorange}{RGB}{255, 127, 14}
\definecolor{mlgreen}{RGB}{44, 160, 44}
\definecolor{mlred}{RGB}{214, 39, 40}
\definecolor{mlgray}{RGB}{127, 127, 127}
```

**Innovation Stage Colors** (Week 1+):
```latex
\definecolor{challenge}{RGB}{148, 103, 189}
\definecolor{explore}{RGB}{52, 152, 219}
\definecolor{generate}{RGB}{46, 204, 113}
\definecolor{peak}{RGB}{241, 196, 15}
\definecolor{filter}{RGB}{230, 126, 34}
\definecolor{refine}{RGB}{231, 76, 60}
\definecolor{strategy}{RGB}{192, 57, 43}
```

### Custom Template Commands
```latex
% Available custom commands from template_beamer_final.tex
\bottomnote{text}              % Lavender annotation at slide bottom
\twocolslide{left}{right}      % Quick two-column layout (0.48/0.48)
\formula{equation}             % Highlighted equation box
\highlight{text}               % Blue bold for emphasis
\keypoint{text}                % Green bold for key insights
```

### Critical Rules
- **Font sizes**: Only `\Large`, `\normalsize`, `\small`, `\footnotesize` allowed
- **Column widths**: 0.48/0.48, 0.55/0.43, or 0.32/0.32/0.32
- **No Unicode**: ASCII only (no emojis, special characters)
  - Replace: ✓ → [OK], ✗ → [X], → → ->, × → x
- **File naming**: YYYYMMDD_HHMM_ prefix mandatory for all .tex files
  - ✅ CORRECT: `20250928_1539_main.tex`, `20250928_1645_main.pdf`
  - ❌ WRONG: `main.tex` (no timestamp), `slides_v2.tex` (version in name)
  - Exception: Beginner versions use `_beginner` suffix: `20250928_1539_main_beginner.tex`
- **Charts**: Python-generated only, NO TikZ
- **Week 0b Finance**: NO Python code on slides (pure theory)
- **Code blocks in Beamer**: Frames with `\begin{lstlisting}` MUST use `[fragile,t]` option
  - ✅ CORRECT: `\begin{frame}[fragile,t]{Title}`
  - ❌ WRONG: `\begin{frame}[t]{Title}` when lstlisting is inside

### Bottom Note Requirements (EDUCATIONAL_PRESENTATION_FRAMEWORK.md)

**Every slide must have a bottom note** providing pedagogical context. Bottom notes must be:

✅ **Compliant Format**:
- Present tense, active voice
- Universal principles (not case-specific)
- Focus on "why this matters" pedagogically
- No company names, quotes, attributions, dates, years
- No template meta-commentary (e.g., "Total: 56 slides")
- No instructional language (e.g., "You'll build", "Work individually")

❌ **Non-Compliant Examples**:
- "Netflix's algorithm understands taste better than users" (company name)
- "Chouldechova (2017) complete proof" (attribution + year)
- "By the end: You'll build a system with 89% accuracy" (instructional + statistic)
- "Total: 56 slides (51 content + 5 appendix)" (meta-commentary)

✅ **Corrected Examples**:
- "Granular categorization reveals preferences - hierarchical topic decomposition exposes latent taste structures"
- "Bayesian derivations expose structural constraints - calibration combined with unequal base rates precludes simultaneous parity"
- "Classification systems enable predictive decision-making - supervised learning transforms historical patterns into forecasts"
- "Measurement transforms ethical concerns into technical problems - quantification enables optimization"

**Note**: All 10 weeks (1-10) and 7 Week 0 variants systematically revised for compliance (Oct 2025)

**Recent Week 0 Improvements (Oct 6, 2025):**
- Fixed polynomial calculation error in Week_00a act3
- Added 41 bottom notes to Week_00_Introduction_ML_AI (all 5 parts)
- Expanded Week_00b_Finance from 124 to 1181 lines (15 new slides: VaR, CVaR, stress testing, HRP, execution algorithms, credit modeling, fraud detection, SR 11-7, MiFID II compliance)
- Eliminated 28 Unicode violations across all variants
- Fixed 12 broken HTML-style LaTeX tags
- Reduced overflows in Week_00b (100%) and Week_00e (partial)

## Pedagogical Framework (4-Act Structure)

Week 0 series (Week_00a-0e) follow the dramatic structure from EDUCATIONAL_PRESENTATION_FRAMEWORK.md:

1. **Act 1: Challenge** (4-5 slides) - Build tension from zero knowledge
   - Real scenario with quantified costs (or qualitative if no verified data)
   - Data tables showing systematic patterns
   - Forward question creating curiosity

2. **Act 2: First Solution** (4-6 slides) - Initial success → failure pattern
   - **CRITICAL**: Success slide BEFORE failure (builds hope)
   - Systematic degradation with data tables
   - Root cause diagnosis

3. **Act 3: Breakthrough** (6-10 slides) - Key insight → validation
   - Human introspection: "How do YOU...?"
   - Hypothesis before mechanism
   - Zero-jargon explanation before technical terms
   - Full numerical walkthrough with actual numbers (or real code examples)
   - Before/after experimental validation (qualitative if no verified metrics)

4. **Act 4: Synthesis** (4-6 slides) - Modern applications + lessons + meta-knowledge
   - Real companies using the approach (NO fake metrics)
   - Universal lessons (transferable principles)
   - **Meta-knowledge slides** (pedagogical_framework_Template.md requirement):
     - "When to Use / When NOT to Use" (judgment criteria)
     - "Common Pitfalls / What Can Go Wrong"
   - Workshop preview

### 8 Critical Pedagogical Beats (All Must Be Present)
1. **Success Before Failure**: Show hope THEN disappointment (Act 2)
2. **Failure Pattern**: Data table with systematic degradation
3. **Root Cause Diagnosis**: Trace specific failure to missing component
4. **Human Introspection**: "How do YOU...?" before technical solution
5. **Hypothesis Before Mechanism**: Natural prediction from observation
6. **Zero-Jargon Explanation**: Plain English before technical terms
7. **Numerical Walkthrough**: Complete trace with actual numbers (or real code)
8. **Experimental Validation**: Controlled before/after comparison (qualitative acceptable)

### Additional Requirements (pedagogical_framework_Template.md)
From Section E (Meta-Knowledge) and Quality Checks:
- ✅ **"When to use / When NOT to use"** - Required for every method (Anti-Pattern #5)
  - Meta-level judgment criteria (when to use METHOD vs alternatives)
  - Technique-level comparison (which variant of METHOD to use)
- ✅ **"Common Pitfalls"** or "What can go wrong" - Required (Quality Check: Content Quality)
- ✅ **Domain applications** - Must be specific, not generic
- ✅ **Bottom notes on every slide** - Contextual annotations required

**Critical for Professional Integrity**:
- **NEVER use unverified statistics** without web-verified sources
- Use qualitative descriptions when quantitative data unavailable
- Conceptual charts preferred over charts with fake data
- Week 8 V2.1, Week 9 V1.1, Week 10 V1.1 serve as reference implementations for compliance

## Week-Specific Architecture

### Week 0 Series (5 Presentations)
Complete 4-act narrative structure presentations (see WEEK_0_SERIES_README.md):
- **Week_00a_ML_Foundations**: Learning journey metaphor (4 acts, 26 slides, 17 charts)
- **Week_00b_Supervised_Learning**: Prediction challenge (4 acts, 25 slides, 25 charts)
- **Week_00c_Unsupervised_Learning**: Discovery without labels (4 acts, 26 slides, 25 charts)
- **Week_00d_Neural_Networks**: Depth challenge (4 acts, 25 slides, 25 charts)
- **Week_00e_Generative_AI**: Creation challenge (4 acts, 29 slides, 20 charts)

**Separate**: Week_00_Finance_Theory - Pure theory for quants (38 slides, 10 parts)

### Weeks with Special Features
- **Week 1**: Innovation Diamond visualization series (18 charts), complete with compile.py
- **Week 3**: NLP with 75 charts, Twitter sentiment workshop
- **Week 4**: Dual versions (advanced + beginner), first to use v2/v3 structure
- **Week 7**: Custom Nature Professional color theme
- **Week 8**: **5-part structure** (49 slides, mllavender palette, pedagogically compliant)
  - Standard week format with part1-5 modular structure
  - Focus on structured output and prompt engineering
  - Includes workshop exercises and handouts
- **Week 9**: V1.1 (Oct 3, 2025): Pedagogical framework compliant
  - Added "When to Use Multi-Metric Validation" judgment criteria slide
  - New chart: validation_depth_decision.pdf (decision tree for validation rigor)
  - 51 slides, 16 charts, matches Week 8 V2.1 meta-knowledge standard
- **Week 10**: V1.1 (Oct 3, 2025): Complete with verified industry statistics
  - Added "When to Use A/B Testing" judgment criteria slide
  - New chart: validation_method_decision.pdf (validation method selection decision tree)
  - 51 slides, 16 charts, fully pedagogical framework compliant

## compile.py Features

The `compile.py` script (present in Week_00_Finance_Theory and Weeks 1-10) provides:
- Auto-detection of latest main.tex file (prioritizes main.tex, then timestamped files)
- Dual pdflatex pass for complete references (resolves citations, TOC, etc.)
- Automatic archiving to `archive/builds/` with timestamps
- Auxiliary file cleanup to `archive/aux/` (moves .aux, .log, .nav, .snm, .toc, .vrb, etc.)
- 60-second timeout per compilation pass
- Clear progress indicators and file size reporting
- Error detection with relevant line extraction

**Usage**:
```bash
python compile.py                    # Auto-detects main.tex or latest file
python compile.py 20250928_1500_main.tex  # Compile specific file
```

## Handout System

### Standard Week Handouts
All weeks (1, 3-10) include 3 skill-level handouts:

1. **Basic** (~150-200 lines): No math/code, checklists, plain English
2. **Intermediate** (~300-400 lines): Python implementation guides, case studies
3. **Advanced** (~400-500 lines): Mathematical proofs, production considerations

### Discovery-Based Handout (Week_00_Introduction_ML_AI)
**New Addition (Oct 2025):** Pre-lecture discovery worksheet system

**Structure:**
- 6 chart-driven discovery activities (10 pages, 45-55 minutes completion)
- Zero prerequisites - pattern discovery before formalization
- Each discovery: Chart → Observation → Conceptual tasks → Summary

**Files:**
- `handouts/20251007_2300_discovery_handout_v2.tex` - Main student worksheet
- `handouts/20251007_2200_discovery_solutions.tex` - Instructor answer key
- `handouts/QUICK_START.md` - Complete deployment guide
- `scripts/create_discovery_chart_*.py` - 6 chart generators

**Chart Generation for Discovery Handout:**
```powershell
cd Week_00_Introduction_ML_AI/scripts
python create_discovery_chart_1_overfitting.py  # True overfitting demo
python create_discovery_chart_2_kmeans.py        # K-means iterations
python create_discovery_chart_3_boundaries.py    # Linear separability
python create_discovery_chart_4_gradient.py      # Optimization landscape
python create_discovery_chart_5_gan.py           # Adversarial training
python create_discovery_chart_6_pca_v2.py        # Dimensionality reduction
```

**Key Design Principles:**
- Conceptual thinking over calculations (30% calculation, 70% reasoning)
- Neutral language (no exclamations, "you" pronouns, metaphors)
- Plain-English explanations for all technical concepts
- True overfitting in Chart 1: Model C has Train=0.0, Test>>Train errors

## Development Workflow

1. **New content**: Copy Week_04 structure as template (most complete)
2. **File creation**: Use timestamp prefix YYYYMMDD_HHMM_
3. **Charts**: Generate via Python scripts → save as PDF + PNG
4. **Compilation**: Always use `compile.py` for consistency
5. **Archiving**: Previous versions auto-moved to `archive/previous/`
6. **Cleanup**: Auxiliary files auto-moved to `archive/aux/`

## Archive System

All weeks use consistent archive structure:

```
Week_##/archive/
├── aux/              # LaTeX auxiliary files (auto-moved by compile.py)
│   ├── *.aux         # Cross-references
│   ├── *.log         # Compilation logs
│   ├── *.nav         # Beamer navigation
│   ├── *.out         # Hyperref outlines
│   ├── *.snm         # Beamer snippets
│   └── *.toc         # Table of contents
├── builds/           # Timestamped PDF archives
│   └── YYYYMMDD_HHMM_*.pdf
└── previous/         # Version history of .tex files
    └── YYYYMMDD_HHMM_*.tex
```

**Key Points**:
- `compile.py` automatically moves aux files to `archive/aux/`
- PDFs are archived to `archive/builds/` with timestamps
- All files preserved (never deleted, only moved)
- `.gitignore` excludes archive directories from version control

## Python Dependencies

### Installation Commands
```powershell
# Required for all weeks (core ML and visualization)
pip install scikit-learn numpy pandas scipy matplotlib seaborn

# Week-specific requirements
pip install textblob transformers nltk wordcloud      # Week 3+ (NLP)
pip install gensim pyLDAvis                          # Week 5 (Topic Modeling)
pip install imblearn                                 # Week 4 (Imbalanced Data)
pip install statsmodels                              # Week 10 (Statistical Testing)

# Development tools (optional)
pip install jupyter ipykernel                        # For notebook development
pip install black flake8                             # Code formatting and linting
```

### Package Usage by Week
```python
# Week 1-2: sklearn.cluster (KMeans, DBSCAN, Hierarchical)
# Week 3: textblob, transformers (BERT), nltk, wordcloud
# Week 4: sklearn.ensemble, imblearn.over_sampling
# Week 5: gensim.models (LDA), pyLDAvis
# Week 6-8: transformers (GPT models), openai API
# Week 9-10: sklearn.metrics, statsmodels.stats
```

## Chart Generation Standards

All charts follow consistent pattern:
```python
plt.style.use('seaborn-v0_8-whitegrid')
plt.savefig('charts/chart_name.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/chart_name.png', dpi=150, bbox_inches='tight')
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
- **Undefined color**: Add color definitions to preamble (see Mandatory Color Definitions section)
- **PDF locked (Windows)**:
  - **Best solution**: Close PDF viewer before recompiling
  - Use `-jobname` parameter: `pdflatex -jobname=temp_name main.tex`
  - Switch to SumatraPDF (lightweight, doesn't lock files)
  - Adobe Acrobat locks files aggressively
- **Unicode error**: Replace with ASCII equivalents (no emojis/special chars)
  - Use sed/PowerShell: `(Get-Content file.tex) -replace '✓','[OK]' | Set-Content file.tex`
- **"Extra }, or forgotten \endgroup" at \end{frame}**: Frame has lstlisting without `[fragile]`
  - Add `[fragile,t]` option to frame: `\begin{frame}[fragile,t]{Title}`
- **"Package Listings Error: language undefined"**: Use supported language
  - JSON not supported → use `[language=Python]` instead
- **Verbatim in beamer**: Use `\texttt{}` or lstlisting with `[fragile,t]`
- **Missing references**: Run pdflatex twice (compile.py does this automatically)

### Python Script Issues
- **Import error**: Check dependencies list and install missing packages
- **Chart not rendering**: Verify matplotlib backend with `matplotlib.use('Agg')`
- **Memory error**: Reduce algorithm sample size or use smaller datasets
- **DPI warnings**: Ignore - charts are intentionally 300dpi (PDF) and 150dpi (PNG)

### Git Issues
- **Auxiliary files**: Automatically ignored by .gitignore (LaTeX aux, Python cache, etc.)
- **Archive directories**: Not tracked by git (`**/archive/aux/` pattern in .gitignore)
- **After cleanup**: If you see deleted aux files in git status, they were moved to archive (safe to commit)
- **Large commits**: Stage files by pattern (e.g., `git add Week_*/charts/*.pdf`)
- **Line endings**: Windows CRLF warnings are normal for .tex files
- **PDF conflicts**: Don't commit intermediate PDFs, only final timestamped versions

## Statistics Verification Protocol

When including industry statistics:
1. Verify all claims via web search
2. Cite sources when available (e.g., "McKinsey 2013")
3. Use conservative estimates when sources conflict
4. Examples of verified stats kept in Week 10 documentation

## Decision Guide: Which Week 0 to Use?

Choose based on course constraints and pedagogical goals:

| Criterion | Week_00_Introduction_ML_AI | Week 0a-0e Series | Week_00_Finance_Theory |
|-----------|---------------------------|-------------------|------------------|
| **Duration** | 90 minutes (single) | 7.5 hours (5 × 90 min) | 90 minutes |
| **Depth** | Broad survey | Deep narrative | Finance theory |
| **Audience** | General BSc students | Pedagogy-focused | Quant/finance majors |
| **Use case** | Time-constrained | Full semester intro | Specialized track |
| **Slides** | 38 comprehensive | 127 total (5 parts) | 38 pure theory |
| **Approach** | Concept → Example | Example → Concept | Mathematical rigor |

**Recommendation**:
- **Tight schedule**: Use Week_00_Introduction_ML_AI (1 session)
- **Deep understanding**: Use Week 0a-0e series (5 sessions, narrative-driven)
- **Finance/quant track**: Use Week_00_Finance_Theory (advanced finance applications)
- **Hybrid**: Teach Week_00_Introduction_ML_AI, offer 0a-0e as optional enrichment

## Key Project Files

- `EDUCATIONAL_PRESENTATION_FRAMEWORK.md`: **CRITICAL** - Complete pedagogical methodology including:
  - Dual-slide pattern (visual anchor + detailed explanation)
  - Zero pre-knowledge principle (6-step pattern for new concepts)
  - Bottom note requirements for contextual annotations
  - Meta-knowledge requirements ("When to use", "Common pitfalls")
  - Quality checks and anti-patterns to avoid
  - Framework designed for concept-introduction presentations (Week 0a-0e series)
  - Applied workshop weeks (1-10) follow core principles selectively
- `WEEK_0_SERIES_README.md`: Overview of Week 0a-0e expansion (5 presentations, 127 slides)
- `GAP_ANALYSIS_REPORT.md`: Course completion tracking (~85% complete, last updated 2025-09-27)
  - **Update (Oct 6, 2025):** Week 0 variants now 100% content-complete
  - All 10 main weeks (1-10) framework-compliant for pedagogical requirements
  - All 7 Week 0 variants verified, mathematically correct, production-ready
- `template_beamer_final.tex`: Standard Beamer template (22 layouts, Madrid theme)
- `fix_chart_sizes.py`: Utility for chart standardization
- `compile.py`: Automated build script available in Weeks 1-10 and Week_00_Finance_Theory

## Template System

### template_beamer_final.tex
Standard template with 22 professional layouts:
- Content layouts (2-column, 3-column, lists)
- Visual layouts (full-width images, mixed media)
- Comparison layouts (definition-example, pros-cons)
- Specialized formats (Q&A, timeline, code examples)
- Data visualization (full-size charts, annotated charts)

**Key Features**:
- Madrid theme customized with mllavender palette
- `\bottomnote{}` command for slide annotations
- Reduced margins (5mm) for more content space
- Clean itemize/enumerate styles
- No navigation symbols

## Windows-Specific Considerations

### Path Handling
- Use forward slashes or backslashes interchangeably: `Week_01\main.tex` or `Week_01/main.tex`
- PowerShell commands shown throughout this guide
- Tab completion works in PowerShell: `cd We<TAB>` → `cd Week_01`

### PDF Viewer Recommendations
1. **SumatraPDF** (Recommended): Lightweight, doesn't lock files, free
2. **Adobe Acrobat Reader**: Full-featured but locks files aggressively
3. **Browser**: Chrome/Edge can open PDFs without locking

### File Locking Issues
- Windows locks files opened by applications
- Close PDF viewer before recompiling to avoid locks
- If locked, use `-jobname` flag or kill viewer process

### Git on Windows
- CRLF line endings are normal for this repository
- Don't commit auxiliary files (*.aux, *.log, etc.)
- Large binary files (PDFs) should be committed selectively

---

**Last Updated**: October 8, 2025
**Recent Changes**:
- **RENAMED**: Week_00b_ML_Finance_Applications → Week_00_Finance_Theory (eliminates confusion)
- **CLEANED**: Moved 696 auxiliary files to archive/aux directories (zero deletions)
- **ENHANCED**: .gitignore now includes LaTeX, Python, Jupyter, OS, and IDE patterns
- Added Quick Troubleshooting Guide section for common issues
- Documented Archive System structure and workflow
- Added Discovery-Based Handout system documentation (Week_00_Introduction_ML_AI)
- Documented 6 discovery chart generators with true overfitting demonstration
- Updated handout system section with chart generation workflow
- Corrected compile.py availability statement (present in Weeks 1-10 and Week_00_Finance_Theory)
- Updated Week_00e slide count (29 slides, not 25)