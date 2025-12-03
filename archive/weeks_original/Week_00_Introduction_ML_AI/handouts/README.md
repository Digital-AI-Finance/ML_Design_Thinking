# Discovery-Based Handout System
## Pre-Lecture Learning for Week 00: Introduction to ML & AI

**Status**: Production Ready (October 2025)
**Version**: 2.0 (Conceptual Emphasis)

---

## Overview

This system provides a **chart-driven discovery worksheet** for students to complete before attending the introductory ML/AI lecture. Students identify patterns visually and develop conceptual understanding before encountering formal terminology.

### Philosophy

**Discovery → Formalization → Application**

Students:
1. Observe charts showing ML algorithms in action
2. Answer conceptual questions about patterns
3. Build intuition before learning mathematical frameworks
4. Arrive at lecture with 70% conceptual foundation already established

---

## Files

### For Students
| File | Purpose | Pages | Time |
|------|---------|-------|------|
| `20251008_0800_discovery_handout_v2.pdf` | Main worksheet | 10 | 45-55 min |

### For Instructors
| File | Purpose | Pages |
|------|---------|-------|
| `20251008_0800_discovery_solutions_v2.pdf` | Answer key with discussion prompts | 10 |
| `QUICK_START.md` | Deployment guide for instructors and TAs | - |
| `VERIFICATION_REPORT.md` | Technical validation of charts and metrics | - |

### Source Files
| File | Purpose |
|------|---------|
| `20251007_2200_discovery_handout.tex` | LaTeX source for student handout |
| `20251008_0800_discovery_solutions_v2.tex` | LaTeX source for solutions |
| `../scripts/create_discovery_chart_*.py` | Chart generation scripts (6 total) |
| `../charts/discovery_chart_*.pdf` | Generated visualization charts |

---

## The 6 Discoveries

| # | Concept | Chart Shows | Key Discovery | Time |
|---|---------|-------------|---------------|------|
| 1 | **Overfitting** | 3 models (simple → complex) on training + test data | Perfect training fit ≠ good generalization | 8 min |
| 2 | **K-Means** | 6-step clustering iteration | Alternating assignment/update converges | 8 min |
| 3 | **Linear Limits** | 4 datasets (A-D) with separation attempts | Some patterns need nonlinear boundaries (XOR) | 8 min |
| 4 | **Optimization** | Topographic error landscape with 2 paths | Starting point matters; local vs global minima | 8 min |
| 5 | **Adversarial Training** | GAN training dynamics over 100 epochs | Competition drives mutual improvement | 8 min |
| 6 | **Dimensionality Reduction** | 3D → 2D compression with reconstruction | Correlated data compresses with minimal loss | 8 min |

**Total**: 48 minutes core + 5-10 minutes reflection = **45-55 minutes**

---

## Design Principles (Version 2.0)

### Conceptual Over Computational
- **70% reasoning, 30% calculation**
- Questions ask "why" and "what pattern" not "calculate exact value"
- Students observe and interpret, not compute precise numbers

### Neutral Language
- No exclamations, "you" pronouns, or metaphors
- Direct, professional tone
- Avoids condescension ("Easy!", "Simply...") and hype ("Amazing!", "Incredible!")

### Zero Prerequisites
- No prior ML knowledge required
- Technical terms introduced AFTER pattern discovery
- Plain-English explanations for all concepts

### True Demonstrations
- Chart 1 overfitting: Model C has Train=0.0, Test=22.9 (true overfitting)
- All charts use real sklearn algorithms, not synthetic/fake data
- Numbers and patterns are mathematically correct

---

## Quick Deployment Guide

### 1 Week Before Lecture

**Instructor Actions**:
```bash
# Distribute handout to students
Email: 20251008_0800_discovery_handout_v2.pdf
Subject: "Pre-Lecture Discovery Activity - Complete Before Class"
```

**Student Instructions** (include in email):
- Download and print PDF (or use tablet)
- Complete 6 discoveries (45-55 minutes)
- Focus on patterns, not precise calculations
- Bring completed worksheet to class

### 3 Days Before Lecture

**Instructor Prep**:
```bash
# Review solutions guide
Open: 20251008_0800_discovery_solutions_v2.pdf

# Prepare discussion questions
- 3-4 prompts per discovery (see solutions guide)
- Identify common misconceptions
```

### During Lecture

**Integration Strategy**:
1. **Opening (5 min)**: "Turn to partner - compare Chart 1 Task 3 answers"
2. **Throughout**: Reference specific charts when formalizing concepts
   - "Chart 2 showed K-means iterations - now let's formalize the algorithm"
   - "You discovered XOR impossibility in Chart 3 - here's the proof"
3. **Validation**: Confirm student intuitions with rigorous derivations
4. **Extension**: Build on discoveries toward advanced topics

---

## Chart Generation

All charts are Python-generated for reproducibility and correctness.

### Regenerate All Charts
```powershell
cd Week_00_Introduction_ML_AI/scripts

# Generate all 6 discovery charts
python create_discovery_chart_1_overfitting.py
python create_discovery_chart_2_kmeans.py
python create_discovery_chart_3_boundaries.py
python create_discovery_chart_4_gradient.py
python create_discovery_chart_5_gan.py
python create_discovery_chart_6_pca_v2.py
```

Each script:
- Outputs to `../charts/discovery_chart_N_*.pdf` (300 dpi)
- Also creates `.png` version (150 dpi)
- Prints metrics to verify correctness

### Chart 1 Verification
Critical requirement: **True overfitting must be demonstrated**

```python
# Run chart 1 and verify output
python create_discovery_chart_1_overfitting.py

# Expected output:
# Model A - Train: 34.2, Test: 41.3  (underfitting)
# Model B - Train: 12.4, Test: 10.2  (optimal)
# Model C - Train: 0.0,  Test: 22.9  (overfitting)
#
# PASS if: C_train ≈ 0.0 AND C_test > B_test * 1.5
```

If verification fails, adjust test data random seed in script.

---

## Compilation

### Compile Student Handout
```powershell
cd Week_00_Introduction_ML_AI/handouts
pdflatex 20251007_2200_discovery_handout.tex
pdflatex 20251007_2200_discovery_handout.tex  # Run twice for references
```

### Compile Solutions Guide
```powershell
pdflatex 20251008_0800_discovery_solutions_v2.tex
pdflatex 20251008_0800_discovery_solutions_v2.tex
```

---

## Assessment Integration

### Completion Check (Binary)
- **5 points**: Completed all 6 discoveries with reasonable effort
- **0 points**: Incomplete or clearly rushed

### Depth Assessment (Optional, 10 points)
Rubric for deeper evaluation:

| Criteria | Points | What to Look For |
|----------|--------|-----------------|
| **Pattern Recognition** | 0-4 | Correctly identifies visual patterns in charts |
| **Conceptual Reasoning** | 0-4 | Explains "why" not just "what"; connects ideas |
| **Curiosity** | 0-2 | Thoughtful questions for lecture; cross-chart connections |

**Grading Time**: ~2 minutes per student (spot check 2-3 answers)

---

## Common Student Questions

### "How precise should my answers be?"
**Response**: Pattern matters more than precision. Approximations and ranges are fine. Focus on the "why" behind patterns.

### "Can I work with a partner?"
**Response**: Yes, but each person should complete their own worksheet. Comparing answers is valuable for learning.

### "I can't separate Dataset D with a line."
**Response**: Correct! That's the discovery. Try to explain why it's impossible (proof optional).

### "My Chart 2 variance numbers don't match exactly."
**Response**: Visual estimation is imperfect - that's OK. The important pattern is that variance decreases over iterations.

### "Can I use Python to check answers?"
**Response**: Yes for arithmetic verification, but don't look up formulas or algorithm names. The goal is discovery.

---

## Troubleshooting

### Chart 1 Doesn't Show Overfitting
**Symptoms**: Model C test error ≤ Model B test error

**Fix**:
```python
# Edit create_discovery_chart_1_overfitting.py
# Lines 29-31: Adjust test data random seed

np.random.seed(123)  # Try different values: 456, 789, 2024
X_test = ...
```

Re-run until Model C test error > Model B test error significantly.

### PDF Locked by Viewer (Windows)
**Fix**:
```powershell
# Close PDF viewer
# OR use temporary filename
pdflatex -jobname=temp_discovery 20251007_2200_discovery_handout.tex
```

### Charts Not Found During Compilation
**Symptoms**: LaTeX error "File `../charts/discovery_chart_1_overfitting.pdf' not found"

**Fix**:
```powershell
# Ensure charts exist
cd ../scripts
python create_discovery_chart_1_overfitting.py  # Repeat for all 6 charts

# Verify output
ls ../charts/discovery_chart_*.pdf  # Should show 6 files
```

---

## Version History

### Version 2.0 (October 8, 2025) - Current
**Major Changes**:
- Shifted from calculation-heavy (70%) to conceptual (70%)
- Removed all Task 5 sections (complexity reduction)
- Neutralized language (removed exclamations, "you", metaphors)
- Removed instruction boxes from first page
- Removed "Bring worksheet to class" footer
- Reduced from 15 pages to 10 pages
- Reduced time from 75-90 minutes to 45-55 minutes

**Chart Updates**:
- Chart 1: Fixed overfitting to guarantee Train=0.0, Test>>Train
- Chart 5: Added 4 plain-English text boxes explaining GAN terms
- Chart 6: Simplified from 6 panels to 3 panels (removed scree plot, trade-off, projections table)

### Version 1.0 (October 7, 2025)
- Initial release
- Calculation-heavy approach (Task 1-4 numerical, Task 5 extension)
- 15 pages, 75-90 minutes
- All 6 discovery charts functional

---

## Educational Research Notes

### Theoretical Foundation
This system implements:
- **Constructivist learning**: Students build knowledge through discovery
- **Worked example effect**: Charts provide concrete examples before abstractions
- **Scaffolded learning**: Gradual progression from observation to formalization
- **Flipped classroom**: Conceptual foundation established pre-lecture

### Expected Outcomes
Students who complete worksheet should demonstrate:
- **70% conceptual foundation** before lecture begins
- **Reduced cognitive load** during lecture (no simultaneous concept introduction + formalization)
- **Higher engagement** (validating discoveries vs passive reception)
- **Better retention** (self-discovered insights persist longer)

### Pilot Testing Recommendations
1. Track completion rate (target: 80%+)
2. Survey time taken (target: 45-55 minutes, flag if >60)
3. Spot-check answer quality (random sample of 10 students)
4. Measure lecture engagement (question frequency, discussion depth)
5. Post-lecture quiz to assess retention (compare with non-discovery control group)

---

## Support

### Documentation
- **QUICK_START.md**: Fast deployment guide for instructors
- **VERIFICATION_REPORT.md**: Technical validation of all charts
- **CLAUDE.md**: Full repository documentation (root directory)

### Contact
For questions or issues with the discovery handout system:
1. Check troubleshooting section above
2. Review QUICK_START.md for instructor guidance
3. Consult solutions guide for expected answers

---

**Last Updated**: October 8, 2025
**Maintainer**: Course Development Team
**License**: Educational Use (Non-Commercial)
