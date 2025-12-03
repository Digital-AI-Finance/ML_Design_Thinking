# Moodle Course Export

## Status: PENDING ANALYSIS - Core Content for Improvement

This folder contains a Moodle course export/backup that represents the course as deployed on the learning management system.

## Contents

**Size**: ~31 MB

**Export Path**: `MLID_(dbmWPM)_HS25_1764742498/`

### Included Materials

- **Learning Resources**: 40+ index.html files with course content
- **Slides**: Embeddings, Support Vector Machines, Decision Trees, Random Forests, etc.
- **Handouts**: N-Grams exercise, Embeddings handout
- **Notebooks**:
  - Supervised Learning (.ipynb + HTML)
  - Descriptive Analysis
  - DBSCAN
  - K-Means
- **Datasets**: innovations.csv
- **Assignments**: Midterm exam
- **Course Pages**: Learning Goals, Streaming, Discussion forums

## Purpose

This is a **backup snapshot** of the course as it appeared on Moodle. It serves as:
1. Reference for what was actually deployed to students
2. Source for identifying gaps between source materials and LMS content
3. Basis for future course improvement analysis

## Analysis Tasks

### Priority: HIGH

This folder contains the **core content needing improvement**. Future analysis should:

1. [ ] Compare Moodle content with source Week_01-10 folders
2. [ ] Identify discrepancies between source and deployed content
3. [ ] Review student-facing materials for quality
4. [ ] Analyze notebook implementations
5. [ ] Check assignment alignment with learning objectives

## Relationship to Source Materials

| Moodle Content | Source Location |
|----------------|-----------------|
| Slides | Week_##/*.pdf |
| Handouts | Week_##/handouts/ |
| Notebooks | Week_##/notebooks/ (if exists) |
| Datasets | Various locations |

## Do Not Modify

This folder should remain read-only as it represents a point-in-time export. Any improvements should be made to the source Week_## folders, then re-exported to Moodle.

## Next Steps

1. Schedule analysis session during semester break
2. Document findings in improvement plan
3. Update source materials
4. Re-export to Moodle for next semester
