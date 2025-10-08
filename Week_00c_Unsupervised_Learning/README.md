# Week 00c: Unsupervised Learning - Discovery Without Labels

## Overview
**Duration**: 90 minutes | **Format**: 4-act narrative | **Slides**: 26 | **Charts**: 25

## Learning Objectives
- Understand clustering without labels
- Master K-means algorithm and optimization
- Apply DBSCAN for density-based clustering
- Use hierarchical clustering and dendrograms
- Choose appropriate clustering methods
- Validate cluster quality

## Structure
1. **Act 1: Challenge** (5 slides) - Customer segmentation without predefined categories
2. **Act 2: K-means** (6 slides) - Centroid-based clustering, success then limitations
3. **Act 3: Density & Hierarchy** (9 slides) - DBSCAN for arbitrary shapes, hierarchical for relationships
4. **Act 4: Synthesis** (6 slides) - Modern applications, method selection, validation

## Key Files
- `act1_challenge.tex` - The discovery problem
- `act2_kmeans.tex` - K-means algorithm
- `act3_density_hierarchy.tex` - Advanced methods
- `act4_synthesis.tex` - Integration & meta-knowledge
- `scripts/create_week0c_charts.py` - Generate all 25 charts

## Compilation
```powershell
cd Week_00c_Unsupervised_Learning
pdflatex 20251007_1640_unsupervised_learning.tex  # Run twice
```

## Key Concepts
- K-means: Centroid-based, fixed K, convex clusters
- DBSCAN: Density-based, auto K, arbitrary shapes
- Hierarchical: Agglomerative, dendrograms, relationships
- Validation: Elbow method, silhouette score, Davies-Bouldin

## Status
✅ Production Ready - Unicode compliant, pedagogically complete

## Dependencies
```powershell
pip install scikit-learn scipy matplotlib seaborn
```
