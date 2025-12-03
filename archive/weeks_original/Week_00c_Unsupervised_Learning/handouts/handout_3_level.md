# Week 00c Handout Level 3: Unsupervised Learning

## Level 3 Focus
Mathematical theory and proofs

## Key Topics
- K-means clustering
- DBSCAN (density-based)
- Hierarchical clustering
- Cluster validation (silhouette, elbow)
- Dimensionality reduction (PCA, t-SNE)



## Mathematical Foundations

### K-means Objective
Minimize within-cluster sum of squares:
\$\$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2\$\$

### DBSCAN Definitions
- Core point: MinPts neighbors within ε
- Density reachable: Chain of core points
- Cluster: Maximal set of density-connected points

## Practice
- Customer segmentation
- Anomaly detection
- Document clustering
