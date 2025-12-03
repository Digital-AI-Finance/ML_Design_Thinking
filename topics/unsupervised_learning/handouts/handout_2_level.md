# Week 00c Handout Level 2: Unsupervised Learning

## Level 2 Focus
Python implementation with scikit-learn

## Key Topics
- K-means clustering
- DBSCAN (density-based)
- Hierarchical clustering
- Cluster validation (silhouette, elbow)
- Dimensionality reduction (PCA, t-SNE)

## Implementation Example

\`\`\`python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# K-means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
print(f"Silhouette: {silhouette_score(X, labels):.3f}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X)

# Hierarchical
hier = AgglomerativeClustering(n_clusters=3)
labels_hier = hier.fit_predict(X)
\`\`\`



## Practice
- Customer segmentation
- Anomaly detection
- Document clustering
