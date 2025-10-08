# Week 2 Handout 3: Advanced Clustering Theory
## Mathematical Foundations and Production Systems

**Target Audience:** ML researchers, senior engineers, PhD students
**Prerequisites:** Linear algebra, probability theory, optimization, Python
**Time to Complete:** 3-4 hours

---

## Overview

This handout covers the mathematical foundations and advanced topics for clustering algorithms:
- K-means as EM algorithm
- DBSCAN formal definitions and properties
- Hierarchical clustering mathematics
- GMM derivations
- Scalability and production considerations

---

## 1. K-Means Mathematical Foundations

### Optimization Formulation

**Objective**: Minimize within-cluster sum of squares (WCSS)

$$J(C, \mu) = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- $K$ = number of clusters
- $C = \\{C_1, ..., C_K\\}$ = cluster assignments
- $\mu = \\{\mu_1, ..., \mu_K\\}$ = cluster centers
- $||·||$ = Euclidean distance (L2 norm)

### Lloyd's Algorithm

K-means is a special case of Expectation-Maximization (EM):

**E-step** (Assignment):
$$C_i^{(t+1)} = \\{x : ||x - \mu_i^{(t)}|| \leq ||x - \mu_j^{(t)}|| \quad \forall j \neq i\\}$$

Assign each point to nearest center.

**M-step** (Update):
$$\mu_i^{(t+1)} = \frac{1}{|C_i^{(t+1)}|} \sum_{x \in C_i^{(t+1)}} x$$

Recompute centers as cluster means.

### Convergence Properties

**Theorem**: Lloyd's algorithm converges to a local minimum.

*Proof sketch*:
1. Assignment step: $J$ decreases (or stays same) because points assigned to nearest center
2. Update step: $J$ decreases because mean minimizes sum of squared distances
3. $J$ is bounded below by 0
4. By monotone convergence, algorithm converges

**Caveats**:
- Converges to *local* minimum (not global)
- Final result depends on initialization
- Can get stuck in poor solutions

### Complexity Analysis

- **Time per iteration**: $O(n \cdot K \cdot d)$
  - $n$ = number of points
  - $K$ = number of clusters
  - $d$ = dimensionality

- **Number of iterations**: Typically $I = O(\log n)$ to $O(n)$ in worst case

- **Total**: $O(n \cdot K \cdot d \cdot I)$

**Practical optimizations**:
- **K-means++** initialization: $O(n \cdot K \cdot d)$ but dramatically reduces iterations
- **Mini-batch K-means**: $O(b \cdot K \cdot d \cdot I)$ where $b$ = batch size
- **Triangle inequality**: Skip distance computations when provably not needed

---

## 2. DBSCAN Formal Definitions

### Core Concepts

**Epsilon-neighborhood**:
$$N_\epsilon(x) = \\{y \in X : d(x, y) \leq \epsilon\\}$$

Set of points within distance $\epsilon$ of point $x$.

**Core point**:
$$|N_\epsilon(x)| \geq \text{minPts}$$

Point with at least minPts neighbors in its $\epsilon$-neighborhood.

**Density-reachable**:

Point $y$ is *directly density-reachable* from $x$ if:
1. $y \in N_\epsilon(x)$
2. $x$ is a core point

Point $y$ is *density-reachable* from $x$ if there exists chain:
$$x = p_1, p_2, ..., p_n = y$$

where $p_{i+1}$ is directly density-reachable from $p_i$.

**Density-connected**:

Points $x$ and $y$ are *density-connected* if there exists point $z$ such that both $x$ and $y$ are density-reachable from $z$.

### Algorithm

```
DBSCAN(X, ε, minPts):
    C = 0  # cluster counter
    labels = [UNVISITED] * |X|

    for each point x in X:
        if labels[x] ≠ UNVISITED:
            continue

        neighbors = rangeQuery(x, ε)

        if |neighbors| < minPts:
            labels[x] = NOISE
            continue

        C = C + 1
        labels[x] = C

        seed_set = neighbors \ {x}

        for each point y in seed_set:
            if labels[y] == NOISE:
                labels[y] = C
            if labels[y] ≠ UNVISITED:
                continue

            labels[y] = C
            neighbors_y = rangeQuery(y, ε)

            if |neighbors_y| >= minPts:
                seed_set = seed_set ∪ neighbors_y

    return labels

rangeQuery(x, ε):
    return {y ∈ X : d(x, y) ≤ ε}
```

### Theoretical Properties

**Theorem 1**: DBSCAN finds all density-connected clusters.

**Theorem 2**: Border points may be assigned to different clusters in different runs, but core points always have consistent clusters.

**Lemma**: If parameters are chosen appropriately, DBSCAN is deterministic for core points.

### Complexity

- **Naive**: $O(n^2)$ for $n$ points (compute all pairwise distances)
- **With spatial index** (kd-tree, R-tree): $O(n \log n)$ average case
- **High dimensions** ($d > 20$): Index structures degrade to $O(n^2)$

---

## 3. Hierarchical Clustering Mathematics

### Linkage Functions

**Single Linkage**:
$$d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$$

Minimum distance between any two points.

**Complete Linkage**:
$$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$$

Maximum distance between any two points.

**Average Linkage**:
$$d(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$$

Average distance between all pairs.

**Ward's Linkage**:
$$d(C_i, C_j) = \frac{|C_i| \cdot |C_j|}{|C_i| + |C_j|} ||\mu_i - \mu_j||^2$$

Increase in within-cluster variance after merge.

### Agglomerative Algorithm

```
AgglomerativeClustering(X, linkage):
    # Initialize: each point is own cluster
    C = {{x₁}, {x₂}, ..., {xₙ}}

    # Compute initial distance matrix
    D = pairwise_distances(X, linkage)

    while |C| > 1:
        # Find closest pair
        (Cᵢ, Cⱼ) = argmin_{Cᵢ, Cⱼ ∈ C, i≠j} D[Cᵢ, Cⱼ]

        # Merge
        C_new = Cᵢ ∪ Cⱼ
        C = (C \ {Cᵢ, Cⱼ}) ∪ {C_new}

        # Update distances
        for Cₖ in C \ {C_new}:
            D[C_new, Cₖ] = update_distance(Cᵢ, Cⱼ, Cₖ, linkage)

    return dendrogram
```

### Lance-Williams Formula

Efficient distance update after merge:

$$d(C_k, C_i \cup C_j) = \alpha_i d(C_k, C_i) + \alpha_j d(C_k, C_j) + \beta d(C_i, C_j) + \gamma |d(C_k, C_i) - d(C_k, C_j)|$$

Where $(\alpha_i, \alpha_j, \beta, \gamma)$ depend on linkage:

| Linkage | $\alpha_i$ | $\alpha_j$ | $\beta$ | $\gamma$ |
|---------|-----------|-----------|---------|---------|
| Single | 0.5 | 0.5 | 0 | -0.5 |
| Complete | 0.5 | 0.5 | 0 | 0.5 |
| Average | $\|C_i\|/\|C_i \cup C_j\|$ | $\|C_j\|/\|C_i \cup C_j\|$ | 0 | 0 |
| Ward | $(\|C_i\|+\|C_k\|)/(\|C_i\|+\|C_j\|+\|C_k\|)$ | $(\|C_j\|+\|C_k\|)/(\|C_i\|+\|C_j\|+\|C_k\|)$ | $-\|C_k\|/(\|C_i\|+\|C_j\|+\|C_k\|)$ | 0 |

### Complexity

- **Naive**: $O(n^3)$ (compute all pairwise distances, $n$ merges, update $O(n^2)$ matrix each time)
- **With priority queue**: $O(n^2 \log n)$
- **SLINK (single linkage)**: $O(n^2)$
- **CLINK (complete linkage)**: $O(n^2)$

---

## 4. Gaussian Mixture Models

### Probabilistic Framework

**Generative model**: Each point generated from one of $K$ Gaussians.

**Model**:
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

Where:
- $\pi_k$ = mixing coefficient (prior probability of cluster $k$), $\sum_k \pi_k = 1$
- $\mathcal{N}(x | \mu_k, \Sigma_k)$ = multivariate Gaussian with mean $\mu_k$ and covariance $\Sigma_k$

**Likelihood**:
$$\mathcal{L}(\theta | X) = \prod_{i=1}^{n} p(x_i | \theta) = \prod_{i=1}^{n} \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)$$

**Log-likelihood** (to maximize):
$$\ell(\theta | X) = \sum_{i=1}^{n} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k) \right)$$

### EM Algorithm for GMM

**Goal**: Maximize log-likelihood $\ell(\theta | X)$ w.r.t. $\theta = (\pi, \mu, \Sigma)$.

**Introduce latent variables** $z_i \in \\{1, ..., K\\}$ indicating which Gaussian generated $x_i$.

**E-step**: Compute posterior probability (responsibility) that cluster $k$ generated point $i$:

$$\gamma_{ik} = p(z_i = k | x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(x_i | \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \mathcal{N}(x_i | \mu_j^{(t)}, \Sigma_j^{(t)})}$$

**M-step**: Update parameters using weighted MLE:

$$\pi_k^{(t+1)} = \frac{1}{n} \sum_{i=1}^{n} \gamma_{ik}$$

$$\mu_k^{(t+1)} = \frac{\sum_{i=1}^{n} \gamma_{ik} x_i}{\sum_{i=1}^{n} \gamma_{ik}}$$

$$\Sigma_k^{(t+1)} = \frac{\sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^T}{\sum_{i=1}^{n} \gamma_{ik}}$$

### Covariance Types

**Full**: $\Sigma_k$ = $d \times d$ positive definite matrix (most flexible, most parameters)

**Tied**: $\Sigma_k = \Sigma$ for all $k$ (same covariance, different means)

**Diagonal**: $\Sigma_k$ = diagonal matrix (features independent within cluster)

**Spherical**: $\Sigma_k = \sigma_k^2 I$ (isotropic, like K-means with soft assignments)

**Parameter count**:
- Full: $K \cdot (d + d(d+1)/2 + 1) - 1 = O(Kd^2)$
- Spherical: $K \cdot (d + 1 + 1) - 1 = O(Kd)$

### Model Selection

**Bayesian Information Criterion (BIC)**:
$$\text{BIC} = -2 \ell(\hat{\theta} | X) + p \log n$$

Where $p$ = number of free parameters.

**Akaike Information Criterion (AIC)**:
$$\text{AIC} = -2 \ell(\hat{\theta} | X) + 2p$$

**Lower is better** for both. BIC penalizes complexity more (prefers simpler models).

---

## 5. Advanced Topics

### Curse of Dimensionality

**Problem**: In high dimensions, distance becomes meaningless.

**Theorem** (concentration of measure): As $d \to \infty$, ratio of distances to nearest and farthest points approaches 1:

$$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$$

**Implication**: All points are approximately equidistant → clustering degenerates.

**Solutions**:
- **Dimensionality reduction**: PCA, t-SNE, UMAP before clustering
- **Feature selection**: Remove irrelevant dimensions
- **Subspace clustering**: Find clusters in different subspaces

### Spectral Clustering

**Idea**: Cluster in eigenspace of graph Laplacian.

**Algorithm**:
1. Construct similarity graph $W$ (e.g., $W_{ij} = \exp(-||x_i - x_j||^2 / 2\sigma^2)$)
2. Compute graph Laplacian $L = D - W$ where $D_{ii} = \sum_j W_{ij}$
3. Compute first $k$ eigenvectors of $L$
4. Apply K-means to eigenvector matrix

**Advantages**:
- Handles non-convex clusters
- Works on graph data (not just Euclidean)
- Theoretical guarantees for graph cuts

---

## 6. Production Scalability

### MiniBatch K-means

**Idea**: Update centers using small random batches instead of full dataset.

**Complexity**: $O(b \cdot K \cdot d \cdot I)$ where $b$ = batch size (typically $b = 100$)

**Trade-off**: 10-100× faster, slightly worse quality (typically <5% decrease in objective).

### BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)

**Idea**: Build compact summary (Clustering Feature tree) in one pass, then cluster summary.

**Clustering Feature**: Compact representation of cluster:
$$CF = (N, LS, SS)$$
- $N$ = number of points
- $LS$ = linear sum $\sum x_i$
- $SS$ = square sum $\sum x_i^2$

**Additivity**: $CF_1 + CF_2 = (N_1 + N_2, LS_1 + LS_2, SS_1 + SS_2)$

**Complexity**: $O(n)$ (single pass through data)

**Use case**: Streaming data, very large datasets ($n > 10^6$)

### HDBSCAN (Hierarchical DBSCAN)

**Improvements over DBSCAN**:
- No need to specify $\epsilon$ (only minPts)
- Handles clusters of varying densities
- Hierarchical structure

**Idea**: Build hierarchy using mutual reachability distance:
$$d_{\text{mreach-k}}(x, y) = \max\\{d(x, y), \text{core}_k(x), \text{core}_k(y)\\}$$

where $\text{core}_k(x)$ = distance to $k$-th nearest neighbor.

---

## 7. Validation Metrics (Mathematical)

### Silhouette Coefficient

For point $i$ in cluster $C_I$:

**a(i)** = average distance to points in same cluster:
$$a(i) = \frac{1}{|C_I| - 1} \sum_{j \in C_I, j \neq i} d(i, j)$$

**b(i)** = minimum average distance to points in other clusters:
$$b(i) = \min_{J \neq I} \frac{1}{|C_J|} \sum_{j \in C_J} d(i, j)$$

**Silhouette** of point $i$:
$$s(i) = \frac{b(i) - a(i)}{\max\\{a(i), b(i)\\}}$$

Range: $[-1, 1]$. Closer to 1 = better clustered.

**Average silhouette**:
$$s = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

### Davies-Bouldin Index

$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)}$$

Where:
- $\sigma_i$ = average distance from points in cluster $i$ to centroid $\mu_i$
- $d(\mu_i, \mu_j)$ = distance between centroids

**Lower is better** (compact, well-separated clusters).

---

## Further Reading

- Bishop (2006): Pattern Recognition and Machine Learning (Chapter 9)
- Hastie et al. (2009): Elements of Statistical Learning (Chapter 14)
- Xu & Wunsch (2005): Survey of Clustering Algorithms (comprehensive review)
- Ester et al. (1996): DBSCAN original paper
- Reynolds (2009): Gaussian Mixture Models tutorial

---

**Status**: Advanced mathematical foundations complete. Ready for research-level work or production deployment at scale.
