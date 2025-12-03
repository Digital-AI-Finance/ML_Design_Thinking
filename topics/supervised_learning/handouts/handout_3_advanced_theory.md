# Week 00b Advanced: Supervised Learning Theory

## Linear Models

### OLS Regression
Minimize: $L(w) = \|Xw - y\|^2$

Solution: $w^* = (X^TX)^{-1}X^Ty$

Assumptions:
- Linearity
- Independence
- Homoscedasticity
- Normality of residuals

### Ridge Regression
$w^* = (X^TX + \lambda I)^{-1}X^Ty$

Shrinks coefficients, prevents overfitting

### Lasso Regression
$\min_w \|Xw - y\|^2 + \lambda\|w\|_1$

Sparse solutions (feature selection)

## Tree-Based Methods

### CART Algorithm
Recursive binary splitting minimizing impurity:

**Gini impurity**:
$$I_G = 1 - \sum_{k=1}^K p_k^2$$

**Entropy**:
$$H = -\sum_{k=1}^K p_k \log_2 p_k$$

### Random Forest
Bootstrap aggregating (bagging):
- Sample data with replacement
- Train tree on each sample
- Average predictions

Reduces variance, prevents overfitting

### Gradient Boosting
Sequential additive model:
$$F_m(x) = F_{m-1}(x) + \nu h_m(x)$$

where $h_m$ fits residuals:
$$h_m = \argmin_h \sum_i L(y_i, F_{m-1}(x_i) + h(x_i))$$

## SVM Theory

### Primal Problem
$$\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$$

Subject to: $y_i(w^Tx_i + b) \geq 1 - \xi_i$

### Dual Problem
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i\alpha_j y_iy_j K(x_i, x_j)$$

Subject to: $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$

## Probabilistic Models

### Logistic Regression
$$P(y=1|x) = \frac{1}{1 + e^{-w^Tx}}$$

Maximum likelihood:
$$\max_w \sum_i [y_i \log \sigma(w^Tx_i) + (1-y_i)\log(1-\sigma(w^Tx_i))]$$

### Naive Bayes
$$P(y|x) \propto P(y)\prod_i P(x_i|y)$$

Independence assumption simplifies computation

## Learning Theory

### PAC Learning
Sample complexity for $\epsilon$-accurate, $(1-\delta)$-confident:
$$m \geq \frac{1}{\epsilon}(\log|\mathcal{H}| + \log(1/\delta))$$

### VC Dimension
- Linear classifiers in $\mathbb{R}^d$: $d+1$
- Decision trees: $\Omega(n)$ for $n$ leaves

## References
- Hastie et al: Elements of Statistical Learning
- Bishop: Pattern Recognition and Machine Learning
