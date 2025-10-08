# Week 00 Finance Theory Handout Level 3

## Level 3 Focus
Mathematical finance + ML theory

## Topics
- Portfolio optimization with ML
- Risk management (VaR, CVaR)
- Algorithmic trading strategies
- Credit scoring and fraud detection
- Regulatory compliance (SR 11-7, MiFID II)



## Portfolio Theory

### Mean-Variance Optimization
\$\$\min_w w^T\Sigma w - \lambda \mu^T w\$\$
Subject to: \$w^T\mathbf{1} = 1\$

### Value at Risk (VaR)
\$\$\text{VaR}_\alpha = \inf\{x : P(L \leq x) \geq \alpha\}\$\$

### Conditional VaR (CVaR)
\$\$\text{CVaR}_\alpha = \mathbb{E}[L | L \geq \text{VaR}_\alpha]\$\$

### Black-Scholes + ML
Learn volatility surface:
\$\$\sigma_{\text{implied}}(K, T) = f_\theta(K, T, S, r)\$\$

## Regulatory Requirements
- SR 11-7: Model validation, ongoing monitoring
- MiFID II: Algorithmic trading controls
- Basel III: Credit risk models
