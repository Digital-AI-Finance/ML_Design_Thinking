# Week 00 Finance Theory Handout Level 2

## Level 2 Focus
Python for quantitative finance

## Topics
- Portfolio optimization with ML
- Risk management (VaR, CVaR)
- Algorithmic trading strategies
- Credit scoring and fraud detection
- Regulatory compliance (SR 11-7, MiFID II)

## Portfolio Optimization

\`\`\`python
import numpy as np
import cvxpy as cp

# Mean-variance optimization
mu = np.array([0.10, 0.12, 0.08])  # Expected returns
Sigma = np.array([[0.04, 0.01, 0.00],
                  [0.01, 0.09, 0.02],
                  [0.00, 0.02, 0.16]])  # Covariance

w = cp.Variable(3)
risk = cp.quad_form(w, Sigma)
ret = mu @ w

prob = cp.Problem(cp.Maximize(ret - 0.5*risk),
                 [cp.sum(w) == 1, w >= 0])
prob.solve()
print(f"Optimal weights: {w.value}")
\`\`\`

## VaR Calculation
\`\`\`python
from scipy.stats import norm

# Parametric VaR (assumes normal returns)
confidence = 0.95
portfolio_value = 1000000
volatility = 0.20
holding_period = 1

VaR = portfolio_value * norm.ppf(confidence) * volatility
print(f"95% VaR: \${VaR:,.0f}")
\`\`\`



## Regulatory Requirements
- SR 11-7: Model validation, ongoing monitoring
- MiFID II: Algorithmic trading controls
- Basel III: Credit risk models
