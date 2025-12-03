# Week 10: A/B Testing & Iterative Improvement
## Machine Learning for Smarter Innovation

**Status:** V1.1 Complete (51 slides, 16 charts, 3 handouts) - Pedagogical framework compliant
**Last Updated:** 2025-10-03

---

## Overview

Week 10 closes the Design Thinking innovation loop by demonstrating how A/B testing and iterative experimentation enable continuous improvement. This week completes the journey from Empathy (Weeks 1-3) through Define, Ideate, Prototype, and Test (Weeks 4-9) to **Iterate** - the final phase that makes innovation sustainable.

**Core Message:** Companies that iterate faster learn faster and win. Spotify runs 1,000+ experiments per year. Amazon attributes 35% of revenue to recommendation experiments. The best ML systems are not built once - they are continuously improved through rigorous experimentation.

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Design rigorous A/B tests**
   - Formulate testable hypotheses
   - Calculate required sample sizes
   - Define primary metrics and guardrails
   - Plan experiment duration

2. **Execute statistical analysis**
   - Classical (frequentist) hypothesis testing
   - Bayesian posterior inference
   - Confidence/credible interval interpretation
   - Multiple testing correction

3. **Implement production experiments**
   - Random assignment with consistent hashing
   - Real-time monitoring and guardrails
   - Canary deployment strategies
   - Rollback mechanisms

4. **Make data-driven decisions**
   - Interpret p-values and effect sizes
   - Balance statistical and practical significance
   - Detect Simpson's paradox
   - Build experimentation culture

5. **Deploy advanced techniques**
   - Multi-armed bandits for exploration-exploitation
   - Thompson sampling for adaptive learning
   - Sequential testing for early stopping
   - Causal inference with DAGs

---

## Course Materials

### Slide Structure

**Total Slides:** 51 (5 modular parts) - V1.1 updated for pedagogical framework compliance

**Part 1: Foundation (10 slides)**
- The iteration advantage: Why speed wins
- Model decay and concept drift
- Cost of being wrong vs cost of being slow
- Building an experimentation mindset
- Case studies: Spotify, Netflix, Amazon

**Part 2: Statistical Algorithms (11 slides)**
- Hypothesis formulation and null/alternative
- Sample size calculation and power analysis
- Classical tests: Z-test (proportions), t-test (continuous)
- Bayesian inference: Beta-Binomial model
- Sequential testing: O'Brien-Fleming boundaries
- Multi-armed bandits: Thompson sampling
- Causal inference: DAGs and identification

**Part 3: Implementation (10 slides)**
- Experiment infrastructure architecture
- Consistent hashing for treatment assignment
- Real-time monitoring and guardrails
- Python implementations:
  - Classical A/B test (scipy.stats)
  - Bayesian A/B test (posterior sampling)
  - Thompson sampling bandit
- Canary deployment and blue-green rollout
- Error handling and automatic rollback

**Part 4: Design Thinking Integration (11 slides)**
- Building experimentation culture (Jeff Bezos: "double experiments, double invention")
- Metric selection: North Star, primary, guardrails
- Communicating results to stakeholders
- Ethics: Informed consent, transparency, harm prevention ("What NOT to Test")
- Experiment velocity: The compounding effect
- Tools & infrastructure: Open source vs enterprise
- **NEW (V1.1)**: When to use A/B testing - Judgment criteria for validation method selection
- Design summary: Key principles and strategic insights

**Part 5: Practice Workshop (9 slides)**
- E-commerce recommendation engine A/B test
- Challenge: Compare 3 algorithms (Collaborative Filtering, Content-Based, Hybrid)
- Dataset: 100,000 daily users, 5% baseline CTR
- Tasks:
  1. Experiment design (hypothesis, metrics, guardrails)
  2. Power analysis and sample size calculation
  3. Simulation of A/B test
  4. Statistical analysis (frequentist + Bayesian)
  5. Guardrail check (latency, error rate, revenue)
  6. Rollout plan and decision memo
- Deliverable: Jupyter notebook with complete analysis

---

## Visualizations

**Total Charts:** 16 (generated via Python scripts)

### Core A/B Testing (Charts 1-6)
1. **ab_testing_fundamentals.pdf**
   - Left: Bar chart of control vs treatment conversion rates
   - Right: Temporal stability check (conversion rate over 14 days)

2. **statistical_power_curves.pdf**
   - Power vs sample size for different effect sizes
   - Shows 80% power threshold

3. **type_i_ii_errors.pdf**
   - Left: Type I error (false positive) visualization
   - Right: Type II error (false negative) and power

4. **confidence_intervals.pdf**
   - 30 repeated experiments with 95% CIs
   - ~5% miss the true value (by design)

5. **bayesian_ab_posterior.pdf**
   - Left: Posterior distributions for control and treatment
   - Right: Posterior distribution of lift

6. **sequential_testing_boundaries.pdf**
   - O'Brien-Fleming, Pocock, and naive boundaries
   - Shows conservative early stopping

### Multi-Armed Bandits (Charts 7-8)
7. **multi_armed_bandit_exploration.pdf**
   - Left: Cumulative pulls per arm over time
   - Right: Cumulative regret vs optimal strategy

8. **thompson_sampling_demo.pdf**
   - 4-panel evolution of Beta posteriors
   - Shows learning progression (rounds 10, 30, 50, 100)

### Causal Inference & Production (Charts 9-12)
9. **causal_inference_dag.pdf**
   - Directed acyclic graph showing confounders, mediators, colliders
   - Illustrates why randomization breaks confounding

10. **experiment_velocity_dashboard.pdf**
    - 4-panel dashboard: Monthly experiments, win rate, duration, by category
    - Shows experimentation maturity metrics

11. **metric_tree_example.pdf**
    - Hierarchical metric decomposition
    - North Star → Primary → Secondary → Guardrails

12. **canary_deployment_timeline.pdf**
    - Top: Progressive rollout stages (1% → 5% → 25% → 50% → 100%)
    - Bottom: Real-time guardrail monitoring (error rate, latency)

### Pitfalls & Decisions (Charts 13-15)
13. **simpsons_paradox.pdf**
    - Left: Treatment wins both segments
    - Right: Control wins overall (imbalanced samples)

14. **experiment_decision_matrix.pdf**
    - 5 scenarios with decision rules (SHIP/ITERATE/STOP/ROLLBACK)

15. **continuous_improvement_loop.pdf**
    - 6-stage cycle: Observe → Hypothesize → Design → Implement → Analyze → Decide

16. **validation_method_decision.pdf** **NEW (V1.1)**
    - Decision tree for when to use A/B testing vs alternative validation methods
    - 3 branches: High traffic (A/B testing), Medium traffic (Adaptive methods), Low traffic (Alternatives)
    - Includes "Skip A/B Testing When" and "Use A/B Testing When" criteria
    - Helps match validation method to problem characteristics (traffic, stakes, reversibility)

---

## Handouts

### Handout 1: Basic A/B Testing (200 lines)
**Target Audience:** Non-technical stakeholders, product managers
**Level:** No code, no mathematics

**Contents:**
- What is A/B testing? (5-minute explanation)
- Why it matters: Companies that test vs those that don't
- The five-step process: Observe, Hypothesize, Design, Implement, Analyze
- Common pitfalls and how to avoid them
- Decision framework: When to ship/iterate/stop/rollback
- Key metrics explained (conversion rate, ARPU, retention)
- Building experimentation culture
- Real-world success stories (Netflix, Booking.com, Etsy)
- Your first A/B test checklist
- FAQ

**Use Case:** Onboarding product managers to experimentation mindset

### Handout 2: Intermediate Experimentation (400 lines)
**Target Audience:** Data scientists, ML engineers
**Level:** Python + statistics required

**Contents:**
- Sample size calculation (statsmodels)
- Z-test for proportions (complete implementation)
- T-test for continuous metrics
- Bayesian A/B test with Beta-Binomial conjugate prior
- Multi-armed bandits: Thompson sampling implementation
- Sequential testing: O'Brien-Fleming boundaries
- Complete A/B testing pipeline class
- Common pitfalls: Multiple testing, non-stationarity
- Variance reduction: CUPED, stratified randomization

**Use Case:** Implementing first A/B test in Python

### Handout 3: Advanced Causal Inference (500 lines)
**Target Audience:** ML researchers, senior engineers
**Level:** Advanced statistics, production systems

**Contents:**
- Potential outcomes framework (Rubin Causal Model)
- Mathematical foundations: ATE, CATE, SUTVA
- Directed acyclic graphs and d-separation
- Heterogeneous treatment effects: S-Learner, T-Learner, X-Learner
- Variance reduction: CUPED mathematical derivation
- Stratified randomization with weighted ATE
- Multi-objective optimization: Pareto frontiers
- Network effects and cluster randomization
- Production systems architecture:
  - Distributed assignment service
  - Real-time guardrail monitoring
  - Statistical power tracking
- Contextual bandits: LinUCB implementation

**Use Case:** Building production-grade experimentation platform

---

## Workshop Exercise

### E-Commerce Recommendation Engine A/B Test

**Scenario:**
You are a data scientist at an e-commerce company with 100,000 daily active users. Current recommendation engine has 5% click-through rate. You want to compare 3 new algorithms:

1. **Collaborative Filtering** (CF): User-user similarity
2. **Content-Based** (CB): Item-item similarity
3. **Hybrid**: Weighted combination of CF and CB

**Your Task:**
Design and analyze an A/B/C/D test to determine which algorithm to deploy.

**Deliverables:**
1. **Experiment Design Document (15 minutes)**
   - Hypothesis for each algorithm
   - Primary metric: CTR
   - Secondary metrics: Conversion rate, revenue per user
   - Guardrails: Page load time, error rate, bounce rate
   - Sample size calculation
   - Duration plan

2. **Power Analysis (10 minutes)**
   - Minimum detectable effect: 10% relative lift (5.0% → 5.5%)
   - Significance level: α = 0.05
   - Power: 80%
   - Calculate required sample size per group

3. **Simulation (15 minutes)**
   - Generate synthetic data for 4 groups
   - Control: 5.0% CTR
   - CF: 5.3% CTR
   - CB: 5.1% CTR
   - Hybrid: 5.6% CTR

4. **Statistical Analysis (15 minutes)**
   - Frequentist: Z-test for each algorithm vs control
   - Bayesian: Posterior P(Algorithm > Control)
   - Bonferroni correction for multiple testing
   - Effect size and confidence intervals

5. **Guardrail Check (5 minutes)**
   - Page load time: p95 < 2 seconds
   - Error rate: < 1%
   - Revenue per user: Not significantly decreased

6. **Decision Memo (5 minutes)**
   - Which algorithm to deploy?
   - Rollout plan (canary → full)
   - Monitoring plan
   - Rollback criteria

**Expected Results:**
- Hybrid algorithm wins with 12% relative lift
- Statistically significant (p < 0.001)
- Guardrails pass
- Recommendation: Deploy to 100% via canary rollout

---

## Technical Implementation

### Dependencies

```python
# Core ML/Statistics
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
statsmodels>=0.13.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Optional (for advanced examples)
pymc3>=3.11.0  # Bayesian inference
```

### File Structure

```
Week_10/
├── 20250927_1218_main.tex          # Master LaTeX file (orchestrates 5 parts)
├── part1_foundation.tex            # 10 slides: Why iterate, model decay
├── part2_algorithms.tex            # 11 slides: Statistical methods
├── part3_implementation.tex        # 10 slides: Python implementations
├── part4_design.tex                # 10 slides: Culture, communication, ethics
├── part5_practice.tex              # 9 slides: Recommendation engine workshop
├── compile.py                      # Automated compilation with cleanup
├── charts/                         # 15 PDF/PNG visualizations
│   ├── ab_testing_fundamentals.pdf
│   ├── statistical_power_curves.pdf
│   ├── type_i_ii_errors.pdf
│   ├── confidence_intervals.pdf
│   ├── bayesian_ab_posterior.pdf
│   ├── sequential_testing_boundaries.pdf
│   ├── multi_armed_bandit_exploration.pdf
│   ├── thompson_sampling_demo.pdf
│   ├── causal_inference_dag.pdf
│   ├── experiment_velocity_dashboard.pdf
│   ├── metric_tree_example.pdf
│   ├── canary_deployment_timeline.pdf
│   ├── simpsons_paradox.pdf
│   ├── experiment_decision_matrix.pdf
│   └── continuous_improvement_loop.pdf
├── scripts/
│   └── generate_all_charts.py      # Chart generation script
├── handouts/
│   ├── handout_1_basic_ab_testing.md
│   ├── handout_2_intermediate_experimentation.md
│   └── handout_3_advanced_causal_inference.md
├── archive/                        # Auxiliary files (.aux, .log, etc.)
└── README.md                       # This file
```

---

## Compilation Instructions

### Using compile.py (Recommended)

```bash
cd Week_10
python compile.py
```

**What it does:**
1. Auto-detects latest main.tex file
2. Runs pdflatex twice (for references and TOC)
3. Moves auxiliary files to archive/aux_TIMESTAMP/
4. Opens PDF automatically (Windows)
5. Displays full path to compiled PDF

### Manual Compilation

```bash
cd Week_10
pdflatex 20250927_1218_main.tex
pdflatex 20250927_1218_main.tex  # Second pass for references

# Clean up auxiliary files
mkdir -p archive/temp
move *.aux *.log *.nav *.snm *.toc *.vrb *.out archive/temp/
```

### Chart Generation

```bash
cd Week_10/scripts
python generate_all_charts.py
```

Generates all 16 charts in both PDF (300 dpi) and PNG (150 dpi) formats.

---

## Key Concepts

### Statistical Foundations

**Hypothesis Testing:**
- Null hypothesis (H0): No difference between groups
- Alternative hypothesis (H1): Treatment differs from control
- P-value: Probability of observing effect if H0 is true
- Significance level (α): Typically 0.05 (5% false positive rate)

**Sample Size Calculation:**
```
n = (z_α/2 + z_β)^2 * 2p(1-p) / δ^2
```
Where:
- z_α/2 = 1.96 (for 95% confidence)
- z_β = 0.84 (for 80% power)
- p = baseline conversion rate
- δ = minimum detectable effect

**Example:**
Detect 1 percentage point increase from 5% to 6%:
- Required sample size: 8,844 per group (17,688 total)

### Bayesian Framework

**Beta-Binomial Model:**
- Prior: Beta(α=1, β=1) = Uniform(0, 1)
- Likelihood: Binomial(n, p)
- Posterior: Beta(α + conversions, β + non-conversions)

**Key Advantage:**
Direct probability statements: "95% probability Treatment is better than Control"

**Compared to Frequentist:**
- Frequentist: P(data | H0) -- probability of seeing this data if null is true
- Bayesian: P(H1 | data) -- probability hypothesis is true given data

### Multi-Armed Bandits

**Exploration-Exploitation Trade-off:**
- Exploration: Try different arms to learn their rewards
- Exploitation: Use best-known arm to maximize immediate reward

**Thompson Sampling:**
1. Maintain Beta posterior for each arm
2. Sample from each posterior
3. Pull arm with highest sample
4. Update posterior with observed reward

**Advantage over A/B Testing:**
- Dynamically allocates more traffic to winners
- Reduces opportunity cost of showing suboptimal variants
- Can run continuously without fixed sample size

### Causal Inference

**Why Randomization Works:**
- Breaks all confounding paths
- Makes treatment assignment independent of potential outcomes
- Enables causal (not just associational) interpretation

**Directed Acyclic Graphs (DAGs):**
- Graphical representation of causal relationships
- Identify confounders, mediators, colliders
- Determine adjustment sets for causal identification

**Example:**
```
    Age
   /   \
  v     v
Treatment → Outcome
```
Age confounds the relationship. Must control for Age or randomize.

---

## Common Pitfalls

### 1. Stopping Too Early

**Mistake:** Check results daily, stop when p < 0.05.

**Why it's wrong:** If you check 20 times, you'll see p < 0.05 by random chance even if there's no real effect.

**Solution:**
- Pre-commit to sample size and duration
- Use sequential testing with adjusted thresholds (O'Brien-Fleming)
- Check only at planned interim analyses

### 2. P-Hacking (Multiple Testing)

**Mistake:** Run 10 tests, only report the 1 that shows p < 0.05.

**Why it's wrong:** With α = 0.05, we expect 1 in 20 tests to be "significant" by chance.

**Solution:**
- Bonferroni correction: Use α/n for n tests
- Pre-register hypotheses before seeing data
- Report all tests, not just significant ones

### 3. Ignoring Simpson's Paradox

**Mistake:** Treatment wins for both new and returning users, but loses overall.

**Why it happens:** Imbalanced sample sizes across segments.

**Solution:**
- Always analyze by key segments
- Use stratified randomization
- Weight segment-specific effects appropriately

### 4. Confusing Statistical and Practical Significance

**Mistake:** P-value < 0.05, so we ship it.

**Why it's wrong:** With large samples, tiny (irrelevant) effects become "significant."

**Solution:**
- Define minimum practical effect size upfront
- Consider cost-benefit analysis
- Ask: "Is a 0.5% lift worth the engineering effort?"

### 5. Ignoring Guardrails

**Mistake:** Primary metric improves, ship immediately.

**Why it's wrong:** May have degraded critical secondary metrics (latency, errors, satisfaction).

**Solution:**
- Define guardrails before experiment
- Automatic rollback if guardrails violated
- Multi-objective decision framework

---

## Industry Best Practices

### Design Phase
- [ ] Clear hypothesis with expected direction
- [ ] Primary metric defined and measurable
- [ ] Guardrails identified (typically 3-5)
- [ ] Sample size calculated (use online calculator or statsmodels)
- [ ] Minimum 1-week duration (capture day-of-week effects)
- [ ] Stakeholder alignment on decision criteria
- [ ] Pre-registration of analysis plan

### Implementation Phase
- [ ] Random assignment is truly random (check 50/50 split)
- [ ] Assignment is consistent (same user always sees same variant)
- [ ] Both variants have no technical errors
- [ ] Guardrails monitored in real-time
- [ ] Canary rollout for high-risk changes
- [ ] Automatic rollback criteria defined

### Analysis Phase
- [ ] Statistical significance calculated (p-value)
- [ ] Practical significance assessed (effect size)
- [ ] Confidence interval reported (not just point estimate)
- [ ] Guardrails reviewed (did we break anything?)
- [ ] Segment analysis (any Simpson's paradox?)
- [ ] Multiple testing correction if applicable

### Decision Phase
- [ ] Document decision rationale (ship/iterate/stop and why)
- [ ] Rollout plan if shipping (canary → 10% → 50% → 100%)
- [ ] Monitoring plan for post-launch
- [ ] Rollback criteria for production
- [ ] Share results with team (even null results)

---

## Connection to Course Narrative

Week 10 completes the Design Thinking innovation cycle:

**Weeks 1-3: Empathize**
- Clustering (Week 1): Discover user segments
- Advanced Clustering (Week 2): Hierarchical patterns
- NLP (Week 3): Emotional understanding

**Week 4: Define**
- Classification: Categorize problems and opportunities

**Week 5: Ideate**
- Topic Modeling: Generate thousands of ideas

**Week 6: Prototype**
- Generative AI: Rapid prototyping with GenAI tools

**Weeks 7-9: Test**
- Ethics (Week 7): Responsible AI evaluation
- Reliability (Week 8): Structured outputs and validation
- Multi-Metric (Week 9): Comprehensive model assessment

**Week 10: Iterate**
- **A/B Testing:** Continuous improvement through experimentation
- **Closing the Loop:** Learn, ship, measure, repeat
- **Sustainable Innovation:** Build systems that evolve

**The Innovation Diamond Revisited:**
- Start: 1 challenge
- Diverge: 5,000 ideas (Weeks 1-5)
- Converge: 5 prototypes (Week 6)
- Test: 1 validated solution (Weeks 7-9)
- **Iterate: Continuous improvement (Week 10)**

---

## Assessment

### Knowledge Check Questions

1. **Sample Size Calculation**
   - Baseline: 5% conversion
   - Target: 6% conversion (1 percentage point lift)
   - α = 0.05, power = 80%
   - How many users per group?

2. **P-Value Interpretation**
   - You run an A/B test and get p = 0.03
   - What does this mean?
   - Should you ship the treatment?

3. **Bayesian vs Frequentist**
   - "95% probability Treatment is better than Control"
   - Is this a Bayesian or Frequentist statement?
   - How would the other framework express this?

4. **Simpson's Paradox**
   - Treatment wins for segment A (10% vs 8%)
   - Treatment wins for segment B (6% vs 5%)
   - Can Control win overall? Explain.

5. **Multi-Armed Bandits**
   - You have 3 recommendation algorithms
   - Traditional A/B/C test: 33% traffic each
   - Thompson Sampling: Adaptive allocation
   - Which finishes faster? Why?

### Practical Assignment

**Task:** Implement a complete A/B testing pipeline in Python

**Requirements:**
1. Sample size calculator function
2. Z-test for proportions
3. Bayesian Beta-Binomial inference
4. Guardrail monitoring
5. Decision framework

**Bonus:**
- Thompson sampling multi-armed bandit
- CUPED variance reduction
- Stratified analysis

---

## Further Reading

### Books
- **Trustworthy Online Controlled Experiments** by Kohavi, Tang, Xu
  - Industry bible for A/B testing
  - Microsoft, Google, Amazon practices

- **Causality: Models, Reasoning, and Inference** by Judea Pearl
  - Theoretical foundations of causal inference
  - DAGs and do-calculus

- **Causal Inference for Statistics, Social, and Biomedical Sciences** by Imbens & Rubin
  - Potential outcomes framework
  - Advanced econometric methods

### Papers
- **Thompson Sampling**: "An Empirical Evaluation of Thompson Sampling" (Chapelle & Li, 2011)
- **CUPED**: "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (Deng et al., 2013)
- **Sequential Testing**: "Group Sequential Methods with Applications to Clinical Trials" (Jennison & Turnbull, 1999)

### Online Resources
- **Evan Miller's A/B Testing Tools**: https://www.evanmiller.org/ab-testing/
- **Optimizely Stats Engine**: https://www.optimizely.com/insights/blog/stats-engine/
- **Netflix Tech Blog**: https://netflixtechblog.com (search "experimentation")

---

## Credits

**Course:** Machine Learning for Smarter Innovation
**Level:** BSc, Week 10 of 10
**Duration:** 90 minutes lecture + 60 minutes workshop
**Prerequisites:** Weeks 1-9 (Design Thinking cycle)

**Tools Used:**
- LaTeX/Beamer (Madrid theme, 8pt)
- Python (numpy, pandas, scipy, scikit-learn, matplotlib)
- Statistical libraries (statsmodels, scipy.stats)

---

## Version History

- **2025-10-03 V1.1**: Pedagogical framework compliance upgrade
  - Added "When to Use A/B Testing" judgment criteria slide to Part 4
  - New chart: validation_method_decision.pdf (decision tree for validation method selection)
  - Total slides: 50 → 51
  - Total charts: 15 → 16
  - Satisfies pedagogical_framework_Template.md requirements (Anti-Pattern #5)
  - Matches Week 8 V2.1 and Week 9 V1.1 meta-knowledge standard

- **2025-09-27 V1.0**: Initial release
  - 50 slides across 5 modular parts
  - 15 visualization charts generated
  - 3 skill-level handouts (basic, intermediate, advanced)
  - E-commerce recommendation engine workshop
  - Complete with verified industry statistics (Spotify, Netflix, Amazon)
  - Ready for teaching

**Last Updated:** 2025-10-03
**Status:** V1.1 Complete - Pedagogical framework compliant and ready for teaching