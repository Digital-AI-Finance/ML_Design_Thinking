# Week 5: Topic Modeling for Ideation

## Discovering Innovation Through Hidden Themes

### Overview
Week 5 explores topic modeling techniques to transform large collections of unstructured text into actionable innovation insights. Students learn to apply LDA, NMF, and modern approaches to discover hidden patterns in customer feedback, innovation workshops, and market research data.

### Learning Objectives
- Master topic modeling algorithms (LDA, NMF, LSA, BERTopic)
- Build end-to-end topic discovery pipelines
- Transform topics into design opportunities
- Apply topic models to real-world ideation challenges
- Create data-driven innovation roadmaps

## Modular Structure (50 Total Slides)

### File Organization
```
Week_05/
├── 20250925_1906_main.tex      # Master controller
├── part1_foundation.tex        # Part 1: Foundation (8 slides)
├── part2_algorithms.tex        # Part 2: Algorithms (10 slides)
├── part3_implementation.tex    # Part 3: Implementation (10 slides)
├── part4_design.tex            # Part 4: Design Applications (10 slides)
├── part5_practice.tex          # Part 5: Workshop & Practice (8 slides)
├── appendix_mathematics.tex    # Mathematical foundations (4 slides)
├── compile.py                  # Automated compilation with cleanup
├── charts/                     # 14+ visualizations
├── scripts/                    # Chart generation scripts
└── archive/                    # Auxiliary file storage
```

### Content Breakdown

#### Part 1: Foundation - From Chaos to Clarity (8 slides)
- The ideation challenge visualization
- Traditional vs ML-enhanced ideation
- What is topic modeling?
- Real-world impact (Netflix, 3M, IDEO, P&G, Spotify, Amazon)
- Innovation funnel with topics
- Types of topic models overview
- Learning objectives
- Foundation summary

#### Part 2: Algorithms - Topic Modeling Techniques (10 slides)
- Latent Dirichlet Allocation (LDA) overview
- LDA intuition with restaurant reviews
- Non-negative Matrix Factorization (NMF)
- Latent Semantic Analysis (LSA)
- Algorithm comparison matrix
- Choosing the number of topics
- Evaluating topic quality
- Modern approach: BERTopic
- Preprocessing for topic models
- Algorithm summary

#### Part 3: Implementation - Building Models (10 slides)
- Data collection strategy
- LDA implementation with Gensim
- NMF implementation with Scikit-learn
- Topic visualization techniques
- Production pipeline architecture
- Hyperparameter optimization
- Handling different text types
- Real-time topic discovery
- Common implementation pitfalls
- Implementation checklist

#### Part 4: Design Applications - Topics to Innovation (10 slides)
- From topics to design opportunities
- Innovation opportunity mapping
- Cross-topic innovation
- Persona-topic alignment
- Emerging trend detection
- Feature prioritization matrix
- Topic-driven innovation workshops
- Competitive intelligence through topics
- ROI of topic-driven innovation
- Design application framework

#### Part 5: Practice - Innovation Mining Workshop (8 slides)
- Case study: Smart home innovation
- Workshop exercise with startup dataset
- Step-by-step implementation guide
- Analyzing results
- Innovation opportunities found
- Common challenges and solutions
- Best practices summary
- Key takeaways

#### Appendix: Mathematical Foundations (4 slides)
- LDA mathematical framework
- NMF matrix factorization
- Topic coherence metrics
- Information theory for topics

## Key Visualizations

### Core Charts Generated
1. **topic_discovery_landscape** - Opening visualization
2. **topic_word_distribution** - Word probability distributions
3. **innovation_funnel_topics** - Innovation process funnel
4. **lda_plate_notation** - LDA graphical model
5. **algorithm_comparison_matrix** - Algorithm performance comparison
6. **topic_coherence_plot** - Optimal topic selection
7. **topic_quality_dashboard** - Quality metrics dashboard
8. **lda_document_topics** - Document-topic distributions
9. **nmf_decomposition** - NMF matrix visualization
10. **topics_to_opportunities** - Transformation process
11. **innovation_opportunity_map** - Quadrant analysis
12. **persona_topic_heatmap** - Persona alignment
13. **trend_evolution** - Topic trends over time
14. **workshop_results** - Practice results visualization

## How to Use

### Compilation
```bash
# Automated compilation with cleanup (RECOMMENDED)
python compile.py

# Manual compilation
pdflatex 20250925_1906_main.tex
pdflatex 20250925_1906_main.tex  # Run twice for references
```

### Generate Charts
```bash
cd scripts
python create_topic_charts.py
python create_additional_charts.py
```

### Requirements
- LaTeX with Beamer
- Python 3.7+
- Libraries: numpy, matplotlib, seaborn, pandas, scipy
- No actual ML models needed (visualizations only)

## Key Concepts Covered

### Technical Components
- Topic modeling fundamentals
- LDA generative process and inference
- NMF matrix factorization
- LSA and semantic analysis
- Hyperparameter optimization
- Coherence and quality metrics
- Production pipeline design

### Design Applications
- Innovation opportunity discovery
- Cross-topic innovation strategies
- Persona-topic alignment
- Trend detection and forecasting
- Feature prioritization frameworks
- Workshop facilitation techniques
- Competitive intelligence

### Practical Skills
- Gensim for LDA implementation
- Scikit-learn for NMF
- pyLDAvis for visualization
- Topic quality evaluation
- Real-time topic modeling
- Scaling to production

## Workshop Exercise

**Innovation Mining Challenge**
- Dataset: 5,000 startup descriptions
- Tasks:
  1. Preprocess and clean text
  2. Build topic model (LDA/NMF)
  3. Extract and interpret topics
  4. Identify innovation opportunities
  5. Create opportunity roadmap

- Duration: 45 minutes
- Deliverables:
  - 10 discovered innovation themes
  - Top 3 prioritized opportunities
  - Action plan for development

## Industry Applications

### Real-World Use Cases
1. **Netflix**: 35 micro-genres discovered, +18% engagement
2. **3M**: 47 new product ideas, $12M revenue
3. **IDEO**: 60% faster insights, 3x more patterns
4. **P&G**: 23 unmet needs found, 5 new products
5. **Spotify**: 1,500 micro-moods, +25% listening time
6. **Amazon**: Product improvements, -30% returns

### ROI Metrics
- **Efficiency**: 70% faster ideation
- **Quality**: 40% better product-market fit
- **Business**: 28% revenue growth
- **Speed**: 45% faster time-to-market

## Learning Outcomes

By the end of Week 5, students will be able to:
1. Build and optimize topic models for ideation
2. Transform unstructured text into innovation insights
3. Apply topic modeling to real business challenges
4. Create data-driven innovation roadmaps
5. Facilitate topic-driven workshops
6. Measure and communicate ROI of topic modeling

## Prerequisites
- Basic Python programming
- Understanding of unsupervised learning
- Familiarity with text processing
- No deep NLP knowledge required

## Resources
- Gensim documentation for LDA
- Scikit-learn NMF tutorials
- pyLDAvis for interactive visualization
- Original LDA paper (Blei et al., 2003)
- BERTopic documentation

## Next Week Preview
**Week 6: Generative AI for Prototyping**
- GPT for design concept generation
- DALL-E for visual prototyping
- Code generation for MVPs
- AI-assisted creativity workflows

## Notes for Instructors
- Focus on practical applications over mathematical details
- Use interactive visualizations (pyLDAvis) for engagement
- Workshop exercise is critical for hands-on learning
- Emphasize the design-to-innovation pipeline
- Show real examples from industry

---

*Created: September 2025*
*Course: Machine Learning for Smarter Innovation*
*Institution: BSc Design & AI Program*
---

## Meta-Knowledge Integration

**NEW (2025-10-03)**: Week 5 now includes systematic meta-knowledge slide:

**Topic Modeling Method Selection**:
- Decision tree chart: `topic_modeling_decision.pdf`
- When to use LDA vs NMF vs BERTopic
- Judgment criteria: Interpretable topics, speed & simplicity, best semantics
- Additional considerations: Dataset size, topic count, languages, real-time needs, coherence, computation
- Principle: "LDA for interpretable topics, NMF for speed, BERTopic for best coherence and modern semantics"
- Bottom note: "Judgment criteria enable systematic topic modeling selection - balance interpretability, computational cost, and semantic quality"

This meta-knowledge slide follows Week 9-10 pedagogical framework standard.

---

## Version History

- **2025-10-03**: Pedagogical framework upgrade
  - Added topic modeling method selection meta-knowledge slide
  - Created decision tree chart: `topic_modeling_decision.pdf`
  - Enhanced README with meta-knowledge documentation
  - Week 5 now has systematic judgment criteria

- **2025-09-25**: Latest revision
  - 5-part modular structure
  - Latest main: 20250925_1906_main.tex
  - Innovation funnel integration
  - Modern approaches (BERTopic) alongside classics

---

**Status**: Week 5 is pedagogically compliant (meta-knowledge slide ✅). Topic modeling foundation complete.
**Last Updated**: 2025-10-03
