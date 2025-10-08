# Week 3: NLP for Emotional Context

## Understanding User Emotions Through Language

### Overview
Week 3 explores Natural Language Processing (NLP) and sentiment analysis to understand user emotions at scale. Students learn to transform unstructured text into actionable design insights using modern transformer models like BERT.

### Learning Objectives
- Master text preprocessing and NLP pipelines
- Implement sentiment and emotion analysis using BERT
- Apply NLP insights to design decisions
- Build production-ready text analysis systems
- Create emotional journey maps from user feedback

## Modular Structure (52 Total Slides)

### File Organization
```
Week_03/
├── 20250118_1630_main.tex      # Master controller
├── part1_foundation.tex        # Part 1: Foundation (8 slides)
├── part2_technical.tex         # Part 2: Technical Deep Dive (15 slides)
├── part3_implementation.tex    # Part 3: Implementation (11 slides)
├── part4_design.tex            # Part 4: Design Applications (12 slides)
├── part5_practice.tex          # Part 5: Case Study & Practice (9 slides)
├── appendix_mathematics.tex    # Mathematical foundations (4 slides)
├── compile.py                  # Automated compilation
├── charts/                     # 18+ visualizations
├── scripts/                    # Chart generation scripts
├── handouts/                   # 3 technical handouts
└── archive/                    # Auxiliary file storage
```

### Content Breakdown

#### Part 1: Foundation (8 slides)
- Opening power visualization: Emotion spectrum heatmap
- Why language reveals emotions
- Context-dependent meaning
- Sentiment vs emotion analysis
- NLP challenges at scale
- Learning objectives
- Real-world impact examples

#### Part 2: Technical Deep Dive (15 slides)
- Text preprocessing pipeline
- Tokenization and vectorization
- Word embeddings evolution
- Transformer architecture
- Attention mechanism
- BERT bidirectional understanding
- Pre-training and fine-tuning
- BERT variants comparison
- Sentiment analysis approaches
- Model evaluation metrics
- Cross-validation strategies
- Handling imbalanced data
- Domain adaptation

#### Part 3: Implementation (11 slides)
- Data collection sources
- Text cleaning techniques
- Feature engineering for NLP
- Model selection framework
- Hugging Face implementation
- Batch processing optimization
- Result interpretation
- Confidence calibration
- Production deployment
- API integration

#### Part 4: Design Applications (12 slides)
- Data to emotions bridge
- Voice of Customer analysis
- Review mining techniques
- Support ticket sentiment
- Social media monitoring
- Emotional journey mapping
- Pain point identification
- Delight moment discovery
- Sentiment-driven prioritization
- Emotional personalization
- Impact measurement

#### Part 5: Practice & Case Study (9 slides)
- Amazon review intelligence case study
- Data pipeline at scale
- Sentiment distribution analysis
- Aspect-based sentiment
- Key insights and patterns
- Business impact metrics
- Practice exercise: Twitter analysis
- Key takeaways

#### Appendix: Mathematical Foundations (4 slides)
- TF-IDF formulation
- Word2Vec objectives
- Attention mechanism mathematics
- Loss functions for classification

## Key Visualizations

### Core Charts (18 total)
1. **emotion_spectrum_heatmap.pdf** - Opening power visualization
2. **text_preprocessing_pipeline.pdf** - Processing flow
3. **word_embedding_space.pdf** - 3D embedding visualization
4. **sentiment_distribution.pdf** - Sentiment breakdown
5. **attention_visualization.pdf** - BERT attention weights
6. **model_comparison.pdf** - Algorithm performance
7. **confusion_matrix.pdf** - Classification results
8. **emotional_journey_map.pdf** - User emotion flow
9. **priority_matrix.pdf** - Issue prioritization
10. **language_emotion_flow.pdf** - Text to insights
11. **context_sentiment_examples.pdf** - Context importance
12. **emotion_wheel.pdf** - Emotion categories
13. **nlp_challenge_pyramid.pdf** - Scale challenges
14. **nlp_impact_metrics.pdf** - Business value
15. **bert_architecture.pdf** - Model structure
16. **amazon_case_overview.pdf** - Case study data
17. **voice_of_customer.pdf** - VoC framework
18. **data_sources_pyramid.pdf** - Data hierarchy

## How to Use

### Compilation
```bash
# Automated compilation with cleanup
python compile.py

# Or specify file directly
python compile.py 20250118_1630_main.tex

# Manual compilation
pdflatex 20250118_1630_main.tex
pdflatex 20250118_1630_main.tex  # Run twice
```

### Generate Charts
```bash
cd scripts
python create_nlp_charts.py
python create_additional_charts.py
```

### Chart Generation Requirements
- Python 3.7+
- numpy, matplotlib, seaborn
- pandas, scikit-learn
- No actual ML models needed (visualizations only)

## Handouts

### 1. Basic: Introduction to Sentiment Analysis
- Understanding sentiment polarity
- Rule-based vs ML approaches
- Simple TextBlob examples
- Hands-on with movie reviews

### 2. Intermediate: BERT for Sentiment
- Transformer architecture basics
- Using pre-trained models
- Fine-tuning for your domain
- Evaluation best practices

### 3. Advanced: Aspect-Based & Emotion Detection
- Multi-aspect sentiment analysis
- 8-emotion classification
- Sarcasm detection techniques
- Production pipeline design

## Practice Exercise

**Twitter Sentiment Analysis Challenge**
- Dataset: 5,000 product tweets
- Tasks:
  1. Preprocess and clean text
  2. Classify sentiment (positive/negative/neutral)
  3. Detect emotions (joy, anger, fear, etc.)
  4. Extract trending topics
  5. Analyze temporal patterns
  6. Generate design recommendations

- Duration: 2 hours
- Tools: Python, Hugging Face, Plotly
- Deliverable: Jupyter notebook with insights dashboard

## Industry Case Study

**Amazon Review Intelligence System**
- Challenge: Process 2M+ reviews daily across 35 languages
- Solution: BERT-based multi-level analysis pipeline
- Implementation:
  - Real-time sentiment scoring
  - Aspect-based analysis
  - Fake review detection
  - Trend identification
- Results:
  - 18% reduction in return rates
  - 35% fewer support tickets
  - 2x faster issue detection
  - $50M annual savings

## Key Concepts Covered

### Technical
- Text preprocessing and normalization
- Tokenization strategies (word, subword, BPE)
- Word embeddings (Word2Vec, GloVe, contextual)
- Transformer architecture and attention
- BERT and its variants
- Transfer learning and fine-tuning
- Evaluation metrics for NLP

### Design Applications
- Voice of Customer frameworks
- Emotional journey mapping
- Pain point prioritization
- Feature discovery from text
- Sentiment-driven design decisions
- Real-time experience adaptation

## Learning Outcomes

By the end of Week 3, students will be able to:
1. Build end-to-end NLP pipelines for sentiment analysis
2. Implement BERT for emotion detection
3. Transform text insights into design actions
4. Deploy production-ready text analysis systems
5. Create data-driven emotional journey maps
6. Prioritize design improvements based on sentiment data

## Prerequisites
- Basic Python programming
- Understanding of machine learning concepts
- Familiarity with classification tasks
- No deep NLP knowledge required

## Resources
- Hugging Face Transformers documentation
- BERT paper: "Attention is All You Need"
- TextBlob for simple sentiment analysis
- spaCy for advanced NLP tasks

## Next Week Preview
**Week 4: Classification for Problem Definition**
- Random Forests and decision trees
- Problem pattern recognition
- Feature importance analysis
- Building problem taxonomies

---

*Created: January 2025*
*Course: Machine Learning for Smarter Innovation*
*Institution: BSc Design & AI Program*
---

## Meta-Knowledge Integration

**NEW (2025-10-03)**: Week 3 now includes systematic meta-knowledge slide:

**NLP Method Selection** (part4_synthesis.tex):
- Decision tree chart: `nlp_method_decision.pdf`
- When to use Rule-based vs Traditional ML vs Transformers
- Judgment criteria: Speed/simplicity, balanced performance, maximum accuracy
- Principle: "Start simple (rules/traditional ML), upgrade to transformers only when context and nuance are critical"
- Bottom note: "Judgment criteria enable wise NLP method selection - balance accuracy needs against computational cost and data requirements"

This meta-knowledge slide follows Week 9-10 pedagogical framework standard.

---

## Version History

- **2025-10-03**: Pedagogical framework upgrade
  - Added NLP method selection meta-knowledge slide (part4_synthesis.tex line 451)
  - Created decision tree chart: `nlp_method_decision.pdf`
  - Enhanced README with meta-knowledge documentation
  - Archived 14 duplicate/old .tex files to archive/previous/
  - Total: 463 lines meta-knowledge slide

- **2025-09-29**: Act-based restructure
  - Converted to 4-act dramatic structure (part1-5)
  - Latest main: 20250929_1000_main.tex
  - 52 total slides

- **2025-09-19**: Original creation
  - Traditional 5-part modular structure
  - 75+ charts (PDF + PNG pairs)
  - Twitter sentiment workshop

---

## Next Steps

This README documents Week 3 structure. For comprehensive enhancement matching Weeks 1-2 format (500+ lines), key additions would include:
- Detailed learning objectives (5 categories)
- Chart-by-chart descriptions
- Handout content summaries
- Common questions/FAQ
- Technical prerequisites
- Quick start guide

**Status**: Week 3 is pedagogically compliant (meta-knowledge slide ✅, bottom notes ⏳). Core documentation complete.
**Last Updated**: 2025-10-03
