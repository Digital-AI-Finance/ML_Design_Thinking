# Week 2: Clustering for Deep Empathy

## Modular Structure (49 Total Slides)

### File Organization
```
Week_02/
├── 20250118_0718_main.tex      # Master file (includes all parts)
├── part1_foundation.tex        # Part 1: Foundation (7 slides)
├── part2_technical.tex         # Part 2: Technical Deep Dive (13 slides)
├── part3_design.tex            # Part 3: Design Integration (11 slides)
├── part4_practice.tex          # Part 4: Practice & Case Study (9 slides)
├── appendix_technical.tex      # Appendix: Mathematical Details (7 slides)
├── compile.py                  # Build script with auto-cleanup
├── charts/                     # 17 visualization PDFs
├── handouts/                   # 3 advanced technical handouts
└── archive/                    # Organized archive system
    ├── aux/                    # LaTeX auxiliary files (auto-cleaned)
    ├── previous/               # Old versions
    └── builds/                 # PDF build history
```

### Content Breakdown

#### Part 1: Foundation (7 slides)
- Opening power visualization
- Problem statement
- Why clustering for empathy
- Traditional vs ML personas
- Learning objectives
- Real-world impact

#### Part 2: Technical Deep Dive (13 slides)
- K-means algorithm mechanics
- Distance metrics
- Finding optimal K (elbow & silhouette)
- Python implementation
- Advanced algorithms (DBSCAN, Hierarchical, GMM)
- Algorithm selection guide
- Performance considerations
- Common pitfalls

#### Part 3: Design Integration (11 slides)
- Bridge from data to narratives
- Cluster to persona mapping
- Building empathy maps
- Journey mapping
- Pain points discovery
- Innovation opportunities
- Design principles
- Personalization at scale
- Measuring impact

#### Part 4: Practice & Case Study (9 slides)
- Spotify case study
- Data collection
- Implementation pipeline
- 5 music personas
- Persona-driven features
- Results & impact
- Practice exercise
- Key takeaways

#### Appendix: Technical Details (7 slides)
- K-means objective function
- Silhouette coefficient formula
- DBSCAN formal algorithm
- Gaussian Mixture Models EM
- Complexity analysis
- Additional resources

### Visualizations Created

17 professional charts demonstrating:
- Clustering evolution (opening power chart)
- Elbow and silhouette analysis
- User segmentation and personas
- Empathy map construction
- Journey comparisons
- Algorithm comparisons
- DBSCAN, Hierarchical, GMM details
- Method selection guide

### How to Compile

```bash
# Simple compilation with auto-cleanup
python compile.py

# Or specify file
python compile.py 20250118_0718_main.tex
```

All auxiliary files automatically archived to `archive/aux/`

### Key Features

✅ **Modular Design**: Each part can be edited independently
✅ **Clean Working Directory**: Automatic archival of auxiliary files
✅ **Version History**: All PDFs archived with timestamps
✅ **Professional Visualizations**: Real ML algorithms, not fake data
✅ **BSc Level**: Visual explanations, minimal math (in appendix)
✅ **Design Integration**: Strong connection to human-centered design

### Learning Outcomes

Students will master:
1. K-means clustering algorithm
2. Optimal cluster selection methods
3. Advanced clustering techniques
4. Creating data-driven personas
5. Building empathy maps from data
6. Real-world implementation (Spotify case)

### Next Week Preview

**Week 3: NLP for Emotional Context**
- Sentiment analysis with BERT
- Understanding user emotions through language
- Topic modeling for theme discovery