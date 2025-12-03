# Week 1: How do we truly understand our users?

## ML Concept: Clustering (K-means)
## Design Stage: Empathize - Discover

### Learning Objectives
By the end of this week, students will be able to:
1. Apply K-means clustering to discover user segments
2. Understand the concept of unsupervised learning
3. Use the elbow method to choose optimal K
4. Transform overwhelming data into actionable insights
5. Combine ML techniques with design empathy

### The Problem
You have 10,000 product reviews but can only read 100. How do you understand ALL your users, not just a sample?

### The Solution
K-means clustering automatically discovers natural groups in your data, revealing user segments you never knew existed.

### Course Materials

#### Slides
- `week01_slides.tex` - 30 slides covering theory and practice
- Compile with: `pdflatex week01_slides.tex`

#### Charts
All charts use real ML calculations, not made-up numbers:
- `clustering_demo.pdf` - Shows chaos to clarity transformation
- `kmeans_process.pdf` - Visualizes the 3-step algorithm
- `empathy_scale.pdf` - Demonstrates elbow method for choosing K
- `results_comparison.pdf` - Compares traditional vs ML approaches

#### Generating Charts
```bash
cd charts
python create_clustering_demo.py
python create_kmeans_process.py
python create_empathy_scale.py
python create_results_comparison.py
```

### Key Concepts Covered

#### ML/AI Topics
- Unsupervised learning
- K-means clustering algorithm
- Distance metrics (Euclidean)
- The elbow method
- Feature extraction from text
- Cluster interpretation

#### Design Thinking Topics
- Digital empathy at scale
- User segmentation
- Voice of customer analysis
- Pattern discovery
- Human-AI collaboration
- From data to insights

### Practical Exercise
Students will:
1. Load a dataset of product reviews
2. Preprocess text data
3. Apply K-means clustering
4. Interpret discovered segments
5. Create actionable design recommendations

### Real-World Example
**Spotify's "Cooking Music" Discovery**
- Used clustering on listening patterns
- Found 12% of users listen while cooking
- Pattern: Upbeat, clean lyrics, 30-45 min sessions
- Created new playlist category
- Result: 8M+ followers, 23% engagement increase

### Assessment Ideas
- Quiz: K-means steps and when to use clustering
- Lab: Cluster a provided dataset and interpret results
- Project: Find hidden segments in real user data

### The Bridge to Week 2
We discovered 5 user segments, but what are they actually FEELING? Next week: Using Transformers and NLP to understand emotion in text.

### Prerequisites
- Basic Python knowledge
- Understanding of mean/average
- No ML experience required

### Tools Needed
- Python 3.8+
- scikit-learn
- matplotlib
- numpy
- pandas

### Time Breakdown (55 minutes)
- Opening Problem: 5 min
- ML Concept (Clustering): 15 min
- Design Concept (Empathy): 15 min
- Integration Demo: 10 min
- Practice Exercise: 10 min

### Key Takeaways
1. ML can process 10,000 voices in the time it takes to read 100
2. Clustering reveals patterns invisible to human analysis
3. The "gift buyer" segment - 15% of users we never knew existed
4. From 2 weeks to 4 hours, $5,000 to $50
5. Empathy at scale is possible with human + machine partnership

### Resources for Further Learning
- [scikit-learn K-means documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Design Thinking and AI Integration](https://www.ideo.com/post/ai-needs-design-thinking)
- [Google's People + AI Guidebook](https://pair.withgoogle.com/)

### Instructor Notes
- Emphasize that clustering is exploration, not classification
- Show that ML amplifies, not replaces, human empathy
- Use the cookie sorting analogy for non-technical students
- Run the live demo even if it fails - debugging is learning
- Connect every technical concept back to design value

### Common Questions
**Q: How is this different from just categorizing manually?**
A: Manual categorization uses predetermined buckets. Clustering discovers natural groups you didn't know existed.

**Q: What if K-means finds weird segments?**
A: That's the point! "Weird" often means "opportunity". The gift buyer segment seemed weird until it drove millions in revenue.

**Q: Do we still need user interviews?**
A: Absolutely! ML shows WHAT patterns exist. Interviews reveal WHY they exist.

---

## Next Week Preview
**Week 2: What are users really saying?**
- ML Concept: NLP with Transformers (BERT)
- Design Stage: Empathize - Feel
- Problem: Understanding emotions in text
- Solution: BERT analyzes sentiment deeply