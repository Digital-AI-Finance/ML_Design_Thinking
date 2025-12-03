"""
Script to create README.md files for each topic folder.
"""

from pathlib import Path

BASE = Path(r"D:\Joerg\Research\slides\ML_Design_Thinking_16")
TOPICS = BASE / "topics"

TOPIC_INFO = {
    "ml_foundations": {
        "title": "ML Foundations",
        "description": "Introduction to machine learning concepts, learning paradigms, and the ML pipeline.",
        "sources": ["Week_00_Introduction_ML_AI", "Week_00a_ML_Foundations"],
        "objectives": [
            "Understand the three learning paradigms (supervised, unsupervised, reinforcement)",
            "Learn the ML pipeline from data to deployment",
            "Recognize bias-variance tradeoff",
            "Identify when ML is appropriate vs traditional programming"
        ]
    },
    "supervised_learning": {
        "title": "Supervised Learning",
        "description": "Prediction and classification using labeled data. Covers regression, decision trees, and ensemble methods.",
        "sources": ["Week_00b_Supervised_Learning"],
        "objectives": [
            "Understand regression vs classification problems",
            "Master linear regression (OLS)",
            "Learn decision tree algorithms",
            "Apply ensemble methods (Random Forest, Gradient Boosting)"
        ]
    },
    "unsupervised_learning": {
        "title": "Unsupervised Learning",
        "description": "Pattern discovery without labels. Covers clustering algorithms and dimensionality reduction.",
        "sources": ["Week_00c_Unsupervised_Learning"],
        "objectives": [
            "Understand clustering vs labeled classification",
            "Apply K-means and hierarchical clustering",
            "Master DBSCAN for density-based clustering",
            "Evaluate clusters using silhouette scores"
        ]
    },
    "clustering": {
        "title": "Clustering & Empathy",
        "description": "Applied clustering for user segmentation and persona creation in design thinking.",
        "sources": ["Week_01", "Week_02"],
        "objectives": [
            "Apply clustering to real-world user data",
            "Create data-driven personas",
            "Integrate ML insights into design process",
            "Select appropriate clustering algorithms"
        ]
    },
    "nlp_sentiment": {
        "title": "NLP & Sentiment Analysis",
        "description": "Natural language processing for understanding text data and user sentiment.",
        "sources": ["Week_03"],
        "objectives": [
            "Process and clean text data",
            "Apply sentiment analysis techniques",
            "Understand BERT and transformer architectures",
            "Build text classification pipelines"
        ]
    },
    "classification": {
        "title": "Classification",
        "description": "Classification algorithms for categorical prediction problems.",
        "sources": ["Week_04"],
        "objectives": [
            "Understand classification metrics (accuracy, precision, recall)",
            "Apply decision trees and random forests",
            "Handle imbalanced datasets",
            "Interpret feature importance"
        ]
    },
    "topic_modeling": {
        "title": "Topic Modeling",
        "description": "Discovering themes and topics in document collections using LDA and related methods.",
        "sources": ["Week_05"],
        "objectives": [
            "Understand topic modeling concepts",
            "Apply Latent Dirichlet Allocation (LDA)",
            "Evaluate topic quality",
            "Visualize topic distributions"
        ]
    },
    "generative_ai": {
        "title": "Generative AI",
        "description": "Large language models, prompt engineering, and generative applications.",
        "sources": ["Week_06", "Week_00e_Generative_AI"],
        "objectives": [
            "Understand transformer architecture",
            "Master prompt engineering techniques",
            "Apply generative AI for prototyping",
            "Evaluate generated content quality"
        ]
    },
    "neural_networks": {
        "title": "Neural Networks",
        "description": "Deep learning fundamentals, architectures, and training.",
        "sources": ["Week_00d_Neural_Networks"],
        "objectives": [
            "Understand neural network architecture",
            "Learn backpropagation and gradient descent",
            "Apply CNNs and RNNs",
            "Train and evaluate deep learning models"
        ]
    },
    "responsible_ai": {
        "title": "Responsible AI",
        "description": "Ethics, fairness, and explainability in machine learning systems.",
        "sources": ["Week_07"],
        "objectives": [
            "Identify and measure algorithmic bias",
            "Apply fairness metrics",
            "Use SHAP and LIME for explainability",
            "Design ethical AI systems"
        ]
    },
    "structured_output": {
        "title": "Structured Output",
        "description": "Generating reliable, structured outputs from AI systems.",
        "sources": ["Week_08"],
        "objectives": [
            "Design structured prompts",
            "Generate JSON and structured data",
            "Validate AI outputs",
            "Build reliable AI pipelines"
        ]
    },
    "validation_metrics": {
        "title": "Validation Metrics",
        "description": "Model evaluation, cross-validation, and performance metrics.",
        "sources": ["Week_09"],
        "objectives": [
            "Understand validation strategies",
            "Apply cross-validation techniques",
            "Select appropriate metrics",
            "Avoid common evaluation pitfalls"
        ]
    },
    "ab_testing": {
        "title": "A/B Testing",
        "description": "Experimental design and statistical testing for product decisions.",
        "sources": ["Week_10"],
        "objectives": [
            "Design controlled experiments",
            "Apply statistical hypothesis testing",
            "Calculate sample sizes and power",
            "Interpret test results"
        ]
    },
    "finance_applications": {
        "title": "Finance Applications",
        "description": "Machine learning applications in quantitative finance and risk management.",
        "sources": ["Week_00_Finance_Theory"],
        "objectives": [
            "Apply ML to financial time series",
            "Understand VaR and risk metrics",
            "Build portfolio optimization models",
            "Apply fraud detection techniques"
        ]
    },
}

def create_readme(topic_name: str, info: dict):
    """Create README.md for a topic."""
    readme_path = TOPICS / topic_name / "README.md"

    objectives_list = "\n".join(f"- {obj}" for obj in info["objectives"])
    sources_list = ", ".join(info["sources"])

    content = f"""# {info['title']}

{info['description']}

## Learning Objectives

{objectives_list}

## Folder Structure

```
{topic_name}/
├── slides/          # Presentation files (.tex, .pdf)
├── charts/          # Visualizations (.pdf, .png)
├── scripts/         # Chart generation scripts (.py)
├── handouts/        # Student materials (.md, .pdf)
└── compile.py       # Compilation script
```

## Source Materials

Original content from: {sources_list}

## Quick Start

```powershell
# Compile slides
python compile.py

# Generate charts
cd scripts
python create_*.py
```

## Key Files

- `slides/main.tex` - Main presentation file
- `handouts/basic.md` - Beginner-level handout
- `handouts/intermediate.md` - Implementation guide
- `handouts/advanced.md` - Theory and proofs
"""

    readme_path.write_text(content, encoding='utf-8')
    print(f"Created: {readme_path}")

def main():
    print("Creating Topic READMEs")
    print("=" * 60)

    for topic_name, info in TOPIC_INFO.items():
        create_readme(topic_name, info)

    print("\n" + "=" * 60)
    print("All READMEs created!")

if __name__ == "__main__":
    main()
