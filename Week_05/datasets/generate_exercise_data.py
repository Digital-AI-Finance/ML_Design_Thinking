#!/usr/bin/env python3
"""
Generate exercise dataset for Week 5 Topic Modeling workshop.
Creates a realistic startup ideas dataset for students to practice with.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Categories and keywords for generating startup descriptions
categories = {
    'Healthcare AI': [
        'health monitoring', 'medical diagnosis', 'patient care', 'telemedicine',
        'health analytics', 'personalized medicine', 'drug discovery', 'mental health',
        'fitness tracking', 'wellness platform', 'elderly care', 'chronic disease'
    ],
    'Sustainable Tech': [
        'renewable energy', 'carbon footprint', 'eco-friendly', 'circular economy',
        'waste reduction', 'sustainable materials', 'green technology', 'solar power',
        'electric vehicles', 'recycling platform', 'environmental monitoring', 'clean energy'
    ],
    'EdTech': [
        'online learning', 'personalized education', 'skill development', 'e-learning platform',
        'virtual classroom', 'educational games', 'student engagement', 'adaptive learning',
        'course marketplace', 'tutoring platform', 'certification program', 'language learning'
    ],
    'FinTech': [
        'digital payments', 'cryptocurrency', 'blockchain technology', 'investment platform',
        'personal finance', 'banking solution', 'financial planning', 'peer-to-peer lending',
        'mobile wallet', 'insurance tech', 'credit scoring', 'wealth management'
    ],
    'Food Delivery': [
        'meal delivery', 'restaurant platform', 'food ordering', 'grocery delivery',
        'meal kit service', 'dark kitchen', 'food subscription', 'dietary preferences',
        'local restaurants', 'quick delivery', 'contactless ordering', 'food waste'
    ],
    'Remote Work': [
        'collaboration tools', 'video conferencing', 'project management', 'team communication',
        'virtual office', 'productivity software', 'time tracking', 'remote hiring',
        'digital workspace', 'cloud storage', 'task automation', 'employee engagement'
    ],
    'IoT & Smart Home': [
        'smart devices', 'home automation', 'connected appliances', 'security system',
        'energy management', 'voice control', 'sensor network', 'smart lighting',
        'temperature control', 'home monitoring', 'IoT platform', 'smart locks'
    ],
    'AI & ML Services': [
        'machine learning', 'artificial intelligence', 'predictive analytics', 'natural language',
        'computer vision', 'deep learning', 'automated decision', 'data analysis',
        'pattern recognition', 'intelligent automation', 'AI assistant', 'neural networks'
    ],
    'Social Media': [
        'content creation', 'social networking', 'influencer marketing', 'community platform',
        'user engagement', 'content sharing', 'social analytics', 'viral marketing',
        'live streaming', 'social commerce', 'content moderation', 'creator economy'
    ],
    'E-commerce': [
        'online marketplace', 'product recommendations', 'shopping experience', 'inventory management',
        'customer reviews', 'price comparison', 'dropshipping platform', 'subscription box',
        'personalized shopping', 'retail analytics', 'order fulfillment', 'customer loyalty'
    ],
    'Cybersecurity': [
        'data protection', 'threat detection', 'security monitoring', 'encryption service',
        'identity verification', 'fraud prevention', 'network security', 'vulnerability assessment',
        'security training', 'compliance management', 'incident response', 'zero trust'
    ],
    'Gaming': [
        'mobile games', 'cloud gaming', 'esports platform', 'game development',
        'virtual reality', 'augmented reality', 'gaming community', 'game streaming',
        'indie games', 'educational gaming', 'multiplayer platform', 'game analytics'
    ],
    'Travel Tech': [
        'travel booking', 'accommodation platform', 'trip planning', 'local experiences',
        'travel insurance', 'flight comparison', 'hotel management', 'tourism analytics',
        'virtual tours', 'travel community', 'expense tracking', 'destination discovery'
    ],
    'Legal Tech': [
        'contract management', 'legal research', 'document automation', 'compliance software',
        'case management', 'legal marketplace', 'dispute resolution', 'intellectual property',
        'legal analytics', 'e-discovery', 'practice management', 'legal advice'
    ],
    'Real Estate Tech': [
        'property listing', 'virtual tours', 'property management', 'real estate investment',
        'home valuation', 'rental platform', 'smart buildings', 'construction tech',
        'mortgage platform', 'property analytics', 'tenant screening', 'facility management'
    ]
}

# Templates for generating descriptions
description_templates = [
    "We are building a {adjective} {category} solution that helps {target} to {action} using {technology}. Our platform {benefit} and {feature}.",
    "A {adjective} startup focused on {category} for {target}. We leverage {technology} to {action} and provide {benefit}.",
    "Our {category} platform enables {target} to {action} through {technology}. Key features include {feature} and {benefit}.",
    "Revolutionary {category} solution that {action} for {target}. Using {technology}, we {benefit} while {feature}.",
    "Next-generation {category} company helping {target} {action}. Our {technology}-powered platform {benefit} and {feature}."
]

adjectives = [
    'innovative', 'cutting-edge', 'revolutionary', 'advanced', 'next-generation',
    'comprehensive', 'intelligent', 'automated', 'scalable', 'efficient',
    'user-friendly', 'powerful', 'seamless', 'integrated', 'modern'
]

targets = [
    'businesses', 'enterprises', 'startups', 'individuals', 'professionals',
    'students', 'healthcare providers', 'financial institutions', 'consumers',
    'developers', 'creators', 'educators', 'organizations', 'communities'
]

actions = [
    'optimize operations', 'increase productivity', 'reduce costs', 'improve efficiency',
    'enhance customer experience', 'streamline processes', 'make better decisions',
    'automate workflows', 'connect and collaborate', 'track and analyze',
    'manage resources', 'scale operations', 'innovate faster', 'achieve goals'
]

technologies = [
    'artificial intelligence', 'machine learning', 'blockchain', 'cloud computing',
    'big data analytics', 'IoT sensors', 'computer vision', 'natural language processing',
    'predictive analytics', 'deep learning', 'edge computing', 'quantum computing',
    'augmented reality', 'virtual reality', '5G technology'
]

benefits = [
    'reduces operational costs by 40%', 'increases efficiency by 3x',
    'saves 10+ hours per week', 'improves accuracy by 95%',
    'delivers real-time insights', 'ensures data security',
    'provides personalized experiences', 'enables data-driven decisions',
    'automates repetitive tasks', 'scales with your business'
]

features = [
    'offers 24/7 support', 'integrates with existing systems',
    'provides detailed analytics', 'ensures compliance',
    'includes mobile apps', 'supports multiple languages',
    'offers custom solutions', 'provides API access',
    'includes training resources', 'offers free trial'
]

def generate_startup_description(category_name, keywords):
    """Generate a realistic startup description."""
    template = random.choice(description_templates)

    description = template.format(
        adjective=random.choice(adjectives),
        category=category_name.lower(),
        target=random.choice(targets),
        action=random.choice(actions),
        technology=random.choice(technologies),
        benefit=random.choice(benefits),
        feature=random.choice(features)
    )

    # Add some category-specific keywords
    additional = f" Our focus areas include {', '.join(random.sample(keywords, min(3, len(keywords))))}."

    return description + additional

def generate_dataset(n_startups=5000):
    """Generate the complete startup dataset."""
    data = []

    # Generate date range for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    for i in range(n_startups):
        # Select category (with some bias towards popular categories)
        weights = np.random.dirichlet(np.ones(len(categories)) * 2)
        category = np.random.choice(list(categories.keys()), p=weights)

        # Generate startup info
        startup = {
            'startup_id': f'STARTUP_{i+1:04d}',
            'name': f'{random.choice(adjectives).title()} {category.split()[0]} {random.choice(["Labs", "Tech", "Solutions", "Platform", "AI", "Hub", "Cloud", "Systems"])}',
            'category': category,
            'description': generate_startup_description(category, categories[category]),
            'founded_date': start_date + timedelta(days=random.randint(0, 730)),
            'funding_stage': random.choice(['Pre-seed', 'Seed', 'Series A', 'Series B', 'Series C']),
            'funding_amount': random.choice([0, 50000, 100000, 500000, 1000000, 5000000, 10000000]),
            'team_size': random.randint(1, 100),
            'location': random.choice(['San Francisco', 'New York', 'London', 'Berlin', 'Singapore', 'Tokyo', 'Bangalore', 'Remote']),
            'status': random.choice(['Active', 'Active', 'Active', 'Acquired', 'Failed']),  # Bias towards active
            'website': f'www.{category.lower().replace(" ", "")}{i}.com',
            'success_score': np.random.beta(2, 5)  # Skewed towards lower scores (realistic)
        }

        data.append(startup)

    return pd.DataFrame(data)

def generate_supplementary_data(df):
    """Generate additional datasets for exercises."""

    # 1. Customer feedback for some startups
    feedback_data = []
    sample_startups = df.sample(n=500)

    feedback_templates = [
        "Great {aspect}, really helps with {benefit}. {sentiment}",
        "The {aspect} could be improved, but overall {sentiment}.",
        "Excellent {aspect}, {sentiment} about the {benefit}.",
        "{sentiment} with the {aspect}, especially the {benefit}."
    ]

    aspects = ['user interface', 'performance', 'features', 'support', 'pricing', 'integration']
    sentiments = ['Very satisfied', 'Happy', 'Impressed', 'Disappointed', 'Neutral', 'Excited']

    for _, startup in sample_startups.iterrows():
        n_feedbacks = random.randint(5, 20)
        for _ in range(n_feedbacks):
            feedback = {
                'startup_id': startup['startup_id'],
                'feedback': random.choice(feedback_templates).format(
                    aspect=random.choice(aspects),
                    benefit=random.choice(benefits),
                    sentiment=random.choice(sentiments)
                ),
                'rating': random.randint(1, 5),
                'date': startup['founded_date'] + timedelta(days=random.randint(30, 365))
            }
            feedback_data.append(feedback)

    # 2. Innovation ideas dataset
    innovation_ideas = []

    for _ in range(2000):
        category = random.choice(list(categories.keys()))
        idea = {
            'idea_id': f'IDEA_{len(innovation_ideas)+1:04d}',
            'title': f'{random.choice(adjectives).title()} solution for {random.choice(categories[category])}',
            'description': generate_startup_description(category, categories[category]),
            'category': category,
            'feasibility': random.choice(['Low', 'Medium', 'High']),
            'impact': random.choice(['Low', 'Medium', 'High', 'Very High']),
            'effort': random.choice(['Low', 'Medium', 'High']),
            'priority_score': round(random.random(), 2)
        }
        innovation_ideas.append(idea)

    return pd.DataFrame(feedback_data), pd.DataFrame(innovation_ideas)

def main():
    """Generate all datasets for the workshop."""
    print("Generating startup dataset...")
    startups_df = generate_dataset(5000)

    print("Generating supplementary datasets...")
    feedback_df, ideas_df = generate_supplementary_data(startups_df)

    # Save datasets
    startups_df.to_csv('startup_ideas.csv', index=False)
    feedback_df.to_csv('customer_feedback.csv', index=False)
    ideas_df.to_csv('innovation_ideas.csv', index=False)

    # Save sample for quick testing (first 100)
    startups_df.head(100).to_csv('startup_ideas_sample.csv', index=False)

    # Create metadata file
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'datasets': {
            'startup_ideas.csv': {
                'rows': len(startups_df),
                'columns': list(startups_df.columns),
                'categories': list(categories.keys()),
                'description': 'Main startup dataset with descriptions for topic modeling'
            },
            'customer_feedback.csv': {
                'rows': len(feedback_df),
                'columns': list(feedback_df.columns),
                'description': 'Customer feedback for sentiment and topic analysis'
            },
            'innovation_ideas.csv': {
                'rows': len(ideas_df),
                'columns': list(ideas_df.columns),
                'description': 'Innovation ideas for ideation workshop'
            }
        }
    }

    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\nDataset Generation Complete!")
    print("=" * 50)
    print(f"Startups: {len(startups_df)} entries")
    print(f"Categories: {startups_df['category'].nunique()}")
    print(f"Feedback: {len(feedback_df)} entries")
    print(f"Ideas: {len(ideas_df)} entries")
    print("\nFiles created:")
    print("- startup_ideas.csv (main dataset)")
    print("- startup_ideas_sample.csv (100 samples for testing)")
    print("- customer_feedback.csv")
    print("- innovation_ideas.csv")
    print("- dataset_metadata.json")

    # Show category distribution
    print("\nCategory Distribution:")
    print(startups_df['category'].value_counts())

if __name__ == "__main__":
    main()