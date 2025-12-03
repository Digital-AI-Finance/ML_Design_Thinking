"""
Generate Product Innovation Success Dataset for Week 4: Classification & Definition
Creates a realistic dataset of 10,000 product launches with success/failure outcomes
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_product_innovation_dataset(n_samples=10000):
    """
    Generate a comprehensive product innovation dataset with realistic patterns.

    Returns:
        DataFrame with product features and success outcomes
    """

    print("Generating Product Innovation Success Dataset...")
    print("=" * 60)

    # Initialize arrays
    data = []

    # Define product segments with realistic distributions
    segments = {
        'tech_innovations': {
            'count': 2500,
            'success_rate': 0.35,
            'characteristics': {
                'novelty_score': (7, 1.5),  # (mean, std)
                'technical_complexity': (8, 1.2),
                'disruption_potential': (7.5, 1.8),
                'market_size': (8, 2),
                'competition_intensity': (7, 1.5),
                'timing_score': (6, 2),
                'team_experience': (7, 1.8),
                'diversity_index': (6.5, 1.5),
                'domain_expertise': (8, 1.2),
                'development_time': (18, 6),  # months
                'iterations': (12, 4),
                'user_testing_hours': (500, 200),
                'initial_funding': (5000000, 2000000),  # dollars
                'marketing_budget': (1000000, 500000),
                'price_point': (500, 300)
            }
        },
        'consumer_products': {
            'count': 2500,
            'success_rate': 0.45,
            'characteristics': {
                'novelty_score': (5, 1.5),
                'technical_complexity': (4, 1.5),
                'disruption_potential': (4, 1.8),
                'market_size': (9, 1),
                'competition_intensity': (8, 1.2),
                'timing_score': (7, 1.5),
                'team_experience': (6, 2),
                'diversity_index': (5.5, 1.8),
                'domain_expertise': (6, 1.8),
                'development_time': (12, 4),
                'iterations': (8, 3),
                'user_testing_hours': (300, 150),
                'initial_funding': (500000, 300000),
                'marketing_budget': (200000, 100000),
                'price_point': (50, 30)
            }
        },
        'b2b_solutions': {
            'count': 2000,
            'success_rate': 0.40,
            'characteristics': {
                'novelty_score': (6, 1.2),
                'technical_complexity': (7, 1.5),
                'disruption_potential': (5, 2),
                'market_size': (6, 2),
                'competition_intensity': (5, 1.8),
                'timing_score': (6.5, 1.5),
                'team_experience': (8, 1.5),
                'diversity_index': (5, 1.5),
                'domain_expertise': (9, 1),
                'development_time': (24, 8),
                'iterations': (10, 4),
                'user_testing_hours': (200, 100),
                'initial_funding': (2000000, 1000000),
                'marketing_budget': (500000, 250000),
                'price_point': (5000, 3000)
            }
        },
        'social_innovations': {
            'count': 1500,
            'success_rate': 0.30,
            'characteristics': {
                'novelty_score': (7, 2),
                'technical_complexity': (5, 2),
                'disruption_potential': (8, 2),
                'market_size': (5, 2.5),
                'competition_intensity': (4, 2),
                'timing_score': (7, 2),
                'team_experience': (5, 2),
                'diversity_index': (8, 1.5),
                'domain_expertise': (6, 2),
                'development_time': (15, 6),
                'iterations': (15, 5),
                'user_testing_hours': (800, 300),
                'initial_funding': (100000, 80000),
                'marketing_budget': (20000, 15000),
                'price_point': (10, 8)
            }
        },
        'breakthrough_successes': {
            'count': 500,
            'success_rate': 0.95,  # These are the unicorns
            'characteristics': {
                'novelty_score': (9, 0.8),
                'technical_complexity': (6, 2),
                'disruption_potential': (9.5, 0.5),
                'market_size': (9.5, 0.5),
                'competition_intensity': (3, 1.5),
                'timing_score': (9, 1),
                'team_experience': (9, 1),
                'diversity_index': (8.5, 1),
                'domain_expertise': (9, 1),
                'development_time': (10, 3),
                'iterations': (20, 5),
                'user_testing_hours': (1000, 300),
                'initial_funding': (10000000, 5000000),
                'marketing_budget': (2000000, 1000000),
                'price_point': (100, 500)
            }
        },
        'failed_ventures': {
            'count': 500,
            'success_rate': 0.0,  # Learning from failures
            'characteristics': {
                'novelty_score': (3, 2),
                'technical_complexity': (8, 2),
                'disruption_potential': (3, 2),
                'market_size': (3, 2),
                'competition_intensity': (9, 1),
                'timing_score': (2, 1.5),
                'team_experience': (3, 2),
                'diversity_index': (3, 1.5),
                'domain_expertise': (3, 2),
                'development_time': (30, 10),
                'iterations': (3, 2),
                'user_testing_hours': (50, 30),
                'initial_funding': (50000, 40000),
                'marketing_budget': (5000, 4000),
                'price_point': (1000, 800)
            }
        }
    }

    # Generate data for each segment
    for segment_name, segment_info in segments.items():
        print(f"  Generating {segment_info['count']} {segment_name.replace('_', ' ')} products...")

        for _ in range(segment_info['count']):
            product = {'segment': segment_name}

            # Generate features based on segment characteristics
            for feature, (mean, std) in segment_info['characteristics'].items():
                if feature in ['initial_funding', 'marketing_budget', 'price_point']:
                    # Log-normal distribution for financial features
                    value = np.random.lognormal(np.log(mean), std/mean)
                    product[feature] = max(1000, value)  # Minimum values
                elif feature in ['iterations', 'development_time']:
                    # Integer values for counts and time
                    product[feature] = max(1, int(np.random.normal(mean, std)))
                else:
                    # Bounded 0-10 scale for scores
                    value = np.random.normal(mean, std)
                    if feature.endswith('_score') or feature.endswith('_index') or \
                       feature == 'domain_expertise' or feature == 'team_experience':
                        product[feature] = np.clip(value, 0, 10)
                    else:
                        product[feature] = value

            # Add some noise and correlations
            # High novelty often correlates with high risk
            if product['novelty_score'] > 7:
                product['technical_complexity'] *= 1.2
                product['development_time'] *= 1.3

            # Market size affects funding
            if product['market_size'] > 8:
                product['initial_funding'] *= 1.5
                product['marketing_budget'] *= 1.8

            # Generate success outcome based on segment and features
            success_probability = segment_info['success_rate']

            # Adjust probability based on key features
            feature_impact = 0
            feature_impact += (product['timing_score'] - 5) * 0.05
            feature_impact += (product['team_experience'] - 5) * 0.04
            feature_impact += (product['market_size'] - 5) * 0.03
            feature_impact += (product['domain_expertise'] - 5) * 0.03
            feature_impact -= (product['competition_intensity'] - 5) * 0.02
            feature_impact += (product['diversity_index'] - 5) * 0.02

            # Cap the impact
            feature_impact = np.clip(feature_impact, -0.3, 0.3)
            final_probability = np.clip(success_probability + feature_impact, 0.05, 0.95)

            # Binary success outcome
            product['success'] = int(np.random.random() < final_probability)

            # Multi-class outcome
            if product['success'] == 0:
                if np.random.random() < 0.6:
                    product['success_level'] = 'failed'
                else:
                    product['success_level'] = 'struggling'
            else:
                if segment_name == 'breakthrough_successes' or \
                   (product['novelty_score'] > 8 and product['market_size'] > 8):
                    product['success_level'] = 'breakthrough'
                elif product['team_experience'] > 7 and product['timing_score'] > 7:
                    product['success_level'] = 'growing'
                else:
                    product['success_level'] = 'growing' if np.random.random() < 0.7 else 'breakthrough'

            # Success probability (for regression)
            product['success_probability'] = final_probability

            # Add some derived features
            product['innovation_intensity'] = (product['novelty_score'] +
                                              product['disruption_potential']) / 2
            product['resource_efficiency'] = (product['initial_funding'] / 1000000) / \
                                            max(1, product['development_time'])
            product['market_readiness'] = (product['market_size'] * product['timing_score']) / \
                                         max(1, product['competition_intensity'])
            product['team_strength'] = (product['team_experience'] +
                                       product['domain_expertise'] +
                                       product['diversity_index']) / 3

            # Add sentiment score (from hypothetical user research)
            if product['user_testing_hours'] > 500:
                product['user_sentiment_score'] = np.random.normal(7, 1.5)
            elif product['user_testing_hours'] > 200:
                product['user_sentiment_score'] = np.random.normal(6, 1.8)
            else:
                product['user_sentiment_score'] = np.random.normal(5, 2)
            product['user_sentiment_score'] = np.clip(product['user_sentiment_score'], 0, 10)

            data.append(product)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add product ID
    df['product_id'] = ['PROD_' + str(i).zfill(5) for i in range(len(df))]

    # Add launch year (for temporal analysis)
    years = np.random.choice(range(2015, 2024), size=len(df),
                           p=[0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.20, 0.15])
    df['launch_year'] = years

    # Add launch quarter
    df['launch_quarter'] = np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], size=len(df))

    # Reorder columns
    column_order = ['product_id', 'segment', 'launch_year', 'launch_quarter',
                    'novelty_score', 'technical_complexity', 'disruption_potential',
                    'market_size', 'competition_intensity', 'timing_score',
                    'team_experience', 'diversity_index', 'domain_expertise',
                    'development_time', 'iterations', 'user_testing_hours',
                    'initial_funding', 'marketing_budget', 'price_point',
                    'innovation_intensity', 'resource_efficiency',
                    'market_readiness', 'team_strength', 'user_sentiment_score',
                    'success', 'success_level', 'success_probability']

    df = df[column_order]

    # Add some missing values for realism (0.3% missing)
    missing_features = ['user_testing_hours', 'user_sentiment_score', 'diversity_index']
    for feature in missing_features:
        missing_idx = np.random.choice(df.index, size=int(0.001 * len(df)), replace=False)
        df.loc[missing_idx, feature] = np.nan

    return df

def create_train_test_splits(df):
    """Create train/test splits for modeling."""

    # Features for modeling
    feature_columns = [col for col in df.columns if col not in
                      ['product_id', 'segment', 'launch_year', 'launch_quarter',
                       'success', 'success_level', 'success_probability']]

    X = df[feature_columns].fillna(df[feature_columns].median())

    # Binary classification
    y_binary = df['success']

    # Multi-class classification
    label_map = {'failed': 0, 'struggling': 1, 'growing': 2, 'breakthrough': 3}
    y_multiclass = df['success_level'].map(label_map)

    # Regression target
    y_regression = df['success_probability']

    # Create splits
    X_train, X_test, y_train_binary, y_test_binary = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

    _, _, y_train_multi, y_test_multi = train_test_split(
        X, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass)

    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_binary': y_train_binary,
        'y_test_binary': y_test_binary,
        'y_train_multi': y_train_multi,
        'y_test_multi': y_test_multi,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'feature_names': feature_columns
    }

def save_dataset(df, splits):
    """Save dataset and splits to files."""

    # Save full dataset
    df.to_csv('innovation_products_full.csv', index=False)
    print(f"\nSaved full dataset to innovation_products_full.csv")

    # Save numpy arrays for modeling
    np.save('innovation_X_train.npy', splits['X_train'].values)
    np.save('innovation_X_test.npy', splits['X_test'].values)
    np.save('innovation_y_train_binary.npy', splits['y_train_binary'].values)
    np.save('innovation_y_test_binary.npy', splits['y_test_binary'].values)
    np.save('innovation_y_train_multi.npy', splits['y_train_multi'].values)
    np.save('innovation_y_test_multi.npy', splits['y_test_multi'].values)
    np.save('innovation_feature_names.npy', splits['feature_names'])

    print("Saved train/test splits to .npy files")

def print_dataset_summary(df):
    """Print summary statistics of the dataset."""

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal products: {len(df):,}")
    print(f"Features: {len(df.columns) - 4}")  # Excluding IDs and targets
    print(f"Time span: {df['launch_year'].min()} - {df['launch_year'].max()}")

    print("\nSegment Distribution:")
    for segment, count in df['segment'].value_counts().items():
        print(f"  {segment.replace('_', ' ').title()}: {count:,} ({count/len(df)*100:.1f}%)")

    print("\nSuccess Metrics:")
    print(f"  Binary success rate: {df['success'].mean():.1%}")

    print("\n  Multi-class distribution:")
    for level, count in df['success_level'].value_counts().sort_index().items():
        print(f"    {level.title()}: {count:,} ({count/len(df)*100:.1f}%)")

    print("\nFeature Statistics (sample):")
    sample_features = ['novelty_score', 'market_size', 'team_experience',
                      'initial_funding', 'user_sentiment_score']
    for feature in sample_features:
        if feature in df.columns:
            values = df[feature].dropna()
            print(f"  {feature}:")
            print(f"    Mean: {values.mean():.2f}, Std: {values.std():.2f}")
            print(f"    Min: {values.min():.2f}, Max: {values.max():.2f}")

    print(f"\nMissing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.2f}%)")

    print("\nNote: This is SIMULATED data for educational purposes")
    print("Purpose: Demonstrate classification techniques for innovation success prediction")

def main():
    """Main execution function."""

    print("\n" + "="*60)
    print("PRODUCT INNOVATION SUCCESS DATASET GENERATOR")
    print("Week 4: Classification & Definition")
    print("="*60)

    # Generate dataset
    df = generate_product_innovation_dataset(n_samples=10000)

    # Create train/test splits
    splits = create_train_test_splits(df)

    # Save files
    save_dataset(df, splits)

    # Print summary
    print_dataset_summary(df)

    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("Ready for classification analysis")
    print("="*60)

if __name__ == "__main__":
    main()