"""
Final verification of Week 2 FinTech dataset and presentation components.
Ensures all files exist and are properly formatted.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def verify_week2_completion():
    """Comprehensive verification of all Week 2 components."""

    print("="*60)
    print("WEEK 2 FINTECH DATASET - FINAL VERIFICATION")
    print("="*60)

    verification_results = []

    # 1. Verify dataset files
    print("\n1. DATASET FILES:")
    print("-" * 40)

    dataset_files = {
        'fintech_user_behavior_full.csv': 'Main dataset',
        'fintech_X.npy': 'Feature matrix',
        'fintech_y_true.npy': 'True labels',
        'fintech_segments.npy': 'Segment descriptions'
    }

    for file, desc in dataset_files.items():
        exists = os.path.exists(file)
        status = "[Y]" if exists else "[N]"
        print(f"  {status} {file:<35} - {desc}")

        if exists and file.endswith('.csv'):
            df = pd.read_csv(file)
            print(f"      Shape: {df.shape}, Columns: {len(df.columns)}")
        elif exists and file.endswith('.npy'):
            arr = np.load(file, allow_pickle=True)
            print(f"      Shape: {arr.shape}")

        verification_results.append(exists)

    # 2. Verify Python scripts
    print("\n2. PYTHON SCRIPTS:")
    print("-" * 40)

    scripts = [
        'generate_fintech_dataset.py',
        'test_all_algorithms.py',
        'validate_clusters.py',
        'create_persona_mapping.py',
        'create_fintech_clustering_suite.py',
        'create_fintech_validation_suite.py',
        'create_descriptive_analysis.py',
        'verify_dataset_implementation.py',
        'compile.py'
    ]

    for script in scripts:
        exists = os.path.exists(script)
        status = "[Y]" if exists else "[N]"

        if exists:
            size = os.path.getsize(script) // 1024
            print(f"  {status} {script:<40} ({size} KB)")
        else:
            print(f"  {status} {script:<40}")

        verification_results.append(exists)

    # 3. Verify visualizations
    print("\n3. VISUALIZATIONS (PDFs):")
    print("-" * 40)

    visualizations = [
        'fintech_dataset_overview_slides.pdf',
        'fintech_algorithm_comparison.pdf',
        'fintech_fraud_detection.pdf',
        'fintech_cluster_quality.pdf',
        'fintech_elbow_comprehensive.pdf',
        'fintech_silhouette_grid.pdf',
        'fintech_descriptive_statistics.pdf',
        'fintech_segment_statistics.pdf',
        'fintech_data_quality.pdf'
    ]

    for viz in visualizations:
        exists = os.path.exists(viz)
        status = "[Y]" if exists else "[N]"

        if exists:
            size = os.path.getsize(viz) // 1024
            print(f"  {status} {viz:<45} ({size} KB)")
        else:
            print(f"  {status} {viz:<45}")

        verification_results.append(exists)

    # 4. Verify LaTeX presentations
    print("\n4. LATEX PRESENTATIONS:")
    print("-" * 40)

    presentations = [
        ('20250921_1800_fintech_complete.tex', 'Complete 35-slide source'),
        ('20250921_1800_fintech_complete.pdf', 'Final presentation PDF')
    ]

    for file, desc in presentations:
        exists = os.path.exists(file)
        status = "[Y]" if exists else "[N]"

        if exists:
            size = os.path.getsize(file) // 1024
            print(f"  {status} {file:<40} - {desc} ({size} KB)")
        else:
            print(f"  {status} {file:<40} - {desc}")

        verification_results.append(exists)

    # 5. Quick data validation
    print("\n5. DATA VALIDATION:")
    print("-" * 40)

    try:
        # Load and validate dataset
        df = pd.read_csv('fintech_user_behavior_full.csv')

        print(f"  Dataset shape: {df.shape}")
        print(f"  Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%)")
        print(f"  Unique segments: {df['true_segment'].nunique()}")
        print(f"  Segment distribution:")

        for segment, count in df['true_segment'].value_counts().items():
            print(f"    - {segment}: {count} ({count/len(df)*100:.1f}%)")

        # Check feature ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\n  Feature statistics (first 5):")
        for col in numeric_cols[:5]:
            print(f"    {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

        data_valid = True
    except Exception as e:
        print(f"  ERROR: Could not validate data - {e}")
        data_valid = False

    verification_results.append(data_valid)

    # 6. Overall summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY:")
    print("="*60)

    total_checks = len(verification_results)
    passed_checks = sum(verification_results)
    success_rate = (passed_checks / total_checks) * 100

    print(f"  Total checks: {total_checks}")
    print(f"  Passed: {passed_checks}")
    print(f"  Failed: {total_checks - passed_checks}")
    print(f"  Success rate: {success_rate:.1f}%")

    if success_rate == 100:
        print("\n  STATUS: ALL COMPONENTS VERIFIED SUCCESSFULLY!")
        print("  The Week 2 FinTech dataset and presentation are complete.")
    else:
        print(f"\n  STATUS: {total_checks - passed_checks} components missing or failed.")
        print("  Please check the output above for details.")

    print("\n" + "="*60)
    print("Note: This is SIMULATED data for educational purposes")
    print("Course: Machine Learning for Smarter Innovation - Week 2")
    print("="*60)

    return success_rate == 100

if __name__ == "__main__":
    success = verify_week2_completion()
    exit(0 if success else 1)