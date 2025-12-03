"""
Test and compare multiple classification algorithms on the Product Innovation dataset.
Demonstrates various classifiers and their performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, precision_recall_curve)

# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the preprocessed data."""
    print("Loading innovation dataset...")

    X_train = np.load('innovation_X_train.npy')
    X_test = np.load('innovation_X_test.npy')
    y_train = np.load('innovation_y_train_binary.npy')
    y_test = np.load('innovation_y_test_binary.npy')
    y_train_multi = np.load('innovation_y_train_multi.npy')
    y_test_multi = np.load('innovation_y_test_multi.npy')
    feature_names = np.load('innovation_feature_names.npy', allow_pickle=True)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_multi': y_train_multi,
        'y_test_multi': y_test_multi,
        'feature_names': feature_names,
        'scaler': scaler
    }

def get_classifiers():
    """Define all classifiers to test."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=10),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

def evaluate_classifier(clf, X_train, X_test, y_train, y_test, name):
    """Train and evaluate a single classifier."""

    print(f"\n  Training {name}...")

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = None

    # Get probability predictions if available
    if hasattr(clf, 'predict_proba'):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Calculate AUC if probabilities are available
    auc = None
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = None

    # Cross-validation score
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results = {
        'name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'model': clf
    }

    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    F1-Score: {f1:.3f}")
    if auc:
        print(f"    AUC: {auc:.3f}")
    print(f"    CV Score: {cv_mean:.3f} (+/- {cv_std:.3f})")

    return results

def compare_classifiers_binary(data):
    """Compare all classifiers on binary classification."""

    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION: Success vs Failure")
    print("=" * 60)

    classifiers = get_classifiers()
    results = []

    # Some algorithms need scaled data
    scaled_algorithms = ['SVM (RBF)', 'SVM (Linear)', 'K-Nearest Neighbors', 'Neural Network']

    for name, clf in classifiers.items():
        if name in scaled_algorithms:
            X_train = data['X_train_scaled']
            X_test = data['X_test_scaled']
        else:
            X_train = data['X_train']
            X_test = data['X_test']

        result = evaluate_classifier(clf, X_train, X_test,
                                    data['y_train'], data['y_test'], name)
        results.append(result)

    return results

def compare_classifiers_multiclass(data):
    """Compare classifiers on multi-class classification."""

    print("\n" + "=" * 60)
    print("MULTI-CLASS CLASSIFICATION: Failed/Struggling/Growing/Breakthrough")
    print("=" * 60)

    # Select subset of classifiers for multi-class
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    results = []
    scaled_algorithms = ['Neural Network']

    for name, clf in classifiers.items():
        if name in scaled_algorithms:
            X_train = data['X_train_scaled']
            X_test = data['X_test_scaled']
        else:
            X_train = data['X_train']
            X_test = data['X_test']

        print(f"\n  Training {name} for multi-class...")

        # Train
        clf.fit(X_train, data['y_train_multi'])

        # Predict
        y_pred = clf.predict(X_test)

        # Metrics
        accuracy = accuracy_score(data['y_test_multi'], y_pred)
        f1 = f1_score(data['y_test_multi'], y_pred, average='weighted')

        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    F1-Score: {f1:.3f}")

        results.append({
            'name': name,
            'accuracy': accuracy,
            'f1': f1,
            'y_pred': y_pred,
            'model': clf
        })

    return results

def analyze_feature_importance(data, binary_results):
    """Analyze feature importance from tree-based models."""

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Get Random Forest model (best for feature importance)
    rf_model = None
    for result in binary_results:
        if result['name'] == 'Random Forest':
            rf_model = result['model']
            break

    if rf_model and hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nTop 10 Most Important Features:")
        for i in range(min(10, len(indices))):
            print(f"  {i+1}. {data['feature_names'][indices[i]]}: {importances[indices[i]]:.4f}")

        return importances, indices
    return None, None

def create_results_summary(binary_results, multiclass_results):
    """Create summary DataFrames of all results."""

    # Binary classification summary
    binary_summary = pd.DataFrame([{
        'Classifier': r['name'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1-Score': r['f1'],
        'AUC': r['auc'] if r['auc'] else np.nan,
        'CV Mean': r['cv_mean'],
        'CV Std': r['cv_std']
    } for r in binary_results])

    binary_summary = binary_summary.sort_values('F1-Score', ascending=False)

    # Multi-class summary
    multi_summary = pd.DataFrame([{
        'Classifier': r['name'],
        'Accuracy': r['accuracy'],
        'F1-Score': r['f1']
    } for r in multiclass_results])

    multi_summary = multi_summary.sort_values('F1-Score', ascending=False)

    return binary_summary, multi_summary

def save_results(binary_results, multiclass_results, binary_summary, multi_summary):
    """Save results to files."""

    # Save summaries as CSV
    binary_summary.to_csv('classification_results_binary.csv', index=False)
    multi_summary.to_csv('classification_results_multiclass.csv', index=False)

    # Save best model
    best_model = max(binary_results, key=lambda x: x['f1'])
    import joblib
    joblib.dump(best_model['model'], 'best_classifier.pkl')

    print("\n" + "=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    print("  - classification_results_binary.csv")
    print("  - classification_results_multiclass.csv")
    print("  - best_classifier.pkl")

def print_final_summary(binary_summary, multi_summary):
    """Print final summary of results."""

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\nBinary Classification (Top 5):")
    print(binary_summary.head().to_string(index=False))

    print("\n\nMulti-class Classification:")
    print(multi_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("BEST PERFORMERS")
    print("=" * 60)

    best_binary = binary_summary.iloc[0]
    print(f"\nBinary Classification Winner: {best_binary['Classifier']}")
    print(f"  - F1-Score: {best_binary['F1-Score']:.3f}")
    print(f"  - Accuracy: {best_binary['Accuracy']:.3f}")
    if not pd.isna(best_binary['AUC']):
        print(f"  - AUC: {best_binary['AUC']:.3f}")

    best_multi = multi_summary.iloc[0]
    print(f"\nMulti-class Classification Winner: {best_multi['Classifier']}")
    print(f"  - F1-Score: {best_multi['F1-Score']:.3f}")
    print(f"  - Accuracy: {best_multi['Accuracy']:.3f}")

def main():
    """Main execution function."""

    print("\n" + "="*60)
    print("CLASSIFICATION ALGORITHMS COMPARISON")
    print("Product Innovation Success Prediction")
    print("="*60)

    # Load data
    data = load_data()

    # Binary classification
    binary_results = compare_classifiers_binary(data)

    # Multi-class classification
    multiclass_results = compare_classifiers_multiclass(data)

    # Feature importance
    analyze_feature_importance(data, binary_results)

    # Create summaries
    binary_summary, multi_summary = create_results_summary(binary_results, multiclass_results)

    # Save results
    save_results(binary_results, multiclass_results, binary_summary, multi_summary)

    # Print final summary
    print_final_summary(binary_summary, multi_summary)

    print("\n" + "="*60)
    print("Classification analysis complete!")
    print("Note: This is SIMULATED data for educational purposes")
    print("="*60)

if __name__ == "__main__":
    main()