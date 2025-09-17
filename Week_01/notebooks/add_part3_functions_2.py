import json

# Load the updated notebook
with open(r'D:\Joerg\Research\slides\ML_Design_Thinking_16\Week_01\Week01_Part3_Practice_Advanced.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the functions cell (should be at index 4)
if nb['cells'][4]['cell_type'] == 'code':
    # Get existing source
    existing_source = nb['cells'][4]['source']
    
    # Add more functions
    additional_functions = '''
# ============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# ============================================================================

def compare_pca_tsne():
    """
    Compare PCA vs t-SNE dimensionality reduction techniques.
    Shows different perspectives on same data.
    """
    print("📊 PCA vs t-SNE: Dimensionality Reduction Comparison\\n")
    
    # Generate complex dataset
    X_complex, y_complex = make_blobs(n_samples=1000, n_features=20, 
                                     centers=5, cluster_std=1.5, 
                                     random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_complex_scaled = scaler.fit_transform(X_complex)
    
    # Apply clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X_complex_scaled)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_complex_scaled)
    
    # t-SNE with different perplexities
    perplexities = [5, 30, 50]
    X_tsne_results = []
    
    for perp in perplexities:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_tsne = tsne.fit_transform(X_complex_scaled)
        X_tsne_results.append(X_tsne)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PCA
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                          cmap='tab10', s=20, alpha=0.7)
    ax1.set_title(f'PCA\\nVariance: {pca.explained_variance_ratio_.sum():.1%}', 
                 fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1)
    
    # t-SNE variations
    for idx, (perp, X_tsne) in enumerate(zip(perplexities, X_tsne_results)):
        ax = axes.flat[idx + 1]
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels,
                            cmap='tab10', s=20, alpha=0.7)
        ax.set_title(f't-SNE (perplexity={perp})', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax)
    
    plt.suptitle('PCA vs t-SNE: Different Perspectives on Same Data', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("📚 Key Differences:")
    print("\\nPCA:")
    print("  • Linear transformation")
    print("  • Preserves global structure")
    print("  • Fast computation")
    print("  • Interpretable axes")
    print("\\nt-SNE:")
    print("  • Non-linear transformation")
    print("  • Preserves local structure")
    print("  • Slower computation")
    print("  • Better cluster separation")
    print("\\n💡 Tip: Use PCA for exploration, t-SNE for presentation")

def create_3d_interactive_clusters():
    """
    Create 3D interactive cluster visualization using Plotly.
    Allows rotation and exploration of clusters in 3D space.
    """
    print("🌐 3D Interactive Cluster Visualization\\n")
    
    # Generate 3D data
    X_3d, y_3d = make_blobs(n_samples=500, n_features=3, centers=4, 
                            cluster_std=0.8, random_state=42)
    
    # Apply clustering
    kmeans_3d = KMeans(n_clusters=4, random_state=42)
    labels_3d = kmeans_3d.fit_predict(X_3d)
    
    # Create interactive 3D plot
    fig = go.Figure()
    
    # Add clusters
    colors_plotly = px.colors.qualitative.Plotly
    for i in range(4):
        mask = labels_3d == i
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0],
            y=X_3d[mask, 1],
            z=X_3d[mask, 2],
            mode='markers',
            name=f'Cluster {i+1}',
            marker=dict(
                size=5,
                color=colors_plotly[i],
                opacity=0.8,
                line=dict(width=0.5, color='white')
            )
        ))
    
    # Add cluster centers
    fig.add_trace(go.Scatter3d(
        x=kmeans_3d.cluster_centers_[:, 0],
        y=kmeans_3d.cluster_centers_[:, 1],
        z=kmeans_3d.cluster_centers_[:, 2],
        mode='markers',
        name='Centers',
        marker=dict(
            size=15,
            color='black',
            symbol='diamond',
            line=dict(width=2, color='white')
        )
    ))
    
    # Update layout
    fig.update_layout(
        title='3D Interactive Clustering (Rotate with Mouse)',
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Feature 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600,
        showlegend=True
    )
    
    fig.show()
    
    print("🖱️ Interaction Tips:")
    print("  • Click and drag to rotate")
    print("  • Scroll to zoom")
    print("  • Double-click to reset view")
    print("  • Click legend items to show/hide clusters")

def create_clustering_dashboard():
    """
    Create comprehensive clustering dashboard with multiple views.
    Shows various aspects of clustering analysis in one view.
    """
    print("📊 Clustering Analysis Dashboard\\n")
    
    # Generate sample data
    X_dashboard, y_dashboard = make_blobs(n_samples=800, n_features=5, 
                                         centers=4, cluster_std=1.2,
                                         random_state=42)
    
    # Feature names
    feature_names_dash = ['Innovation', 'Quality', 'Cost', 'Time', 'Risk']
    df_dashboard = pd.DataFrame(X_dashboard, columns=feature_names_dash)
    
    # Apply clustering
    scaler_dash = StandardScaler()
    X_scaled_dash = scaler_dash.fit_transform(X_dashboard)
    kmeans_dash = KMeans(n_clusters=4, random_state=42)
    labels_dash = kmeans_dash.fit_predict(X_scaled_dash)
    df_dashboard['Cluster'] = labels_dash
    
    # Import required function
    from sklearn.metrics import calinski_harabasz_score
    
    # Create dashboard with subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Cluster Distribution', 'Feature Comparison', 
                       'Correlation Matrix', 'PCA Projection',
                       'Cluster Quality', 'Feature Importance'),
        specs=[[{'type': 'bar'}, {'type': 'box'}, {'type': 'heatmap'}],
               [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    colors_plotly = px.colors.qualitative.Plotly
    
    # 1. Cluster distribution
    cluster_counts = df_dashboard['Cluster'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=[f'C{i}' for i in cluster_counts.index], 
              y=cluster_counts.values,
              marker_color=colors_plotly[:4]),
        row=1, col=1
    )
    
    # 2. Feature comparison (box plots)
    for i in range(4):
        cluster_data = df_dashboard[df_dashboard['Cluster'] == i]['Innovation']
        fig.add_trace(
            go.Box(y=cluster_data, name=f'C{i}',
                  marker_color=colors_plotly[i]),
            row=1, col=2
        )
    
    # 3. Correlation matrix
    corr_matrix = df_dashboard[feature_names_dash].corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values,
                  x=feature_names_dash,
                  y=feature_names_dash,
                  colorscale='RdBu',
                  zmid=0),
        row=1, col=3
    )
    
    # 4. PCA projection
    pca_dash = PCA(n_components=2)
    X_pca_dash = pca_dash.fit_transform(X_scaled_dash)
    for i in range(4):
        mask = labels_dash == i
        fig.add_trace(
            go.Scatter(x=X_pca_dash[mask, 0],
                      y=X_pca_dash[mask, 1],
                      mode='markers',
                      name=f'C{i}',
                      marker=dict(color=colors_plotly[i], size=5)),
            row=2, col=1
        )
    
    # 5. Cluster quality metrics
    metrics = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
    scores = [
        silhouette_score(X_scaled_dash, labels_dash),
        -davies_bouldin_score(X_scaled_dash, labels_dash) / 5,  # Normalize
        calinski_harabasz_score(X_scaled_dash, labels_dash) / 500  # Normalize
    ]
    fig.add_trace(
        go.Bar(x=metrics, y=scores,
              marker_color=['green', 'orange', 'blue']),
        row=2, col=2
    )
    
    # 6. Feature importance
    feature_var = df_dashboard.groupby('Cluster')[feature_names_dash].mean().std()
    fig.add_trace(
        go.Bar(x=feature_names_dash, y=feature_var.values,
              marker_color='purple'),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text='Clustering Analysis Dashboard',
        showlegend=False,
        height=700
    )
    
    fig.show()
    
    print("\\n📈 Dashboard Components:")
    print("  1. Cluster sizes and distribution")
    print("  2. Feature variations across clusters")
    print("  3. Feature correlations")
    print("  4. 2D projection of clusters")
    print("  5. Quality metrics")
    print("  6. Feature importance for clustering")

# ============================================================================
# EXERCISE FUNCTIONS
# ============================================================================

def exercise_basic_clustering():
    """
    Exercise 1: Basic clustering task with guided solution.
    Learn fundamentals of clustering workflow.
    """
    print("📝 Exercise 1: Basic Clustering\\n")
    print("Task: Generate 500 innovation ideas and apply K-means with K=3")
    print("-" * 50)
    
    print("\\n💡 Solution:")
    
    # Generate data
    X_ex1, y_ex1 = make_blobs(n_samples=500, centers=3, n_features=5, 
                             cluster_std=1.0, random_state=42)
    
    # Standardize
    scaler_ex1 = StandardScaler()
    X_ex1_scaled = scaler_ex1.fit_transform(X_ex1)
    
    # Apply K-means
    kmeans_ex1 = KMeans(n_clusters=3, random_state=42)
    labels_ex1 = kmeans_ex1.fit_predict(X_ex1_scaled)
    
    # Calculate score
    score_ex1 = silhouette_score(X_ex1_scaled, labels_ex1)
    
    # Visualize
    pca_ex1 = PCA(n_components=2)
    X_pca_ex1 = pca_ex1.fit_transform(X_ex1_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca_ex1[:, 0], X_pca_ex1[:, 1], 
                         c=labels_ex1, cmap='viridis', s=30, alpha=0.7)
    plt.scatter(pca_ex1.transform(kmeans_ex1.cluster_centers_)[:, 0],
               pca_ex1.transform(kmeans_ex1.cluster_centers_)[:, 1],
               c='red', marker='*', s=300, edgecolors='black', linewidth=2)
    plt.title(f'Exercise 1: K-Means Clustering (Silhouette: {score_ex1:.3f})', fontweight='bold')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter)
    plt.show()
    
    print(f"\\n✅ Results:")
    print(f"  • Number of samples: 500")
    print(f"  • Number of clusters: 3")
    print(f"  • Silhouette score: {score_ex1:.3f}")
    print(f"  • Cluster sizes: {np.bincount(labels_ex1)}")

def exercise_finding_optimal_k():
    """
    Exercise 2: Find optimal K using multiple methods.
    Learn to determine the right number of clusters.
    """
    print("📝 Exercise 2: Finding Optimal K\\n")
    print("Task: Use elbow method and silhouette analysis to find optimal K")
    print("-" * 50)
    
    # Generate data with unknown optimal K
    X_ex2, y_ex2 = make_blobs(n_samples=600, centers=5, n_features=4,
                             cluster_std=1.3, random_state=42)
    X_ex2_scaled = StandardScaler().fit_transform(X_ex2)
    
    print("\\n💡 Solution:")
    
    k_range = range(2, 10)
    inertias = []
    silhouettes = []
    davies_bouldins = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_ex2_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_ex2_scaled, labels))
        davies_bouldins.append(davies_bouldin_score(X_ex2_scaled, labels))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
    best_k_sil = list(k_range)[np.argmax(silhouettes)]
    ax2.axvline(x=best_k_sil, color='red', linestyle='--', label=f'Best K={best_k_sil}')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Davies-Bouldin plot
    ax3.plot(k_range, davies_bouldins, 'ro-', linewidth=2, markersize=8)
    best_k_db = list(k_range)[np.argmin(davies_bouldins)]
    ax3.axvline(x=best_k_db, color='green', linestyle='--', label=f'Best K={best_k_db}')
    ax3.set_xlabel('K')
    ax3.set_ylabel('Davies-Bouldin Index')
    ax3.set_title('Davies-Bouldin Index (lower is better)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Exercise 2: Finding Optimal K', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\\n✅ Results:")
    print(f"  • Silhouette suggests K = {best_k_sil}")
    print(f"  • Davies-Bouldin suggests K = {best_k_db}")
    print(f"  • True number of centers = 5")

def exercise_dbscan_challenge():
    """
    Exercise 3: DBSCAN clustering on complex shapes.
    Learn to handle non-spherical clusters and outliers.
    """
    print("📝 Exercise 3: DBSCAN Challenge\\n")
    print("Task: Cluster complex shapes and handle outliers with DBSCAN")
    print("-" * 50)
    
    # Generate complex data with outliers
    from sklearn.datasets import make_moons
    X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_outliers = np.random.uniform(-3, 3, (30, 2))
    X_ex3 = np.vstack([X_moons, X_outliers])
    
    print("\\n💡 Solution:")
    
    # Test different parameters
    eps_values = [0.1, 0.2, 0.3]
    min_samples_values = [5, 10]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for i, min_samples in enumerate(min_samples_values):
        for j, eps in enumerate(eps_values):
            ax = axes[i, j]
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_ex3)
            
            # Count clusters and noise
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Plot
            unique_labels = set(labels)
            colors_db = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors_db):
                if k == -1:
                    col = 'black'
                    marker = 'x'
                else:
                    marker = 'o'
                
                class_member_mask = (labels == k)
                xy = X_ex3[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                          s=30, alpha=0.7)
            
            ax.set_title(f'eps={eps}, min_samples={min_samples}\\n'
                        f'Clusters: {n_clusters}, Noise: {n_noise}',
                        fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
    
    plt.suptitle('Exercise 3: DBSCAN Parameter Tuning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n✅ Best Parameters:")
    print("  • eps = 0.2 (good balance)")
    print("  • min_samples = 5 (less restrictive)")
    print("  • Successfully separated two moons")
    print("  • Identified outliers as noise")

# ============================================================================
# REFERENCE GUIDE FUNCTIONS
# ============================================================================

def show_algorithm_selection_guide():
    """
    Display clustering algorithm selection flowchart.
    Help users choose the right algorithm for their data.
    """
    print("🎯 Clustering Algorithm Selection Guide\\n")
    
    decision_tree = \"\"\"
Start: What kind of data do you have?
│
├─ Known number of clusters?
│  │
│  ├─ YES → Spherical clusters expected?
│  │        │
│  │        ├─ YES → K-Means ✓
│  │        └─ NO → GMM or Spectral Clustering
│  │
│  └─ NO → Need hierarchical structure?
│          │
│          ├─ YES → Hierarchical Clustering ✓
│          └─ NO → DBSCAN or Mean Shift
│
├─ Have outliers/noise?
│  │
│  ├─ YES → DBSCAN ✓
│  └─ NO → Continue to next question
│
├─ Overlapping clusters?
│  │
│  ├─ YES → GMM ✓
│  └─ NO → K-Means or Hierarchical
│
└─ Very large dataset (>100k points)?
   │
   ├─ YES → Mini-Batch K-Means ✓
   └─ NO → Any algorithm based on above criteria
\"\"\"
    
    print(decision_tree)
    
    # Quick comparison table
    comparison = pd.DataFrame({
        'Algorithm': ['K-Means', 'DBSCAN', 'Hierarchical', 'GMM'],
        'Speed': ['Fast', 'Medium', 'Slow', 'Medium'],
        'Scalability': ['Excellent', 'Good', 'Poor', 'Good'],
        'Handles_Noise': ['No', 'Yes', 'No', 'Partially'],
        'Cluster_Shape': ['Spherical', 'Any', 'Any', 'Elliptical'],
        'Need_K': ['Yes', 'No', 'No*', 'Yes'],
        'Best_For': ['General', 'Outliers', 'Taxonomy', 'Overlapping']
    })
    
    print("\\n📊 Algorithm Comparison:")
    display(comparison)

def show_code_snippets():
    """
    Display copy-paste ready code snippets for common clustering tasks.
    Quick reference for implementation.
    """
    print("📋 Copy-Paste Ready Code Snippets\\n")
    
    snippets = {
        "K-Means Basic": \"\"\"
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Get cluster centers
centers = kmeans.cluster_centers_
\"\"\",
        
        "Find Optimal K": \"\"\"
from sklearn.metrics import silhouette_score

k_range = range(2, 10)
scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

best_k = k_range[np.argmax(scores)]
\"\"\",
        
        "DBSCAN": \"\"\"
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Count clusters and noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
\"\"\",
        
        "PCA Visualization": \"\"\"
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters in PCA Space')
plt.colorbar()
plt.show()
\"\"\"
    }
    
    for name, code in snippets.items():
        print(f"### {name}")
        print("```python")
        print(code.strip())
        print("```\\n")

def show_common_issues_solutions():
    """
    Display common clustering issues and their solutions.
    Troubleshooting guide for clustering problems.
    """
    print("⚠️ Common Issues & Solutions\\n")
    
    issues = [
        {
            'Issue': 'Poor clustering results',
            'Causes': ['Unscaled features', 'Wrong K', 'Outliers'],
            'Solutions': ['Always standardize data', 'Use elbow/silhouette', 'Try DBSCAN']
        },
        {
            'Issue': 'Clusters change each run',
            'Causes': ['Random initialization', 'Local minima'],
            'Solutions': ['Set random_state', 'Increase n_init', 'Try different algorithm']
        },
        {
            'Issue': 'Memory/speed issues',
            'Causes': ['Large dataset', 'High dimensions'],
            'Solutions': ['Use MiniBatchKMeans', 'Apply PCA first', 'Sample data']
        },
        {
            'Issue': 'All points in one cluster',
            'Causes': ['Data scale issues', 'Wrong parameters'],
            'Solutions': ['Check data distribution', 'Standardize features', 'Adjust parameters']
        }
    ]
    
    for issue in issues:
        print(f"❌ {issue['Issue']}")
        print(f"   Causes: {', '.join(issue['Causes'])}")
        print(f"   ✅ Solutions: {', '.join(issue['Solutions'])}")
        print()
    
    print("\\n💡 Pro Tips:")
    print("  1. Always visualize your data first")
    print("  2. Try multiple algorithms")
    print("  3. Validate with domain knowledge")
    print("  4. Document your choices")
    print("  5. Test stability with different seeds")

def show_course_summary():
    """
    Display course completion summary and next steps.
    Celebrate learning achievements.
    """
    print("\\n" + "="*60)
    print("🎯 Week 1 Complete: Clustering for Innovation Mastered!")
    print("="*60)
    print("\\nTotal content covered:")
    print("  • 8 sections")
    print("  • 5 clustering algorithms")
    print("  • 4 evaluation metrics")
    print("  • 3 comprehensive notebooks")
    print("  • Countless visualizations")
    print("\\n🚀 You're ready to find patterns in any innovation dataset!")
    print("\\n📚 Continue your journey with Week 2: Advanced Clustering")

# Ensure calinski_harabasz_score is imported
from sklearn.metrics import calinski_harabasz_score

print("✅ All Part 3 functions loaded successfully!")
print(f"Total functions: {len([name for name in dir() if callable(eval(name)) and not name.startswith('_')])}")
'''
    
    # Append additional functions to existing source
    existing_source.extend(additional_functions.split('\n'))
    nb['cells'][4]['source'] = existing_source
    
    # Save the updated notebook
    with open(r'D:\Joerg\Research\slides\ML_Design_Thinking_16\Week_01\Week01_Part3_Practice_Advanced.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print('Successfully added all remaining functions to Part 3')
else:
    print('Error: Functions cell not found at expected position')