import json
import re

# Load Part 1 notebook
with open(r'D:\Joerg\Research\slides\ML_Design_Thinking_16\Week_01\Week01_Part1_Setup_Foundation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the main functions cell (should be at index 5)
functions_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'def setup_plot_style' in source:
            functions_cell_idx = i
            break

if functions_cell_idx is None:
    print("Error: Could not find main functions cell")
    exit(1)

# Get existing functions
existing_source = nb['cells'][functions_cell_idx]['source']
existing_text = ''.join(existing_source)

# Add missing functions at the end of existing functions
additional_functions = '''

# ============================================================================
# PRE-DISCOVERY DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_innovation_data_types():
    """Show different types of innovation data."""
    print("🔍 Types of Innovation Data\\n")
    
    # Create sample datasets
    data_types = {
        'Customer Feedback': pd.DataFrame({
            'satisfaction': np.random.uniform(1, 10, 100),
            'engagement': np.random.uniform(0, 100, 100),
            'retention': np.random.uniform(0, 1, 100)
        }),
        'Product Features': pd.DataFrame({
            'complexity': np.random.uniform(1, 10, 50),
            'cost': np.random.uniform(100, 10000, 50),
            'time_to_market': np.random.uniform(1, 24, 50)
        }),
        'Market Segments': pd.DataFrame({
            'size': np.random.uniform(1000, 100000, 30),
            'growth_rate': np.random.uniform(-0.1, 0.5, 30),
            'competition': np.random.uniform(1, 10, 30)
        })
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for idx, (name, df) in enumerate(data_types.items()):
        ax = axes[idx]
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.6, s=50)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel(df.columns[0].replace('_', ' ').title())
        ax.set_ylabel(df.columns[1].replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Different Types of Innovation Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n💡 Each type requires different clustering approaches!")
    return data_types

def demonstrate_scale_challenge():
    """Demonstrate the challenge of scale in innovation analysis."""
    print("📊 The Scale Challenge in Innovation\\n")
    
    # Show increasing data sizes
    sizes = [10, 100, 1000, 10000]
    times = []
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, n in enumerate(sizes):
        # Generate data
        import time
        start = time.time()
        X = np.random.randn(n, 2)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(5, n//10), random_state=42)
        labels = kmeans.fit_predict(X)
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Visualize
        ax = axes[idx]
        if n <= 1000:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, alpha=0.6, cmap='viridis')
        else:
            # Sample for visualization
            sample_idx = np.random.choice(n, 1000, replace=False)
            ax.scatter(X[sample_idx, 0], X[sample_idx, 1], 
                      c=labels[sample_idx], s=20, alpha=0.6, cmap='viridis')
        
        ax.set_title(f'{n:,} Data Points\\nTime: {elapsed:.3f}s', fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    plt.suptitle('Scaling Challenge: More Data = More Complexity', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\\n📈 Processing time scaling: {times}")
    print("💡 Clustering helps manage large-scale innovation data efficiently!")

def demonstrate_hidden_patterns():
    """Show how patterns are hidden in raw data."""
    print("🔮 Finding Hidden Patterns\\n")
    
    # Generate data with hidden structure
    np.random.seed(42)
    
    # Create three hidden groups
    group1 = np.random.multivariate_normal([2, 2], [[0.5, 0.3], [0.3, 0.5]], 100)
    group2 = np.random.multivariate_normal([6, 2], [[0.5, -0.3], [-0.3, 0.5]], 100)
    group3 = np.random.multivariate_normal([4, 6], [[0.8, 0], [0, 0.8]], 100)
    
    X_hidden = np.vstack([group1, group2, group3])
    
    # Add noise dimension
    X_hidden = np.column_stack([X_hidden, np.random.randn(300)])
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Raw data - patterns not visible
    ax1 = axes[0]
    ax1.scatter(X_hidden[:, 0], X_hidden[:, 1], c='gray', alpha=0.5, s=30)
    ax1.set_title('Raw Data\\n(Patterns Hidden)', fontweight='bold')
    ax1.set_xlabel('Innovation Score')
    ax1.set_ylabel('Market Potential')
    
    # With color coding (if we knew the groups)
    ax2 = axes[1]
    true_labels = np.repeat([0, 1, 2], 100)
    ax2.scatter(X_hidden[:, 0], X_hidden[:, 1], c=true_labels, 
               cmap='viridis', alpha=0.6, s=30)
    ax2.set_title('Hidden Structure\\n(If We Knew)', fontweight='bold')
    ax2.set_xlabel('Innovation Score')
    ax2.set_ylabel('Market Potential')
    
    # After clustering
    ax3 = axes[2]
    kmeans = KMeans(n_clusters=3, random_state=42)
    predicted_labels = kmeans.fit_predict(X_hidden)
    ax3.scatter(X_hidden[:, 0], X_hidden[:, 1], c=predicted_labels, 
               cmap='viridis', alpha=0.6, s=30)
    ax3.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='*', s=300, edgecolors='black', linewidth=2)
    ax3.set_title('Patterns Revealed\\n(After Clustering)', fontweight='bold')
    ax3.set_xlabel('Innovation Score')
    ax3.set_ylabel('Market Potential')
    
    plt.suptitle('Clustering Reveals Hidden Innovation Patterns', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n✨ Clustering transforms chaos into clarity!")
    return X_hidden

def interactive_pattern_exercise():
    """Interactive exercise for pattern recognition."""
    print("🎯 Interactive Exercise: Can You Spot the Patterns?\\n")
    
    # Generate ambiguous data
    np.random.seed(42)
    n_samples = 200
    
    # Create data that could be 2, 3, or 4 clusters
    X_ambiguous = np.random.randn(n_samples, 2)
    X_ambiguous[:50] += [2, 2]
    X_ambiguous[50:100] += [2, -2]
    X_ambiguous[100:150] += [-2, 0]
    X_ambiguous[150:] += [0, 0]  # Overlapping with others
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Show raw data
    axes[0, 0].scatter(X_ambiguous[:, 0], X_ambiguous[:, 1], 
                      c='gray', alpha=0.5, s=30)
    axes[0, 0].set_title('Raw Data\\nHow many groups do you see?', fontweight='bold')
    
    # Try different K values
    for idx, k in enumerate([2, 3, 4]):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_ambiguous)
        
        # Top row: just clusters
        ax_top = axes[0, idx]
        ax_top.scatter(X_ambiguous[:, 0], X_ambiguous[:, 1], 
                      c=labels, cmap='viridis', alpha=0.6, s=30)
        ax_top.set_title(f'K = {k}', fontweight='bold')
        
        # Bottom row: with metrics
        ax_bottom = axes[1, idx]
        ax_bottom.scatter(X_ambiguous[:, 0], X_ambiguous[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6, s=30)
        
        silhouette = silhouette_score(X_ambiguous, labels)
        ax_bottom.set_title(f'K = {k}\\nSilhouette: {silhouette:.3f}', 
                           fontweight='bold')
    
    plt.suptitle('Pattern Recognition Exercise: Finding the Right K', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n📊 Results:")
    print("• K=2: Clear separation (might be too simple)")
    print("• K=3: Good balance")
    print("• K=4: Possibly overfitting")
    print("\\n💡 Lesson: Use metrics + domain knowledge to decide!")

# ============================================================================
# FOUNDATION DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_innovation_discovery():
    """Show the innovation discovery process."""
    print("🚀 Innovation Discovery Process\\n")
    
    # Simulate innovation discovery pipeline
    stages = {
        'Raw Ideas': np.random.randn(500, 2) * 3,
        'Initial Filter': np.random.randn(300, 2) * 2.5,
        'Refined Concepts': np.random.randn(150, 2) * 2,
        'Final Innovations': np.random.randn(50, 2) * 1.5
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, (stage, data) in enumerate(stages.items()):
        ax = axes[idx]
        
        # Apply clustering at each stage
        n_clusters = max(2, len(data) // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, 
                           cmap='viridis', alpha=0.6, s=30)
        ax.set_title(f'{stage}\\n({len(data)} items)', fontweight='bold')
        ax.set_xlabel('Innovation Score')
        ax.set_ylabel('Feasibility')
        
        # Add stage number
        ax.text(0.5, 0.95, f'Stage {idx+1}', 
               transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Innovation Discovery Pipeline: From Ideas to Implementation', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n📈 Pipeline Benefits:")
    print("• Stage 1: Capture all ideas")
    print("• Stage 2: Group similar concepts")
    print("• Stage 3: Identify promising clusters")
    print("• Stage 4: Focus on best opportunities")
    
    return stages

def demonstrate_hidden_complexity():
    """Demonstrate hidden complexity in simple-looking data."""
    print("🌐 Hidden Complexity in Innovation Data\\n")
    
    # Create data with hidden complexity
    np.random.seed(42)
    
    # Simple 2D view
    X_2d = np.random.randn(300, 2)
    
    # Actually has complex 5D structure
    X_5d = np.random.randn(300, 5)
    X_5d[:100] = np.random.randn(100, 5) + [2, 0, 1, -1, 0]
    X_5d[100:200] = np.random.randn(100, 5) + [-2, 1, 0, 0, 2]
    X_5d[200:] = np.random.randn(100, 5) + [0, -1, -1, 2, -2]
    
    # Project to 2D for visualization
    pca = PCA(n_components=2)
    X_2d_projection = pca.fit_transform(X_5d)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Row 1: Different 2D projections
    for i in range(3):
        ax = axes[0, i]
        ax.scatter(X_5d[:, i], X_5d[:, (i+1)%5], c='gray', alpha=0.5, s=30)
        ax.set_title(f'Features {i+1} vs {(i+2)%5+1}', fontweight='bold')
        ax.set_xlabel(f'Feature {i+1}')
        ax.set_ylabel(f'Feature {(i+2)%5+1}')
    
    # Row 2: Clustering results
    # 2D clustering
    kmeans_2d = KMeans(n_clusters=3, random_state=42)
    labels_2d = kmeans_2d.fit_predict(X_2d_projection)
    axes[1, 0].scatter(X_2d_projection[:, 0], X_2d_projection[:, 1], 
                      c=labels_2d, cmap='viridis', alpha=0.6, s=30)
    axes[1, 0].set_title('2D Clustering', fontweight='bold')
    
    # 5D clustering
    kmeans_5d = KMeans(n_clusters=3, random_state=42)
    labels_5d = kmeans_5d.fit_predict(X_5d)
    axes[1, 1].scatter(X_2d_projection[:, 0], X_2d_projection[:, 1], 
                      c=labels_5d, cmap='viridis', alpha=0.6, s=30)
    axes[1, 1].set_title('5D Clustering (shown in 2D)', fontweight='bold')
    
    # True structure
    true_labels = np.repeat([0, 1, 2], 100)
    axes[1, 2].scatter(X_2d_projection[:, 0], X_2d_projection[:, 1], 
                      c=true_labels, cmap='viridis', alpha=0.6, s=30)
    axes[1, 2].set_title('True Structure', fontweight='bold')
    
    plt.suptitle('Hidden Complexity: What You See Is Not All There Is', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n🔍 Key Insights:")
    print("• 2D views miss important patterns")
    print("• Higher dimensions reveal true structure")
    print("• Multiple features capture full complexity")
    print("• Clustering works in high dimensions!")

def compare_traditional_vs_ai():
    """Compare traditional analysis vs AI-powered clustering."""
    print("⚖️ Traditional Analysis vs AI-Powered Clustering\\n")
    
    # Generate complex innovation data
    np.random.seed(42)
    n_innovations = 200
    
    # Create data with non-linear patterns
    theta = np.random.uniform(0, 2*np.pi, n_innovations)
    r = np.random.uniform(0, 3, n_innovations)
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Add third cluster in center
    X[::3] = np.random.randn(len(X[::3]), 2) * 0.3
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Traditional approach (manual thresholds)
    ax1 = axes[0, 0]
    traditional_labels = np.zeros(len(X))
    traditional_labels[X[:, 0] > 1] = 1
    traditional_labels[X[:, 1] > 1] = 2
    ax1.scatter(X[:, 0], X[:, 1], c=traditional_labels, 
               cmap='viridis', alpha=0.6, s=30)
    ax1.set_title('Traditional\\n(Manual Rules)', fontweight='bold')
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    # K-means
    ax2 = axes[0, 1]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels, 
               cmap='viridis', alpha=0.6, s=30)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='*', s=300, edgecolors='black', linewidth=2)
    ax2.set_title('AI: K-Means', fontweight='bold')
    
    # DBSCAN
    ax3 = axes[0, 2]
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    ax3.scatter(X[:, 0], X[:, 1], c=dbscan_labels, 
               cmap='viridis', alpha=0.6, s=30)
    ax3.set_title('AI: DBSCAN', fontweight='bold')
    
    # Performance comparison
    ax4 = axes[1, 0]
    methods = ['Traditional', 'K-Means', 'DBSCAN']
    scores = [
        -0.1,  # Traditional performs poorly
        silhouette_score(X, kmeans_labels),
        silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -0.2
    ]
    bars = ax4.bar(methods, scores, color=['red', 'green', 'blue'])
    ax4.set_title('Performance Comparison', fontweight='bold')
    ax4.set_ylabel('Silhouette Score')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Scalability
    ax5 = axes[1, 1]
    data_sizes = [100, 1000, 10000, 100000]
    traditional_time = [0.01, 0.1, 10, 1000]  # Exponential growth
    ai_time = [0.01, 0.05, 0.2, 1]  # Much better scaling
    ax5.semilogy(data_sizes, traditional_time, 'r-o', label='Traditional')
    ax5.semilogy(data_sizes, ai_time, 'g-o', label='AI-Powered')
    ax5.set_xlabel('Data Size')
    ax5.set_ylabel('Processing Time (s)')
    ax5.set_title('Scalability', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Adaptability
    ax6 = axes[1, 2]
    capabilities = ['Linear\\nPatterns', 'Non-linear\\nPatterns', 'Outlier\\nHandling', 
                   'Auto\\nOptimization', 'High\\nDimensions']
    traditional_scores = [3, 1, 1, 0, 1]
    ai_scores = [5, 5, 4, 5, 5]
    
    x = np.arange(len(capabilities))
    width = 0.35
    ax6.bar(x - width/2, traditional_scores, width, label='Traditional', color='red', alpha=0.7)
    ax6.bar(x + width/2, ai_scores, width, label='AI-Powered', color='green', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(capabilities, fontsize=9)
    ax6.set_ylabel('Capability Score')
    ax6.set_title('Capability Comparison', fontweight='bold')
    ax6.legend()
    
    plt.suptitle('Traditional vs AI-Powered Innovation Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n🏆 AI-Powered Advantages:")
    print("• Finds complex patterns automatically")
    print("• Scales to massive datasets")
    print("• Adapts to data structure")
    print("• No manual rule creation needed")

def demonstrate_dual_pipeline():
    """Show the dual pipeline: ML + Design Thinking."""
    print("🔄 Dual Pipeline: ML + Design Thinking\\n")
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    
    # ML Pipeline (top row)
    ml_stages = ['Data', 'Preprocess', 'Model', 'Evaluate', 'Deploy']
    colors_ml = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA77B']
    
    for i, (stage, color) in enumerate(zip(ml_stages, colors_ml)):
        ax = axes[0, i]
        
        # Create visual representation
        circle = plt.Circle((0.5, 0.5), 0.3, color=color, alpha=0.7)
        ax.add_patch(circle)
        ax.text(0.5, 0.5, stage, ha='center', va='center', 
               fontsize=11, fontweight='bold')
        
        # Add connections
        if i < len(ml_stages) - 1:
            ax.arrow(0.85, 0.5, 0.1, 0, head_width=0.05, 
                    head_length=0.03, fc='gray', ec='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add description
        descriptions = [
            'Innovation\\nData',
            'Clean &\\nScale',
            'Cluster\\nAnalysis',
            'Validate\\nResults',
            'Innovation\\nInsights'
        ]
        ax.text(0.5, 0.1, descriptions[i], ha='center', fontsize=9, 
               style='italic', color='gray')
    
    # Design Pipeline (bottom row)
    design_stages = ['Empathize', 'Define', 'Ideate', 'Prototype', 'Test']
    colors_design = ['#FF6B9D', '#C44569', '#F8961E', '#90BE6D', '#577590']
    
    for i, (stage, color) in enumerate(zip(design_stages, colors_design)):
        ax = axes[1, i]
        
        # Create visual representation
        diamond = plt.Polygon([(0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.2, 0.5)],
                             color=color, alpha=0.7)
        ax.add_patch(diamond)
        ax.text(0.5, 0.5, stage, ha='center', va='center', 
               fontsize=11, fontweight='bold')
        
        # Add connections
        if i < len(design_stages) - 1:
            ax.arrow(0.85, 0.5, 0.1, 0, head_width=0.05, 
                    head_length=0.03, fc='gray', ec='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add ML integration points
        ml_integration = [
            'Cluster\\nUsers',
            'Pattern\\nAnalysis',
            'Generate\\nConcepts',
            'Test\\nVariations',
            'Validate\\nResults'
        ]
        ax.text(0.5, 0.9, ml_integration[i], ha='center', fontsize=9, 
               style='italic', color='blue')
    
    # Add pipeline labels
    fig.text(0.08, 0.75, 'ML Pipeline', fontsize=14, fontweight='bold', 
            rotation=90, va='center')
    fig.text(0.08, 0.25, 'Design Pipeline', fontsize=14, fontweight='bold', 
            rotation=90, va='center')
    
    plt.suptitle('Integrated Innovation Process: ML + Design Thinking', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\\n🔗 Integration Benefits:")
    print("• ML provides data-driven insights")
    print("• Design ensures human-centered solutions")
    print("• Parallel processing accelerates innovation")
    print("• Continuous feedback improves both pipelines")

def demonstrate_convergence_flow():
    """Demonstrate how insights converge through the pipeline."""
    print("🎯 Convergence: From Many to Few\\n")
    
    # Create convergence visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define stages and their data points
    stages = [
        ('Raw Ideas', 1000, 0),
        ('Clustered Concepts', 100, 1),
        ('Refined Groups', 20, 2),
        ('Key Themes', 5, 3),
        ('Innovation Focus', 1, 4)
    ]
    
    # Create funnel visualization
    y_positions = np.linspace(0.9, 0.1, len(stages))
    
    for i, (stage, count, x_pos) in enumerate(stages):
        y = y_positions[i]
        width = np.log10(count + 1) / 4  # Width based on count
        
        # Draw trapezoid for funnel effect
        if i < len(stages) - 1:
            next_width = np.log10(stages[i+1][1] + 1) / 4
            next_y = y_positions[i+1]
            
            trapezoid = plt.Polygon([
                (0.5 - width, y),
                (0.5 + width, y),
                (0.5 + next_width, next_y),
                (0.5 - next_width, next_y)
            ], alpha=0.3, color=plt.cm.viridis(i/len(stages)))
            ax.add_patch(trapezoid)
        
        # Add stage labels
        ax.text(0.5, y, f'{stage}\\n({count:,} items)', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add process labels on the side
        processes = ['Collect', 'Cluster', 'Analyze', 'Synthesize', 'Focus']
        if i < len(stages) - 1:
            ax.text(0.9, (y + y_positions[i+1])/2, processes[i], 
                   ha='center', va='center', fontsize=10, style='italic',
                   color='gray')
            
            # Add arrow
            ax.annotate('', xy=(0.85, y_positions[i+1] + 0.02), 
                       xytext=(0.85, y - 0.02),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title and description
    ax.text(0.5, 0.95, 'Innovation Convergence Process', 
           ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.02, 'Clustering helps filter and focus innovation efforts', 
           ha='center', fontsize=12, style='italic', color='gray')
    
    # Add benefit annotations
    benefits = [
        (0.1, 0.8, 'Volume'),
        (0.1, 0.5, 'Quality'),
        (0.1, 0.2, 'Focus')
    ]
    
    for x, y, text in benefits:
        ax.text(x, y, text, fontsize=11, fontweight='bold', color='blue')
    
    plt.show()
    
    print("\\n📊 Convergence Metrics:")
    for stage, count, _ in stages:
        print(f"• {stage}: {count:,} items")
    
    print("\\n✨ Result: 1000x reduction while preserving key insights!")

# Add these functions to the existing source
'''

# Insert the additional functions into the cell source
existing_source.extend(additional_functions.split('\n'))
nb['cells'][functions_cell_idx]['source'] = existing_source

# Save the updated notebook
with open(r'D:\Joerg\Research\slides\ML_Design_Thinking_16\Week_01\Week01_Part1_Setup_Foundation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully added missing functions to Part 1")
print(f"Functions cell now at index: {functions_cell_idx}")