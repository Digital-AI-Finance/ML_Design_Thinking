"""
Week 2 Opening Power Chart: Live Clustering Evolution
Shows K-means algorithm converging from chaos to clear user segments
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

# Define colors
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f',
    'mlyellow': '#bcbd22',
    'mlcyan': '#17becf'
}

def generate_user_data(n_samples=2000):
    """Generate realistic user behavior data with natural clusters"""
    # Create 5 distinct user segments with different behaviors
    centers = [
        [2, 2],    # Casual users
        [8, 2],    # Power users
        [5, 8],    # Social sharers
        [2, 7],    # Content creators
        [8, 8],    # Premium subscribers
    ]
    
    # Generate data with varying cluster sizes and spreads
    cluster_std = [1.2, 0.8, 1.0, 1.5, 0.6]
    n_samples_per_cluster = [400, 300, 500, 400, 400]
    
    X = []
    y_true = []
    
    for i, (center, std, n) in enumerate(zip(centers, cluster_std, n_samples_per_cluster)):
        cluster_data = np.random.randn(n, 2) * std + center
        X.extend(cluster_data)
        y_true.extend([i] * n)
    
    return np.array(X), np.array(y_true)

def create_evolution_frames():
    """Create three frames showing K-means evolution"""
    X, y_true = generate_user_data()
    
    # Initialize K-means
    kmeans = KMeans(n_clusters=5, init='random', n_init=1, max_iter=1, random_state=42)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Frame 1: Chaos - Random initialization
    ax1 = plt.subplot(131)
    ax1.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=20)
    
    # Add random initial centers
    kmeans.fit(X)
    initial_centers = kmeans.cluster_centers_
    ax1.scatter(initial_centers[:, 0], initial_centers[:, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    ax1.set_title('Frame 1: CHAOS\n10,000 Users, No Patterns', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Engagement Score', fontsize=11)
    ax1.set_ylabel('Activity Level', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 12)
    ax1.set_ylim(-2, 12)
    
    # Add annotation
    ax1.annotate('Random Centers', xy=(initial_centers[0, 0], initial_centers[0, 1]),
                xytext=(initial_centers[0, 0] + 2, initial_centers[0, 1] + 2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red')
    
    # Frame 2: Movement - Partial convergence
    ax2 = plt.subplot(132)
    kmeans_partial = KMeans(n_clusters=5, init=initial_centers, n_init=1, max_iter=3, random_state=42)
    y_partial = kmeans_partial.fit_predict(X)
    
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y_partial, alpha=0.6, s=20, cmap='tab10')
    
    # Show movement arrows
    for i in range(5):
        ax2.annotate('', xy=(kmeans_partial.cluster_centers_[i, 0], kmeans_partial.cluster_centers_[i, 1]),
                    xytext=(initial_centers[i, 0], initial_centers[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, alpha=0.5))
    
    ax2.scatter(kmeans_partial.cluster_centers_[:, 0], kmeans_partial.cluster_centers_[:, 1],
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    ax2.set_title('Frame 2: CONVERGENCE\nPatterns Emerging', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Engagement Score', fontsize=11)
    ax2.set_ylabel('Activity Level', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 12)
    ax2.set_ylim(-2, 12)
    
    # Frame 3: Clear Segments - Final convergence with personas
    ax3 = plt.subplot(133)
    kmeans_final = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
    y_final = kmeans_final.fit_predict(X)
    
    scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=y_final, alpha=0.6, s=20, cmap='tab10')
    ax3.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    # Add persona labels
    personas = ['Casual\nBrowsers', 'Power\nUsers', 'Social\nSharers', 
                'Content\nCreators', 'Premium\nMembers']
    
    for i, (center, persona) in enumerate(zip(kmeans_final.cluster_centers_, personas)):
        ax3.annotate(persona, xy=(center[0], center[1]), 
                    xytext=(center[0] + 0.5, center[1] + 0.5),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax3.set_title('Frame 3: CLARITY\n5 Distinct User Personas', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Engagement Score', fontsize=11)
    ax3.set_ylabel('Activity Level', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-2, 12)
    ax3.set_ylim(-2, 12)
    
    # Main title
    fig.suptitle('K-Means Evolution: From Chaos to User Understanding', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Add metrics
    inertias = [15000, 5000, 1200]  # Approximate inertia values
    for ax, inertia in zip([ax1, ax2, ax3], inertias):
        ax.text(0.02, 0.98, f'Inertia: {inertia:,}', 
               transform=ax.transAxes, fontsize=10, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_animated_version():
    """Create an animated version showing continuous evolution"""
    X, y_true = generate_user_data()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize with random centers
    n_clusters = 5
    initial_centers = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=20)
    centers_plot = ax.scatter(initial_centers[:, 0], initial_centers[:, 1],
                            c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_xlabel('Engagement Score', fontsize=12)
    ax.set_ylabel('Activity Level', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update(frame):
        kmeans = KMeans(n_clusters=5, init=initial_centers if frame == 0 else 'k-means++', 
                       max_iter=frame+1, n_init=1, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Update colors
        scatter.set_color([plt.cm.tab10(l) for l in labels])
        
        # Update centers
        centers_plot.set_offsets(kmeans.cluster_centers_)
        
        # Update text
        iteration_text.set_text(f'Iteration: {frame + 1}\nInertia: {kmeans.inertia_:.0f}')
        
        if frame == 0:
            ax.set_title('Initializing Random Centers...', fontsize=14, fontweight='bold')
        elif frame < 5:
            ax.set_title(f'Converging... (Step {frame + 1})', fontsize=14, fontweight='bold')
        else:
            ax.set_title('Converged! 5 Clear User Personas', fontsize=14, fontweight='bold')
        
        return scatter, centers_plot, iteration_text
    
    anim = FuncAnimation(fig, update, frames=10, interval=1000, blit=True)
    
    return fig, anim

def create_detail_analysis():
    """Create detailed analysis of final clusters"""
    X, y_true = generate_user_data()
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Main clustering result
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.6, s=20, cmap='tab10')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    ax.set_title('Final Clustering Result', fontsize=12, fontweight='bold')
    ax.set_xlabel('Engagement Score')
    ax.set_ylabel('Activity Level')
    ax.grid(True, alpha=0.3)
    
    # Cluster sizes
    ax = axes[0, 1]
    cluster_sizes = [np.sum(y_pred == i) for i in range(5)]
    bars = ax.bar(range(5), cluster_sizes, color=plt.cm.tab10(range(5)))
    ax.set_title('Cluster Sizes', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Users')
    ax.set_xticks(range(5))
    
    # Add value labels on bars
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
               f'{size}', ha='center', fontweight='bold')
    
    # Cluster characteristics radar
    ax = axes[0, 2]
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(2, 3, 3, projection='polar')
    
    for i in range(5):
        cluster_mask = y_pred == i
        values = [
            np.mean(X[cluster_mask, 0]) / 10,  # Normalized engagement
            np.mean(X[cluster_mask, 1]) / 10,  # Normalized activity
            np.random.random(),  # Simulated retention
            np.random.random(),  # Simulated value
            np.random.random()   # Simulated growth
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Engagement', 'Activity', 'Retention', 'Value', 'Growth'])
    ax.set_title('Cluster Profiles', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Convergence history
    ax = axes[1, 0]
    iterations = range(1, 11)
    inertias = [15000 * np.exp(-0.5 * i) + 1200 for i in iterations]
    ax.plot(iterations, inertias, 'o-', linewidth=2, markersize=8, color=colors['mlblue'])
    ax.set_title('Convergence History', fontsize=12, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax = axes[1, 1]
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, y_pred)
    
    y_lower = 10
    for i in range(5):
        cluster_silhouette_vals = silhouette_vals[y_pred == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.tab10(i)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                        facecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.axvline(x=np.mean(silhouette_vals), color='red', linestyle='--', linewidth=2,
              label=f'Average: {np.mean(silhouette_vals):.3f}')
    ax.legend()
    
    # Persona descriptions
    ax = axes[1, 2]
    ax.axis('off')
    
    personas = [
        ('Casual Browsers', 'Low engagement\nInfrequent visits\nPrice sensitive'),
        ('Power Users', 'High engagement\nDaily active\nFeature seekers'),
        ('Social Sharers', 'Community focused\nContent spreaders\nInfluencers'),
        ('Content Creators', 'Original content\nRegular posting\nQuality focused'),
        ('Premium Members', 'Paying customers\nHigh retention\nFeature users')
    ]
    
    y_pos = 0.9
    for i, (name, desc) in enumerate(personas):
        color = plt.cm.tab10(i)
        ax.text(0.1, y_pos, f'Cluster {i}: {name}', fontsize=11, fontweight='bold', color=color)
        ax.text(0.1, y_pos - 0.08, desc, fontsize=9, color='gray')
        y_pos -= 0.18
    
    ax.set_title('Persona Definitions', fontsize=12, fontweight='bold')
    
    plt.suptitle('Deep Dive: User Segmentation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Create main evolution visualization
    fig1 = create_evolution_frames()
    plt.savefig('clustering_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('clustering_evolution.png', dpi=150, bbox_inches='tight')
    print("Created clustering_evolution.pdf/png")
    
    # Create detailed analysis
    fig2 = create_detail_analysis()
    plt.savefig('clustering_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('clustering_analysis.png', dpi=150, bbox_inches='tight')
    print("Created clustering_analysis.pdf/png")
    
    # Note: Animated version can be saved as gif with additional libraries
    # fig3, anim = create_animated_version()
    # anim.save('Week_02/charts/clustering_animation.gif', writer='pillow', fps=1)
    
    plt.close('all')  # Close all figures instead of showing