#!/usr/bin/env python3
"""
Create all remaining visualization charts for Week 1
This script generates all the visualizations that haven't been created yet
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx

# Set global style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def create_user_empathy_visual():
    """Create user empathy visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Traditional empathy (left)
    ax1.add_patch(plt.Circle((0.5, 0.5), 0.3, color='lightblue', alpha=0.5))
    ax1.text(0.5, 0.5, 'Traditional\nEmpathy\n\n5-20 users\nInterviews', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Traditional Approach', fontsize=14)
    ax1.axis('off')
    
    # ML-enhanced empathy (right)
    for i in range(50):
        x = np.random.random()
        y = np.random.random()
        ax2.add_patch(plt.Circle((x, y), 0.02, color='darkblue', alpha=0.3))
    ax2.text(0.5, 0.5, 'ML-Enhanced\nEmpathy\n\n1000s of users\nAutomated', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('ML-Enhanced Approach', fontsize=14)
    ax2.axis('off')
    
    plt.suptitle('Scaling Empathy with Machine Learning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('user_empathy_visual.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('user_empathy_visual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("User empathy visual created")

def create_clustering_examples():
    """Create real-world clustering examples grid"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    examples = [
        ('Spotify\nListeners', 'Music Taste'),
        ('Netflix\nViewers', 'Genre Preference'),
        ('Amazon\nShoppers', 'Purchase Behavior'),
        ('Instagram\nUsers', 'Content Type'),
        ('Gmail\nEmails', 'Message Category'),
        ('App\nUsers', 'Usage Pattern')
    ]
    
    for idx, (ax, (title, subtitle)) in enumerate(zip(axes.flat, examples)):
        # Generate sample clusters
        X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=idx)
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for i in range(3):
            mask = y == i
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=20)
        ax.set_title(f'{title}\n({subtitle})', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Clustering in Real-World Applications', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clustering_examples.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('clustering_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Clustering examples created")

def create_distance_visual():
    """Create distance measurement visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create three users
    users = {
        'User A': [2, 2],
        'User B': [2.5, 2.5],
        'User C': [8, 8]
    }
    
    colors = {'User A': '#3498db', 'User B': '#2ecc71', 'User C': '#e74c3c'}
    
    for name, pos in users.items():
        ax.scatter(pos[0], pos[1], s=200, c=colors[name], label=name, 
                  edgecolors='black', linewidth=2)
        ax.text(pos[0], pos[1]-0.5, name, ha='center', fontsize=10)
    
    # Draw distance lines
    ax.plot([users['User A'][0], users['User B'][0]], 
           [users['User A'][1], users['User B'][1]], 
           'k--', alpha=0.5, linewidth=2)
    ax.text(2.25, 2.0, 'Close\n(Similar)', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.plot([users['User A'][0], users['User C'][0]], 
           [users['User A'][1], users['User C'][1]], 
           'k--', alpha=0.5, linewidth=2)
    ax.text(5, 5, 'Far\n(Different)', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    ax.set_xlabel('Hours per Day', fontsize=12)
    ax.set_ylabel('Features Used', fontsize=12)
    ax.set_title('Measuring Similarity Between Users', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('distance_visual.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('distance_visual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Distance visual created")

def create_kmeans_animation():
    """Create K-means steps visualization"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Generate data
    X, y_true = make_blobs(n_samples=150, centers=3, random_state=42)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    titles = ['Step 1: Random Centers', 'Step 2: Assign Points', 
              'Step 3: Update Centers', 'Step 4: Converged']
    
    for step, (ax, title) in enumerate(zip(axes, titles)):
        kmeans = KMeans(n_clusters=3, random_state=42, max_iter=step+1, n_init=1)
        kmeans.fit(X)
        
        # Plot points
        for i in range(3):
            mask = kmeans.labels_ == i
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=30)
        
        # Plot centers
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                  c='black', marker='X', s=200, edgecolors='white', linewidth=2)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('K-means Algorithm: Step by Step', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kmeans_animation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('kmeans_animation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" K-means animation created")

def create_dendrogram_cut():
    """Create dendrogram with cut line"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate data
    X, _ = make_blobs(n_samples=20, centers=3, random_state=42)
    
    # Hierarchical clustering
    linkage_matrix = linkage(X, method='ward')
    
    # Plot dendrogram
    dendrogram(linkage_matrix, ax=ax, color_threshold=7)
    
    # Add cut line
    ax.axhline(y=7, c='red', linestyle='--', linewidth=2, 
              label='Cut here for 3 clusters')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Cutting the Dendrogram to Get Clusters', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('dendrogram_cut.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('dendrogram_cut.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Dendrogram cut created")

def create_cluster_quality():
    """Create cluster quality visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Good clusters (tight and separated)
    X1, y1 = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i in range(3):
        mask = y1 == i
        axes[0].scatter(X1[mask, 0], X1[mask, 1], c=colors[i], alpha=0.7, s=30)
    axes[0].set_title('Y Tight Groups\nLow variance within clusters', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Separated clusters
    X2, y2 = make_blobs(n_samples=150, centers=3, cluster_std=0.5, 
                       center_box=(-10, 10), random_state=42)
    for i in range(3):
        mask = y2 == i
        axes[1].scatter(X2[mask, 0], X2[mask, 1], c=colors[i], alpha=0.7, s=30)
    axes[1].set_title('Y Separated Groups\nClear boundaries', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Meaningful clusters (with labels)
    axes[2].scatter([1, 1.2, 0.8], [1, 0.8, 1.2], c=colors[0], s=100, label='New Users')
    axes[2].scatter([5, 5.2, 4.8], [5, 4.8, 5.2], c=colors[1], s=100, label='Regular')
    axes[2].scatter([9, 9.2, 8.8], [9, 8.8, 9.2], c=colors[2], s=100, label='Power Users')
    axes[2].set_title('Y Makes Sense\nBusiness meaning', fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 10)
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Three Checks for Good Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_quality.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cluster_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Cluster quality created")

def create_dbscan_shapes():
    """Create DBSCAN non-spherical clusters visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # K-means on non-spherical data
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    axes[0].set_title('K-means: Assumes Round Clusters\n(Poor result on crescents)', 
                     fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # DBSCAN on same data
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
    axes[1].set_title('DBSCAN: Finds Any Shape\n(Perfect on crescents)', 
                     fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('DBSCAN Advantage: Finding Non-Round Clusters', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dbscan_shapes.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('dbscan_shapes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" DBSCAN shapes created")

def create_feature_importance():
    """Create feature importance chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = ['Daily Usage Hours', 'Features Used', 'Days Active', 
                'Messages Sent', 'Files Shared', 'Sessions per Day',
                'Peak Hour', 'Device Type', 'Account Age', 'Location']
    importance = [0.95, 0.88, 0.82, 0.75, 0.68, 0.62, 0.45, 0.38, 0.25, 0.15]
    colors = ['green' if x > 0.7 else 'orange' if x > 0.4 else 'red' for x in importance]
    
    bars = ax.barh(features, importance, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars, importance):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{val:.0%}', va='center', fontsize=9)
    
    ax.set_xlabel('Feature Importance for Clustering', fontsize=12)
    ax.set_title('Choosing the Right Features for User Clustering', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    ax.text(0.5, -2, 'Green: High importance (>70%)  Orange: Medium (40-70%)  Red: Low (<40%)',
           fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Feature importance created")

def create_customer_segments():
    """Create customer segmentation visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Segment distribution pie chart
    segments = ['VIP Shoppers', 'Bargain Hunters', 'Window Shoppers', 
                'Loyal Regulars', 'One-timers']
    sizes = [15, 25, 20, 30, 10]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    ax1.pie(sizes, labels=segments, colors=colors, autopct='%1.0f%%',
           startangle=90)
    ax1.set_title('Customer Segment Distribution', fontsize=12, fontweight='bold')
    
    # Segment characteristics
    X, y = make_blobs(n_samples=500, centers=5, random_state=42)
    
    for i in range(5):
        mask = y == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                   label=segments[i], alpha=0.6, s=30)
    
    ax2.set_xlabel('Average Order Value', fontsize=11)
    ax2.set_ylabel('Purchase Frequency', fontsize=11)
    ax2.set_title('Segment Characteristics', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('E-commerce Customer Segmentation Results', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('customer_segments.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('customer_segments.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Customer segments created")

def create_pain_points_heatmap():
    """Create pain points heatmap"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    clusters = ['Power Users', 'New Users', 'Mobile Users', 'Free Users']
    pain_points = ['Complex UI', 'Slow Loading', 'Missing Features', 
                   'Poor Documentation', 'Sync Issues']
    
    # Create intensity matrix
    data = np.array([[0.2, 0.8, 0.9, 0.1, 0.7],
                     [0.9, 0.3, 0.2, 0.8, 0.3],
                     [0.5, 0.9, 0.7, 0.4, 0.8],
                     [0.4, 0.5, 0.8, 0.6, 0.2]])
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pain_points)))
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_xticklabels(pain_points)
    ax.set_yticklabels(clusters)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pain Intensity', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(clusters)):
        for j in range(len(pain_points)):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Pain Points Heatmap by User Cluster', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pain_points_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('pain_points_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Pain points heatmap created")

def create_remaining_charts():
    """Create all other remaining charts with simplified versions"""
    
    # Behavior patterns
    fig, ax = plt.subplots(figsize=(10, 6))
    times = np.arange(0, 24, 1)
    patterns = {
        'Morning Rushers': np.exp(-(times - 7)**2 / 10),
        'Deep Workers': np.exp(-(times - 14)**2 / 20),
        'Night Owls': np.exp(-(times - 22)**2 / 15)
    }
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for (name, pattern), color in zip(patterns.items(), colors):
        ax.plot(times, pattern, label=name, linewidth=2, color=color)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Activity Level', fontsize=12)
    ax.set_title('User Behavior Patterns Throughout the Day', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('behavior_patterns.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('behavior_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Behavior patterns created")
    
    # Journey map
    fig, ax = plt.subplots(figsize=(12, 6))
    stages = ['Discover', 'Explore', 'Onboard', 'Engage', 'Master']
    cluster1 = [3, 5, 7, 9, 10]
    cluster2 = [2, 3, 4, 6, 8]
    cluster3 = [1, 2, 3, 4, 5]
    x = np.arange(len(stages))
    ax.plot(x, cluster1, 'o-', label='Power Users', linewidth=2, markersize=8)
    ax.plot(x, cluster2, 's-', label='Regular Users', linewidth=2, markersize=8)
    ax.plot(x, cluster3, '^-', label='Casual Users', linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel('Engagement Level', fontsize=12)
    ax.set_title('User Journey Maps by Cluster', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('journey_map_clusters.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('journey_map_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Journey map created")
    
    # Stakeholder network
    fig, ax = plt.subplots(figsize=(10, 8))
    G = nx.Graph()
    stakeholders = ['Decision\nMakers', 'Daily\nUsers', 'Influencers', 'Evaluators']
    G.add_nodes_from(stakeholders)
    G.add_edges_from([('Decision\nMakers', 'Daily\nUsers'),
                      ('Decision\nMakers', 'Influencers'),
                      ('Influencers', 'Daily\nUsers'),
                      ('Influencers', 'Evaluators'),
                      ('Evaluators', 'Daily\nUsers')])
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, ax=ax)
    ax.set_title('Stakeholder Network from Cluster Analysis', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('stakeholder_network.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('stakeholder_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Stakeholder network created")
    
    # Persona cards
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    personas = [
        {'name': 'Power Paula', 'age': '32', 'role': 'Manager', 'usage': '7h/day'},
        {'name': 'Regular Rob', 'age': '28', 'role': 'Developer', 'usage': '4h/day'},
        {'name': 'Casual Carl', 'age': '24', 'role': 'Student', 'usage': '1h/day'}
    ]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for ax, persona, color in zip(axes, personas, colors):
        ax.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor=color, alpha=0.2))
        ax.text(0.5, 0.7, persona['name'], ha='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.5, f"Age: {persona['age']}", ha='center', fontsize=10)
        ax.text(0.5, 0.4, f"Role: {persona['role']}", ha='center', fontsize=10)
        ax.text(0.5, 0.3, f"Usage: {persona['usage']}", ha='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.suptitle('Data-Driven Persona Cards', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('persona_cards.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('persona_cards.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Persona cards created")
    
    # Design priority matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    features = ['Bulk Operations', 'Better Onboarding', 'Mobile Features', 
                'Advanced Analytics', 'Social Sharing', 'API Access',
                'Dark Mode', 'Shortcuts', 'Templates', 'Integrations']
    effort = np.random.uniform(1, 10, len(features))
    impact = np.random.uniform(1, 10, len(features))
    
    colors = ['green' if i > 6 and e < 5 else 'orange' if i > 5 else 'red' 
             for i, e in zip(impact, effort)]
    
    ax.scatter(effort, impact, s=200, c=colors, alpha=0.6, edgecolors='black')
    
    for i, txt in enumerate(features):
        ax.annotate(txt, (effort[i], impact[i]), fontsize=8, ha='center')
    
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Implementation Effort', fontsize=12)
    ax.set_ylabel('User Impact', fontsize=12)
    ax.set_title('Design Priority Matrix from Cluster Insights', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.text(2.5, 7.5, 'Quick Wins', fontsize=11, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.text(7.5, 7.5, 'Major Projects', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.text(2.5, 2.5, 'Fill-ins', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax.text(7.5, 2.5, 'Low Priority', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral'))
    
    plt.tight_layout()
    plt.savefig('design_priority_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('design_priority_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Design priority matrix created")
    
    # Clustering methods comparison
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    methods = ['K-means', 'Hierarchical', 'DBSCAN', 'GMM', 'Mean Shift', 'Spectral']
    
    for ax, method in zip(axes.flat, methods):
        if method == 'K-means':
            X, y = make_blobs(n_samples=100, centers=3, random_state=42)
        elif method == 'DBSCAN':
            X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
        else:
            X, y = make_blobs(n_samples=100, centers=3, random_state=42)
        
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, s=20)
        ax.set_title(method, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Clustering Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clustering_methods_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('clustering_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Clustering methods comparison created")
    
    # Spotify clustering diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a simple flow diagram
    boxes = {
        'Users': (2, 3),
        'Songs': (2, 1),
        'Clustering': (5, 2),
        'Playlists': (8, 2)
    }
    
    for name, (x, y) in boxes.items():
        rect = patches.FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                                      boxstyle="round,pad=0.1",
                                      facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Add arrows
    ax.arrow(3, 3, 1.5, -0.8, head_width=0.1, head_length=0.1, fc='gray', ec='gray')
    ax.arrow(3, 1, 1.5, 0.8, head_width=0.1, head_length=0.1, fc='gray', ec='gray')
    ax.arrow(6, 2, 1.5, 0, head_width=0.1, head_length=0.1, fc='gray', ec='gray')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title("Spotify's Discover Weekly: Clustering in Action", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('spotify_clustering.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('spotify_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Spotify clustering created")
    
    # Week 2 preview
    fig, ax = plt.subplots(figsize=(10, 6))
    
    topics = ['Time-Series\nClustering', 'Multi-View\nClustering', 
              'Online\nClustering', 'Validation\nTechniques', 'A/B Testing\nwith Clusters']
    x = np.arange(len(topics))
    heights = [8, 7, 9, 6, 8]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    bars = ax.bar(x, heights, color=colors, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(topics, fontsize=10)
    ax.set_ylabel('Complexity Level', fontsize=12)
    ax.set_title('Week 2 Preview: Advanced Clustering Techniques', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('week2_preview.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('week2_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Week 2 preview created")

def main():
    """Generate all charts"""
    print("Generating all Week 1 visualization charts...")
    print("="*50)
    
    # Run all creation functions
    create_user_empathy_visual()
    create_clustering_examples()
    create_distance_visual()
    create_kmeans_animation()
    create_dendrogram_cut()
    create_cluster_quality()
    create_dbscan_shapes()
    create_feature_importance()
    create_customer_segments()
    create_pain_points_heatmap()
    create_remaining_charts()
    
    print("="*50)
    print("All charts created successfully!")

if __name__ == "__main__":
    main()