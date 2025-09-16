import json

# Load notebook
with open(r'D:\Joerg\Research\slides\ML_Design_Thinking_16\Week_01\Week01_Part3_Practice_Advanced.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Insert Section 0 with all functions after cell 2 (imports)
section_0_markdown = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '---\n',
        '# Section 0: Complete Function Library\n',
        'All functions for Part 3 are defined here for modularity and reusability.'
    ]
}

# Create comprehensive function definitions cell
functions_cell = {
    'cell_type': 'code',
    'metadata': {},
    'outputs': [],
    'source': []
}

# Define all function content
function_definitions = '''# Part 3: Complete Function Library

# ============================================================================
# SPOTIFY CASE STUDY FUNCTIONS
# ============================================================================

def generate_spotify_music_data(n_songs=2000):
    """
    Generate Spotify-like music data with realistic features.
    Returns DataFrame with song features and genre labels.
    """
    print("🎵 Spotify Case Study: Music Clustering for Recommendations\\n")
    
    # Music features (what Spotify actually uses)
    music_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # Generate different music genres with distinct characteristics
    genre_profiles = {
        'Pop': {'centers': [0.7, 0.7, -5, 0.1, 0.1, 0, 0.1, 0.8, 120], 'std': 0.1, 'count': 400},
        'Rock': {'centers': [0.5, 0.8, -3, 0.05, 0.05, 0.1, 0.2, 0.5, 130], 'std': 0.15, 'count': 350},
        'Electronic': {'centers': [0.8, 0.9, -4, 0.03, 0.01, 0.8, 0.1, 0.6, 128], 'std': 0.1, 'count': 300},
        'Classical': {'centers': [0.3, 0.3, -20, 0.02, 0.9, 0.9, 0.1, 0.3, 80], 'std': 0.1, 'count': 250},
        'Hip-Hop': {'centers': [0.8, 0.6, -6, 0.3, 0.1, 0, 0.1, 0.6, 90], 'std': 0.12, 'count': 300},
        'Jazz': {'centers': [0.5, 0.4, -15, 0.05, 0.8, 0.3, 0.3, 0.5, 110], 'std': 0.15, 'count': 200},
        'Country': {'centers': [0.6, 0.5, -8, 0.05, 0.6, 0.1, 0.1, 0.7, 100], 'std': 0.1, 'count': 200}
    }
    
    # Generate songs
    all_songs = []
    all_genres = []
    
    for genre, profile in genre_profiles.items():
        n_genre_songs = profile['count']
        genre_data = np.random.normal(profile['centers'], 
                                     [profile['std']] * len(music_features),
                                     (n_genre_songs, len(music_features)))
        
        # Clip values to realistic ranges
        genre_data[:, [0,1,3,4,5,6,7]] = np.clip(genre_data[:, [0,1,3,4,5,6,7]], 0, 1)
        genre_data[:, 2] = np.clip(genre_data[:, 2], -60, 0)  # Loudness
        genre_data[:, 8] = np.clip(genre_data[:, 8], 40, 200)  # Tempo
        
        all_songs.append(genre_data)
        all_genres.extend([genre] * n_genre_songs)
    
    # Create DataFrame
    X_music = np.vstack(all_songs)
    music_df = pd.DataFrame(X_music, columns=music_features)
    music_df['genre_true'] = all_genres
    music_df['song_id'] = [f'SONG_{i:04d}' for i in range(len(music_df))]
    
    print(f"Generated {len(music_df)} songs across {len(genre_profiles)} genres")
    print("\\nSample of music data:")
    display(music_df.head())
    print("\\nGenre distribution:")
    print(music_df['genre_true'].value_counts())
    
    return music_df, music_features

def cluster_spotify_music(music_df, music_features):
    """
    Apply clustering to music data and visualize results.
    Returns clustered DataFrame and model.
    """
    print("🎯 Clustering Music for Recommendations\\n")
    
    # Prepare data
    X_music_features = music_df[music_features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_music_scaled = scaler.fit_transform(X_music_features)
    
    # Find optimal number of clusters
    k_range = range(3, 12)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_music_scaled)
        silhouette_scores.append(silhouette_score(X_music_scaled, labels))
    
    # Best K
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {best_k}")
    
    # Apply final clustering
    kmeans_music = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    music_clusters = kmeans_music.fit_predict(X_music_scaled)
    music_df['cluster'] = music_clusters
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_music_scaled)
    
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=music_clusters, 
                         cmap='tab10', s=10, alpha=0.6)
    ax1.set_title('Music Clusters (PCA)', fontsize=12, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, ax=ax1)
    
    # Cluster sizes
    ax2 = axes[0, 1]
    cluster_counts = music_df['cluster'].value_counts().sort_index()
    ax2.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.tab10(np.arange(best_k)))
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Songs')
    ax2.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    
    # Feature comparison across clusters
    ax3 = axes[1, 0]
    feature_means = music_df.groupby('cluster')[['energy', 'valence', 'danceability', 'acousticness']].mean()
    feature_means.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Average Value')
    ax3.set_title('Key Features by Cluster', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    
    # Genre distribution per cluster
    ax4 = axes[1, 1]
    genre_cluster_matrix = pd.crosstab(music_df['genre_true'], music_df['cluster'], normalize='columns')
    im = ax4.imshow(genre_cluster_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(best_k))
    ax4.set_xticklabels([f'C{i}' for i in range(best_k)])
    ax4.set_yticks(range(len(genre_cluster_matrix)))
    ax4.set_yticklabels(genre_cluster_matrix.index)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('True Genre')
    ax4.set_title('Genre Distribution in Clusters', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    plt.suptitle('Spotify Music Clustering Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\\n📊 Clustering Performance:")
    print(f"Silhouette Score: {silhouette_score(X_music_scaled, music_clusters):.3f}")
    
    return music_df, kmeans_music, scaler

def generate_music_recommendations(music_df, kmeans_music, scaler, music_features, n_users=100):
    """
    Generate music recommendations based on user profiles.
    Shows recommendation system in action.
    """
    print("🎧 Generating Music Recommendations\\n")
    
    best_k = kmeans_music.n_clusters
    
    # Create user profiles based on listening history
    user_profiles = []
    user_preferences = []
    
    # Simulate user preferences
    for i in range(n_users):
        preferred_clusters = np.random.choice(range(best_k), 
                                            size=np.random.randint(1, 4), 
                                            replace=False)
        user_preferences.append(preferred_clusters)
        
        # Create user profile as weighted average of preferred cluster centers
        weights = np.random.dirichlet(np.ones(len(preferred_clusters)))
        user_profile = np.zeros(len(music_features))
        for cluster, weight in zip(preferred_clusters, weights):
            cluster_center = kmeans_music.cluster_centers_[cluster]
            user_profile += weight * cluster_center
        user_profiles.append(user_profile)
    
    # Recommendation function
    def recommend_songs(user_profile, music_df, kmeans_model, n_recommendations=10):
        distances = np.linalg.norm(kmeans_model.cluster_centers_ - user_profile, axis=1)
        closest_clusters = np.argsort(distances)[:2]
        
        candidate_songs = music_df[music_df['cluster'].isin(closest_clusters)]
        
        song_features = candidate_songs[music_features].values
        song_features_scaled = scaler.transform(song_features)
        similarities = np.dot(song_features_scaled, user_profile) / (
            np.linalg.norm(song_features_scaled, axis=1) * np.linalg.norm(user_profile)
        )
        
        top_indices = np.argsort(similarities)[-n_recommendations:][::-1]
        recommendations = candidate_songs.iloc[top_indices]
        
        return recommendations[['song_id', 'genre_true', 'cluster', 'energy', 'valence', 'danceability']]
    
    # Example recommendations
    example_user = 0
    recommendations = recommend_songs(user_profiles[example_user], music_df, kmeans_music)
    
    print(f"Recommendations for User {example_user}:")
    print(f"User prefers clusters: {user_preferences[example_user]}\\n")
    display(recommendations)
    
    # Generate mood playlists
    print("\\n🎵 Generated Playlists by Mood:")
    
    mood_profiles = {
        'Workout': {'energy': 0.9, 'valence': 0.7, 'danceability': 0.8, 'tempo': 130},
        'Study': {'energy': 0.3, 'instrumentalness': 0.8, 'speechiness': 0.1, 'loudness': -20},
        'Party': {'danceability': 0.9, 'energy': 0.8, 'valence': 0.8, 'loudness': -5},
        'Relax': {'energy': 0.2, 'acousticness': 0.7, 'valence': 0.5, 'tempo': 80}
    }
    
    for mood, profile in mood_profiles.items():
        mood_vector = np.zeros(len(music_features))
        for feature, value in profile.items():
            if feature in music_features:
                idx = music_features.index(feature)
                mood_vector[idx] = value
        
        mood_vector_scaled = (mood_vector - scaler.mean_) / scaler.scale_
        
        distances = np.linalg.norm(kmeans_music.cluster_centers_ - mood_vector_scaled, axis=1)
        best_cluster = np.argmin(distances)
        
        print(f"\\n{mood} Playlist (Cluster {best_cluster}):")
        playlist_songs = music_df[music_df['cluster'] == best_cluster].sample(5)
        print(playlist_songs[['song_id', 'genre_true', 'energy', 'valence']].to_string())
    
    return user_profiles, user_preferences

# ============================================================================
# INNOVATION CHALLENGE FUNCTIONS  
# ============================================================================

def create_innovation_challenge():
    """
    Create innovation challenge dataset for hands-on practice.
    Returns startup ideas DataFrame.
    """
    print("🚀 Innovation Challenge: Build Your Own Clustering Solution\\n")
    
    print("Choose your dataset:")
    print("1. Startup Ideas Classification")
    print("2. Product Feature Requests")
    print("3. Customer Feedback Analysis")
    print("4. Research Paper Categorization")
    print("\\nFor this demo, we'll use Startup Ideas...\\n")
    
    n_startups = 500
    
    startup_features = [
        'tech_innovation', 'market_size', 'competition', 'funding_needed',
        'time_to_market', 'scalability', 'team_expertise', 'regulatory_risk'
    ]
    
    startup_categories = {
        'FinTech': {'features': [8, 9, 7, 8, 6, 9, 8, 9], 'count': 100},
        'HealthTech': {'features': [9, 8, 6, 9, 8, 7, 9, 10], 'count': 80},
        'EdTech': {'features': [6, 7, 5, 5, 4, 8, 6, 4], 'count': 90},
        'E-commerce': {'features': [5, 8, 9, 6, 3, 9, 5, 3], 'count': 70},
        'SaaS': {'features': [7, 6, 8, 4, 5, 10, 7, 2], 'count': 85},
        'GreenTech': {'features': [8, 7, 4, 9, 7, 6, 7, 6], 'count': 75}
    }
    
    startup_data = []
    startup_types = []
    
    for category, specs in startup_categories.items():
        category_data = np.random.normal(specs['features'], 1.5, 
                                        (specs['count'], len(startup_features)))
        category_data = np.clip(category_data, 0, 10)
        startup_data.append(category_data)
        startup_types.extend([category] * specs['count'])
    
    X_startups = np.vstack(startup_data)
    startups_df = pd.DataFrame(X_startups, columns=startup_features)
    startups_df['category_true'] = startup_types
    startups_df['startup_id'] = [f'STARTUP_{i:03d}' for i in range(len(startups_df))]
    
    print(f"Generated {len(startups_df)} startup ideas")
    print("\\nSample data:")
    display(startups_df.head())
    
    print("\\n📝 Your Challenge Tasks:")
    print("1. ✅ Load and explore the data")
    print("2. ⬜ Preprocess and scale features")
    print("3. ⬜ Choose clustering algorithm")
    print("4. ⬜ Find optimal parameters")
    print("5. ⬜ Validate results")
    print("6. ⬜ Create visualizations")
    print("7. ⬜ Generate insights")
    
    return startups_df, startup_features

def complete_clustering_pipeline(startups_df, startup_features):
    """
    Complete clustering pipeline for startup ideas.
    Demonstrates full workflow from data to insights.
    """
    print("📊 Complete Clustering Pipeline\\n")
    
    # Step 1: Data Preparation
    print("Step 1: Data Preparation")
    X_startup_features = startups_df[startup_features].values
    scaler = StandardScaler()
    X_startup_scaled = scaler.fit_transform(X_startup_features)
    print("✅ Data scaled\\n")
    
    # Step 2: Algorithm Selection
    print("Step 2: Testing Multiple Algorithms")
    algorithms = {
        'K-Means': KMeans(n_clusters=6, random_state=42),
        'DBSCAN': DBSCAN(eps=1.5, min_samples=5),
        'Hierarchical': AgglomerativeClustering(n_clusters=6),
        'GMM': GaussianMixture(n_components=6, random_state=42)
    }
    
    results = {}
    for name, algo in algorithms.items():
        if hasattr(algo, 'fit_predict'):
            labels = algo.fit_predict(X_startup_scaled)
        else:
            labels = algo.fit(X_startup_scaled).predict(X_startup_scaled)
        
        if len(np.unique(labels[labels != -1])) > 1:
            score = silhouette_score(X_startup_scaled, labels)
        else:
            score = -1
        
        results[name] = {'labels': labels, 'score': score}
        print(f"{name}: Silhouette = {score:.3f}")
    
    best_algo = max(results.keys(), key=lambda k: results[k]['score'])
    best_labels = results[best_algo]['labels']
    print(f"\\n✅ Best Algorithm: {best_algo}\\n")
    
    # Step 3: Visualization
    print("Step 3: Creating Visualizations")
    
    fig = plt.figure(figsize=(15, 10))
    
    # PCA visualization
    ax1 = plt.subplot(2, 3, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_startup_scaled)
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, 
                         cmap='tab10', s=30, alpha=0.7)
    ax1.set_title('Startup Clusters (PCA)', fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax1)
    
    # t-SNE visualization
    ax2 = plt.subplot(2, 3, 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_startup_scaled)
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=best_labels,
                          cmap='tab10', s=30, alpha=0.7)
    ax2.set_title('Startup Clusters (t-SNE)', fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=ax2)
    
    # Feature importance
    ax3 = plt.subplot(2, 3, 3)
    startups_df['cluster'] = best_labels
    feature_importance = startups_df.groupby('cluster')[startup_features].mean()
    feature_std = feature_importance.std(axis=0).sort_values(ascending=False)
    ax3.bar(range(len(feature_std)), feature_std.values)
    ax3.set_xticks(range(len(feature_std)))
    ax3.set_xticklabels(feature_std.index, rotation=45, ha='right')
    ax3.set_title('Feature Importance (Variance)', fontweight='bold')
    ax3.set_ylabel('Standard Deviation')
    
    # Cluster characteristics
    ax4 = plt.subplot(2, 3, 4)
    cluster_means = startups_df.groupby('cluster')[startup_features].mean()
    im = ax4.imshow(cluster_means.T, cmap='RdYlGn', aspect='auto')
    ax4.set_yticks(range(len(startup_features)))
    ax4.set_yticklabels(startup_features)
    ax4.set_xticks(range(len(cluster_means)))
    ax4.set_xticklabels([f'C{i}' for i in cluster_means.index])
    ax4.set_title('Cluster Characteristics', fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    # Category distribution
    ax5 = plt.subplot(2, 3, 5)
    category_cluster = pd.crosstab(startups_df['category_true'], 
                                  startups_df['cluster'], normalize='columns')
    category_cluster.plot(kind='bar', stacked=True, ax=ax5, width=0.8)
    ax5.set_title('True Categories in Clusters', fontweight='bold')
    ax5.set_xlabel('True Category')
    ax5.set_ylabel('Proportion')
    ax5.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # Investment vs Risk
    ax6 = plt.subplot(2, 3, 6)
    for cluster in np.unique(best_labels[best_labels != -1]):
        mask = startups_df['cluster'] == cluster
        ax6.scatter(startups_df.loc[mask, 'funding_needed'],
                   startups_df.loc[mask, 'regulatory_risk'],
                   label=f'Cluster {cluster}', s=50, alpha=0.6)
    ax6.set_xlabel('Funding Needed')
    ax6.set_ylabel('Regulatory Risk')
    ax6.set_title('Investment vs Risk by Cluster', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Startup Clustering Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizations complete\\n")
    
    # Step 4: Generate Insights
    print("Step 4: Key Insights")
    print("\\n🔍 Cluster Insights:")
    for cluster in range(len(cluster_means)):
        top_features = cluster_means.iloc[cluster].nlargest(3).index.tolist()
        size = (startups_df['cluster'] == cluster).sum()
        print(f"\\nCluster {cluster} ({size} startups):")
        print(f"  Top features: {', '.join(top_features)}")
        
        if 'tech_innovation' in top_features and 'funding_needed' in top_features:
            print("  Type: High-Tech, High-Investment")
        elif 'market_size' in top_features and 'scalability' in top_features:
            print("  Type: Market Leaders")
        elif 'team_expertise' in top_features:
            print("  Type: Expert-Driven")
        else:
            print("  Type: Balanced Portfolio")
    
    return startups_df, results
'''

# Add function content to cell source (split by newlines to create list)
functions_cell['source'] = function_definitions.split('\n')

# Insert the new cells at position 3 
nb['cells'].insert(3, section_0_markdown)
nb['cells'].insert(4, functions_cell)

# Save the updated notebook
with open(r'D:\Joerg\Research\slides\ML_Design_Thinking_16\Week_01\Week01_Part3_Practice_Advanced.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Successfully added Section 0 with all function definitions to Part 3')