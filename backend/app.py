def initialize_data():
    """Initialize all data and models with hyperparameter tuning"""
    global movies_df, ratings_df, movies_metadata_df, tfidf_matrix, tfidf_vectorizer, kmeans, nn_model, content_model, cluster_names
    
    print("Loading MovieLens 100k dataset...")
    ratings_df = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies_df = pd.read_csv('../data/ml-100k/u.item', sep='|', encoding='latin-1', 
                            names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
                                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

    print(f"Ratings shape: {ratings_df.shape}")
    print(f"Movies shape: {movies_df.shape}")

    # Create synthetic movie metadata dataset to simulate 'The Movies Dataset'
    print("Creating synthetic movie metadata dataset...")

    # Sample movie titles and generate synthetic overviews
    sample_titles = movies_df['title'].tolist()
    sample_genres = []
    sample_overviews = []

    # Define some genre combinations for synthetic data
    genre_combinations = [
        ['Action', 'Adventure'],
        ['Comedy', 'Romance'],
        ['Drama', 'Romance'],
        ['Action', 'Thriller'],
        ['Horror', 'Thriller'],
        ['Sci-Fi', 'Action'],
        ['Drama', 'Crime'],
        ['Animation', 'Children'],
        ['Documentary'],
        ['Fantasy', 'Adventure']
    ]

    # Define some sample overviews for different genres
    overview_templates = {
        'Action': [
            "An action-packed adventure with intense fight scenes and high-speed chases.",
            "Fast-paced thriller with explosive action sequences and heroic protagonists.",
            "Epic battle between good and evil with spectacular stunts and special effects."
        ],
        'Comedy': [
            "A hilarious comedy that will make you laugh out loud with witty dialogue.",
            "Charming romantic comedy with delightful characters and funny situations.",
            "Satirical comedy that pokes fun at modern society with clever humor."
        ],
        'Drama': [
            "A deeply emotional drama exploring human relationships and personal growth.",
            "Powerful story of triumph over adversity with compelling character development.",
            "Moving tale of love, loss, and redemption set against historical events."
        ],
        'Romance': [
            "A heartwarming love story with beautiful cinematography and touching moments.",
            "Passionate romance between two souls destined to be together.",
            "Timeless love story that transcends all obstacles and challenges."
        ],
        'Sci-Fi': [
            "Mind-bending science fiction exploring futuristic technologies and alien worlds.",
            "Space opera with interstellar adventures and advanced civilizations.",
            "Speculative fiction that questions the nature of reality and humanity."
        ],
        'Horror': [
            "Bone-chilling horror that will keep you awake at night with terrifying scenes.",
            "Supernatural horror with psychological elements and spine-tingling suspense.",
            "Classic horror tale with unexpected twists and shocking revelations."
        ],
        'Thriller': [
            "Gripping thriller with unexpected plot twists and intense suspense.",
            "Psychological thriller that keeps you guessing until the very end.",
            "Action-packed thriller with dangerous missions and high stakes."
        ]
    }

    # Generate synthetic data
    synthetic_data = []
    for idx, title in enumerate(sample_titles):  # Use all titles from the dataset
        # Randomly select a genre combination
        genres = genre_combinations[np.random.randint(0, len(genre_combinations))]
        
        # Generate an overview based on the primary genre
        primary_genre = genres[0]
        if primary_genre in overview_templates:
            overview = np.random.choice(overview_templates[primary_genre])
        else:
            overview = f"A compelling film in the {', '.join(genres)} genre with engaging plot."
        
        synthetic_data.append({
            'title': title,
            'overview': overview,
            'genres': json.dumps(genres),
            'movie_id': idx + 1
        })

    # Create the synthetic movies metadata dataframe
    movies_metadata_df = pd.DataFrame(synthetic_data)
    print(f"Synthetic movies metadata shape: {movies_metadata_df.shape}")

    # Apply preprocessing
    print("Preprocessing text data...")
    movies_metadata_df['clean_overview'] = movies_metadata_df['overview'].apply(preprocess_text)
    movies_metadata_df['genre_tokens'] = movies_metadata_df['genres'].apply(extract_genres)

    # Combine overview and genres
    movies_metadata_df['combined_text'] = (movies_metadata_df['clean_overview'] + ' ' + 
                                          movies_metadata_df['genre_tokens']).str.strip()

    print("Text preprocessing completed!")

    # HYPERPARAMETER TUNING: Find optimal TF-IDF parameters
    print("Performing hyperparameter tuning for TF-IDF vectorizer...")
    
    # Test different TF-IDF parameters
    tfidf_params = {
        'max_features': [3000, 5000, 8000],
        'ngram_range': [(1, 1), (1, 2), (1, 3)],
        'min_df': [1, 2, 3],
        'max_df': [0.8, 0.9, 1.0]
    }
    
    best_score = -1
    best_params = None
    
    # For efficiency, test a subset of parameter combinations
    for max_features in [3000, 5000]:
        for ngram_range in [(1, 2), (1, 3)]:
            for min_df in [1, 2]:
                for max_df in [0.8, 0.9]:
                    try:
                        temp_vectorizer = TfidfVectorizer(
                            max_features=max_features,
                            ngram_range=ngram_range,
                            stop_words=None,
                            lowercase=True,
                            strip_accents='unicode',
                            analyzer='word',
                            min_df=min_df,
                            max_df=max_df,
                            token_pattern=r'\b[a-zA-Z]{2,}\b'
                        )
                        
                        temp_matrix = temp_vectorizer.fit_transform(movies_metadata_df['combined_text'])
                        
                        # Check if the matrix is not too sparse or empty
                        if temp_matrix.shape[1] > 100:  # Ensure we have enough features
                            # Use a sample for quick evaluation
                            sample_size = min(500, temp_matrix.shape[0])
                            sample_indices = np.random.choice(temp_matrix.shape[0], sample_size, replace=False)
                            sample_matrix = temp_matrix[sample_indices]
                            
                            # Determine optimal clusters for this configuration
                            K_range = range(2, 8)
                            silhouette_scores = []
                            
                            for k in K_range:
                                if sample_matrix.shape[0] > k:  # Ensure we have enough samples
                                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                                    cluster_labels = kmeans_temp.fit_predict(sample_matrix)
                                    
                                    if len(np.unique(cluster_labels)) > 1:  # Ensure multiple clusters
                                        sil_score = silhouette_score(sample_matrix, cluster_labels)
                                        silhouette_scores.append(sil_score)
                            
                            if silhouette_scores:
                                avg_score = np.mean(silhouette_scores)
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_params = {
                                        'max_features': max_features,
                                        'ngram_range': ngram_range,
                                        'min_df': min_df,
                                        'max_df': max_df
                                    }
                    except:
                        continue  # Skip invalid parameter combinations
    
    print(f"Best TF-IDF parameters: {best_params}")
    print(f"Best silhouette score: {best_score:.3f}")

    # Initialize TF-IDF vectorizer with best parameters
    print("Initializing TF-IDF vectorizer with optimal parameters...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=best_params['max_features'] if best_params else 5000,
        ngram_range=best_params['ngram_range'] if best_params else (1, 2),
        stop_words=None,
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        min_df=best_params['min_df'] if best_params else 1,
        max_df=best_params['max_df'] if best_params else 0.9,
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )

    # Fit and transform the combined text
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_metadata_df['combined_text'])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    # HYPERPARAMETER TUNING: Find optimal number of clusters using multiple methods
    print("Performing hyperparameter tuning for clustering...")
    
    # Use elbow method and silhouette analysis to find optimal number of clusters
    sample_size = min(800, tfidf_matrix.shape[0])  # Use larger sample for better tuning
    sample_indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
    sample_tfidf = tfidf_matrix[sample_indices]

    K_range = range(2, 15)  # Test more cluster options
    inertias = []
    silhouette_scores = []

    for k in K_range:
        if sample_tfidf.shape[0] > k:  # Ensure we have enough samples
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10, n_jobs=-1)  # Use all CPU cores
            cluster_labels = kmeans_temp.fit_predict(sample_tfidf.toarray())  # Convert to dense for better stability
            
            if len(np.unique(cluster_labels)) > 1:  # Ensure multiple clusters
                inertias.append(kmeans_temp.inertia_)
                
                # Calculate silhouette score
                sil_score = silhouette_score(sample_tfidf, cluster_labels)
                silhouette_scores.append(sil_score)
                print(f"K={k}, Inertia: {kmeans_temp.inertia_:.2f}, Silhouette Score: {sil_score:.3f}")
            else:
                silhouette_scores.append(-1)  # Invalid clustering
        else:
            silhouette_scores.append(-1)  # Invalid clustering

    # Find optimal K based on silhouette score
    valid_scores = [(i, score) for i, score in enumerate(silhouette_scores) if score > -1]
    if valid_scores:
        best_idx, best_silhouette = max(valid_scores, key=lambda x: x[1])
        optimal_k = K_range[best_idx]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k} (score: {best_silhouette:.3f})")
    else:
        optimal_k = 8  # Default if no valid clustering found
        print(f"Using default number of clusters: {optimal_k}")

    # Apply K-Means clustering with the optimal number of clusters
    print(f"Applying K-Means clustering with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, n_jobs=-1)  # More iterations for better convergence
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())  # Convert to dense for stability

    # Add cluster labels to the dataframe
    movies_metadata_df['cluster'] = cluster_labels

    # Print cluster statistics
    cluster_counts = movies_metadata_df['cluster'].value_counts().sort_index()
    print("Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} movies")

    # Assign human-readable names to clusters based on their characteristics
    cluster_names = {}
    for cluster_id in range(optimal_k):
        cluster_names[cluster_id] = f"Cluster {cluster_id}"

    # More descriptive cluster names based on content
    # Get top terms for each cluster to create descriptive names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    def get_top_terms_per_cluster(tfidf_matrix, feature_names, cluster_labels, n_terms=5):
        """Get top terms for each cluster based on mean TF-IDF scores"""
        top_terms = {}
        
        for cluster_id in np.unique(cluster_labels):
            # Get indices for this cluster
            cluster_mask = cluster_labels == cluster_id
            
            # Get TF-IDF values for this cluster
            cluster_tfidf = tfidf_matrix[cluster_mask]
            
            # Calculate mean TF-IDF score for each term in this cluster
            mean_scores = np.array(cluster_tfidf.mean(axis=0)).flatten()
            
            # Get top terms
            top_indices = mean_scores.argsort()[-n_terms:][::-1]
            top_terms[cluster_id] = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        return top_terms

    # Get top terms for each cluster
    top_terms_per_cluster = get_top_terms_per_cluster(tfidf_matrix, feature_names, cluster_labels, n_terms=5)
    
    # Create more descriptive cluster names based on top terms
    for cluster_id, terms in top_terms_per_cluster.items():
        top_words = [term for term, score in terms[:3]]  # Top 3 terms
        cluster_names[cluster_id] = f"{', '.join(top_words).title()} Cluster"

    # Map cluster names to the dataframe
    movies_metadata_df['cluster_name'] = movies_metadata_df['cluster'].map(cluster_names)

    # Build item-user sparse matrix from MovieLens ratings for collaborative filtering
    print("Building item-user sparse matrix from MovieLens ratings...")
    
    # Convert explicit ratings to implicit feedback (rating >= 4 → 1)
    implicit_ratings = ratings_df.copy()
    implicit_ratings['implicit_rating'] = (implicit_ratings['rating'] >= 4).astype(int)

    # Filter to keep only positive interactions
    positive_interactions = implicit_ratings[implicit_ratings['implicit_rating'] == 1]

    # Create item-user matrix
    n_users = implicit_ratings['user_id'].nunique()
    n_items = min(len(movies_metadata_df), implicit_ratings['movie_id'].nunique())

    print(f"Number of unique users: {n_users}")
    print(f"Number of unique items: {n_items}")
    print(f"Number of positive interactions: {len(positive_interactions)}")

    # Create sparse matrix
    user_item_matrix = csr_matrix((positive_interactions['implicit_rating'].values,
                                  (positive_interactions['movie_id'].values - 1,  # Adjust for 0-indexing
                                   positive_interactions['user_id'].values - 1)),
                                 shape=(n_items, n_users))

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {(1 - user_item_matrix.nnz / (n_items * n_users)) * 100:.2f}%")

    # Transpose to get user-item relationships (users are rows, items are columns)
    user_item_matrix = user_item_matrix.T

    # HYPERPARAMETER TUNING: Optimize collaborative filtering model
    print("Performing hyperparameter tuning for collaborative filtering model...")
    
    # Test different parameters for nearest neighbors
    nn_params = [
        {'n_neighbors': 10, 'metric': 'cosine', 'algorithm': 'brute'},
        {'n_neighbors': 15, 'metric': 'cosine', 'algorithm': 'brute'},
        {'n_neighbors': 20, 'metric': 'cosine', 'algorithm': 'brute'},
        {'n_neighbors': 10, 'metric': 'euclidean', 'algorithm': 'brute'},
        {'n_neighbors': 15, 'metric': 'euclidean', 'algorithm': 'brute'},
    ]
    
    best_cf_score = -1
    best_cf_params = None
    
    # Use a sample for quick evaluation
    sample_cf_matrix = user_item_matrix[:min(500, user_item_matrix.shape[0])]
    
    for params in nn_params:
        try:
            temp_model = NearestNeighbors(**params)
            temp_model.fit(sample_cf_matrix)
            
            # Evaluate by finding neighbors for a few samples
            n_samples = min(10, sample_cf_matrix.shape[0])
            sample_indices = np.random.choice(sample_cf_matrix.shape[0], n_samples, replace=False)
            
            avg_distances = []
            for idx in sample_indices:
                distances, indices = temp_model.kneighbors(sample_cf_matrix[idx], n_neighbors=min(5, params['n_neighbors']))
                avg_distances.append(np.mean(distances))
            
            avg_score = np.mean(avg_distances) if avg_distances else 0
            if avg_score < best_cf_score or best_cf_score == -1:  # Lower distance is better
                best_cf_score = avg_score
                best_cf_params = params
        except:
            continue
    
    print(f"Best collaborative filtering parameters: {best_cf_params}")
    
    # Train the final collaborative filtering model with best parameters
    collaborative_filtering_model = NearestNeighbors(
        n_neighbors=best_cf_params['n_neighbors'] if best_cf_params else 20,
        metric=best_cf_params['metric'] if best_cf_params else 'cosine',
        algorithm=best_cf_params['algorithm'] if best_cf_params else 'brute'
    )
    collaborative_filtering_model.fit(user_item_matrix)  # Fit on the user-item matrix

    # HYPERPARAMETER TUNING: Optimize content-based similarity model
    print("Performing hyperparameter tuning for content-based similarity model...")
    
    best_content_score = -1
    best_content_params = None
    
    # Test different parameters for content-based model
    content_params = [
        {'n_neighbors': 10, 'metric': 'cosine', 'algorithm': 'brute'},
        {'n_neighbors': 15, 'metric': 'cosine', 'algorithm': 'brute'},
        {'n_neighbors': 20, 'metric': 'cosine', 'algorithm': 'brute'},
        {'n_neighbors': 10, 'metric': 'euclidean', 'algorithm': 'brute'},
        {'n_neighbors': 15, 'metric': 'euclidean', 'algorithm': 'brute'},
    ]
    
    for params in content_params:
        try:
            temp_model = NearestNeighbors(**params)
            temp_model.fit(tfidf_matrix)
            
            # Evaluate by finding neighbors for a few samples
            n_samples = min(10, tfidf_matrix.shape[0])
            sample_indices = np.random.choice(tfidf_matrix.shape[0], n_samples, replace=False)
            
            avg_distances = []
            for idx in sample_indices:
                distances, indices = temp_model.kneighbors(tfidf_matrix[idx], n_neighbors=min(5, params['n_neighbors']))
                avg_distances.append(np.mean(distances))
            
            avg_score = np.mean(avg_distances) if avg_distances else 0
            if avg_score < best_content_score or best_content_score == -1:  # Lower distance is better
                best_content_score = avg_score
                best_content_params = params
        except:
            continue
    
    print(f"Best content-based similarity parameters: {best_content_params}")
    
    # Create content-based similarity model with best parameters
    content_similarity_model = NearestNeighbors(
        n_neighbors=best_content_params['n_neighbors'] if best_content_params else 20,
        metric=best_content_params['metric'] if best_content_params else 'cosine',
        algorithm=best_content_params['algorithm'] if best_content_params else 'brute'
    )
    content_similarity_model.fit(tfidf_matrix)  # Fit on the TF-IDF matrix

    # Store both models globally
    global nn_model, content_model
    nn_model = collaborative_filtering_model
    content_model = content_similarity_model

    print("Data initialization completed successfully!")