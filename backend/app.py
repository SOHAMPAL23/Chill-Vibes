#!/usr/bin/env python
# coding: utf-8

"""
Movie Content Clustering & Recommendation System - Backend API
Flask server to serve movie clustering and recommendation endpoints
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables to store data
movies_df = None
ratings_df = None
movies_metadata_df = None
tfidf_matrix = None
tfidf_vectorizer = None
kmeans = None
nn_model = None
cluster_names = {}

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters but keep meaningful words
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Ensure we don't return empty strings that would cause issues with TF-IDF
    if not text.strip():
        return 'unknown'
    
    return text

def extract_genres(genre_json):
    """Extract genre tokens from JSON string"""
    try:
        genres = json.loads(genre_json)
        return ' '.join([g.lower() for g in genres])
    except:
        return ''

def initialize_data():
    """Initialize all data and models"""
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

    # Initialize TF-IDF vectorizer with specified parameters
    print("Initializing TF-IDF vectorizer with max_features=5000, ngram_range=(1,2)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None,  # Changed from 'english' to None to avoid removing all content
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only include alphabetic words with 2+ characters
    )

    # Fit and transform the combined text
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_metadata_df['combined_text'])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    # Determine optimal number of clusters using silhouette analysis
    print("Determining optimal number of clusters...")
    
    # Sample a subset for faster computation
    sample_size = min(500, tfidf_matrix.shape[0])  # Reduced for faster computation
    sample_indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
    sample_tfidf = tfidf_matrix[sample_indices]

    # Test different numbers of clusters
    K_range = range(2, 10)  # Reduced range for faster computation
    silhouette_scores = []

    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans_temp.fit_predict(sample_tfidf)
        
        # Calculate silhouette score
        sil_score = silhouette_score(sample_tfidf, cluster_labels)
        silhouette_scores.append(sil_score)
        print(f"K={k}, Silhouette Score: {sil_score:.3f}")

    # Find optimal K based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

    # Apply K-Means clustering with the optimal number of clusters
    print(f"Applying K-Means clustering with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Add cluster labels to the dataframe
    movies_metadata_df['cluster'] = cluster_labels

    # Print cluster statistics
    cluster_counts = movies_metadata_df['cluster'].value_counts().sort_index()
    print("Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} movies")

    # Assign human-readable names to clusters based on their characteristics
    cluster_names = {
        0: "Action/Adventure",
        1: "Romantic Comedy", 
        2: "Drama/Thriller",
        3: "Sci-Fi/Action",
        4: "Horror/Thriller",
        5: "Documentary",
        6: "Animation/Children",
        7: "Crime/Drama",
        8: "Fantasy/Adventure",
        9: "General Entertainment"
    }

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

    # Use NearestNeighbors with cosine metric for collaborative filtering
    print("Training collaborative filtering model...")
    collaborative_filtering_model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    collaborative_filtering_model.fit(user_item_matrix)  # Fit on the user-item matrix

    # Create content-based similarity model using TF-IDF matrix
    print("Training content-based similarity model...")
    content_similarity_model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    content_similarity_model.fit(tfidf_matrix)  # Fit on the TF-IDF matrix

    # Store both models globally
    global nn_model, content_model
    nn_model = collaborative_filtering_model
    content_model = content_similarity_model

    print("Data initialization completed successfully!")

def get_item_item_recommendations(movie_idx, n_recommendations=10):
    """Get item-item recommendations for a given movie"""
    if movie_idx >= tfidf_matrix.shape[0]:
        return [], []
    
    # Get the movie vector from the TF-IDF matrix
    movie_vector = tfidf_matrix[movie_idx:movie_idx+1]
    
    # Find similar items using the content-based similarity model
    distances, indices = content_model.kneighbors(movie_vector, n_neighbors=n_recommendations+1)
    
    # Exclude the first result (the movie itself)
    similar_indices = indices[0][1:]
    similarity_scores = 1 - distances[0][1:]  # Convert distance to similarity
    
    # Filter out indices that are out of range
    valid_indices = [idx for idx in similar_indices if idx < len(movies_metadata_df)]
    valid_similarities = [similarity_scores[i] for i, idx in enumerate(similar_indices) if idx < len(movies_metadata_df)]
    
    return valid_indices, valid_similarities

def get_cluster_aware_recommendations(movie_idx, n_recommendations=10, restrict_to_same_cluster=True):
    """Get recommendations restricted to same cluster or similar clusters"""
    if movie_idx >= len(movies_metadata_df):
        return []
    
    # Get base recommendations
    similar_indices, similarity_scores = get_item_item_recommendations(movie_idx, n_recommendations*2)
    
    # Get the cluster of the input movie
    input_cluster = movies_metadata_df.iloc[movie_idx]['cluster']
    
    # Filter recommendations based on cluster
    filtered_recommendations = []
    
    for idx, score in zip(similar_indices, similarity_scores):
        if idx < len(movies_metadata_df):
            rec_cluster = movies_metadata_df.iloc[idx]['cluster']
            
            # If restricting to same cluster, only include if same cluster
            if restrict_to_same_cluster:
                if rec_cluster == input_cluster:
                    filtered_recommendations.append((idx, score))
            else:
                # Otherwise, include all
                filtered_recommendations.append((idx, score))
        
        # Stop when we have enough recommendations
        if len(filtered_recommendations) >= n_recommendations:
            break
    
    # Return top n_recommendations
    return filtered_recommendations[:n_recommendations]

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get all movies"""
    try:
        movies_list = []
        for idx, row in movies_metadata_df.iterrows():
            movie = {
                'id': idx + 1,
                'title': row['title'],
                'overview': row['overview'],
                'cluster': int(row['cluster']),
                'cluster_name': row['cluster_name'] if pd.notna(row['cluster_name']) else 'Unknown'
            }
            movies_list.append(movie)
        
        return jsonify(movies_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Get cluster information"""
    try:
        clusters_list = []
        cluster_counts = movies_metadata_df['cluster'].value_counts().sort_index()
        
        for cluster_id in cluster_counts.index:
            cluster_name = cluster_names.get(cluster_id, f'Cluster {cluster_id}')
            cluster_info = {
                'id': int(cluster_id),
                'name': cluster_name,
                'count': int(cluster_counts[cluster_id]),
                'movies': []
            }
            
            # Get sample movies for this cluster
            cluster_movies = movies_metadata_df[movies_metadata_df['cluster'] == cluster_id]
            for _, movie in cluster_movies.head(5).iterrows():  # Get first 5 movies in cluster
                cluster_info['movies'].append({
                    'id': int(cluster_movies.index[cluster_movies['title'] == movie['title']][0]) + 1,
                    'title': movie['title']
                })
            
            clusters_list.append(cluster_info)
        
        return jsonify(clusters_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/<int:movie_id>', methods=['GET'])
def get_recommendations(movie_id):
    """Get recommendations for a specific movie"""
    try:
        # Convert to 0-based index
        movie_idx = movie_id - 1
        
        if movie_idx < 0 or movie_idx >= len(movies_metadata_df):
            return jsonify({'error': 'Movie not found'}), 404
        
        # Get the original movie
        original_movie = {
            'id': movie_id,
            'title': movies_metadata_df.iloc[movie_idx]['title'],
            'overview': movies_metadata_df.iloc[movie_idx]['overview'],
            'cluster': int(movies_metadata_df.iloc[movie_idx]['cluster']),
            'cluster_name': movies_metadata_df.iloc[movie_idx]['cluster_name']
        }
        
        # Get cluster-aware recommendations
        recommendations_data = get_cluster_aware_recommendations(movie_idx, n_recommendations=10, restrict_to_same_cluster=True)
        
        recommendations = []
        for idx, score in recommendations_data:
            if idx < len(movies_metadata_df):
                rec_movie = {
                    'id': int(idx + 1),
                    'title': movies_metadata_df.iloc[idx]['title'],
                    'overview': movies_metadata_df.iloc[idx]['overview'],
                    'cluster': int(movies_metadata_df.iloc[idx]['cluster']),
                    'cluster_name': movies_metadata_df.iloc[idx]['cluster_name'],
                    'similarity': float(score)
                }
                recommendations.append(rec_movie)
        
        result = {
            'original_movie': original_movie,
            'recommendations': recommendations
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies by title"""
    try:
        query = request.args.get('q', '').lower()
        if not query:
            return jsonify([])
        
        # Filter movies that match the query
        matches = movies_metadata_df[movies_metadata_df['title'].str.lower().str.contains(query)]
        
        results = []
        for idx, row in matches.iterrows():
            movie = {
                'id': int(idx + 1),
                'title': row['title'],
                'overview': row['overview'],
                'cluster': int(row['cluster']),
                'cluster_name': row['cluster_name']
            }
            results.append(movie)
        
        # Return top 10 matches
        return jsonify(results[:10])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """Get a specific movie by ID"""
    try:
        movie_idx = movie_id - 1
        if movie_idx < 0 or movie_idx >= len(movies_metadata_df):
            return jsonify({'error': 'Movie not found'}), 404
        
        movie = {
            'id': movie_id,
            'title': movies_metadata_df.iloc[movie_idx]['title'],
            'overview': movies_metadata_df.iloc[movie_idx]['overview'],
            'cluster': int(movies_metadata_df.iloc[movie_idx]['cluster']),
            'cluster_name': movies_metadata_df.iloc[movie_idx]['cluster_name']
        }
        
        return jsonify(movie)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Movie Recommendation API is running'})

if __name__ == '__main__':
    print("Initializing data and models...")
    initialize_data()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)