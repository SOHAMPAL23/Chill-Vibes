import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MovieList from './components/MovieList';
import MovieSearch from './components/MovieSearch';
import RecommendationPanel from './components/RecommendationPanel';
import ClusterVisualization from './components/ClusterVisualization';
import './App.css';

function App() {
  const [movies, setMovies] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [activeTab, setActiveTab] = useState('movies'); // Added for tab navigation
  const [selectedCluster, setSelectedCluster] = useState(null); // Added for cluster filtering

  useEffect(() => {
    // Load initial data
    fetchMovies();
    fetchClusters();
  }, []);

  const fetchMovies = async () => {
    try {
      const response = await axios.get('/api/movies');
      setMovies(response.data);
    } catch (error) {
      console.error('Error fetching movies:', error);
      // Fallback to empty array if API fails
      setMovies([]);
    }
  };

  const fetchClusters = async () => {
    try {
      const response = await axios.get('/api/clusters');
      setClusters(response.data);
    } catch (error) {
      console.error('Error fetching clusters:', error);
      // Fallback to empty array if API fails
      setClusters([]);
    }
  };

  const fetchRecommendations = async (movieId) => {
    setLoading(true);
    try {
      const response = await axios.get(`/api/recommendations/${movieId}`);
      setRecommendations(response.data.recommendations);
      setSelectedMovie(response.data.original_movie);
      setShowSearchResults(false);
      setActiveTab('recommendations'); // Switch to recommendations tab
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setRecommendations([]);
    } finally {
      setLoading(false);
    }
  };

  const handleMovieSelect = (movie) => {
    fetchRecommendations(movie.id);
  };

  const handleSearch = async (query) => {
    if (query.trim() === '') {
      setSearchResults([]);
      setShowSearchResults(false);
      return;
    }

    try {
      const response = await axios.get(`/api/search?q=${query}`);
      setSearchResults(response.data);
      setShowSearchResults(true);
      setActiveTab('search'); // Switch to search results tab
    } catch (error) {
      console.error('Error searching movies:', error);
    }
  };

  const handleSearchResultSelect = (movie) => {
    handleMovieSelect(movie);
    setSearchResults([]);
    setShowSearchResults(false);
  };

  // Filter movies by selected cluster
  const filteredMovies = selectedCluster 
    ? movies.filter(movie => movie.cluster === selectedCluster.id)
    : movies;

  // Get movies in the same cluster as selected movie
  const clusterMovies = selectedMovie 
    ? movies.filter(movie => movie.cluster === selectedMovie.cluster)
    : [];

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎬 Movie Recommendation System</h1>
        <p>Powered by Content Clustering & Collaborative Filtering</p>
        
        {/* Enhanced navigation tabs */}
        <div className="tab-navigation">
          <button 
            className={activeTab === 'movies' ? 'active-tab' : ''}
            onClick={() => setActiveTab('movies')}
          >
            All Movies
          </button>
          <button 
            className={activeTab === 'clusters' ? 'active-tab' : ''}
            onClick={() => setActiveTab('clusters')}
          >
            Clusters
          </button>
          <button 
            className={activeTab === 'recommendations' ? 'active-tab' : ''}
            onClick={() => setActiveTab('recommendations')}
          >
            Recommendations
          </button>
          <button 
            className={activeTab === 'search' ? 'active-tab' : ''}
            onClick={() => setActiveTab('search')}
          >
            Search
          </button>
        </div>
      </header>

      <main className="app-main">
        <div className="search-section">
          <MovieSearch 
            onSearch={handleSearch}
            onResultSelect={handleSearchResultSelect}
            results={searchResults}
            showResults={showSearchResults}
          />
        </div>

        <div className="content-section">
          <div className="left-panel">
            {/* Show different content based on active tab */}
            {activeTab === 'movies' && (
              <>
                <div className="cluster-filter">
                  <label>Filter by Cluster: </label>
                  <select 
                    value={selectedCluster ? selectedCluster.id : ''}
                    onChange={(e) => {
                      const cluster = clusters.find(c => c.id === parseInt(e.target.value));
                      setSelectedCluster(cluster || null);
                    }}
                  >
                    <option value="">All Clusters</option>
                    {clusters.map(cluster => (
                      <option key={cluster.id} value={cluster.id}>
                        {cluster.name} ({cluster.count} movies)
                      </option>
                    ))}
                  </select>
                </div>
                
                <h3>
                  {selectedCluster 
                    ? `Movies in ${selectedCluster.name}` 
                    : 'All Movies'}
                </h3>
                
                <MovieList 
                  movies={filteredMovies}
                  onMovieSelect={handleMovieSelect}
                  selectedMovie={selectedMovie}
                />
              </>
            )}

            {activeTab === 'clusters' && (
              <>
                <h3>Movie Clusters</h3>
                <ClusterVisualization 
                  clusters={clusters} 
                  onClusterSelect={setSelectedCluster}
                  selectedCluster={selectedCluster}
                />
                
                {selectedCluster && (
                  <div className="cluster-movies">
                    <h4>Movies in {selectedCluster.name}:</h4>
                    <MovieList 
                      movies={movies.filter(m => m.cluster === selectedCluster.id).slice(0, 10)}
                      onMovieSelect={handleMovieSelect}
                      selectedMovie={selectedMovie}
                    />
                  </div>
                )}
              </>
            )}

            {activeTab === 'recommendations' && (
              <>
                <h3>Recommendations</h3>
                {selectedMovie ? (
                  <div className="selected-movie-info">
                    <h4>Based on: {selectedMovie.title}</h4>
                    <p>Cluster: {selectedMovie.cluster_name}</p>
                  </div>
                ) : (
                  <p>Select a movie to see recommendations</p>
                )}
                
                <RecommendationPanel 
                  selectedMovie={selectedMovie}
                  recommendations={recommendations}
                  loading={loading}
                  onMovieSelect={handleMovieSelect}
                />
                
                {selectedMovie && clusterMovies.length > 0 && (
                  <div className="cluster-suggestions">
                    <h4>Other movies in the same cluster:</h4>
                    <MovieList 
                      movies={clusterMovies.slice(0, 5)}
                      onMovieSelect={handleMovieSelect}
                      selectedMovie={selectedMovie}
                    />
                  </div>
                )}
              </>
            )}

            {activeTab === 'search' && (
              <>
                <h3>Search Results</h3>
                {showSearchResults ? (
                  <MovieList 
                    movies={searchResults}
                    onMovieSelect={handleSearchResultSelect}
                    selectedMovie={selectedMovie}
                  />
                ) : (
                  <p>Enter a search term to see results</p>
                )}
              </>
            )}
          </div>

          <div className="right-panel">
            {selectedMovie && (
              <div className="movie-detail-panel">
                <h3>{selectedMovie.title}</h3>
                <p><strong>Cluster:</strong> {selectedMovie.cluster_name}</p>
                <p><strong>Overview:</strong> {selectedMovie.overview}</p>
                
                {recommendations.length > 0 && (
                  <div className="quick-recommendations">
                    <h4>Top Recommendations:</h4>
                    <ul>
                      {recommendations.slice(0, 3).map((rec, index) => (
                        <li key={rec.id} onClick={() => handleMovieSelect(rec)}>
                          {index + 1}. {rec.title} (Sim: {(rec.similarity * 100).toFixed(1)}%)
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
            
            {clusters.length > 0 && (
              <div className="cluster-summary">
                <h3>Cluster Summary</h3>
                <div className="cluster-stats">
                  <p><strong>Total Clusters:</strong> {clusters.length}</p>
                  <p><strong>Total Movies:</strong> {movies.length}</p>
                  <div className="top-clusters">
                    <h4>Top 3 Clusters:</h4>
                    {clusters
                      .sort((a, b) => b.count - a.count)
                      .slice(0, 3)
                      .map(cluster => (
                        <div key={cluster.id} className="cluster-stat-item">
                          <span>{cluster.name}</span>
                          <span>{cluster.count} movies</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Movie Content Clustering & Item-Item Recommendation System</p>
        <p>Enhanced with hyperparameter tuning and interactive features</p>
      </footer>
    </div>
  );
}

export default App;