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
    }
  };

  const fetchClusters = async () => {
    try {
      const response = await axios.get('/api/clusters');
      setClusters(response.data);
    } catch (error) {
      console.error('Error fetching clusters:', error);
    }
  };

  const fetchRecommendations = async (movieId) => {
    setLoading(true);
    try {
      const response = await axios.get(`/api/recommendations/${movieId}`);
      setRecommendations(response.data.recommendations);
      setSelectedMovie(response.data.original_movie);
      setShowSearchResults(false);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
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
    } catch (error) {
      console.error('Error searching movies:', error);
    }
  };

  const handleSearchResultSelect = (movie) => {
    handleMovieSelect(movie);
    setSearchResults([]);
    setShowSearchResults(false);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎬 Movie Recommendation System</h1>
        <p>Powered by Content Clustering & Collaborative Filtering</p>
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
            <ClusterVisualization clusters={clusters} />
            <MovieList 
              movies={showSearchResults ? searchResults : movies}
              onMovieSelect={showSearchResults ? handleSearchResultSelect : handleMovieSelect}
              selectedMovie={selectedMovie}
            />
          </div>

          <div className="right-panel">
            <RecommendationPanel 
              selectedMovie={selectedMovie}
              recommendations={recommendations}
              loading={loading}
            />
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Movie Content Clustering & Item-Item Recommendation System</p>
      </footer>
    </div>
  );
}

export default App;