import React from 'react';
import './MovieList.css';

const MovieList = ({ movies, onMovieSelect, selectedMovie }) => {
  const handleMovieClick = (movie) => {
    onMovieSelect(movie);
  };

  return (
    <div className="movie-list">
      {movies.length > 0 ? (
        movies.map((movie) => (
          <div
            key={movie.id}
            className={`movie-item ${selectedMovie && selectedMovie.id === movie.id ? 'selected' : ''}`}
            onClick={() => handleMovieClick(movie)}
            title={`${movie.title} - ${movie.cluster_name}`}
          >
            <h4>{movie.title}</h4>
            <p className="cluster-info"><strong>Cluster:</strong> {movie.cluster_name}</p>
            <p>{movie.overview.substring(0, 100)}{movie.overview.length > 100 ? '...' : ''}</p>
          </div>
        ))
      ) : (
        <div className="no-movies">
          <p>No movies available. Please try adjusting your filters.</p>
        </div>
      )}
    </div>
  );
};

export default MovieList;