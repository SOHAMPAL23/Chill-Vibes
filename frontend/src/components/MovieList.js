import React from 'react';

const MovieList = ({ movies, onMovieSelect, selectedMovie }) => {
  return (
    <div className="movie-list">
      <h3>All Movies</h3>
      <div className="movies-container">
        {movies.slice(0, 50).map((movie) => (
          <div
            key={movie.id}
            className={`movie-item ${selectedMovie && selectedMovie.id === movie.id ? 'selected' : ''}`}
            onClick={() => onMovieSelect(movie)}
          >
            <div className="movie-title">{movie.title}</div>
            <div className="movie-cluster">{movie.cluster_name}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MovieList;