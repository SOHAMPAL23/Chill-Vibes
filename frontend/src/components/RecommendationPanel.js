import React from 'react';
import './RecommendationPanel.css';

const RecommendationPanel = ({ selectedMovie, recommendations, loading, onMovieSelect }) => {
  const handleRecommendationClick = (movie) => {
    onMovieSelect(movie);
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Finding personalized recommendations...</p>
      </div>
    );
  }

  if (!selectedMovie) {
    return (
      <div className="no-recommendations">
        <p>Select a movie to see personalized recommendations</p>
      </div>
    );
  }

  return (
    <div className="recommendation-panel">
      <div className="selected-movie">
        <h3>Based on: {selectedMovie.title}</h3>
        <p>Cluster: {selectedMovie.cluster_name}</p>
      </div>
      
      {recommendations.length > 0 ? (
        <div className="recommendations-list">
          <h4>Recommended Movies:</h4>
          {recommendations.map((rec, index) => (
            <div 
              key={rec.id} 
              className="recommendation-item"
              onClick={() => handleRecommendationClick(rec)}
              title={`Similarity: ${(rec.similarity * 100).toFixed(1)}%`}
            >
              <div className="rec-header">
                <h4>{index + 1}. {rec.title}</h4>
                <span className="similarity-score">
                  {(rec.similarity * 100).toFixed(1)}% match
                </span>
              </div>
              <p>Cluster: {rec.cluster_name}</p>
              <p>{rec.overview.substring(0, 120)}{rec.overview.length > 120 ? '...' : ''}</p>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-recommendations">
          <p>No recommendations available for this movie. Try another selection.</p>
        </div>
      )}
    </div>
  );
};

export default RecommendationPanel;