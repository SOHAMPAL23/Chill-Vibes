import React from 'react';

const RecommendationPanel = ({ selectedMovie, recommendations, loading }) => {
  if (loading) {
    return (
      <div className="recommendation-panel">
        <div className="loading">Loading recommendations...</div>
      </div>
    );
  }

  return (
    <div className="recommendation-panel">
      <h3>Recommendations</h3>
      
      {selectedMovie && (
        <div className="selected-movie">
          <h4>Selected Movie: {selectedMovie.title}</h4>
          <div className="cluster-tag">
            Cluster: {selectedMovie.cluster_name}
          </div>
        </div>
      )}

      {recommendations.length > 0 ? (
        <div className="recommendations-list">
          {recommendations.map((rec, index) => (
            <div key={rec.id} className="recommendation-item">
              <div className="movie-info">
                <div className="movie-title">{rec.title}</div>
                <div className="movie-cluster">{rec.cluster_name}</div>
              </div>
              <div className="similarity-score">
                {(rec.similarity * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-recommendations">
          {selectedMovie 
            ? "No recommendations available for this movie." 
            : "Select a movie to see recommendations."}
        </div>
      )}
    </div>
  );
};

export default RecommendationPanel;