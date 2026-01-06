import React from 'react';
import './ClusterVisualization.css';

const ClusterVisualization = ({ clusters, onClusterSelect, selectedCluster }) => {
  const handleClusterClick = (cluster) => {
    onClusterSelect(cluster);
  };

  return (
    <div className="cluster-visualization">
      <h3>Movie Clusters</h3>
      {clusters.length > 0 ? (
        clusters.map((cluster) => (
          <div
            key={cluster.id}
            className={`cluster-item ${selectedCluster && selectedCluster.id === cluster.id ? 'selected' : ''}`}
            onClick={() => handleClusterClick(cluster)}
            title={`${cluster.name} - ${cluster.count} movies`}
          >
            <h4>{cluster.name}</h4>
            <p>{cluster.count} movies</p>
            <div className="cluster-movies-preview">
              {cluster.movies.slice(0, 3).map((movie) => (
                <span key={movie.id} className="movie-preview">
                  {movie.title.split(' ')[0]}
                </span>
              ))}
            </div>
          </div>
        ))
      ) : (
        <div className="no-clusters">
          <p>Loading clusters...</p>
        </div>
      )}
    </div>
  );
};

export default ClusterVisualization;