import React from 'react';

const ClusterVisualization = ({ clusters }) => {
  return (
    <div className="cluster-visualization">
      <h3>Movie Clusters</h3>
      {clusters.length > 0 ? (
        <div className="clusters-container">
          {clusters.map((cluster) => (
            <div key={cluster.id} className="cluster-item">
              <span className="cluster-name">{cluster.name}</span>
              <span className="cluster-count">{cluster.count} movies</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-clusters">Loading clusters...</div>
      )}
    </div>
  );
};

export default ClusterVisualization;