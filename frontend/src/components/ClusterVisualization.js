import React from 'react';
import './ClusterVisualization.css';

const ClusterVisualization = ({ clusters, onClusterSelect, selectedCluster }) => {
  const handleClusterClick = (cluster) => {
    onClusterSelect(cluster);
  };

  return (
    <div className="cluster-visualization">
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