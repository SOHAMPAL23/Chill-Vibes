import React, { useState, useRef, useEffect } from 'react';

const MovieSearch = ({ onSearch, onResultSelect, results, showResults }) => {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const searchRef = useRef(null);

  useEffect(() => {
    setIsOpen(showResults);
  }, [showResults]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleInputChange = (e) => {
    const value = e.target.value;
    setQuery(value);
    onSearch(value);
  };

  const handleResultSelect = (result) => {
    setQuery('');
    onResultSelect(result);
    setIsOpen(false);
  };

  return (
    <div className="search-container" ref={searchRef}>
      <input
        type="text"
        className="search-input"
        placeholder="Search for movies..."
        value={query}
        onChange={handleInputChange}
        onFocus={() => query && setIsOpen(true)}
      />
      {isOpen && results.length > 0 && (
        <div className="search-results">
          {results.map((result) => (
            <div
              key={result.id}
              className="search-result-item"
              onClick={() => handleResultSelect(result)}
            >
              <div className="movie-title">{result.title}</div>
              <div className="movie-cluster">{result.cluster_name}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MovieSearch;