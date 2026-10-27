import React, { useState, useMemo } from 'react';

/**
 * Preview table displaying matched entity IDs per Source 1 entity.
 */
export default function ResultsTable({ items, isLoading }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all' | 'matched' | 'unmatched'
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 15;

  const filteredItems = useMemo(() => {
    if (!items || items.length === 0) return [];

    return items.filter((row) => {
      const s1 = row.source1_entity_id.toLowerCase();
      const m = row.matched_entity_ids.toLowerCase();
      const query = searchTerm.toLowerCase().trim();

      const matchesSearch = !query || s1.includes(query) || m.includes(query);
      if (!matchesSearch) return false;

      if (filterType === 'matched') {
        return row.count > 0;
      }
      if (filterType === 'unmatched') {
        return row.count === 0;
      }
      return true;
    });
  }, [items, searchTerm, filterType]);

  const totalPages = Math.ceil(filteredItems.length / pageSize) || 1;
  const paginatedItems = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredItems.slice(start, start + pageSize);
  }, [filteredItems, currentPage]);

  if (isLoading) {
    return (
      <div className="table-card" style={{ padding: '3rem', textAlign: 'center' }}>
        <div className="spinner-ring" style={{ margin: '0 auto 1rem' }} aria-hidden="true" />
        <p style={{ color: 'var(--text-secondary)' }}>Loading matching results preview...</p>
      </div>
    );
  }

  if (!items || items.length === 0) {
    return null;
  }

  return (
    <div className="table-card">
      <div className="table-toolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Matching Results Preview
          </h4>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Showing {filteredItems.length} of {items.length} entities
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <input
            type="text"
            className="table-search-input"
            placeholder="Search entity ID (e.g. S1-00001)..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
          />

          <div className="table-filter-group">
            <button
              type="button"
              className={`filter-btn ${filterType === 'all' ? 'active' : ''}`}
              onClick={() => {
                setFilterType('all');
                setCurrentPage(1);
              }}
            >
              All
            </button>
            <button
              type="button"
              className={`filter-btn ${filterType === 'matched' ? 'active' : ''}`}
              onClick={() => {
                setFilterType('matched');
                setCurrentPage(1);
              }}
            >
              Matched Only
            </button>
            <button
              type="button"
              className={`filter-btn ${filterType === 'unmatched' ? 'active' : ''}`}
              onClick={() => {
                setFilterType('unmatched');
                setCurrentPage(1);
              }}
            >
              Singletons
            </button>
          </div>
        </div>
      </div>

      <div className="table-wrapper">
        <table className="results-table">
          <thead>
            <tr>
              <th style={{ width: '220px' }}>Source 1 Entity ID</th>
              <th>Matched Entity IDs</th>
              <th style={{ width: '140px', textAlign: 'right' }}>Match Count</th>
            </tr>
          </thead>
          <tbody>
            {paginatedItems.length === 0 ? (
              <tr>
                <td colSpan="3" style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--text-muted)' }}>
                  No entities found matching your search criteria.
                </td>
              </tr>
            ) : (
              paginatedItems.map((row) => {
                const matchTokens = row.matched_entity_ids ? row.matched_entity_ids.split(',').filter(Boolean) : [];

                return (
                  <tr key={row.source1_entity_id}>
                    <td>
                      <span className="entity-id-pill">{row.source1_entity_id}</span>
                    </td>
                    <td>
                      {matchTokens.length > 0 ? (
                        <div className="match-tag-list">
                          {matchTokens.map((mid) => (
                            <span key={mid} className="match-tag" title={`Matched Candidate: ${mid}`}>
                              {mid}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="no-match-badge">
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="15" y1="9" x2="9" y2="15" />
                          </svg>
                          No matches
                        </span>
                      )}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <span style={{ fontWeight: 600, color: matchTokens.length > 0 ? 'var(--accent-emerald)' : 'var(--text-muted)' }}>
                        {matchTokens.length}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <div className="table-footer">
        <span>
          Page {currentPage} of {totalPages}
        </span>
        <div style={{ display: 'flex', gap: '0.4rem' }}>
          <button
            type="button"
            className="filter-btn"
            disabled={currentPage <= 1}
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            style={{ opacity: currentPage <= 1 ? 0.4 : 1 }}
          >
            Previous
          </button>
          <button
            type="button"
            className="filter-btn"
            disabled={currentPage >= totalPages}
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            style={{ opacity: currentPage >= totalPages ? 0.4 : 1 }}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
