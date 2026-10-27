import React from 'react';

/**
 * Clean responsive Footer component.
 */
export default function Footer({ onNavigate }) {
  return (
    <footer className="footer" role="contentinfo">
      <div className="footer-container">
        <div className="footer-brand">
          <span className="footer-title">Business Entity Resolution</span>
          <span className="footer-subtitle">
            AI-powered entity matching across independent data sources
          </span>
        </div>

        <div className="footer-links">
          <button
            type="button"
            className="footer-link"
            onClick={() => onNavigate('dashboard')}
            style={{ background: 'none', border: 'none', cursor: 'pointer' }}
          >
            Dashboard
          </button>
          <button
            type="button"
            className="footer-link"
            onClick={() => onNavigate('results')}
            style={{ background: 'none', border: 'none', cursor: 'pointer' }}
          >
            Results
          </button>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="footer-link"
          >
            API Docs
          </a>
        </div>

        <div className="footer-meta">
          <span>XGBoost ML Pipeline • Threshold 0.70</span>
        </div>
      </div>
    </footer>
  );
}
