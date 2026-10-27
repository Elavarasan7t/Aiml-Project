import React, { useState } from 'react';

export default function Navbar({ activePage, onNavigate, backendStatus, modelInfo }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const isOnline = backendStatus === 'online';

  return (
    <nav className="navbar" role="navigation" aria-label="Main Navigation">
      <div className="nav-container">
        {/* Brand */}
        <div
          className="nav-brand"
          onClick={() => onNavigate('dashboard')}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && onNavigate('dashboard')}
          title="Return to Dashboard"
        >
          <div className="nav-logo-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
              <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
              <line x1="6" y1="6" x2="6.01" y2="6" />
              <line x1="6" y1="18" x2="6.01" y2="18" />
            </svg>
          </div>
          <div className="nav-title-block">
            <span className="nav-title">Business Entity Resolution</span>
            <span className="nav-subtitle">
              {modelInfo ? `${modelInfo.model} • Threshold ${modelInfo.threshold.toFixed(2)}` : 'AI Matching Engine'}
            </span>
          </div>
        </div>

        {/* Desktop Navigation Links */}
        <div className={`nav-links ${mobileOpen ? 'mobile-open' : ''}`}>
          <button
            type="button"
            className={`nav-item ${activePage === 'dashboard' ? 'active' : ''}`}
            onClick={() => {
              onNavigate('dashboard');
              setMobileOpen(false);
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <polyline points="9 22 9 12 15 12 15 22" />
            </svg>
            Dashboard
          </button>

          <button
            type="button"
            className={`nav-item ${activePage === 'results' ? 'active' : ''}`}
            onClick={() => {
              onNavigate('results');
              setMobileOpen(false);
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="20" x2="18" y2="10" />
              <line x1="12" y1="20" x2="12" y2="4" />
              <line x1="6" y1="20" x2="6" y2="14" />
            </svg>
            Results
          </button>

          {/* Backend Status Indicator */}
          <div
            className={`nav-status-badge ${isOnline ? '' : 'offline'}`}
            title={isOnline ? 'FastAPI Backend Online (http://localhost:8000)' : 'FastAPI Backend Offline'}
          >
            <span className="status-dot" aria-hidden="true" />
            <span>{isOnline ? 'Backend Connected' : 'Backend Offline'}</span>
          </div>
        </div>

        {/* Mobile menu button */}
        <button
          type="button"
          className="mobile-menu-btn"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation menu"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            {mobileOpen ? (
              <line x1="18" y1="6" x2="6" y2="18" />
            ) : (
              <>
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </>
            )}
          </svg>
        </button>
      </div>
    </nav>
  );
}
