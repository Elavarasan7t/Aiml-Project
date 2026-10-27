import React from 'react';

/**
 * Model information section displaying verified backend parameters.
 */
export default function ModelInfoSection({ modelInfo }) {
  if (!modelInfo) return null;

  const thresholdVal = typeof modelInfo.threshold === 'number'
    ? modelInfo.threshold.toFixed(2)
    : '0.70';

  const featureCount = modelInfo.feature_count ?? 20;

  return (
    <div className="model-info-card">
      <div className="model-info-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="model-badge-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Model Architecture & Parameters
            </h3>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Verified metadata loaded from backend JSON artifacts
            </span>
          </div>
        </div>
        <span className="source-badge s1">Status: {modelInfo.status}</span>
      </div>

      <div className="model-info-grid">
        <div className="model-info-item">
          <span className="model-info-label">Model:</span>
          <strong className="model-info-value">{modelInfo.model}</strong>
        </div>

        <div className="model-info-item">
          <span className="model-info-label">Task:</span>
          <strong className="model-info-value">{modelInfo.task || 'Binary Entity Classification'}</strong>
        </div>

        <div className="model-info-item">
          <span className="model-info-label">Classes:</span>
          <strong className="model-info-value">{modelInfo.classes || 'Different Business / Same Business'}</strong>
        </div>

        <div className="model-info-item">
          <span className="model-info-label">Threshold:</span>
          <strong className="model-info-value" style={{ color: 'var(--accent-emerald)' }}>
            {thresholdVal}
          </strong>
        </div>

        <div className="model-info-item">
          <span className="model-info-label">Feature Count:</span>
          <strong className="model-info-value">{featureCount} Features</strong>
        </div>
      </div>
    </div>
  );
}
