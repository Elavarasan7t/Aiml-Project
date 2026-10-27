import React from 'react';

/**
 * Visual representation of the Entity Resolution pipeline workflow.
 * Status states: 'idle' | 'processing' | 'completed' | 'error'
 */
export default function VisualPipeline({ status = 'idle' }) {
  const steps = [
    { id: 'upload', name: 'Data Upload', desc: 'TSV validation & ingest' },
    { id: 'clean', name: 'Data Cleaning', desc: 'NFKD normalization & missing flags' },
    { id: 'blocking', name: 'Candidate Generation', desc: 'Inverted index blocking' },
    { id: 'features', name: 'Feature Engineering', desc: '20 pairwise similarity signals' },
    { id: 'xgboost', name: 'XGBoost Prediction', desc: 'Batch tree inference' },
    { id: 'threshold', name: 'F0.5 Threshold', desc: 'Optimal cutoff (0.70)' },
    { id: 'results', name: 'Matching Results', desc: 'Entity-level aggregation' },
  ];

  const getStatusBadge = () => {
    switch (status) {
      case 'processing':
        return { text: 'Processing Active', className: 'status-processing' };
      case 'completed':
        return { text: 'Pipeline Completed', className: 'status-completed' };
      case 'error':
        return { text: 'Execution Error', className: 'status-error' };
      default:
        return { text: 'Workflow Idle', className: 'status-idle' };
    }
  };

  const badge = getStatusBadge();

  return (
    <div className="pipeline-workflow-card">
      <div className="pipeline-workflow-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Entity Resolution Pipeline
          </h3>
          <span className={`workflow-status-badge ${badge.className}`}>
            {status === 'processing' && <span className="status-dot pulsing" aria-hidden="true" />}
            {badge.text}
          </span>
        </div>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          End-to-End ML Architecture
        </span>
      </div>

      <div className="pipeline-steps-flow">
        {steps.map((step, idx) => (
          <React.Fragment key={step.id}>
            <div className={`pipeline-flow-step ${status}`}>
              <div className="step-circle" aria-hidden="true">
                {idx + 1}
              </div>
              <div className="step-text-wrap">
                <span className="step-flow-name">{step.name}</span>
                <span className="step-flow-desc">{step.desc}</span>
              </div>
            </div>
            {idx < steps.length - 1 && (
              <div className={`step-arrow-divider ${status}`} aria-hidden="true">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
