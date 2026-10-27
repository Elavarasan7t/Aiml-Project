import React from 'react';

/**
 * Animated indeterminate pipeline progress indicator.
 */
export default function ProcessingStatus() {
  const steps = [
    { num: '1', name: 'Validating TSV files & schemas' },
    { num: '2', name: 'Cleaning data & normalizing text' },
    { num: '3', name: 'Generating candidates via multi-strategy blocking' },
    { num: '4', name: 'Calculating 20 similarity features' },
    { num: '5', name: 'Running XGBoost model inference' },
    { num: '6', name: 'Applying optimal F0.5 threshold & saving results' },
  ];

  return (
    <div className="processing-card" role="status" aria-live="polite">
      <div className="processing-header">
        <div className="spinner-ring" aria-hidden="true" />
        <div className="processing-title-block">
          <h3>Processing entity matches...</h3>
          <p>The ML entity resolution pipeline is running. Please wait while records are matched.</p>
        </div>
      </div>

      <div className="pipeline-steps-grid">
        {steps.map((step) => (
          <div key={step.num} className="pipeline-step-item">
            <span className="step-num-badge">{step.num}</span>
            <span>{step.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
