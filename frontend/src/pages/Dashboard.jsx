import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import ProcessingStatus from '../components/ProcessingStatus';
import StatisticsCards from '../components/StatisticsCards';
import ModelInfoSection from '../components/ModelInfoSection';
import VisualPipeline from '../components/VisualPipeline';
import ResultsTable from '../components/ResultsTable';
import {
  processFiles,
  downloadMatchingResults,
  downloadCandidatePairs,
  fetchMatchingResultsPreview,
} from '../services/api';

export default function Dashboard({ modelInfo, backendStatus, onNavigate, onProcessingSuccess }) {
  const [files, setFiles] = useState({
    source1: null,
    source2: null,
    source3: null,
  });

  const [isProcessing, setIsProcessing] = useState(false);
  const [pipelineState, setPipelineState] = useState('idle'); // 'idle' | 'processing' | 'completed' | 'error'
  const [errorMessage, setErrorMessage] = useState('');
  const [lastResults, setLastResults] = useState(null);
  const [previewRows, setPreviewRows] = useState([]);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [downloading, setDownloading] = useState({ matching: false, candidates: false });

  const allFilesSelected = Boolean(files.source1 && files.source2 && files.source3);

  const handleFileSelect = (key, file) => {
    setErrorMessage('');
    setFiles((prev) => ({ ...prev, [key]: file }));
  };

  const handleFileRemove = (key) => {
    setFiles((prev) => ({ ...prev, [key]: null }));
  };

  const handleRunMatching = async () => {
    if (!allFilesSelected) {
      setErrorMessage('Please select all three TSV files before running entity resolution.');
      return;
    }

    if (backendStatus !== 'online') {
      setErrorMessage('Unable to connect to the Entity Resolution server. Make sure the FastAPI backend is running on http://localhost:8000.');
      return;
    }

    setIsProcessing(true);
    setPipelineState('processing');
    setErrorMessage('');
    setLastResults(null);
    setPreviewRows([]);

    try {
      const responseData = await processFiles(files.source1, files.source2, files.source3);
      setLastResults(responseData);
      setPipelineState('completed');

      if (onProcessingSuccess) {
        onProcessingSuccess(responseData);
      }

      // Fetch preview rows for immediate inspection
      setIsLoadingPreview(true);
      try {
        const rows = await fetchMatchingResultsPreview(50);
        setPreviewRows(rows);
      } catch (previewErr) {
        console.warn('Could not load preview table:', previewErr);
      } finally {
        setIsLoadingPreview(false);
      }
    } catch (err) {
      console.error('Matching pipeline error:', err);
      setErrorMessage(err.message || 'An error occurred while executing the entity resolution pipeline.');
      setPipelineState('error');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = async (type) => {
    try {
      setDownloading((prev) => ({ ...prev, [type]: true }));
      if (type === 'matching') {
        await downloadMatchingResults();
      } else {
        await downloadCandidatePairs();
      }
    } catch (err) {
      alert(`Download failed: ${err.message}`);
    } finally {
      setDownloading((prev) => ({ ...prev, [type]: false }));
    }
  };

  return (
    <div className="dashboard-page">
      {/* Hero Header */}
      <section className="hero-section">
        <div className="hero-badge">
          <span className="hero-badge-tag">AI Pipeline</span>
          <span>Multi-Source Entity Resolution</span>
        </div>

        <h1 className="hero-title">
          Business <span>Entity Resolution</span>
        </h1>

        <p className="hero-subtitle">
          AI-powered matching of business records across independent data sources.
        </p>

        {modelInfo && (
          <div className="model-pill-banner">
            <div className="model-pill-item">
              <span>Model:</span>
              <strong>{modelInfo.model}</strong>
            </div>
            <div className="model-pill-item">
              <span>Features:</span>
              <strong>{modelInfo.feature_count} Pairwise Similarity Signals</strong>
            </div>
            <div className="model-pill-item">
              <span>Validation Metric:</span>
              <strong>F0.5 (Macro: 0.9835)</strong>
            </div>
            <div className="model-pill-item">
              <span>Working Threshold:</span>
              <strong>{modelInfo.threshold.toFixed(2)}</strong>
            </div>
          </div>
        )}
      </section>

      {/* Task 12: Dedicated Model Information Section */}
      <ModelInfoSection modelInfo={modelInfo} />

      {/* Task 13: Visual Pipeline Flow Workflow */}
      <VisualPipeline status={pipelineState} />

      {/* Backend connection warning if offline */}
      {backendStatus === 'offline' && (
        <div className="alert-box error" role="alert">
          <div className="alert-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <div>
            <strong>Backend Unavailable:</strong> Unable to connect to the Entity Resolution server. Please make sure the FastAPI backend is running on <code>http://localhost:8000</code>.
          </div>
        </div>
      )}

      {/* Error alert box */}
      {errorMessage && (
        <div className="alert-box error" role="alert">
          <div className="alert-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <div>
            <strong>Validation / Execution Error:</strong> {errorMessage}
          </div>
        </div>
      )}

      {/* Upload Section Header */}
      <div className="section-header">
        <h2 className="section-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          Source Datasets
        </h2>
        <span className="section-helper">Required format: Tab-separated .tsv files</span>
      </div>

      {/* 3 Upload Cards */}
      <div className="upload-grid">
        <FileUpload
          sourceKey="source1"
          sourceNumber="1"
          badgeTitle="Reference Master"
          badgeClass="s1"
          description="Master reference dataset with unique business entities."
          file={files.source1}
          onFileSelect={handleFileSelect}
          onFileRemove={handleFileRemove}
          disabled={isProcessing}
        />

        <FileUpload
          sourceKey="source2"
          sourceNumber="2"
          badgeTitle="Secondary Source"
          badgeClass="s2"
          description="Independent business dataset with noisy attributes."
          file={files.source2}
          onFileSelect={handleFileSelect}
          onFileRemove={handleFileRemove}
          disabled={isProcessing}
        />

        <FileUpload
          sourceKey="source3"
          sourceNumber="3"
          badgeTitle="Tertiary Source"
          badgeClass="s3"
          description="Independent registry records with abbreviations."
          file={files.source3}
          onFileSelect={handleFileSelect}
          onFileRemove={handleFileRemove}
          disabled={isProcessing}
        />
      </div>

      {/* Trigger Action Card */}
      <div className="action-card">
        <div className="action-info">
          <span className="action-title">Run ML Entity Resolution</span>
          <span className="action-desc">
            {allFilesSelected
              ? 'All 3 datasets selected. Ready to execute blocking, feature extraction, and XGBoost matching.'
              : 'Please select Source 1, Source 2, and Source 3 TSV files to enable execution.'}
          </span>
        </div>

        <button
          type="button"
          className="btn-run"
          disabled={!allFilesSelected || isProcessing || backendStatus === 'offline'}
          onClick={handleRunMatching}
        >
          {isProcessing ? (
            <>
              <div className="spinner-ring" style={{ width: '18px', height: '18px', borderWidth: '2px' }} aria-hidden="true" />
              Processing Pipeline...
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              Run Entity Matching
            </>
          )}
        </button>
      </div>

      {/* Processing Animation */}
      {isProcessing && <ProcessingStatus />}

      {/* Execution Results Section */}
      {lastResults && (
        <section style={{ marginTop: '2rem' }}>
          <div className="alert-box success" role="status">
            <div className="alert-icon" aria-hidden="true">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
            </div>
            <div>
              <strong>Entity matching completed successfully.</strong> Applied optimal threshold {Number(lastResults.threshold).toFixed(2)}.
            </div>
          </div>

          {/* Task 14: Matching Summary Section Header */}
          <div className="section-header">
            <h3 className="section-title">Matching Summary</h3>
            <span className="section-helper">Real execution metrics returned by backend API</span>
          </div>

          <StatisticsCards data={lastResults} />

          {/* Quick Downloads & Navigation */}
          <div className="download-card">
            <div className="download-info">
              <h4>Download Artifacts</h4>
              <p>Export the newly generated matching results and candidate pairs as TSV files.</p>
            </div>

            <div className="download-btn-group">
              <button
                type="button"
                className="btn-download primary"
                onClick={() => handleDownload('matching')}
                disabled={downloading.matching}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                {downloading.matching ? 'Downloading...' : 'Download Matching Results'}
              </button>

              <button
                type="button"
                className="btn-download"
                onClick={() => handleDownload('candidates')}
                disabled={downloading.candidates}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                {downloading.candidates ? 'Downloading...' : 'Download Candidate Pairs'}
              </button>

              <button
                type="button"
                className="btn-download"
                onClick={() => onNavigate('results')}
                style={{ borderColor: 'var(--primary)', color: '#a5b4fc' }}
              >
                View Full Results Page →
              </button>
            </div>
          </div>

          {/* Task 11: Results Preview Table on Dashboard */}
          <div style={{ marginTop: '2rem' }}>
            <ResultsTable items={previewRows} isLoading={isLoadingPreview} />
          </div>
        </section>
      )}
    </div>
  );
}
