import React, { useState, useEffect } from 'react';
import StatisticsCards from '../components/StatisticsCards';
import ResultsTable from '../components/ResultsTable';
import {
  getResults,
  downloadMatchingResults,
  downloadCandidatePairs,
  fetchMatchingResultsPreview,
} from '../services/api';

export default function Results({ onNavigate }) {
  const [jobData, setJobData] = useState(null);
  const [tableItems, setTableItems] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isTableLoading, setIsTableLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [downloading, setDownloading] = useState({ matching: false, candidates: false });

  const loadJobResults = async () => {
    setIsLoading(true);
    setErrorMessage('');
    try {
      const res = await getResults();
      setJobData(res);

      if (res && res.results && Array.isArray(res.results) && res.results.length > 0) {
        const formatted = res.results.map((r) => ({
          source1_entity_id: r.source1_entity_id,
          matched_entity_ids: Array.isArray(r.matched_entity_ids) ? r.matched_entity_ids.join(',') : (r.matched_entity_ids || ''),
          count: r.match_count ?? (Array.isArray(r.matched_entity_ids) ? r.matched_entity_ids.length : 0),
        }));
        setTableItems(formatted);
      } else if (res && res.status === 'completed' && res.matching_file_available) {
        setIsTableLoading(true);
        try {
          const previewRows = await fetchMatchingResultsPreview(150);
          setTableItems(previewRows);
        } catch (tableErr) {
          console.warn('Could not load table preview:', tableErr);
        } finally {
          setIsTableLoading(false);
        }
      }
    } catch (err) {
      console.error('Failed to load results:', err);
      setErrorMessage(err.message || 'Unable to connect to the backend server.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadJobResults();
  }, []);

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

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: '5rem 0' }}>
        <div className="spinner-ring" style={{ margin: '0 auto 1.5rem' }} aria-hidden="true" />
        <h3 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>Loading Results...</h3>
        <p style={{ color: 'var(--text-muted)' }}>Retrieving latest execution status from API...</p>
      </div>
    );
  }

  // Error state
  if (errorMessage) {
    return (
      <div className="empty-state-box">
        <div className="empty-icon-circle" style={{ color: 'var(--accent-rose)' }} aria-hidden="true">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>
        <h2 className="empty-title">Connection Error</h2>
        <p className="empty-desc">{errorMessage}</p>
        <button type="button" className="btn-primary" onClick={loadJobResults}>
          Retry Connection
        </button>
      </div>
    );
  }

  // Empty state if no job has been processed yet
  const hasNoData = !jobData || jobData.status === 'idle' || (jobData.source1_count === 0 && !jobData.matching_file_available);

  if (hasNoData) {
    return (
      <div className="empty-state-box">
        <div className="empty-icon-circle" aria-hidden="true">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
            <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
            <line x1="6" y1="6" x2="6.01" y2="6" />
            <line x1="6" y1="18" x2="6.01" y2="18" />
          </svg>
        </div>
        <h2 className="empty-title">No results available yet.</h2>
        <p className="empty-desc">
          Return to Dashboard and upload your Source 1, Source 2, and Source 3 datasets to execute the ML matching pipeline.
        </p>
        <button
          type="button"
          className="btn-primary"
          onClick={() => onNavigate('dashboard')}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <line x1="19" y1="12" x2="5" y2="12" />
            <polyline points="12 19 5 12 12 5" />
          </svg>
          Go to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="results-page">
      {/* Header */}
      <div className="section-header" style={{ marginBottom: '1.75rem' }}>
        <div>
          <h2 className="section-title">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <line x1="18" y1="20" x2="18" y2="10" />
              <line x1="12" y1="20" x2="12" y2="4" />
              <line x1="6" y1="20" x2="6" y2="14" />
            </svg>
            Matching Execution Results
          </h2>
          <span className="section-helper">
            {jobData.processed_at
              ? `Executed on ${new Date(jobData.processed_at).toLocaleString()}`
              : 'Latest completed pipeline run'}
          </span>
        </div>

        <button
          type="button"
          className="filter-btn"
          onClick={loadJobResults}
          title="Refresh results from server"
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Real Statistics from GET /api/results */}
      <StatisticsCards data={jobData} />

      {/* Download Card */}
      <div className="download-card">
        <div className="download-info">
          <h4>Export Processed Data</h4>
          <p>Download the official TSV outputs generated by the XGBoost model at threshold {jobData.threshold ? Number(jobData.threshold).toFixed(2) : '0.70'}.</p>
        </div>

        <div className="download-btn-group">
          <button
            type="button"
            className="btn-download primary"
            onClick={() => handleDownload('matching')}
            disabled={downloading.matching || !jobData.matching_file_available}
            title="Download matching_results.tsv"
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
            disabled={downloading.candidates || !jobData.candidates_file_available}
            title="Download candidate_pairs.tsv"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            {downloading.candidates ? 'Downloading...' : 'Download Candidate Pairs'}
          </button>
        </div>
      </div>

      {/* Interactive Results Table Preview */}
      <ResultsTable items={tableItems} isLoading={isTableLoading} />
    </div>
  );
}
