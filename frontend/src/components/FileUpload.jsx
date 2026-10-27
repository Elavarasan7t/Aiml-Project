import React, { useRef, useState } from 'react';

/**
 * Single File Upload Card with drag-and-drop and .tsv validation.
 */
export default function FileUpload({
  sourceKey,
  sourceNumber,
  badgeTitle,
  badgeClass,
  description,
  file,
  onFileSelect,
  onFileRemove,
  disabled,
}) {
  const inputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const formatFileSize = (bytes) => {
    if (!bytes && bytes !== 0) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleValidateAndSelect = (selectedFile) => {
    setErrorMsg('');
    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith('.tsv')) {
      setErrorMsg('Only tab-separated .tsv files are accepted.');
      return;
    }

    onFileSelect(sourceKey, selectedFile);
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleValidateAndSelect(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleValidateAndSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className={`upload-card ${file ? 'has-file' : ''} ${isDragOver ? 'drag-over' : ''}`}>
      <div className="source-header">
        <div>
          <span className={`source-badge ${badgeClass}`}>{badgeTitle}</span>
          <h3 className="source-name">Source {sourceNumber}</h3>
          <p className="source-desc">{description}</p>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".tsv"
        className="file-input-hidden"
        onChange={handleInputChange}
        disabled={disabled}
        id={`file-input-${sourceKey}`}
        aria-label={`Upload TSV dataset for Source ${sourceNumber}`}
      />

      {file ? (
        <div className="file-selected-box">
          <div className="file-info-header">
            <div className="file-type-icon" aria-hidden="true">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="9" y1="15" x2="15" y2="15" />
              </svg>
            </div>
            <div className="file-details">
              <div className="file-name" title={file.name}>
                {file.name}
              </div>
              <div className="file-size">{formatFileSize(file.size)}</div>
            </div>
          </div>

          <div className="file-actions">
            <button
              type="button"
              className="btn-file-action"
              onClick={() => inputRef.current && inputRef.current.click()}
              disabled={disabled}
              title="Select a different TSV file"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
              </svg>
              Replace
            </button>
            <button
              type="button"
              className="btn-file-action remove"
              onClick={() => {
                setErrorMsg('');
                onFileRemove(sourceKey);
                if (inputRef.current) inputRef.current.value = '';
              }}
              disabled={disabled}
              title="Remove selected file"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
              Remove
            </button>
          </div>
        </div>
      ) : (
        <div
          className="dropzone"
          onClick={() => inputRef.current && inputRef.current.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && inputRef.current && inputRef.current.click()}
          aria-label={`Upload TSV file for Source ${sourceNumber}`}
        >
          <div className="dropzone-icon" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <div className="dropzone-text">Choose TSV File</div>
          <div className="dropzone-subtext">Click to browse or drag & drop</div>
        </div>
      )}

      {errorMsg && (
        <div className="alert-box error" style={{ marginTop: '0.85rem', padding: '0.65rem 0.85rem', fontSize: '0.78rem' }}>
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
}
