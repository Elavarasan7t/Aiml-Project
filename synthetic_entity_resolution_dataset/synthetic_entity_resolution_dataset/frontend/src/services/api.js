/**
 * Centralized API Service for Business Entity Resolution Backend.
 * Communicates with FastAPI server at http://localhost:8000.
 */

export const API_BASE_URL = 'http://localhost:8000';

/**
 * Check backend health status.
 * @returns {Promise<{ status: string, service: string }>}
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

/**
 * Fetch trained model information, feature count, and selected threshold.
 * @returns {Promise<{ model: string, threshold: number, feature_count: number, status: string, metric: string, features: string[] }>}
 */
export async function getModelInfo() {
  const response = await fetch(`${API_BASE_URL}/api/model-info`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || errorData.error || `Model info failed with status ${response.status}`);
  }
  return response.json();
}

/**
 * Upload Source 1, Source 2, and Source 3 files to execute the ML entity matching pipeline.
 * @param {File} source1 - Source 1 TSV file
 * @param {File} source2 - Source 2 TSV file
 * @param {File} source3 - Source 3 TSV file
 * @returns {Promise<{ status: string, source1_records: number, source2_records: number, source3_records: number, candidate_pairs: number, matched_entities: number, unmatched_entities: number, threshold: number }>}
 */
export async function processFiles(source1, source2, source3) {
  const formData = new FormData();
  formData.append('source1', source1);
  formData.append('source2', source2);
  formData.append('source3', source3);

  // Note: Do not set Content-Type header manually when sending FormData,
  // the browser will automatically set multipart/form-data with proper boundary.
  const response = await fetch(`${API_BASE_URL}/api/process`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const message = errorData.message || errorData.error || `Processing failed with status ${response.status}`;
    const err = new Error(message);
    err.details = errorData;
    throw err;
  }

  return response.json();
}

/**
 * Fetch latest processing job statistics.
 * @returns {Promise<{ status: string, source1_count: number, source2_count: number, source3_count: number, candidate_pair_count: number, matched_entity_count: number, unmatched_entity_count: number, threshold: number, matching_file_available: boolean, candidates_file_available: boolean }>}
 */
export async function getResults() {
  const response = await fetch(`${API_BASE_URL}/api/results`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || errorData.error || `Results request failed with status ${response.status}`);
  }
  return response.json();
}

/**
 * Trigger browser download of matching_results.tsv.
 */
export async function downloadMatchingResults() {
  const response = await fetch(`${API_BASE_URL}/api/download/matching`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || errorData.error || 'Failed to download matching results.');
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'matching_results.tsv';
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

/**
 * Trigger browser download of candidate_pairs.tsv.
 */
export async function downloadCandidatePairs() {
  const response = await fetch(`${API_BASE_URL}/api/download/candidates`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || errorData.error || 'Failed to download candidate pairs.');
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'candidate_pairs.tsv';
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

/**
 * Fetch preview of matching results for interactive table display.
 * Uses dedicated backend endpoint /api/results/preview with fallback to TSV parsing.
 * @param {number} limit - Maximum rows to return (default 50)
 * @returns {Promise<Array<{ source1_entity_id: string, matched_entity_ids: string, count: number }>>}
 */
export async function fetchMatchingResultsPreview(limit = 50) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/results/preview?limit=${limit}`);
    if (response.ok) {
      const data = await response.json();
      if (data && Array.isArray(data.rows) && data.rows.length > 0) {
        return data.rows;
      }
    }
  } catch (err) {
    console.warn('API preview fetch failed, attempting download parsing fallback:', err);
  }

  // Fallback to downloading matching_results.tsv and parsing text
  try {
    const response = await fetch(`${API_BASE_URL}/api/download/matching`);
    if (!response.ok) {
      return [];
    }

    const text = await response.text();
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (lines.length <= 1) return [];

    const headers = lines[0].split('\t');
    const s1Idx = headers.indexOf('source1_entity_id');
    const matchIdx = headers.indexOf('matched_entity_ids');

    const parsed = [];
    const maxLines = Math.min(lines.length, limit + 1);

    for (let i = 1; i < maxLines; i++) {
      const cols = lines[i].split('\t');
      const s1Id = cols[s1Idx >= 0 ? s1Idx : 0] || '';
      const matchIds = cols[matchIdx >= 0 ? matchIdx : 1] || '';
      const count = matchIds.trim() ? matchIds.split(',').filter(Boolean).length : 0;

      parsed.push({
        source1_entity_id: s1Id,
        matched_entity_ids: matchIds.trim(),
        count,
      });
    }

    return parsed;
  } catch (fallbackErr) {
    console.warn('Fallback download parsing also failed:', fallbackErr);
    return [];
  }
}

