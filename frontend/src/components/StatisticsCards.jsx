import React from 'react';

/**
 * Metric cards component displaying real processing counts from the API.
 * Satisfies Task 10 and Task 14 (Matching Summary).
 */
export default function StatisticsCards({ data }) {
  if (!data) return null;

  const s1Count = data.source1_records ?? data.source1_count ?? 0;
  const s2Count = data.source2_records ?? data.source2_count ?? 0;
  const s3Count = data.source3_records ?? data.source3_count ?? 0;
  const totalBusinesses = s1Count + s2Count + s3Count;

  const candsCount = data.candidate_pairs ?? data.candidate_pair_count ?? 0;
  const matchedCount = data.matched_entities ?? data.matched_entity_count ?? 0;
  const unmatchedCount = data.unmatched_entities ?? data.unmatched_entity_count ?? 0;
  const threshold = data.threshold !== undefined && data.threshold !== null ? Number(data.threshold).toFixed(2) : '0.70';

  const matchRate = s1Count > 0 ? ((matchedCount / s1Count) * 100).toFixed(1) : '0.0';
  const formatNum = (n) => Number(n).toLocaleString();

  const cards = [
    {
      label: 'Total Businesses',
      value: formatNum(totalBusinesses),
      subtext: `S1: ${formatNum(s1Count)} | S2: ${formatNum(s2Count)} | S3: ${formatNum(s3Count)}`,
      color: 'indigo',
      icon: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
          <circle cx="9" cy="7" r="4" />
          <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
          <path d="M16 3.13a4 4 0 0 1 0 7.75" />
        </svg>
      ),
    },
    {
      label: 'Candidate Pairs',
      value: formatNum(candsCount),
      subtext: 'High-recall candidate pairs generated',
      color: 'amber',
      icon: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="16 18 22 12 16 6" />
          <polyline points="8 6 2 12 8 18" />
        </svg>
      ),
    },
    {
      label: 'Matched Entities',
      value: formatNum(matchedCount),
      subtext: `${matchRate}% of Source 1 entities resolved`,
      color: 'emerald',
      icon: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      ),
    },
    {
      label: 'Unmatched Entities',
      value: formatNum(unmatchedCount),
      subtext: 'Singletons (correctly no match assigned)',
      color: 'rose',
      icon: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
        </svg>
      ),
    },
    {
      label: 'Selected Threshold',
      value: threshold,
      subtext: 'Optimized validation F0.5 decision cutoff',
      color: 'cyan',
      icon: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 14 14" />
        </svg>
      ),
    },
    {
      label: 'Source 1 Records',
      value: formatNum(s1Count),
      subtext: 'Master reference entities evaluated',
      color: 'indigo',
      icon: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      ),
    },
    {
      label: 'Source 2 Records',
      value: formatNum(s2Count),
      subtext: 'Secondary source records processed',
      color: 'cyan',
      icon: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      ),
    },
    {
      label: 'Source 3 Records',
      value: formatNum(s3Count),
      subtext: 'Tertiary source records processed',
      color: 'amber',
      icon: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      ),
    },
  ];

  return (
    <div className="stats-grid">
      {cards.map((c) => (
        <div key={c.label} className="stat-card">
          <div className={`stat-icon-wrapper ${c.color}`} aria-hidden="true">
            {c.icon}
          </div>
          <div className="stat-content">
            <span className="stat-label">{c.label}</span>
            <span className="stat-value">{c.value}</span>
            <span className="stat-subtext">{c.subtext}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
