export default function LLMDiagnosis({ result }) {
  const diag = result?.layer2_diagnosis;

  if (!diag) {
    return (
      <div className="empty-state fade-in">
        <div className="empty-state__icon">🔍</div>
        <div className="empty-state__text">No diagnostic report</div>
        <div className="empty-state__hint">
          Pipeline is either normal or hasn't been run yet
        </div>
      </div>
    );
  }

  const fields = [
    { label: 'Root Cause', value: diag.cause, icon: '🎯' },
    { label: 'Affected Field', value: diag.affected_field, icon: '📌', mono: true },
    { label: 'Anomaly Category', value: diag.anomaly_category, icon: '📂' },
    { label: 'Expected Type', value: diag.expected_type, icon: '✅' },
    { label: 'Observed Type', value: diag.actual_type, icon: '❌' },
    { label: 'Replacement Key Found', value: diag.found_replacement_key, icon: '🔑', mono: true },
    { label: 'Suggested Fix', value: diag.suggested_fix, icon: '🛠️' },
  ];

  return (
    <div className="fade-in">
      <div className="glass-card" style={{ borderLeft: '3px solid var(--accent-violet)' }}>
        <div className="glass-card__title">
          <span className="glass-card__title-icon">🔍</span>
          Root Cause Diagnosis Report
          <span className="badge badge--anomaly" style={{ marginLeft: 'auto' }}>
            {diag.anomaly_category}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          {fields.map((f) => (
            <div key={f.label} style={{
              background: 'rgba(3, 7, 18, 0.4)',
              borderRadius: 'var(--radius-md)',
              padding: '14px 16px',
              border: '1px solid rgba(55, 65, 81, 0.25)',
            }}>
              <div style={{
                fontSize: '0.72rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: 'var(--text-muted)',
                marginBottom: 6,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}>
                <span>{f.icon}</span> {f.label}
              </div>
              <div style={{
                fontSize: '0.92rem',
                color: 'var(--text-primary)',
                fontFamily: f.mono ? "'JetBrains Mono', 'Fira Code', monospace" : 'inherit',
                fontWeight: f.mono ? 500 : 400,
                lineHeight: 1.5,
              }}>
                {f.value || '—'}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-card">
        <div className="glass-card__title">
          <span className="glass-card__title-icon">📋</span>
          Full Diagnostic JSON
        </div>
        <div className="json-viewer">{JSON.stringify(diag, null, 2)}</div>
      </div>
    </div>
  );
}
