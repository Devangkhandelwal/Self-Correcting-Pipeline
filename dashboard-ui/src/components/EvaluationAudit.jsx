import { useState } from 'react';

const API = '/api';

export default function EvaluationAudit() {
  const [metrics, setMetrics] = useState(null);
  const [logs, setLogs] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState(false);

  async function runEvaluation() {
    setLoadingMetrics(true);
    try {
      const res = await fetch(`${API}/evaluate`);
      const data = await res.json();
      setMetrics(data);
    } catch (e) {
      console.error('Evaluation failed:', e);
    } finally {
      setLoadingMetrics(false);
    }
  }

  async function fetchLogs() {
    setLoadingLogs(true);
    try {
      const res = await fetch(`${API}/audit-logs?limit=30`);
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (e) {
      console.error('Fetching logs failed:', e);
    } finally {
      setLoadingLogs(false);
    }
  }

  const ml = metrics?.ml_model;
  const cm = metrics?.confusion_matrix;
  const classes = metrics?.classes;

  return (
    <div className="fade-in">
      {/* Evaluation Section */}
      <div className="glass-card" style={{ borderLeft: '3px solid var(--accent-indigo)' }}>
        <div className="glass-card__title" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="glass-card__title-icon">📊</span>
            Layer 1 ML Model Benchmark
          </span>
          <button
            className="btn btn--primary"
            onClick={runEvaluation}
            disabled={loadingMetrics}
          >
            {loadingMetrics ? (
              <>
                <span className="btn__spinner" /> Evaluating...
              </>
            ) : (
              '⚡ Run Evaluation'
            )}
          </button>
        </div>

        {ml && (
          <div className="slide-up">
            <div className="metrics-row" style={{ marginTop: 16 }}>
              <div className="metric-card">
                <div className="metric-card__label">Precision</div>
                <div className="metric-card__value metric-card__value--green">
                  {(ml.precision * 100).toFixed(2)}%
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-card__label">Recall</div>
                <div className="metric-card__value metric-card__value--blue">
                  {(ml.recall * 100).toFixed(2)}%
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-card__label">F1-Score</div>
                <div className="metric-card__value metric-card__value--purple">
                  {(ml.f1_score * 100).toFixed(2)}%
                </div>
              </div>
            </div>

            {/* Confusion Matrix */}
            {cm && classes && (
              <div style={{ marginTop: 20 }}>
                <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  Confusion Matrix
                </h4>
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Actual \ Predicted</th>
                        {classes.map((c) => <th key={c}>{c}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {cm.map((row, i) => (
                        <tr key={i}>
                          <td style={{ fontWeight: 600, color: 'var(--accent-indigo)' }}>{classes[i]}</td>
                          {row.map((val, j) => (
                            <td key={j} style={{
                              fontWeight: i === j ? 700 : 400,
                              color: i === j ? 'var(--status-green)' : val > 0 ? 'var(--status-red)' : 'var(--text-muted)',
                              background: i === j ? 'rgba(52, 211, 153, 0.08)' : 'transparent',
                            }}>
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {!ml && !loadingMetrics && (
          <div style={{ textAlign: 'center', padding: 30, color: 'var(--text-muted)' }}>
            Click "Run Evaluation" to benchmark the ML model on synthetic data
          </div>
        )}
      </div>

      {/* Audit Logs */}
      <div className="glass-card" style={{ borderLeft: '3px solid var(--accent-emerald)' }}>
        <div className="glass-card__title" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="glass-card__title-icon">📜</span>
            Pipeline Audit Trail
          </span>
          <button
            className="btn btn--ghost"
            onClick={fetchLogs}
            disabled={loadingLogs}
          >
            {loadingLogs ? 'Loading...' : '🔄 Refresh Logs'}
          </button>
        </div>

        {logs && logs.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  {Object.keys(logs[0]).map((k) => (
                    <th key={k}>{k.replace(/_/g, ' ')}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {logs.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((v, j) => (
                      <td key={j} style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : logs && logs.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 30, color: 'var(--text-muted)' }}>
            No audit events recorded yet
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: 30, color: 'var(--text-muted)' }}>
            Click "Refresh Logs" to load audit trail
          </div>
        )}
      </div>
    </div>
  );
}
