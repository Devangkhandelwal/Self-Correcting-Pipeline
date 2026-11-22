import { useCallback, useEffect, useRef, useState } from 'react';
import ErrorNotice from './ErrorNotice';
import { getAuditLogs, getEvaluation } from '../api';

const LOG_COLUMNS = ['timestamp', 'event_type', 'details'];

export default function EvaluationAudit() {
  const [metrics, setMetrics] = useState(null);
  const [logs, setLogs] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [metricsError, setMetricsError] = useState(null);
  const [logsError, setLogsError] = useState(null);

  const abortRef = useRef(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  const runEvaluation = useCallback(async () => {
    setLoadingMetrics(true);
    setMetricsError(null);
    try {
      setMetrics(await getEvaluation());
    } catch (error) {
      if (error?.name !== 'AbortError') {
        setMetricsError(error.message);
        setMetrics(null);
      }
    } finally {
      setLoadingMetrics(false);
    }
  }, []);

  const fetchLogs = useCallback(async () => {
    setLoadingLogs(true);
    setLogsError(null);
    try {
      const data = await getAuditLogs(30);
      setLogs(data.logs || []);
    } catch (error) {
      if (error?.name !== 'AbortError') {
        setLogsError(error.message);
        setLogs(null);
      }
    } finally {
      setLoadingLogs(false);
    }
  }, []);

  const model = metrics?.ml_model;
  const baseline = metrics?.rule_baseline;
  const matrix = metrics?.confusion_matrix;
  const classes = metrics?.classes;
  const comparison = metrics?.model_vs_baseline;

  return (
    <div className="fade-in">
      <div className="glass-card" style={{ borderLeft: '3px solid var(--accent-indigo)' }}>
        <div className="glass-card__title" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="glass-card__title-icon" aria-hidden="true">📊</span>
            Layer 1 Benchmark
          </span>
          <button
            type="button"
            className="btn btn--primary"
            onClick={runEvaluation}
            disabled={loadingMetrics}
          >
            {loadingMetrics ? (
              <>
                <span className="btn__spinner" aria-hidden="true" /> Evaluating…
              </>
            ) : (
              'Run Evaluation'
            )}
          </button>
        </div>

        {metricsError && (
          <ErrorNotice
            title="Evaluation failed"
            message={metricsError}
            onRetry={runEvaluation}
          />
        )}

        {model && (
          <div className="slide-up">
            <div className="metrics-row" style={{ marginTop: 16 }}>
              <div className="metric-card">
                <div className="metric-card__label">Precision</div>
                <div className="metric-card__value metric-card__value--green">
                  {(model.precision * 100).toFixed(2)}%
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-card__label">Recall</div>
                <div className="metric-card__value metric-card__value--blue">
                  {(model.recall * 100).toFixed(2)}%
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-card__label">F1-Score</div>
                <div className="metric-card__value metric-card__value--purple">
                  {(model.f1_score * 100).toFixed(2)}%
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-card__label">Rows Evaluated</div>
                <div className="metric-card__value">
                  {metrics.evaluated_rows}
                  {metrics.held_out ? ' (held-out)' : ' (NOT held-out)'}
                </div>
              </div>
            </div>

            {baseline && (
              <div style={{ marginTop: 20 }}>
                <h4 className="section-heading">Model vs Rule Baseline</h4>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Engine</th>
                      <th scope="col">Precision</th>
                      <th scope="col">Recall</th>
                      <th scope="col">F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { label: 'ML model', values: model },
                      { label: 'Rule baseline', values: baseline },
                    ].map(({ label, values }) => (
                      <tr key={label}>
                        <th scope="row">{label}</th>
                        <td>{(values.precision * 100).toFixed(2)}%</td>
                        <td>{(values.recall * 100).toFixed(2)}%</td>
                        <td>{(values.f1_score * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {comparison && (
                  <p className="metric-note">{comparison.note}</p>
                )}
              </div>
            )}

            {matrix && classes && (
              <div style={{ marginTop: 20 }}>
                <h4 className="section-heading">Confusion Matrix</h4>
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table">
                    <caption className="sr-only">
                      Rows are the true anomaly class; columns are the predicted class.
                    </caption>
                    <thead>
                      <tr>
                        <th scope="col">Actual \ Predicted</th>
                        {classes.map((name) => (
                          <th scope="col" key={name}>{name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {matrix.map((row, rowIndex) => (
                        <tr key={classes[rowIndex]}>
                          <th scope="row" style={{ color: 'var(--accent-indigo)' }}>
                            {classes[rowIndex]}
                          </th>
                          {row.map((value, colIndex) => {
                            const onDiagonal = rowIndex === colIndex;
                            return (
                              <td
                                key={classes[colIndex]}
                                style={{
                                  fontWeight: onDiagonal ? 700 : 400,
                                  color: onDiagonal
                                    ? 'var(--status-green)'
                                    : value > 0
                                      ? 'var(--status-red)'
                                      : 'var(--text-muted)',
                                  background: onDiagonal
                                    ? 'rgba(52, 211, 153, 0.08)'
                                    : 'transparent',
                                }}
                              >
                                {value}
                                {!onDiagonal && value > 0 && (
                                  <span className="sr-only"> misclassified</span>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {metrics.split_note && <p className="metric-note">{metrics.split_note}</p>}
            {metrics.caveat && (
              <p className="metric-note metric-note--warning">
                <strong>Measurement caveat: </strong>{metrics.caveat}
              </p>
            )}
          </div>
        )}

        {!model && !loadingMetrics && !metricsError && (
          <div className="empty-state__hint" style={{ textAlign: 'center', padding: 30 }}>
            Run the evaluation to benchmark Layer 1 on its held-out split.
          </div>
        )}
      </div>

      <div className="glass-card" style={{ borderLeft: '3px solid var(--accent-emerald)' }}>
        <div className="glass-card__title" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="glass-card__title-icon" aria-hidden="true">📜</span>
            Pipeline Audit Trail
          </span>
          <button
            type="button"
            className="btn btn--ghost"
            onClick={fetchLogs}
            disabled={loadingLogs}
          >
            {loadingLogs ? 'Loading…' : 'Refresh Logs'}
          </button>
        </div>

        {logsError && (
          <ErrorNotice title="Could not load the audit trail" message={logsError} onRetry={fetchLogs} />
        )}

        {logs && logs.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  {LOG_COLUMNS.map((column) => (
                    <th scope="col" key={column}>{column.replace(/_/g, ' ')}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {logs.map((entry, index) => (
                  <tr key={`${entry.timestamp}-${index}`}>
                    {LOG_COLUMNS.map((column) => {
                      const value = entry[column];
                      return (
                        <td
                          key={column}
                          style={{
                            maxWidth: 360,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={typeof value === 'object' ? JSON.stringify(value) : String(value ?? '')}
                        >
                          {typeof value === 'object'
                            ? JSON.stringify(value)
                            : String(value ?? '—')}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {logs && logs.length === 0 && !logsError && (
          <div className="empty-state__hint" style={{ textAlign: 'center', padding: 30 }}>
            No audit events recorded yet. Run the pipeline to generate some.
          </div>
        )}

        {!logs && !loadingLogs && !logsError && (
          <div className="empty-state__hint" style={{ textAlign: 'center', padding: 30 }}>
            Load the audit trail to see recorded detection, diagnosis and healing events.
          </div>
        )}
      </div>
    </div>
  );
}
