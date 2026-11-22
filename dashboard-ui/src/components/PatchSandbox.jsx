export default function PatchSandbox({ result }) {
  const patch = result?.layer3_patch;
  const verif = result?.layer3_verification;
  const success = result?.heal_success;

  if (!patch) {
    return (
      <div className="empty-state fade-in">
        <div className="empty-state__icon">🛠️</div>
        <div className="empty-state__text">No patch generated</div>
        <div className="empty-state__hint">
          Patches are only generated when an anomaly is detected
        </div>
      </div>
    );
  }

  return (
    <div className="fade-in">
      {/* Status bar */}
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-card__label">Sandbox Result</div>
          <div className={`metric-card__value ${success ? 'metric-card__value--green' : 'metric-card__value--red'}`}>
            {success ? '✓ PASSED' : '✗ FAILED'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label">Records Verified</div>
          <div className="metric-card__value metric-card__value--blue">
            {verif?.passed_count ?? 0} / {verif?.total_count ?? 0}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label">Success Rate</div>
          <div className="metric-card__value metric-card__value--purple">
            {((verif?.success_rate ?? 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label">Parser Status</div>
          <div
            className={`metric-card__value ${success ? 'metric-card__value--green' : 'metric-card__value--red'}`}
          >
            {success ? 'HOT-SWAPPED' : 'INACTIVE'}
          </div>
        </div>
      </div>

      <div className="columns columns--2">
        {/* Code Patch */}
        <div className="glass-card" style={{ borderLeft: '3px solid var(--accent-teal)' }}>
          <div className="glass-card__title">
            <span className="glass-card__title-icon">💻</span>
            Auto-Generated Python Patch
          </div>
          <div className="code-block">{patch}</div>
        </div>

        {/* Verification */}
        <div className="glass-card" style={{ borderLeft: success ? '3px solid var(--status-green)' : '3px solid var(--status-red)' }}>
          <div className="glass-card__title">
            <span className="glass-card__title-icon">🧪</span>
            Sandbox Verification
          </div>

          <div style={{
            background: success ? 'rgba(6, 95, 70, 0.2)' : 'rgba(153, 27, 27, 0.2)',
            border: `1px solid ${success ? 'rgba(52, 211, 153, 0.3)' : 'rgba(248, 113, 113, 0.3)'}`,
            borderRadius: 'var(--radius-md)',
            padding: '16px 20px',
            marginBottom: 16,
            display: 'flex',
            alignItems: 'center',
            gap: 12,
          }}>
            <span style={{ fontSize: '1.5rem' }}>{success ? '✅' : '❌'}</span>
            <div>
              <div style={{ fontWeight: 700, fontSize: '1rem', color: success ? 'var(--status-green)' : 'var(--status-red)' }}>
                Sandbox Verification {success ? 'PASSED' : 'FAILED'}
              </div>
              <div style={{ fontSize: '0.83rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                {success
                  ? 'Patch AST-screened, executed in an isolated interpreter under a timeout, and scored against held-out records before activation.'
                  : 'Patch failed verification. The previous parser remains in effect.'}
              </div>
            </div>
          </div>

          {success && (
            <div style={{
              background: 'rgba(30, 64, 175, 0.15)',
              border: '1px solid rgba(96, 165, 250, 0.3)',
              borderRadius: 'var(--radius-md)',
              padding: '14px 18px',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
            }}>
              <span className="status-dot status-dot--blue" />
              <span className="badge badge--healed">ACTIVE HOT-SWAPPED PARSER</span>
            </div>
          )}

          {verif && (
            <div className="json-viewer" style={{ marginTop: 16 }}>
              {JSON.stringify(verif, null, 2)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
