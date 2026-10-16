export default function LivePipeline({ result, loading, selectedPipeline, pipelines = [] }) {
  const currentPipelineObj = pipelines.find((p) => p.id === selectedPipeline);

  if (!result && !loading) {
    return (
      <div className="empty-state fade-in">
        <div className="empty-state__icon">{currentPipelineObj?.icon || '🛰️'}</div>
        <div className="empty-state__text">
          {currentPipelineObj ? `${currentPipelineObj.name} Ready` : 'No pipeline run yet'}
        </div>
        <div className="empty-state__hint">
          {currentPipelineObj?.description || 'Select an anomaly type above and click "Run Pipeline" to begin'}
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="empty-state fade-in">
        <div className="empty-state__icon">⚡</div>
        <div className="empty-state__text" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
          <span className="pulse-ring" /> Running {result?.pipeline_info?.name || 'pipeline'}...
        </div>
        <div className="empty-state__hint">Fetching → Injecting → Detecting → Diagnosing → Patching</div>
      </div>
    );
  }

  const det = result.layer1_detection || {};
  const isAnomaly = det.is_anomaly;
  const healed = result.heal_success;
  const pipeInfo = result.pipeline_info || {};

  return (
    <div className="fade-in">
      {/* Status overview */}
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-card__label">Active Stream</div>
          <div className="metric-card__value metric-card__value--blue">
            {pipeInfo.icon} {pipeInfo.name || 'Pipeline'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label">Injected Anomaly</div>
          <div className="metric-card__value metric-card__value--purple">
            {result.injected_anomaly}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label">Detection Status</div>
          <div className={`metric-card__value ${isAnomaly ? 'metric-card__value--red' : 'metric-card__value--green'}`}>
            {isAnomaly ? '⚠ ANOMALY' : '✓ NORMAL'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label">Auto-Heal Status</div>
          <div className={`metric-card__value ${healed ? 'metric-card__value--green' : healed === false ? 'metric-card__value--red' : ''}`}>
            {healed === true ? '✓ HEALED' : healed === false ? '✗ FAILED' : '—'}
          </div>
        </div>
      </div>

      {/* Payload comparison */}
      <div className="columns columns--3">
        <div className="glass-card">
          <div className="glass-card__title">
            <span className="glass-card__title-icon">🟢</span>
            Clean Stream Record
          </div>
          <div className="json-viewer">
            {JSON.stringify(result.clean_payload, null, 2)}
          </div>
        </div>

        <div className="glass-card">
          <div className="glass-card__title">
            <span className="glass-card__title-icon">🔴</span>
            Corrupted Ingestion
            <span className="badge badge--anomaly" style={{ marginLeft: 'auto' }}>
              {result.injected_anomaly}
            </span>
          </div>
          <div className="json-viewer">
            {JSON.stringify(result.corrupted_payload, null, 2)}
          </div>
        </div>

        <div className="glass-card">
          <div className="glass-card__title">
            <span className="glass-card__title-icon">🔵</span>
            Healed Parser Output
            {healed && (
              <span className="badge badge--healed" style={{ marginLeft: 'auto' }}>HEALED</span>
            )}
            {healed === false && (
              <span className="badge badge--anomaly" style={{ marginLeft: 'auto' }}>BROKEN</span>
            )}
            {!isAnomaly && (
              <span className="badge badge--normal" style={{ marginLeft: 'auto' }}>CLEAN</span>
            )}
          </div>
          <div className="json-viewer">
            {result.healed_payload
              ? JSON.stringify(result.healed_payload, null, 2)
              : !isAnomaly
                ? JSON.stringify(result.clean_payload, null, 2)
                : '"Pipeline break unresolved"'}
          </div>
        </div>
      </div>
    </div>
  );
}
