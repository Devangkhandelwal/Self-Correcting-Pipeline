export default function MLDetector({ result }) {
  if (!result) {
    return (
      <div className="empty-state fade-in">
        <div className="empty-state__icon">🧠</div>
        <div className="empty-state__text">No detection data</div>
        <div className="empty-state__hint">Run the pipeline first to see Layer 1 ML results</div>
      </div>
    );
  }

  const det = result.layer1_detection || {};
  const features = det.features || {};
  const featureKeys = Object.keys(features);

  return (
    <div className="fade-in">
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-card__label">Detection</div>
          <div className={`metric-card__value ${det.is_anomaly ? 'metric-card__value--red' : 'metric-card__value--green'}`}>
            {det.is_anomaly ? '⚠ ANOMALY' : '✓ NORMAL'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label">Predicted Category</div>
          <div className="metric-card__value metric-card__value--purple">{det.anomaly_type}</div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label">Confidence Score</div>
          <div className="metric-card__value metric-card__value--blue">
            {((det.confidence || 0) * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="glass-card">
        <div className="glass-card__title">
          <span className="glass-card__title-icon">📊</span>
          Extracted Feature Vector
        </div>
        {featureKeys.length > 0 ? (
          <table className="data-table">
            <thead>
              <tr>
                {featureKeys.map((k) => (
                  <th key={k}>{k.replace(/_/g, ' ')}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                {featureKeys.map((k) => (
                  <td key={k}>
                    {typeof features[k] === 'number'
                      ? features[k].toFixed(4)
                      : String(features[k])}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        ) : (
          <div className="json-viewer">{JSON.stringify(det, null, 2)}</div>
        )}
      </div>
    </div>
  );
}
