import { useState, useEffect } from 'react';
import Plasma from './components/Plasma';
import ScrollExpand from './components/ScrollExpand';
import ParticleText from './components/ParticleText';
import { BrainMatrixBlock, ShieldMatrixBlock, KeyMatrixBlock } from './components/CyberBlocks';
import LivePipeline from './components/LivePipeline';
import MLDetector from './components/MLDetector';
import LLMDiagnosis from './components/LLMDiagnosis';
import PatchSandbox from './components/PatchSandbox';
import EvaluationAudit from './components/EvaluationAudit';

const API = '/api';

const TABS = [
  { id: 'pipeline', label: 'Live Pipeline', icon: '🛰️' },
  { id: 'detector', label: 'Layer 1: ML Detector', icon: '🧠' },
  { id: 'diagnosis', label: 'Layer 2: LLM Diagnosis', icon: '🔍' },
  { id: 'patch', label: 'Layer 3: Patch & Sandbox', icon: '🛠️' },
  { id: 'evaluation', label: 'Evaluation & Audit', icon: '📊' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline');
  const [pipelines, setPipelines] = useState([]);
  const [selectedPipeline, setSelectedPipeline] = useState('weather');
  const [anomalyClasses, setAnomalyClasses] = useState([]);
  const [selectedAnomaly, setSelectedAnomaly] = useState('schema_drift');
  const [pipelineResult, setPipelineResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/pipelines`)
      .then((r) => r.json())
      .then((data) => {
        setPipelines(data.pipelines || []);
        if (data.pipelines?.length) setSelectedPipeline(data.pipelines[0].id);
      })
      .catch((err) => console.error('Failed to fetch pipelines:', err));

    fetch(`${API}/anomaly-classes`)
      .then((r) => r.json())
      .then((data) => {
        setAnomalyClasses(data.classes || []);
        if (data.classes?.length) setSelectedAnomaly(data.classes[1] || data.classes[0]);
      })
      .catch((err) => console.error('Failed to fetch anomaly classes:', err));
  }, []);

  async function runPipeline() {
    setLoading(true);
    setPipelineResult(null);
    try {
      const res = await fetch(`${API}/run-pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pipeline_id: selectedPipeline,
          anomaly_type: selectedAnomaly
        }),
      });
      const data = await res.json();
      setPipelineResult(data);
    } catch (e) {
      console.error('Pipeline run failed:', e);
    } finally {
      setLoading(false);
    }
  }

  function renderTab() {
    switch (activeTab) {
      case 'pipeline':
        return <LivePipeline result={pipelineResult} loading={loading} selectedPipeline={selectedPipeline} pipelines={pipelines} />;
      case 'detector':
        return <MLDetector result={pipelineResult} />;
      case 'diagnosis':
        return <LLMDiagnosis result={pipelineResult} />;
      case 'patch':
        return <PatchSandbox result={pipelineResult} />;
      case 'evaluation':
        return <EvaluationAudit />;
      default:
        return null;
    }
  }

  const activePipelineObj = pipelines.find((p) => p.id === selectedPipeline);

  return (
    <>
      {/* Background Shader */}
      <Plasma
        color="#06B6D4"
        speed={1}
        direction="forward"
        scale={0.6}
        opacity={0.95}
        mouseInteractive
        iterations={60}
        renderScale={0.55}
        targetFps={60}
        maxDpr={1.5}
      />

      <div className="landing-page">
        {/* Navigation Bar */}
        <nav className="nav-bar">
          <div className="nav-bar__brand">
            <span className="nav-bar__logo">⚡</span>
            <span className="nav-bar__title">Self-Healing Data Agent</span>
          </div>
          <div className="nav-bar__links">
            <a href="#hero" className="nav-bar__link">Home</a>
            <a href="#features" className="nav-bar__link">Architecture</a>
            <a href="#workspace" className="nav-bar__link nav-bar__link--active">Live Sandbox</a>
          </div>
          <button className="btn btn--primary" onClick={() => {
            const el = document.getElementById('workspace');
            el?.scrollIntoView({ behavior: 'smooth' });
          }}>
            Launch Agent Console
          </button>
        </nav>

        {/* Hero Section with ParticleText */}
        <section id="hero">
          <div style={{ width: '100%', height: 210, paddingTop: '5.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ParticleText
              text="Future Interfaces"
              particleSize={2.0}
              density={3}
              color="#f8fafc"
              highlightColor="#8b5cf6"
              scatter={160}
              gatherDuration={1400}
              stagger={300}
              pointerRepel={40}
              repelRadius={110}
              idleDrift={0.6}
              glow
            />
          </div>

          <ScrollExpand
            src="/hero.jpg"
            alt="Self-Correcting MLOps Pipeline Agent"
            title="Self-Correcting Data Pipeline"
            scrollHint="Scroll down inside the frame to expand"
            useWindowScroll
            startWidth={50}
            startHeight={50}
            startRadius={24}
            endRadius={8}
            mediaZoom={1.3}
            overlayScrim={0.45}
            enabled
          >

            {/* Dark Cyber Block Cards matching Image 2 */}
            <div id="features" className="features-grid">
              {/* Card 1 */}
              <div className="cyber-card">
                <BrainMatrixBlock />
                <p className="cyber-card__desc">
                  Every alert, log, and payload event flows into one vectorizer — catching statistical drift, invalid types, and out-of-range outliers in under 5ms.
                </p>
                <div className="cyber-card__footer">
                  <span className="cyber-card__tag">Detection · Sentinel Core</span>
                  <button className="cyber-card__arrow" aria-label="Detection core details">→</button>
                </div>
              </div>

              {/* Card 2 */}
              <div className="cyber-card">
                <ShieldMatrixBlock />
                <p className="cyber-card__desc">
                  Automated LLM diagnostician compares corrupted payloads against schema specifications to pinpoint field drifts and missing keys before analyst intervention.
                </p>
                <div className="cyber-card__footer">
                  <span className="cyber-card__tag">Response · Sentinel Flow</span>
                  <button className="cyber-card__arrow" aria-label="Response flow details">→</button>
                </div>
              </div>

              {/* Card 3 */}
              <div className="cyber-card">
                <KeyMatrixBlock />
                <p className="cyber-card__desc">
                  Hardware-backed verification sandbox generates Python parser patches, executes 30+ regression tests, and hot-swaps active parser modules with zero downtime.
                </p>
                <div className="cyber-card__footer">
                  <span className="cyber-card__tag">Governance · Sentinel Vault</span>
                  <button className="cyber-card__arrow" aria-label="Governance vault details">→</button>
                </div>
              </div>
            </div>
          </ScrollExpand>
        </section>

        {/* Live Workspace Section */}
        <section id="workspace" className="workspace-section">
          <div className="app-container">
            {/* Control Bar */}
            <div className="control-bar">
              <span className="control-bar__label">Target Stream</span>
              <select
                className="control-bar__select"
                value={selectedPipeline}
                onChange={(e) => setSelectedPipeline(e.target.value)}
              >
                {pipelines.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.icon} {p.name}
                  </option>
                ))}
              </select>

              <div className="control-bar__divider" />

              <span className="control-bar__label">Inject Anomaly</span>
              <select
                className="control-bar__select"
                value={selectedAnomaly}
                onChange={(e) => setSelectedAnomaly(e.target.value)}
              >
                {anomalyClasses.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>

              <div className="control-bar__divider" />

              <button
                className="btn btn--primary"
                onClick={runPipeline}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="btn__spinner" /> Running Pipeline...
                  </>
                ) : (
                  `🚀 Run ${activePipelineObj ? activePipelineObj.icon : ''} Pipeline`
                )}
              </button>

              {pipelineResult && (
                <>
                  <div className="control-bar__divider" />
                  <span className={`badge ${pipelineResult.layer1_detection?.is_anomaly ? 'badge--anomaly' : 'badge--normal'}`}>
                    {pipelineResult.layer1_detection?.is_anomaly
                      ? `⚠ ${pipelineResult.layer1_detection.anomaly_type}`
                      : '✓ Normal'}
                  </span>
                  {pipelineResult.heal_success && (
                    <span className="badge badge--healed">✓ Auto-Healed</span>
                  )}
                </>
              )}
            </div>

            {/* Main Content */}
            <main className="app-content">
              {/* Tabs */}
              <nav className="tabs" role="tablist" aria-label="Pipeline layers">
                {TABS.map((t) => (
                  <button
                    key={t.id}
                    className={`tab ${activeTab === t.id ? 'tab--active' : ''}`}
                    onClick={() => setActiveTab(t.id)}
                    role="tab"
                    aria-selected={activeTab === t.id}
                    id={`tab-${t.id}`}
                  >
                    <span className="tab__icon">{t.icon}</span>
                    {t.label}
                  </button>
                ))}
              </nav>

              {/* Tab Panel */}
              <div role="tabpanel" aria-labelledby={`tab-${activeTab}`}>
                {renderTab()}
              </div>
            </main>
          </div>
        </section>
      </div>
    </>
  );
}
