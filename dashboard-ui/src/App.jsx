import { useCallback, useEffect, useRef, useState } from 'react';
import Plasma from './components/Plasma';
import ScrollExpand from './components/ScrollExpand';
import ParticleText from './components/ParticleText';
import { BrainMatrixBlock, ShieldMatrixBlock, KeyMatrixBlock } from './components/CyberBlocks';
import LivePipeline from './components/LivePipeline';
import MLDetector from './components/MLDetector';
import LLMDiagnosis from './components/LLMDiagnosis';
import PatchSandbox from './components/PatchSandbox';
import EvaluationAudit from './components/EvaluationAudit';
import ErrorNotice from './components/ErrorNotice';
import { getAnomalyClasses, getHealth, getPipelines, runPipeline } from './api';

const TABS = [
  { id: 'pipeline', label: 'Live Pipeline', icon: '🛰️' },
  { id: 'detector', label: 'Layer 1: ML Detector', icon: '🧠' },
  { id: 'diagnosis', label: 'Layer 2: Diagnosis', icon: '🔍' },
  { id: 'patch', label: 'Layer 3: Patch & Sandbox', icon: '🛠️' },
  { id: 'evaluation', label: 'Evaluation & Audit', icon: '📊' },
];

const ARCHITECTURE_CARDS = [
  {
    Visual: BrainMatrixBlock,
    tag: 'Layer 1 · ML Anomaly Detector',
    description:
      'Each payload is vectorised into ten structural features — missing and ' +
      'unexpected keys, type mismatches, null counts, key-similarity scores and ' +
      'range violations — then classified into one of five failure modes.',
  },
  {
    Visual: ShieldMatrixBlock,
    tag: 'Layer 2 · Root-Cause Diagnosis',
    description:
      'A diagnostic agent compares the failing payload against the declared ' +
      'schema to pinpoint the affected field and the drift that caused it. ' +
      'Gemini drives it when configured, with a deterministic engine as fallback.',
  },
  {
    Visual: KeyMatrixBlock,
    tag: 'Layer 3 · Sandboxed Patching',
    description:
      'Candidate parser patches are AST-screened, executed in an isolated ' +
      'interpreter under a timeout, scored against held-out records, and only ' +
      'then hot-swapped into the live pipeline.',
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline');
  const [pipelines, setPipelines] = useState([]);
  const [selectedPipeline, setSelectedPipeline] = useState('');
  const [anomalyClasses, setAnomalyClasses] = useState([]);
  const [selectedAnomaly, setSelectedAnomaly] = useState('schema_drift');
  const [health, setHealth] = useState(null);

  const [pipelineResult, setPipelineResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [runError, setRunError] = useState(null);

  const [bootstrapError, setBootstrapError] = useState(null);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [reloadToken, setReloadToken] = useState(0);

  // Aborts the in-flight pipeline run if the component unmounts or a new run
  // starts, so a late response cannot overwrite fresher state.
  const runAbortRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    async function bootstrap() {
      setBootstrapping(true);
      setBootstrapError(null);
      try {
        const [pipelineData, anomalyData] = await Promise.all([
          getPipelines(controller.signal),
          getAnomalyClasses(controller.signal),
        ]);
        if (cancelled) return;

        const list = pipelineData.pipelines || [];
        setPipelines(list);
        if (list.length) setSelectedPipeline(list[0].id);

        const classes = anomalyData.classes || [];
        setAnomalyClasses(classes);
        if (classes.length) {
          setSelectedAnomaly(classes.includes('schema_drift') ? 'schema_drift' : classes[0]);
        }

        // Non-fatal: the console still works without the health probe.
        try {
          setHealth(await getHealth(controller.signal));
        } catch {
          setHealth(null);
        }
      } catch (error) {
        if (!cancelled && error?.name !== 'AbortError') setBootstrapError(error.message);
      } finally {
        if (!cancelled) setBootstrapping(false);
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [reloadToken]);

  useEffect(() => () => runAbortRef.current?.abort(), []);

  // Clearing the result when the target changes prevents weather output from
  // sitting on screen labelled as a finance run.
  const handlePipelineChange = useCallback((event) => {
    setSelectedPipeline(event.target.value);
    setPipelineResult(null);
    setRunError(null);
  }, []);

  const handleAnomalyChange = useCallback((event) => {
    setSelectedAnomaly(event.target.value);
    setPipelineResult(null);
    setRunError(null);
  }, []);

  const handleRun = useCallback(async () => {
    if (!selectedPipeline || !selectedAnomaly) return;

    runAbortRef.current?.abort();
    const controller = new AbortController();
    runAbortRef.current = controller;

    setLoading(true);
    setRunError(null);
    setPipelineResult(null);
    try {
      const data = await runPipeline(selectedPipeline, selectedAnomaly, controller.signal);
      setPipelineResult(data);
    } catch (error) {
      if (error?.name !== 'AbortError') setRunError(error.message);
    } finally {
      if (runAbortRef.current === controller) setLoading(false);
    }
  }, [selectedPipeline, selectedAnomaly]);

  const activePipeline = pipelines.find((item) => item.id === selectedPipeline);
  const detection = pipelineResult?.layer1_detection;

  function renderTab() {
    switch (activeTab) {
      case 'pipeline':
        return (
          <LivePipeline
            result={pipelineResult}
            loading={loading}
            selectedPipeline={selectedPipeline}
            pipelines={pipelines}
          />
        );
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

  return (
    <>
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
        <nav className="nav-bar">
          <div className="nav-bar__brand">
            <span className="nav-bar__logo" aria-hidden="true">⚡</span>
            <span className="nav-bar__title">Self-Healing Data Agent</span>
          </div>
          <div className="nav-bar__links">
            <a href="#hero" className="nav-bar__link">Home</a>
            <a href="#features" className="nav-bar__link">Architecture</a>
            <a href="#workspace" className="nav-bar__link nav-bar__link--active">Live Sandbox</a>
          </div>
          <a className="btn btn--primary" href="#workspace">
            Launch Agent Console
          </a>
        </nav>

        <section id="hero">
          <div className="hero-title">
            <ParticleText
              text="Self-Healing Pipelines"
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
            alt="Abstract visualisation of a data pipeline healing itself"
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
            <div id="features" className="features-grid">
              {ARCHITECTURE_CARDS.map(({ Visual, tag, description }) => (
                <div className="cyber-card" key={tag}>
                  <Visual />
                  <p className="cyber-card__desc">{description}</p>
                  <div className="cyber-card__footer">
                    <span className="cyber-card__tag">{tag}</span>
                  </div>
                </div>
              ))}
            </div>
          </ScrollExpand>
        </section>

        <section id="workspace" className="workspace-section">
          <div className="app-container">
            {bootstrapError ? (
              <ErrorNotice
                title="Could not load the agent console"
                message={bootstrapError}
                onRetry={() => setReloadToken((token) => token + 1)}
              />
            ) : (
              <>
                <div className="control-bar">
                  <label className="control-bar__label" htmlFor="pipeline-select">
                    Target Stream
                  </label>
                  <select
                    id="pipeline-select"
                    className="control-bar__select"
                    value={selectedPipeline}
                    onChange={handlePipelineChange}
                    disabled={bootstrapping || !pipelines.length}
                  >
                    {pipelines.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.icon} {item.name}
                      </option>
                    ))}
                  </select>

                  <div className="control-bar__divider" />

                  <label className="control-bar__label" htmlFor="anomaly-select">
                    Inject Anomaly
                  </label>
                  <select
                    id="anomaly-select"
                    className="control-bar__select"
                    value={selectedAnomaly}
                    onChange={handleAnomalyChange}
                    disabled={bootstrapping || !anomalyClasses.length}
                  >
                    {anomalyClasses.map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>

                  <div className="control-bar__divider" />

                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={handleRun}
                    disabled={loading || bootstrapping || !selectedPipeline}
                  >
                    {loading ? (
                      <>
                        <span className="btn__spinner" aria-hidden="true" /> Running Pipeline…
                      </>
                    ) : (
                      `Run ${activePipeline ? activePipeline.icon : ''} Pipeline`
                    )}
                  </button>

                  {detection && (
                    <>
                      <div className="control-bar__divider" />
                      <span
                        className={`badge ${detection.is_anomaly ? 'badge--anomaly' : 'badge--normal'}`}
                      >
                        {detection.is_anomaly
                          ? `Anomaly: ${detection.anomaly_type}`
                          : 'Normal'}
                      </span>
                      {pipelineResult.heal_success === true && (
                        <span className="badge badge--healed">Auto-Healed</span>
                      )}
                      {pipelineResult.heal_success === false && (
                        <span className="badge badge--anomaly">Heal Failed</span>
                      )}
                    </>
                  )}
                </div>

                {health && !health.model_loaded && (
                  <ErrorNotice
                    title="Layer 1 is running on rule-based heuristics"
                    message="No trained model was found. Run `python run_data_and_train.py` to train the classifier."
                  />
                )}

                {runError && (
                  <ErrorNotice
                    title="Pipeline run failed"
                    message={runError}
                    onRetry={handleRun}
                  />
                )}

                <main className="app-content">
                  <nav className="tabs" role="tablist" aria-label="Pipeline layers">
                    {TABS.map((tab) => (
                      <button
                        type="button"
                        key={tab.id}
                        className={`tab ${activeTab === tab.id ? 'tab--active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                        role="tab"
                        aria-selected={activeTab === tab.id}
                        aria-controls="tab-panel"
                        id={`tab-${tab.id}`}
                      >
                        <span className="tab__icon" aria-hidden="true">{tab.icon}</span>
                        {tab.label}
                      </button>
                    ))}
                  </nav>

                  <div
                    id="tab-panel"
                    role="tabpanel"
                    tabIndex={-1}
                    aria-labelledby={`tab-${activeTab}`}
                  >
                    {renderTab()}
                  </div>
                </main>
              </>
            )}
          </div>
        </section>
      </div>
    </>
  );
}
