import os
import sys
import json
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.api_client import WeatherAPIClient
from src.ingestion.injector import SyntheticAnomalyInjector, ANOMALY_CLASSES
from src.layer1_detector.model import MLAnomalyDetector
from src.layer1_detector.evaluator import evaluate_layer1
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator
from src.utils.logger import get_recent_audit_logs

# Streamlit Page Config & Custom CSS Styling
st.set_page_config(
    page_title="Self-Correcting Data Pipeline Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ============================================
       AURORA BEAM ANIMATED BACKGROUND
       Sweeping layered sheets of light effect
       ============================================ */

    /* Force dark base for entire app */
    .stApp {
        background: #030712 !important;
    }

    /* Aurora Beam Container — fixed behind all content */
    .aurora-beam-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: 0;
        pointer-events: none;
    }

    /* Individual aurora light sheets */
    .aurora-sheet {
        position: absolute;
        width: 200%;
        height: 60%;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0;
        mix-blend-mode: screen;
    }

    /* Sheet 1: Deep indigo sweep */
    .aurora-sheet-1 {
        background: radial-gradient(ellipse at center, rgba(99, 102, 241, 0.35) 0%, rgba(79, 70, 229, 0.15) 40%, transparent 70%);
        top: -30%;
        left: -50%;
        animation: aurora-sweep-1 8s ease-in-out infinite alternate;
    }

    /* Sheet 2: Violet / purple beam */
    .aurora-sheet-2 {
        background: radial-gradient(ellipse at center, rgba(139, 92, 246, 0.30) 0%, rgba(124, 58, 237, 0.12) 40%, transparent 70%);
        top: -20%;
        left: -30%;
        animation: aurora-sweep-2 10s ease-in-out infinite alternate;
    }

    /* Sheet 3: Teal / cyan ribbon */
    .aurora-sheet-3 {
        background: radial-gradient(ellipse at center, rgba(6, 182, 212, 0.25) 0%, rgba(20, 184, 166, 0.10) 40%, transparent 70%);
        top: -25%;
        left: -20%;
        animation: aurora-sweep-3 12s ease-in-out infinite alternate;
    }

    /* Sheet 4: Magenta / pink accent */
    .aurora-sheet-4 {
        background: radial-gradient(ellipse at center, rgba(236, 72, 153, 0.20) 0%, rgba(219, 39, 119, 0.08) 40%, transparent 70%);
        top: -15%;
        left: -40%;
        animation: aurora-sweep-4 14s ease-in-out infinite alternate;
    }

    /* Sheet 5: Emerald shimmer */
    .aurora-sheet-5 {
        background: radial-gradient(ellipse at center, rgba(52, 211, 153, 0.18) 0%, rgba(16, 185, 129, 0.06) 40%, transparent 70%);
        top: -35%;
        left: 0%;
        animation: aurora-sweep-5 9s ease-in-out infinite alternate;
    }

    @keyframes aurora-sweep-1 {
        0%   { transform: translateX(-10%) translateY(0%) rotate(-8deg) scale(1);   opacity: 0.6; }
        50%  { transform: translateX(15%) translateY(5%) rotate(4deg) scale(1.1);   opacity: 0.85; }
        100% { transform: translateX(40%) translateY(-5%) rotate(-3deg) scale(1.05); opacity: 0.5; }
    }
    @keyframes aurora-sweep-2 {
        0%   { transform: translateX(20%) translateY(10%) rotate(5deg) scale(1.05);  opacity: 0.5; }
        50%  { transform: translateX(-5%) translateY(-8%) rotate(-6deg) scale(1.15); opacity: 0.75; }
        100% { transform: translateX(-25%) translateY(3%) rotate(8deg) scale(0.95);  opacity: 0.55; }
    }
    @keyframes aurora-sweep-3 {
        0%   { transform: translateX(-15%) translateY(-5%) rotate(3deg) scale(1.1);  opacity: 0.45; }
        50%  { transform: translateX(25%) translateY(12%) rotate(-5deg) scale(1);    opacity: 0.7; }
        100% { transform: translateX(10%) translateY(-10%) rotate(6deg) scale(1.2);  opacity: 0.4; }
    }
    @keyframes aurora-sweep-4 {
        0%   { transform: translateX(30%) translateY(5%) rotate(-4deg) scale(0.9);   opacity: 0.35; }
        50%  { transform: translateX(-10%) translateY(-3%) rotate(7deg) scale(1.1);  opacity: 0.6; }
        100% { transform: translateX(-30%) translateY(8%) rotate(-2deg) scale(1);    opacity: 0.3; }
    }
    @keyframes aurora-sweep-5 {
        0%   { transform: translateX(5%) translateY(-8%) rotate(2deg) scale(1);      opacity: 0.3; }
        50%  { transform: translateX(-20%) translateY(5%) rotate(-4deg) scale(1.15); opacity: 0.55; }
        100% { transform: translateX(15%) translateY(10%) rotate(5deg) scale(0.95);  opacity: 0.25; }
    }

    /* Subtle noise grain overlay for realism */
    .aurora-grain {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        pointer-events: none;
        opacity: 0.03;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        background-repeat: repeat;
    }

    /* ============================================
       GLASSMORPHISM UI COMPONENTS
       ============================================ */

    /* Main content container — ensure above aurora */
    .main .block-container {
        position: relative;
        z-index: 1;
    }

    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818CF8 0%, #A78BFA 25%, #C084FC 50%, #F472B6 75%, #34D399 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 6s ease infinite;
        margin-bottom: 0.2rem;
        text-shadow: 0 0 40px rgba(139, 92, 246, 0.3);
    }

    @keyframes gradient-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .sub-header {
        color: rgba(156, 163, 175, 0.9);
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.02em;
    }

    /* Glassmorphism card */
    .card {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(75, 85, 99, 0.4);
        margin-bottom: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .metric-box {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(75, 85, 99, 0.3);
        text-align: center;
    }

    /* Status badges with glow effects */
    .badge-normal {
        background: rgba(6, 95, 70, 0.6);
        color: #34D399;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.25);
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-anomaly {
        background: rgba(153, 27, 27, 0.6);
        color: #F87171;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        box-shadow: 0 0 12px rgba(248, 113, 113, 0.25);
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    .badge-healed {
        background: rgba(30, 64, 175, 0.6);
        color: #60A5FA;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        box-shadow: 0 0 12px rgba(96, 165, 250, 0.25);
        border: 1px solid rgba(96, 165, 250, 0.3);
    }

    /* Sidebar glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.75) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(75, 85, 99, 0.3) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(17, 24, 39, 0.4);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        backdrop-filter: blur(8px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: rgba(156, 163, 175, 0.8);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #A5B4FC !important;
        border-bottom: 2px solid #818CF8 !important;
    }

    /* Metric widgets */
    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(75, 85, 99, 0.3);
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(75, 85, 99, 0.3);
    }

    /* Code blocks */
    .stCodeBlock {
        border-radius: 12px !important;
        border: 1px solid rgba(75, 85, 99, 0.3) !important;
    }

    /* JSON viewer */
    [data-testid="stJson"] {
        background: rgba(17, 24, 39, 0.5) !important;
        border-radius: 12px;
        border: 1px solid rgba(75, 85, 99, 0.3);
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6, #A855F7) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 28px rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-1px) !important;
    }
</style>

<!-- Aurora Beam Background Layers -->
<div class="aurora-beam-bg">
    <div class="aurora-sheet aurora-sheet-1"></div>
    <div class="aurora-sheet aurora-sheet-2"></div>
    <div class="aurora-sheet aurora-sheet-3"></div>
    <div class="aurora-sheet aurora-sheet-4"></div>
    <div class="aurora-sheet aurora-sheet-5"></div>
</div>
<div class="aurora-grain"></div>
""", unsafe_allow_html=True)

# Initialize Session State
if "api_client" not in st.session_state:
    st.session_state.api_client = WeatherAPIClient()
if "injector" not in st.session_state:
    st.session_state.injector = SyntheticAnomalyInjector()
if "detector" not in st.session_state:
    st.session_state.detector = MLAnomalyDetector()
if "diagnostician" not in st.session_state:
    st.session_state.diagnostician = LLMDiagnosticAgent()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = SelfHealingOrchestrator()

# Header
st.markdown('<div class="main-header">⚡ Self-Correcting Data Pipeline Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous MLOps Agent: ML Anomaly Detection (Layer 1) ➔ LLM Diagnosis (Layer 2) ➔ Sandboxed Patching & Auto-Healing (Layer 3)</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Controls")
api_mode = st.sidebar.selectbox("Data Source", ["Open-Meteo Live API", "Cached Weather Baseline"])
selected_anomaly = st.sidebar.selectbox("Inject Synthetic Anomaly", ANOMALY_CLASSES)
st.sidebar.markdown("---")
st.sidebar.info("💡 **How it works**: Trigger an anomaly, watch Layer 1 detect it, Layer 2 diagnose the root cause, and Layer 3 generate + sandbox-verify a code patch to heal the pipeline!")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛰️ Live Pipeline & Injector",
    "🧠 Layer 1: ML Detector",
    "🔍 Layer 2: LLM Diagnosis",
    "🛠️ Layer 3: Patch & Sandbox",
    "📊 Evaluation & Audit Trail"
])

# ----------------------------------------------------
# TAB 1: LIVE PIPELINE & INJECTOR
# ----------------------------------------------------
with tab1:
    st.subheader("1. Ingestion Stream & Synthetic Anomaly Injector")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🚀 Fetch Live Sample & Process Pipeline", type="primary", use_container_width=True):
            clean_payload = st.session_state.api_client.fetch_live_sample()
            corrupted, anomaly_label, meta = st.session_state.injector.inject_anomaly(
                clean_payload, target_anomaly=selected_anomaly
            )
            st.session_state.current_clean = clean_payload
            st.session_state.current_corrupted = corrupted
            st.session_state.current_label = anomaly_label
            st.session_state.current_meta = meta

            # Step 1: Layer 1 ML Detection
            detection_res = st.session_state.detector.detect(corrupted)
            st.session_state.last_detection = detection_res

            # Step 2 & 3 if Anomaly
            if detection_res["is_anomaly"]:
                diag_res = st.session_state.diagnostician.diagnose(
                    failing_payload=corrupted,
                    expected_schema=st.session_state.api_client.get_expected_schema_spec(),
                    layer1_result=detection_res
                )
                st.session_state.last_diagnosis = diag_res

                success, patch_code, verif_res = st.session_state.orchestrator.heal_pipeline(
                    diagnostic=diag_res,
                    failing_payload=corrupted
                )
                st.session_state.last_patch = patch_code
                st.session_state.last_verif = verif_res
                st.session_state.last_heal_success = success
            else:
                st.session_state.last_diagnosis = None
                st.session_state.last_patch = None
                st.session_state.last_verif = None
                st.session_state.last_heal_success = None

    if "current_corrupted" in st.session_state:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Clean API Baseline Payload")
            st.json(st.session_state.current_clean)
        with c2:
            st.markdown(f"### Injected Payload (`{st.session_state.current_label}`)")
            st.json(st.session_state.current_corrupted)
        with c3:
            st.markdown("### Pipeline Output (Post-Heal)")
            if st.session_state.get("last_heal_success"):
                parsed = st.session_state.orchestrator.parse(st.session_state.current_corrupted)
                st.success("✅ Cleanly Parsed & Healed!")
                st.json(parsed)
            elif not st.session_state.get("last_detection", {}).get("is_anomaly"):
                st.info("ℹ️ Payload Normal - Clean Ingestion")
                st.json(st.session_state.current_corrupted)
            else:
                st.error("❌ Pipeline Break Unresolved")

# ----------------------------------------------------
# TAB 2: LAYER 1 ML DETECTOR
# ----------------------------------------------------
with tab2:
    st.subheader("Layer 1: Supervised ML Anomaly Detector")
    if "last_detection" in st.session_state:
        det = st.session_state.last_detection
        m1, m2, m3 = st.columns(3)
        with m1:
            status_html = '<span class="badge-anomaly">ANOMALY DETECTED</span>' if det["is_anomaly"] else '<span class="badge-normal">NORMAL PAYLOAD</span>'
            st.markdown(f"**Detection Status**: {status_html}", unsafe_allow_html=True)
        with m2:
            st.metric("Predicted Anomaly Category", det["anomaly_type"])
        with m3:
            st.metric("Model Confidence Score", f"{det['confidence'] * 100:.1f}%")

        st.markdown("#### Extracted Payload Schema Feature Vector")
        feats_df = pd.DataFrame([det["features"]])
        st.dataframe(feats_df, use_container_width=True)

    else:
        st.info("Fetch a sample from Tab 1 to run Layer 1 ML Detector.")

# ----------------------------------------------------
# TAB 3: LAYER 2 LLM DIAGNOSIS
# ----------------------------------------------------
with tab3:
    st.subheader("Layer 2: Diagnostic Agent (LLM Reasoning)")
    if st.session_state.get("last_diagnosis"):
        diag = st.session_state.last_diagnosis
        st.markdown(f"""
        <div class="card">
            <h4>🔍 Root Cause Diagnosis Report</h4>
            <p><b>Cause:</b> {diag.cause}</p>
            <p><b>Affected Field:</b> <code>{diag.affected_field}</code></p>
            <p><b>Category:</b> {diag.anomaly_category}</p>
            <p><b>Expected Type:</b> {diag.expected_type} | <b>Observed:</b> {diag.actual_type}</p>
            <p><b>Found Replacement Key:</b> <code>{diag.found_replacement_key}</code></p>
            <p><b>Suggested Fix Strategy:</b> {diag.suggested_fix}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Structured JSON Diagnostic Output")
        st.json(diag.model_dump())
    else:
        st.info("No active pipeline diagnosis. (Either pipeline is normal or fetch sample from Tab 1).")

# ----------------------------------------------------
# TAB 4: LAYER 3 PATCH & SANDBOX
# ----------------------------------------------------
with tab4:
    st.subheader("Layer 3: Code Patch Generation & Sandboxed Self-Verification")
    if st.session_state.get("last_patch"):
        col1, col2 = st.columns([1.2, 0.8])
        with col1:
            st.markdown("#### Auto-Generated Python Code Patch")
            st.code(st.session_state.last_patch, language="python")
        with col2:
            st.markdown("#### Sandbox Verification Execution")
            verif = st.session_state.last_verif
            if verif.get("passed"):
                st.success(f"✅ Sandbox Verification PASSED! ({verif['passed_count']}/{verif['total_count']} records)")
            else:
                st.error(f"❌ Sandbox Verification FAILED")

            st.metric("Held-Out Verification Success Rate", f"{verif.get('success_rate', 0.0) * 100:.1f}%")
            st.markdown("#### Active Parser Status")
            st.markdown('<span class="badge-healed">ACTIVE HOT-SWAPPED PARSER IN PRODUCTION</span>', unsafe_allow_html=True)
    else:
        st.info("No patch generated yet.")

# ----------------------------------------------------
# TAB 5: EVALUATION & AUDIT TRAIL
# ----------------------------------------------------
with tab5:
    st.subheader("System Evaluation Metrics & Audit Trail")
    if st.button("📊 Run Layer 1 ML Model Benchmark Evaluation"):
        with st.spinner("Evaluating model on synthetic dataset..."):
            metrics = evaluate_layer1()
            if metrics:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("ML Model Precision", f"{metrics['ml_model']['precision'] * 100:.2f}%")
                with c2:
                    st.metric("ML Model Recall", f"{metrics['ml_model']['recall'] * 100:.2f}%")
                with c3:
                    st.metric("ML Model F1-Score", f"{metrics['ml_model']['f1_score'] * 100:.2f}%")

                st.markdown("### Confusion Matrix (ML Anomaly Model)")
                cm_df = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["classes"],
                    columns=metrics["classes"]
                )
                st.dataframe(cm_df, use_container_width=True)

    st.markdown("### 📜 Pipeline Audit Logs")
    logs = get_recent_audit_logs(limit=25)
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No audit log events recorded yet.")
