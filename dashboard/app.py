"""Streamlit dashboard for the self-healing pipeline agent.

Rewritten against the pipeline registry. Previously this UI predated the
multi-pipeline work and was weather-only, so it could not exercise three of
the four domains the backend supports.

Other fixes:
  * the "Data Source" selector now does something -- it was read into a
    variable and never used, a control that appeared functional but was not;
  * the "ACTIVE HOT-SWAPPED PARSER IN PRODUCTION" badge no longer renders
    when verification has FAILED (it sat outside the pass/fail branch);
  * 284 lines of inline CSS moved to dashboard/styles.css;
  * the orchestrator is no longer handed session state directly, which let a
    generated parser mutate the payload being displayed for comparison.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.dedup import DuplicateTracker
from src.ingestion.injector import ANOMALY_CLASSES, SyntheticAnomalyInjector
from src.ingestion.pipeline_registry import registry
from src.layer1_detector.evaluator import evaluate_layer1
from src.layer1_detector.model import MLAnomalyDetector
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator
from src.utils.logger import get_recent_audit_logs

ASSETS = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Self-Correcting Data Pipeline Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    """Load the stylesheet and animated background from disk."""
    try:
        css = (ASSETS / "styles.css").read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except OSError:
        pass  # Styling is cosmetic; never block the dashboard on it.
    try:
        st.markdown((ASSETS / "background.html").read_text(encoding="utf-8"),
                    unsafe_allow_html=True)
    except OSError:
        pass


@st.cache_resource
def get_components():
    """Build the pipeline components once per server process."""
    return {
        "injector": SyntheticAnomalyInjector(),
        "detector": MLAnomalyDetector(),
        "diagnostician": LLMDiagnosticAgent(),
        "orchestrator": SelfHealingOrchestrator(),
    }


@st.cache_data(show_spinner=False)
def cached_evaluation():
    """Layer 1 metrics, computed once rather than on every rerun."""
    return evaluate_layer1()


def run_pipeline(pipeline_id: str, anomaly: str, use_live_data: bool) -> None:
    """Execute one full detect -> diagnose -> heal cycle into session state."""
    components = get_components()
    pipeline = registry.require_pipeline(pipeline_id)
    known_aliases = {
        alias: canonical for canonical, alias in pipeline.field_renames.items()
    }

    if use_live_data or not hasattr(pipeline, "cached_baseline"):
        clean = pipeline.fetch_clean_sample()
    else:
        clean = pipeline.cached_baseline()

    corrupted, label, meta = components["injector"].inject_anomaly(
        clean,
        target_anomaly=anomaly,
        field_renames=pipeline.field_renames,
        valid_ranges=pipeline.valid_ranges,
    )

    # Each run simulates one isolated stream event, so it gets its own
    # duplicate tracker; sharing a long-lived one would report a repeated
    # `normal` weather run as a duplicate, since that upstream returns the
    # same observation for an hour.
    request_tracker = DuplicateTracker()
    if label == "duplicate_record":
        request_tracker.observe(corrupted)

    detection = components["detector"].detect(
        corrupted,
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
        duplicate_tracker=request_tracker,
    )
    is_duplicate = bool(detection.get("features", {}).get("duplicate_flag", 0.0))

    state = st.session_state
    state.pipeline_id = pipeline_id
    state.pipeline_name = pipeline.name
    state.data_source = pipeline.last_fetch_source
    state.current_clean = clean
    state.current_corrupted = corrupted
    state.current_label = label
    state.current_meta = meta
    state.last_detection = detection
    state.last_diagnosis = None
    state.last_patch = None
    state.last_verif = None
    state.last_heal_success = None
    state.healed_payload = None
    state.is_duplicate = is_duplicate

    if not detection["is_anomaly"]:
        return

    diagnosis = components["diagnostician"].diagnose(
        failing_payload=corrupted,
        expected_schema=pipeline.get_schema_spec(),
        layer1_result=detection,
        valid_ranges=pipeline.valid_ranges,
        known_aliases=known_aliases,
    )
    state.last_diagnosis = diagnosis

    success, patch_code, verification = components["orchestrator"].heal_pipeline(
        diagnostic=diagnosis,
        failing_payload=corrupted,
        pipeline_id=pipeline_id,
        expected_schema=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
        known_aliases=known_aliases,
    )
    state.last_patch = patch_code
    state.last_verif = verification
    state.last_heal_success = success

    if success:
        # Pass a copy: the parser must not touch the record on display.
        state.healed_payload = components["orchestrator"].parse(
            dict(corrupted), pipeline_id=pipeline_id, is_duplicate=is_duplicate
        )


def render_sidebar():
    """Simulation controls. Returns (pipeline_id, anomaly, use_live_data)."""
    st.sidebar.header("⚙️ Simulation Controls")

    options = registry.list_pipelines()
    labels = {item["id"]: f"{item['icon']} {item['name']}" for item in options}
    pipeline_id = st.sidebar.selectbox(
        "Target Stream",
        [item["id"] for item in options],
        format_func=lambda pid: labels[pid],
    )

    pipeline = registry.require_pipeline(pipeline_id)
    st.sidebar.caption(pipeline.description)

    anomaly = st.sidebar.selectbox("Inject Synthetic Anomaly", list(ANOMALY_CLASSES))

    # Only the weather pipeline has a live upstream; the others are simulated,
    # so the control is disabled rather than silently doing nothing.
    is_live_capable = pipeline_id == "weather"
    use_live_data = st.sidebar.radio(
        "Data Source",
        ["Live upstream API", "Cached baseline"],
        index=0 if is_live_capable else 1,
        disabled=not is_live_capable,
        help="Only the weather pipeline fetches from a live API; the other "
             "streams are locally simulated.",
    ) == "Live upstream API"

    st.sidebar.markdown("---")
    diagnostician = get_components()["diagnostician"]
    detector = get_components()["detector"]
    st.sidebar.markdown("**System status**")
    st.sidebar.markdown(
        f"- Layer 1 engine: `{'trained model' if detector.is_trained else 'rule-based'}`\n"
        f"- Layer 2 engine: `{'Gemini LLM' if diagnostician.llm_enabled else 'deterministic'}`"
    )
    if not detector.is_trained:
        st.sidebar.warning(
            "No trained model found. Run `python run_data_and_train.py` for ML detection."
        )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 Trigger an anomaly, watch Layer 1 detect it, Layer 2 diagnose the "
        "root cause, and Layer 3 generate and sandbox-verify a code patch."
    )
    return pipeline_id, anomaly, use_live_data


def render_live_pipeline(pipeline_id: str, anomaly: str, use_live_data: bool) -> None:
    st.subheader("1. Ingestion Stream & Synthetic Anomaly Injector")

    pipeline = registry.require_pipeline(pipeline_id)
    if st.button(
        f"🚀 Run {pipeline.icon} {pipeline.name} Pipeline",
        type="primary",
        width="stretch",
    ):
        with st.spinner("Fetching → injecting → detecting → diagnosing → patching..."):
            try:
                run_pipeline(pipeline_id, anomaly, use_live_data)
            except Exception as exc:
                st.error(f"Pipeline run failed: {type(exc).__name__}: {exc}")
                return

    state = st.session_state
    if "current_corrupted" not in state:
        st.info("Select a stream and an anomaly, then run the pipeline to begin.")
        return

    st.markdown("---")
    st.caption(
        f"Stream: **{state.pipeline_name}** · data source: `{state.data_source}`"
    )

    left, middle, right = st.columns(3)
    with left:
        st.markdown("### Clean Baseline Payload")
        st.json(state.current_clean)
    with middle:
        st.markdown(f"### Injected Payload (`{state.current_label}`)")
        st.json(state.current_corrupted)
    with right:
        st.markdown("### Pipeline Output (Post-Heal)")
        if state.get("last_heal_success"):
            st.success("✅ Cleanly parsed and healed")
            st.json(state.healed_payload)
        elif not state.get("last_detection", {}).get("is_anomaly"):
            st.info("ℹ️ Payload normal — clean ingestion")
            st.json(state.current_corrupted)
        else:
            st.error("❌ Pipeline break unresolved")
            trace = (state.get("last_verif") or {}).get("error_trace")
            if trace:
                st.caption(trace[:600])


def render_detector() -> None:
    st.subheader("Layer 1: Supervised ML Anomaly Detector")
    detection = st.session_state.get("last_detection")
    if not detection:
        st.info("Run the pipeline to populate Layer 1 results.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Detection Status", "⚠ ANOMALY" if detection["is_anomaly"] else "✓ NORMAL")
    col2.metric("Predicted Category", detection["anomaly_type"])
    col3.metric("Model Confidence", f"{detection['confidence'] * 100:.1f}%")
    st.caption(f"Inference engine: `{detection.get('engine', 'unknown')}`")

    st.markdown("#### Extracted Feature Vector")
    st.dataframe(
        pd.DataFrame([detection["features"]]).T.rename(columns={0: "value"}),
        width="stretch",
    )

    if detection.get("class_probabilities"):
        st.markdown("#### Class Probabilities")
        st.bar_chart(pd.Series(detection["class_probabilities"], name="probability"))


def render_diagnosis() -> None:
    st.subheader("Layer 2: Diagnostic Agent")
    diagnosis = st.session_state.get("last_diagnosis")
    if not diagnosis:
        st.info("No active diagnosis — the payload is normal or the pipeline has not run.")
        return

    report = diagnosis.model_dump()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Root cause**  \n{report['cause']}")
        st.markdown(f"**Affected field**  \n`{report['affected_field']}`")
        st.markdown(f"**Category**  \n`{report['anomaly_category']}`")
    with col2:
        st.markdown(f"**Expected type**  \n`{report['expected_type']}`")
        st.markdown(f"**Observed type**  \n`{report['actual_type']}`")
        st.markdown(
            f"**Replacement key**  \n`{report['found_replacement_key'] or '—'}`"
        )
    st.markdown(f"**Suggested fix**  \n{report['suggested_fix']}")

    st.markdown("#### Structured JSON Output")
    st.json(report)


def render_patch() -> None:
    st.subheader("Layer 3: Patch Generation & Sandboxed Verification")
    patch = st.session_state.get("last_patch")
    if not patch:
        st.info("No patch generated yet.")
        return

    verification = st.session_state.get("last_verif") or {}
    succeeded = bool(st.session_state.get("last_heal_success"))

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown("#### Auto-Generated Python Patch")
        st.code(patch, language="python")
    with right:
        st.markdown("#### Sandbox Verification")
        if succeeded:
            st.success(
                f"✅ Verification PASSED "
                f"({verification.get('passed_count')}/{verification.get('total_count')} records)"
            )
        else:
            st.error("❌ Verification FAILED")
            if verification.get("error_trace"):
                st.code(verification["error_trace"][:1000])

        st.metric(
            "Held-Out Pass Rate",
            f"{verification.get('success_rate', 0.0) * 100:.1f}%",
        )

        st.markdown("#### Active Parser Status")
        # Only claim an active hot-swapped parser when one actually activated.
        if succeeded:
            st.markdown(
                '<span class="badge-healed">ACTIVE HOT-SWAPPED PARSER</span>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No new parser activated — the previous parser remains in effect.")


def render_evaluation() -> None:
    st.subheader("System Evaluation Metrics & Audit Trail")

    if st.button("📊 Run Layer 1 Benchmark Evaluation"):
        with st.spinner("Evaluating on the held-out split..."):
            metrics = cached_evaluation()

        if not metrics:
            st.warning(
                "No evaluation data. Run `python run_data_and_train.py` to generate "
                "the dataset and train the model."
            )
        else:
            model, baseline = metrics["ml_model"], metrics["rule_baseline"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{model['precision'] * 100:.2f}%")
            col2.metric("Recall", f"{model['recall'] * 100:.2f}%")
            col3.metric("F1-Score", f"{model['f1_score'] * 100:.2f}%")

            st.markdown("#### Model vs Rule Baseline")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"engine": "ML model", **model},
                        {"engine": "Rule baseline", **baseline},
                    ]
                ).set_index("engine"),
                width="stretch",
            )
            st.info(metrics["model_vs_baseline"]["note"])

            st.markdown("#### Confusion Matrix")
            st.dataframe(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["classes"],
                    columns=metrics["classes"],
                ),
                width="stretch",
            )
            st.caption(metrics["split_note"])
            st.caption(f"⚠️ {metrics['caveat']}")

    st.markdown("### 📜 Pipeline Audit Trail")
    logs = get_recent_audit_logs(limit=25)
    if logs:
        st.dataframe(pd.DataFrame(logs), width="stretch")
    else:
        st.info("No audit events recorded yet.")


def main() -> None:
    inject_theme()
    st.markdown(
        '<div class="main-header">⚡ Self-Correcting Data Pipeline Agent</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Autonomous MLOps agent: ML anomaly detection '
        "(Layer 1) ➔ root-cause diagnosis (Layer 2) ➔ sandboxed patching and "
        'auto-healing (Layer 3)</div>',
        unsafe_allow_html=True,
    )

    pipeline_id, anomaly, use_live_data = render_sidebar()

    tabs = st.tabs([
        "🛰️ Live Pipeline & Injector",
        "🧠 Layer 1: ML Detector",
        "🔍 Layer 2: Diagnosis",
        "🛠️ Layer 3: Patch & Sandbox",
        "📊 Evaluation & Audit",
    ])

    with tabs[0]:
        render_live_pipeline(pipeline_id, anomaly, use_live_data)
    with tabs[1]:
        render_detector()
    with tabs[2]:
        render_diagnosis()
    with tabs[3]:
        render_patch()
    with tabs[4]:
        render_evaluation()


main()
