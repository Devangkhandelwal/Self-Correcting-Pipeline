# Autonomous Self-Healing Data Pipeline Agent 🤖⚡

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19.x-61dafb.svg?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.x-646cff.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-104%20passing-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **self-healing data ingestion agent**. When an upstream provider silently
changes its API, the pipeline detects the anomaly with a trained classifier,
diagnoses the root cause, synthesises a replacement parser, verifies that
parser in an isolated interpreter against held-out records, and hot-swaps it
into the running pipeline — without a restart and without a human.

---

## 📑 Table of Contents

- [The Problem](#-the-problem-silent-data-pipeline-outages)
- [Architecture](#-architecture)
- [Layer Breakdown](#-layer-breakdown)
- [Multi-Domain Pipelines](#-multi-domain-pipelines)
- [Anomaly Taxonomy](#-anomaly-taxonomy)
- [Dashboards](#-dashboards)
- [REST API](#-rest-api)
- [Measured Performance](#-measured-performance)
- [Measurement Caveats](#-measurement-caveats-read-before-quoting-the-numbers)
- [Security Model](#-security-model)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Running the Tests](#-running-the-tests)
- [CLI Demo](#-cli-demo)
- [Known Limitations](#-known-limitations)
- [License](#-contributing--license)

---

## 🚨 The Problem: Silent Data Pipeline Outages

Ingestion pipelines pulling from third-party APIs break from **silent upstream
changes**:

1. **Schema drift** — a vendor renames a field (`precipitation` → `precip`).
2. **Type mismatches** — numbers start arriving as strings (`"22.5"`) or booleans.
3. **Nulls and omissions** — required attributes go missing on sensor dropout.
4. **Outliers** — hardware glitches inject values far outside physical range.
5. **Duplicates** — retry storms replay identical events.

Traditional pipelines either crash outright or, worse, quietly poison the
downstream data lake. Either way a human spends hours on triage.

---

## 🏗️ Architecture

```
                    ┌────────────────────────────────────┐
                    │   Ingestion APIs / Simulated Feeds │
                    │  Weather │ Finance │ Retail │ IoT  │
                    └─────────────────┬──────────────────┘
                                      ▼
                    ┌────────────────────────────────────┐
                    │     Synthetic Anomaly Injector     │
                    └─────────────────┬──────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 · ML Anomaly Detector                                                │
│  10-feature payload vectoriser → RandomForest → class + confidence           │
│  Falls back to a rule baseline when no model is trained                       │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      ▼ anomaly flagged
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2 · Root-Cause Diagnosis                                               │
│  Gemini 2.5 Flash with Pydantic-validated output, or a deterministic engine   │
│  Range-driven outlier detection + declared-alias and fuzzy rename inference   │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      ▼ structured DiagnosticReport
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3 · Patch Synthesis & Verification                                      │
│  Schema-driven codegen → AST screening → isolated subprocess under timeout    │
│  Scored against held-out records; 3 escalating retry strategies               │
│  Verified patch is hot-swapped into the live pipeline                         │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      ▼
                    ┌────────────────────────────────────┐
                    │  Normalised output + audit trail   │
                    └────────────────────────────────────┘
```

Every layer is driven by the **selected pipeline's own schema**, supplied by
`src/ingestion/pipeline_registry.py`. Nothing downstream hardcodes a domain.

---

## 🔬 Layer Breakdown

### Layer 1 · ML Anomaly Detector (`src/layer1_detector/`)

Ten structural features per payload, computed against the pipeline's declared
schema and ranges:

| # | Feature | Meaning |
|---|---------|---------|
| 1 | `total_keys` | number of keys present |
| 2 | `missing_expected_count` | declared fields absent |
| 3 | `unexpected_extra_count` | undeclared keys present |
| 4 | `type_mismatch_count` | values whose type contradicts the schema |
| 5 | `null_val_count` | null values |
| 6 | `max_key_similarity` | best string similarity between a missing and an extra key (rename signal) |
| 7 | `outlier_count` | numeric values outside their declared range |
| 8 | `duplicate_flag` | duplicate marker present (**see caveats**) |
| 9 | `string_type_count` | string-valued fields |
| 10 | `bool_type_count` | boolean-valued fields |

A `RandomForestClassifier` maps that vector to one of six classes. When no
trained model is present the detector falls back to an explicit rule baseline,
and `detect()` reports which engine produced the verdict.

### Layer 2 · Root-Cause Diagnosis (`src/layer2_diagnostician/`)

Returns a Pydantic `DiagnosticReport`: cause, affected field, category,
expected vs observed type, replacement key, and a suggested fix.

Rename recovery uses three sources in descending order of trust:

1. **the pipeline's declared rename table** — authoritative configuration, and
   the only thing that catches a *semantic* rename. `reorder_level` →
   `min_stock` scores **0.09** on string similarity and is invisible to
   lexical matching;
2. **the diagnosis** for this specific failure;
3. **lexical fuzzy matching**, for renames nobody anticipated.

Gemini drives this layer when `GEMINI_API_KEY` is set. LLM output is
**validated against the schema before use** — a hallucinated field name is
rejected and the deterministic engine takes over, because Layer 3 generates
code from these values.

### Layer 3 · Patch Synthesis & Verification (`src/layer3_patcher/`)

Generates a table-driven `parse_payload(payload: dict) -> dict` that applies
alias recovery, type coercion, schema-safe defaults, range clamping, duplicate
normalisation, and pass-through of unrecognised fields.

Retries **escalate** rather than repeat:

| Attempt | Strategy | Behaviour |
|---------|----------|-----------|
| 1 | `strict` | declared aliases + the diagnosed rename |
| 2 | `fuzzy_alias` | adds lexical matching for every still-missing field |
| 3 | `defensive` | adds salvage parsing of dirty numeric strings; drops unrecognised keys |

Verification runs in a **separate isolated interpreter** under a wall-clock
timeout, scoring the patch on the failing record plus held-out batches for that
pipeline. A patch is activated only at a 100% pass rate.

---

## 🌐 Multi-Domain Pipelines

Defined in `src/ingestion/pipeline_registry.py`:

| Domain | ID | Source | Monitored fields |
|---|---|---|---|
| **Weather Forecast** | `weather` | Live Open-Meteo REST API | `latitude`, `longitude`, `temperature_2m`, `relative_humidity_2m`, `precipitation`, `surface_pressure`, `wind_speed_10m`, `time` |
| **Financial Transactions** | `finance` | Simulated payment/fraud stream | `transaction_id`, `user_id`, `amount`, `currency`, `merchant_id`, `risk_score`, `timestamp` |
| **E-Commerce Inventory** | `ecommerce` | Simulated warehouse stream | `item_id`, `sku`, `stock_quantity`, `unit_price`, `category`, `reorder_level`, `timestamp` |
| **Industrial IoT Telemetry** | `iot` | Simulated turbine sensors | `device_id`, `vibration_rms`, `temperature_c`, `pressure_psi`, `motor_rpm`, `battery_pct`, `timestamp` |

Only `weather` performs live HTTP. The other three are locally simulated; each
pipeline reports its provenance via `last_fetch_source` (`live`,
`cached_baseline` or `simulated`), surfaced in both dashboards. A network
failure falls back to a cached baseline and **says so** rather than passing
fabricated data off as live.

---

## ⚠️ Anomaly Taxonomy

Injected by `src/ingestion/injector.py` (accepts a seeded RNG for reproducibility):

| Class | Simulates |
|---|---|
| `normal` | clean control record |
| `schema_drift` | provider renamed a field (`amount` → `tx_amount`) |
| `type_mismatch` | number serialised as a string or boolean |
| `null_field` | key set to null, or omitted entirely |
| `outlier` | sensor spike outside the declared range |
| `duplicate_record` | replayed event from a retry storm |

---

## 🖥️ Dashboards

### React Agent Console (`dashboard-ui/`)

React 19 + Vite 8 with **hand-written CSS** (no CSS framework). Features a
WebGL shader background, a canvas particle headline, pipeline/anomaly
selectors, and five panels: payload comparison, Layer 1 verdict and feature
vector, Layer 2 report, generated patch with verification result, and the
benchmark plus audit trail.

Every panel handles its own loading, empty and error states; `src/api.js` is
the only place that calls `fetch`.

### Streamlit Dashboard (`dashboard/app.py`)

Operational view over the same registry: all four pipelines, Layer 1 class
probabilities, model-vs-baseline comparison, confusion matrix and audit trail.
Styling lives in `dashboard/styles.css`.

---

## ⚡ REST API

Served by `api/server.py`. Built via a `create_app()` factory so tests drive it
without binding a port.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | readiness; reports `model_loaded`, `llm_enabled`, active parsers |
| `GET /api/pipelines` | all registered pipelines with schema and ranges |
| `GET /api/anomaly-classes` | supported anomaly classes |
| `GET /api/schema?pipeline_id=weather` | one pipeline's schema and ranges |
| `POST /api/run-pipeline` | full detect → diagnose → heal cycle |
| `GET /api/evaluate` | Layer 1 benchmark (cached; `?refresh=1` to recompute) |
| `GET /api/audit-logs?limit=30` | recent audit events (limit clamped) |

`POST /api/run-pipeline` body:

```json
{ "pipeline_id": "weather", "anomaly_type": "schema_drift" }
```

Both fields are validated against the registry and the anomaly taxonomy;
anything unrecognised returns **400** with the list of valid values rather
than silently defaulting.

---

## 📊 Measured Performance

Measured on this repository, 12 runs across all four pipelines, deterministic
Layer 2, no live HTTP. Reproduce with `python run_data_and_train.py` and
`pytest tests/`.

### Latency

| Stage | Median | p95 |
|---|---|---|
| Layer 1 detect | 28.7 ms | 29.6 ms |
| Layer 2 diagnose (deterministic) | 0.9 ms | 1.0 ms |
| Layer 3 generate + verify + activate | 136.9 ms | 147.4 ms |
| Healed parse, per record | 0.02 ms | 0.03 ms |
| **Total, detect → healed** | **167.3 ms** | **178.0 ms** |

Layer 3 dominates because verification spawns an isolated interpreter. That is
a deliberate trade: roughly 130 ms per heal buys process isolation for
executing generated code. A Gemini-backed Layer 2 adds network latency.

### Detection quality

240-row held-out split, seed 42 — rows the model never saw:

| Engine | Precision | Recall | F1 |
|---|---|---|---|
| RandomForest | 100% | 100% | 100% |
| Rule baseline | 100% | 100% | 100% |

**The trained model shows zero measurable lift over the handwritten rule
baseline**: the two agree on 100% of held-out rows for an F1 lift of
`+0.0000`. Read the caveats below before quoting any of this.

### Healing coverage

All 24 combinations of 4 pipelines × 6 anomaly classes heal successfully, each
verified against 101–121 records. Covered by `tests/test_multidomain_healing.py`.

---

## 🔍 Measurement Caveats (read before quoting the numbers)

Being blunt, because the headline figures look better than they are:

1. **The benchmark is close to separable by construction.** Feature
   `duplicate_flag` is set by the *injector's own marker keys*
   (`_is_duplicate_flag`, `duplicate_uuid`) — for that class the feature
   effectively *is* the label. Every other feature is computed against the same
   schema the injector perturbs.
2. **The RandomForest earns nothing here.** It agrees with the rule baseline on
   every held-out row. The engineered features already encode the anomaly
   definitions the rules test directly, so on this benchmark the ML layer is
   decoration. It is retained because it generalises to feature *combinations*
   the rules do not enumerate — but that benefit is unmeasured.
3. **100% is a property of the synthetic data, not evidence of real-world
   accuracy.** There is no real-world labelled evaluation set.
4. **Declared-alias recovery is configuration, not inference.** When the
   injector renames via the registry's table and the healer consults that same
   table, the loop is partly circular. The genuinely autonomous part is the
   fuzzy fallback for *undeclared* renames.
5. **Duplicate detection is marker-based, not content-based.** It does not hash
   payloads or track recently-seen records. See
   [Known Limitations](#-known-limitations).

`GET /api/evaluate` returns these caveats inline, and both dashboards display
them next to the metrics.

---

## 🔒 Security Model

Layer 3 generates and executes Python, so this is load-bearing.

**Verification** — candidate patches are screened by
`sandbox.validate_patch_source` before anything runs. Rejected outright:
`import` / `from`, `eval`, `exec`, `compile`, `open`, `__import__`, `globals`,
`locals`, `getattr`/`setattr`, any dunder attribute access (blocking
`{}.__class__.__bases__`-style escapes), lambdas, classes, `with`, `async`,
generators, and sources without a top-level `parse_payload`. Screened code then
runs in a **separate interpreter** launched with `-I` (isolated: no
`PYTHONPATH`, no user site-packages, no inherited env influence) under a
wall-clock timeout. A hostile or merely non-terminating patch cannot corrupt or
hang the API process.

**Activation** necessarily executes the patch in-process — that is what
hot-swapping means — so every patch is re-screened on load, including patches
read back from `data/patches/` at startup.

**The API** defaults to `127.0.0.1` with the debugger **off**. `API_DEBUG=true`
enables the Werkzeug console, which grants arbitrary code execution to anyone
who can reach the port; never combine it with `API_HOST=0.0.0.0`. CORS is
restricted to `CORS_ORIGINS` rather than every origin. Flask's development
server is not a production server — `waitress` is included for that.

**Secrets** are read from the environment only; nothing is hardcoded, and
`.env` is gitignored.

### Residual risk

The AST screen is a denylist over a restricted subset, not a capability
sandbox. It is appropriate for template-generated code and a meaningful
obstacle to LLM-generated code, but it is not a security boundary against a
determined attacker. For untrusted patch sources, run verification in a
container or VM with seccomp and no network.

---

## 📁 Project Structure

```
├── .env.example               # every configuration value, documented
├── requirements.txt           # pinned with upper bounds
├── conftest.py                # pytest path setup
├── demo_pipeline.py           # CLI demo: --pipeline, --anomaly
├── run_data_and_train.py      # generate -> train -> evaluate
│
├── api/
│   └── server.py              # Flask REST API (create_app factory)
│
├── dashboard/
│   ├── app.py                 # Streamlit dashboard
│   ├── styles.css             # extracted stylesheet
│   └── background.html        # aurora background markup
│
├── dashboard-ui/              # React 19 + Vite 8 console
│   └── src/
│       ├── api.js             # single fetch wrapper, raises ApiError
│       ├── App.jsx            # shell: nav, hero, controls, tabs
│       └── components/        # panels + ErrorNotice + visual components
│
├── data/                      # ALL generated; gitignored
│   ├── synthetic/dataset.csv  # labelled training data
│   ├── heldout/               # per-pipeline verification batches
│   ├── patches/               # parsers synthesised at runtime
│   ├── model.joblib           # trained Layer 1 classifier
│   └── train_test_split.json  # persisted split, for honest evaluation
│
├── logs/                      # rotating pipeline + audit logs (gitignored)
│
├── src/
│   ├── ingestion/
│   │   ├── api_client.py      # thin facade over WeatherPipeline
│   │   ├── generate_data.py   # seeded, varied, per-pipeline generation
│   │   ├── injector.py        # anomaly injection (seedable RNG)
│   │   └── pipeline_registry.py  # single source of truth for all schemas
│   ├── layer1_detector/
│   │   ├── feature_extractor.py  # 10-feature vectoriser
│   │   ├── model.py           # classifier wrapper + rule baseline
│   │   ├── train.py           # training; persists the split
│   │   └── evaluator.py       # held-out evaluation + lift reporting
│   ├── layer2_diagnostician/
│   │   └── agent.py           # Gemini + deterministic engine
│   ├── layer3_patcher/
│   │   ├── auto_healer.py     # orchestrator, retries, hot-swap
│   │   ├── generator.py       # schema-driven codegen
│   │   ├── sandbox.py         # AST screening + subprocess verification
│   │   └── _sandbox_runner.py # isolated-interpreter entry point
│   └── utils/
│       ├── config.py          # all configuration, read from env
│       └── logger.py          # rotating logs + JSONL audit trail
│
└── tests/                     # 104 tests
    ├── test_api_validation.py      # request validation, bounds, health
    ├── test_e2e_pipeline.py        # end-to-end weather cycle
    ├── test_heldout_data.py        # verification-data integrity
    ├── test_injector.py            # all anomaly classes injectable
    ├── test_layer1.py              # extractor + detector
    ├── test_multidomain_healing.py # every rename pair; every anomaly x pipeline
    ├── test_outlier_healing.py     # range-driven diagnosis + clamping
    └── test_sandbox_security.py    # malicious patches rejected; timeout enforced
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- Node.js **18+** (for the React console)

### 1. Install

```bash
git clone https://github.com/Devangkhandelwal/Self-Correcting-Pipeline.git
cd Self-Correcting-Pipeline

python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Windows CMD:         .\.venv\Scripts\activate.bat
# Linux / macOS:       source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Build the dataset and model — **required**

The dataset, model and held-out batches are build artifacts and are **not**
committed. Generate them before first run:

```bash
python run_data_and_train.py
```

Without this, Layer 1 falls back to rule-based heuristics (the API reports
`model_loaded: false` and both dashboards warn about it) and Layer 3 falls back
to a small synthetic verification set.

### 3. Run it

**Full stack** — API plus React console:

```bash
python api/server.py          # terminal 1 → http://127.0.0.1:5000

cd dashboard-ui               # terminal 2
npm install
npm run dev                   # → http://localhost:3000
```

**Streamlit dashboard:**

```bash
streamlit run dashboard/app.py   # → http://localhost:8501
```

**Production WSGI server** (Flask's dev server is not one):

```bash
waitress-serve --host 127.0.0.1 --port 5000 api.server:app
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit; every value is optional and documented
there. Real environment variables take precedence over `.env`.

```bash
cp .env.example .env
```

The settings worth knowing:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | enables Gemini for Layer 2; unset uses the deterministic engine |
| `API_HOST` / `PORT` | `127.0.0.1` / `5000` | API bind address |
| `API_DEBUG` | `false` | **Werkzeug debugger — arbitrary code execution. Keep off.** |
| `CORS_ORIGINS` | `localhost:3000` | origins permitted to call the API |
| `LOG_LEVEL` | `INFO` | `DEBUG` also logs full audit payloads |
| `RANDOM_SEED` | `42` | makes dataset generation reproducible |
| `MAX_HEAL_RETRIES` | `3` | escalating patch attempts |
| `SANDBOX_TIMEOUT_SECONDS` | `15` | wall-clock limit per verification run |
| `KEY_SIMILARITY_THRESHOLD` | `0.4` | minimum similarity to infer an undeclared rename |

---

## 🧪 Running the Tests

```bash
pytest tests/ -v
```

104 tests, roughly 35 seconds. What they cover:

```text
tests/test_api_validation.py ........                           [  8%]
tests/test_e2e_pipeline.py .                                     [  9%]
tests/test_heldout_data.py ..                                    [ 11%]
tests/test_injector.py .                                         [ 12%]
tests/test_layer1.py ..                                          [ 13%]
tests/test_multidomain_healing.py ..........................     [ 69%]
tests/test_outlier_healing.py ............                       [ 81%]
tests/test_sandbox_security.py ......................            [100%]

============================= 104 passed in ~35s ==============================
```

The suite deliberately enumerates **every declared rename pair** rather than
letting the injector pick at random. An earlier random version masked a real
defect: two semantic renames were never recovered, but the test only hit them
when the RNG happened to choose those fields.

---

## 💻 CLI Demo

```bash
python demo_pipeline.py                                  # weather, schema drift
python demo_pipeline.py --pipeline iot --anomaly outlier
python demo_pipeline.py --pipeline finance --anomaly type_mismatch
```

Prints each stage: the clean payload and its provenance, the corruption, the
Layer 1 verdict and engine, the Layer 2 report, the generated patch with its
verification score, and the healed output. Exits non-zero when healing fails,
so it works as a smoke check in CI.

---

## 🚧 Known Limitations

Honest about what this does not do:

- **Duplicate detection is marker-based.** It looks for injector marker keys
  rather than hashing payloads or tracking recently-seen records. Real
  content-based deduplication needs state across records and is not
  implemented.
- **The ML layer adds no measured value** over the rule baseline on this
  benchmark (see caveats). It is not currently justified by evidence.
- **The AST screen is a denylist, not a capability sandbox.** See
  [Security Model](#-security-model).
- **Three of four pipelines are simulated.** Only `weather` hits a real API.
- **Defaults are invented values.** A backfilled field gets a schema-safe
  default (`0.01` for finance `amount`, clamped into range). Healing keeps the
  pipeline flowing; it does not recover data that never arrived. Downstream
  consumers cannot currently distinguish a real value from a backfilled one —
  emitting per-field provenance would be the fix.
- **No authentication on the API.** It is bound to loopback by design. Exposing
  it requires adding auth.
- **No live-traffic evaluation.** All metrics come from synthetic data.

---

## 🤝 Contributing & License

Contributions, issues and feature requests are welcome.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Make your change, with tests.
4. Run `pytest tests/` and, for frontend changes, `npm run lint && npm run build`.
5. Open a pull request.

Licensed under the **MIT License** — see [LICENSE](LICENSE).
