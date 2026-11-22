# Autonomous Self-Healing Data Pipeline Agent 🤖⚡

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19.x-61dafb.svg?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.x-646cff.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-242%20passing-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
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
│  11-feature payload vectoriser → RandomForest → class + confidence           │
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

Eleven structural features per payload, computed against the pipeline's
declared schema and ranges:

| # | Feature | Meaning |
|---|---------|---------|
| 1 | `total_keys` | number of keys present |
| 2 | `missing_expected_count` | declared fields absent |
| 3 | `unexpected_extra_count` | undeclared keys present |
| 4 | `type_mismatch_count` | values whose type contradicts the schema |
| 5 | `null_val_count` | null values |
| 6 | `max_key_similarity` | best string similarity between a missing and an extra key (rename signal) |
| 7 | `outlier_count` | numeric values outside their declared range |
| 8 | `duplicate_flag` | record content has been seen before in this stream |
| 9 | `string_type_count` | string-valued fields |
| 10 | `bool_type_count` | boolean-valued fields |
| 11 | `best_alias_value_compatibility` | does an unexpected key's *value* fit a missing field's declared type and range? |

A `RandomForestClassifier` maps that vector to one of six classes. When no
trained model is present the detector falls back to an explicit rule baseline,
and `detect()` reports which engine produced the verdict.

Feature 11 exists because features 2, 3 and 6 cannot separate two different
situations. A real rename and a plain missing field that merely arrived beside
a lookalike key both give `missing=1`, `extra=1` and a low similarity ratio —
they are *feature-identical*, so no classifier can tell them apart, and Layer 1
reported `reorder_level` → `min_stock` (similarity 0.09) as `null_field`.
Value compatibility carries signal the name does not: a renamed
`reorder_level` arriving as `min_stock: 50.0` has a value that fits
`reorder_level`'s declared float range of 1–500, whereas a coincidental
`reorder_mode: "auto"` does not. Scored 1.0 for an exact type-and-range fit,
0.5 for a recoverable one (numeric string, or right type but out of range), and
0.0 for incompatible.

Feature 8 comes from a `DuplicateTracker` (`src/ingestion/dedup.py`), which
keeps a bounded set of SHA-256 content fingerprints over schema-relevant
fields. Key order does not affect a fingerprint and annotation keys are
excluded, so a record does not change identity merely by being annotated.
Deduplication is inherently stateful — a duplicate is only a duplicate
*relative to a stream* — so it lives in the ingestion stage. A single record
in isolation can never be known to be a repeat, and the API's
`/api/run-pipeline` therefore gives each request its own tracker and primes it
explicitly when asked to simulate a replay.

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
alias recovery, type coercion, schema-safe defaults, range clamping, and
pass-through of unrecognised fields.

**Every field the parser touches is disclosed.** The healed record carries a
`_healing` block naming what changed and what was there before, so a consumer
can tell a value that genuinely arrived from one the parser invented:

```json
{
  "amount": 0.01,
  "risk_score": 1.0,
  "_healing": {
    "amount":     { "actions": ["defaulted"],       "original": null },
    "risk_score": { "actions": ["clamped_to_max"],  "original": 870.5 }
  }
}
```

Fields absent from that block passed through untouched, and a record that
needed no healing carries no `_healing` key at all. Actions are
`aliased_from:<key>`, `defaulted`, `defaulted_blank`, `coerced_to_number`,
`coerced_to_str`, `rejected_bool`, `clamped_to_min`, `clamped_to_max` and
`dropped_unrecognised`.

**Some fields cannot tolerate an invented value at all.** Backfilling a
measurement is reasonable — a dropped sensor reading is an ordinary
operational event. Backfilling an *identity* or a *monetary amount* is not: a
transaction with an empty `transaction_id` and an `amount` of 0.01 passes
schema validation, reaches the warehouse, and then enters `SUM(amount)` and
`COUNT(*) GROUP BY merchant_id`. It is not a repaired record; it is a
corrupted one that now looks valid — which is the exact failure this project
exists to prevent.

Each pipeline therefore declares a **missing-value policy** per field:

```python
missing_value_policy={
    "transaction_id": POLICY_QUARANTINE,   # unjoinable, double-counts in GROUP BY
    "amount":         POLICY_QUARANTINE,   # corrupts SUM and AVG
    # currency, risk_score, timestamp -> POLICY_BACKFILL (the default)
}
```

When a quarantined field's value is **invented or altered** — defaulted, or
clamped out of range — the record is flagged:

```json
{ "transaction_id": "", "amount": 0.01, "currency": "USD",
  "_quarantine": ["transaction_id", "amount"] }
```

The record keeps a complete, correctly typed shape, so it still flows through
typed systems; `_quarantine` means it must be kept out of aggregates and
downstream consumers. A *coercion* does not quarantine: `"42.5"` → `42.5`
preserves the value rather than inventing one. Declared policies:

| Pipeline | Quarantined | Backfilled |
|---|---|---|
| `weather` | `latitude`, `longitude`, `time` | the five metrics |
| `finance` | `transaction_id`, `user_id`, `merchant_id`, `amount` | `currency`, `risk_score`, `timestamp` |
| `ecommerce` | `item_id`, `sku`, `unit_price`, `stock_quantity` | `category`, `reorder_level`, `timestamp` |
| `iot` | `device_id`, `timestamp` | the five telemetry channels |

Weather location is quarantined because `0.0, 0.0` is a real point in the Gulf
of Guinea — the "null island" artefact that shows up in any geospatial dataset
with backfilled coordinates.

`GET /api/pipelines` publishes each policy, `POST /api/run-pipeline` returns
`quarantined_fields`, and both dashboards show an amber state — deliberately
not green or red, because the heal succeeded while the record stayed
untrustworthy.

Duplicate records are the one case a generated parser cannot handle: it sees
one record at a time and cannot know about a repeat. Detection owns that
state, and `parse(..., is_duplicate=True)` records its verdict as
`_is_duplicate` on the output.

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
| `duplicate_record` | a verbatim replay of an earlier record (no marker keys) |

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
| `GET /api/pipelines` | all registered pipelines with schema, ranges and quarantine policy |
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

240-row held-out split, seed 42 — rows the model never saw. Two benchmarks,
because the answer differs:

**`--difficulty standard`** (the default)

| Engine | Precision | Recall | F1 |
|---|---|---|---|
| RandomForest | 100% | 100% | 100% |
| Rule baseline | 100% | 100% | 100% |

F1 lift `+0.0000`; the two engines agree on every row. This benchmark is
trivially separable: each anomaly class maps to a distinct feature signature,
so there is nothing for a model to learn that the rules do not already encode.
**On this dataset the RandomForest earns nothing.**

**`--difficulty hard`** adds confounders that attack the baseline's fixed 0.4
key-similarity threshold from both sides — genuine renames whose similarity
falls *below* it (mirroring `reorder_level` → `min_stock` at 0.09), and plain
missing fields accompanied by a coincidental lookalike key that scores *above*
it. The labels are unchanged; only the features become ambiguous.

| Engine | Precision | Recall | F1 |
|---|---|---|---|
| RandomForest | 100% | 100% | 100% |
| Rule baseline | 98.48% | 98.33% | 98.33% |

F1 lift **`+0.0167`**, agreement 98.3%. The model is ahead, but only just.

**That gap used to be `+0.1091`, and closing it was the more useful result.**
The original 10.9-point lead disappeared when feature 11 was added: the rule
baseline went from 89.09% to 98.33% F1 on this same benchmark. Most of what
looked like "ML value" was in fact a *missing feature* — the model was
compensating, via an opaque decision boundary, for information the feature
vector should have carried in the first place. Adding the feature improved the
explainable baseline by 9.2 points and left the model with 1.7.

The lesson generalises: a large model-over-baseline gap is as likely to
indicate an inadequate feature set as a clever model. It is worth trying to
close with a feature before concluding the model earned it.

`GET /api/evaluate` reports the measured lift and a verdict on it, so the claim
is never stated without its evidence. Reproduce either benchmark with
`python run_data_and_train.py [--difficulty hard]`.

### Healing coverage

All 24 combinations of 4 pipelines × 6 anomaly classes heal successfully, each
verified against 101–121 records. Covered by `tests/test_multidomain_healing.py`.

---

## 🔍 Measurement Caveats (read before quoting the numbers)

Being blunt, because the headline figures look better than they are:

1. **No feature encodes a label any more.** `duplicate_flag` used to be set by
   the injector's own marker keys (`_is_duplicate_flag`, `duplicate_uuid`),
   which for that class made the feature *be* the label. It is now derived from
   a content fingerprint, and the injector emits genuine replays instead of
   planting markers.
2. **The standard benchmark is still separable by construction.** Every feature
   is computed against the same schema definitions the injector perturbs, so
   the classes are far cleaner here than in real traffic. 100% is a property of
   this dataset, not evidence of real-world accuracy — which is exactly why
   `--difficulty hard` exists.
3. **The model's remaining advantage is small and benchmark-specific.**
   The +0.017 lift is measured against confounders this project invented, and
   it is what survived after a missing feature was supplied. The explainable
   rule baseline is within 1.7 F1 points, so the RandomForest is not doing
   heavy lifting here — on this evidence, shipping the rule baseline alone
   would be a defensible choice.
4. **There is no real-world labelled evaluation set.** Everything above is
   synthetic.
5. **Declared-alias recovery is configuration, not inference.** When the
   injector renames via the registry's table and the healer consults that same
   table, the loop is partly circular. The genuinely autonomous part is the
   fuzzy fallback for *undeclared* renames — which is what hard mode stresses.

`GET /api/evaluate` returns these caveats inline, and both dashboards display
them next to the metrics.

---

## 🔒 Security Model

Layer 3 generates and executes Python, so this is load-bearing.

**Verification** — candidate patches are screened by
`sandbox.validate_patch_source` before anything runs, against an **allowlist**
of permitted AST node types. This matters: the screen was originally a
denylist, and a denylist fails open. Every Python release adds syntax, so the
set of forms to forbid is open-ended, and the original list silently permitted
walrus assignments, `match` statements, comprehension scopes, starred
unpacking, f-strings, decorators and nested functions. An allowlist rejects
unknown syntax by default.

The permitted set was derived empirically rather than guessed: 432 patches
generated across every pipeline, strategy and anomaly category use exactly 39
node types, and all 432 still validate. On top of the allowlist:

- dangerous builtin *names* are denied — allowlisting cannot cover these,
  since `eval(...)` is syntactically an ordinary `Call` on a `Name`;
- any dunder attribute access is blocked, defeating
  `{}.__class__.__bases__`-style escapes;
- the module top level may contain only constants and function definitions,
  because that module is executed on import when a patch is activated;
- decorators and nested functions are rejected, so the entrypoint is the plain
  function the sandbox believes it is calling.

Screened code then runs in a **separate interpreter** launched with `-I`
(isolated: no `PYTHONPATH`, no user site-packages, no inherited env influence)
under a wall-clock timeout. `while` is not in the permitted subset, so a
non-terminating patch has to loop via `for` — the timeout covers that, and is
tested against it.

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
│   │   ├── dedup.py           # content-fingerprint duplicate tracker
│   │   ├── generate_data.py   # seeded generation; standard & hard difficulty
│   │   ├── injector.py        # anomaly injection (seedable RNG)
│   │   └── pipeline_registry.py  # schemas, ranges, renames, missing-value policy
│   ├── layer1_detector/
│   │   ├── feature_extractor.py  # 11-feature vectoriser
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
└── tests/                     # 242 tests
    ├── test_api_validation.py      # request validation, bounds, health
    ├── test_dedup.py               # fingerprinting, tracker, replay injection
    ├── test_e2e_pipeline.py        # end-to-end weather cycle
    ├── test_heldout_data.py        # verification-data integrity
    ├── test_injector.py            # all anomaly classes injectable
    ├── test_layer1.py              # extractor + detector
    ├── test_multidomain_healing.py # every rename pair; every anomaly x pipeline
    ├── test_outlier_healing.py     # range-driven diagnosis + clamping
    ├── test_provenance.py          # healed records disclose what changed
    ├── test_quarantine.py          # per-field missing-value policy
    ├── test_sandbox_security.py    # allowlist screening; timeout enforced
    └── test_value_compatibility.py # rename inference beyond name similarity
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
python run_data_and_train.py                    # standard benchmark
python run_data_and_train.py --difficulty hard  # confounded benchmark
```

Without this, Layer 1 falls back to rule-based heuristics (the API reports
`model_loaded: false` and both dashboards warn about it) and Layer 3 falls back
to a small synthetic verification set.

Both runs print precision, recall, F1, the model-versus-baseline lift and a
verdict on whether the trained model is carrying its weight.

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

Deduplication memory is set in code (`dedup.DEFAULT_CAPACITY`, 2048
fingerprints) rather than by environment variable, since it is a property of
the ingestion stage rather than a deployment knob.

---

## 🧪 Running the Tests

```bash
pytest tests/ -v
```

242 tests, roughly 28 seconds. The suite runs **entirely offline**: a
`conftest.py` fixture serves the weather pipeline from its cached baseline, so
no test depends on Open-Meteo being reachable. Before that fixture, live HTTP
dominated the runtime and made the suite fail without a network connection.

What they cover:

```text
tests/test_api_validation.py ..........                          [  4%]
tests/test_dedup.py .....................                        [ 15%]
tests/test_e2e_pipeline.py .                                     [ 15%]
tests/test_heldout_data.py ..                                    [ 16%]
tests/test_injector.py .                                         [ 17%]
tests/test_layer1.py ..                                          [ 18%]
tests/test_multidomain_healing.py ..........................     [ 30%]
tests/test_outlier_healing.py ............                       [ 36%]
tests/test_provenance.py .........                               [ 35%]
tests/test_quarantine.py ................................        [ 48%]
tests/test_sandbox_security.py ........................................ [ 65%]
tests/test_value_compatibility.py .....................................[100%]

============================= 242 passed in ~28s ==============================
```

The suite deliberately enumerates **every declared rename pair** rather than
letting the injector pick at random. An earlier random version masked a real
defect: two semantic renames were never recovered, but the test only hit them
when the RNG happened to choose those fields.

`test_sandbox_security.py` validates all 432 patches the generator can emit
against the allowlist, so hardening the screen cannot silently break patch
generation. `test_value_compatibility.py` asserts the rule baseline — not only
the model — recognises every declared rename, on the principle that a feature
only the model can exploit is suspect.

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

- **Deduplication is bounded and order-dependent.** The tracker remembers the
  last 2048 fingerprints, so a duplicate separated from its original by more
  than that is missed — the normal trade for streaming deduplication. It also
  cannot see a repeat across process restarts, since the memory is in-process.
- **Backfilled measurements are still invented values.** Fields under
  `POLICY_BACKFILL` get a schema-safe default and flow on, disclosed via
  `_healing` but not quarantined. That is a judgement call: a dropped sensor
  reading is an ordinary event, and quarantining every one of them would stop
  the pipeline for routine gaps. It does mean an averaged metric can still
  include substituted values, so a consumer computing statistics should filter
  on `_healing` as well as `_quarantine`.
- **Quarantine is advisory, not enforced.** The record is annotated and
  returned; routing it to a dead-letter path is the caller's job. A stricter
  design would have the orchestrator withhold it from the main stream.
- **The AST screen is a denylist, not a capability sandbox.** See
  [Security Model](#-security-model).
- **Three of four pipelines are simulated.** Only `weather` hits a real API,
  and it returns the same observation for up to an hour — so repeated live
  fetches are genuine content duplicates.
- **Rename inference is only as good as its two signals.** Name similarity
  plus value compatibility correctly identify all 17 renames the registry
  declares, but an undeclared rename to a field of the same type and range as
  another missing field is still ambiguous — nothing in the payload
  distinguishes which of two interchangeable fields the value belongs to.
- **No authentication on the API.** It is bound to loopback by design.
  Exposing it requires adding auth.
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
