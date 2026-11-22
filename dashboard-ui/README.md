# Agent Console (React + Vite)

The browser console for the Autonomous Self-Healing Data Pipeline Agent. It
talks to the Flask API in `../api/server.py`.

## Running it

The backend must be up first, because every panel is driven by the API:

```bash
# from the repository root, in one terminal
python run_data_and_train.py     # once, to build the dataset and model
python api/server.py

# in another terminal
cd dashboard-ui
npm install
npm run dev                      # http://localhost:3000
```

`vite.config.js` proxies `/api` to `http://127.0.0.1:5000`, so no CORS
configuration is needed in development. If you change the API port, set
`PORT` for the backend and update the proxy target to match.

## Scripts

| Command           | Purpose                                  |
| ----------------- | ---------------------------------------- |
| `npm run dev`     | Dev server with HMR on port 3000         |
| `npm run build`   | Production build into `dist/`            |
| `npm run preview` | Serve the production build locally       |
| `npm run lint`    | Oxlint over `src/`                       |

## Layout

```
src/
  api.js                  Single fetch wrapper; checks res.ok and raises ApiError
  App.jsx                 Shell: nav, hero, controls, tab routing
  components/
    LivePipeline.jsx      Clean / corrupted / healed payload comparison
    MLDetector.jsx        Layer 1 verdict and feature vector
    LLMDiagnosis.jsx      Layer 2 root-cause report
    PatchSandbox.jsx      Generated patch and verification result
    EvaluationAudit.jsx   Layer 1 benchmark and audit trail
    ErrorNotice.jsx       Shared inline error panel with retry
    Plasma.jsx            WebGL background shader
    ParticleText.jsx      Canvas particle headline
    ScrollExpand.jsx      Scroll-driven hero expansion
    CyberBlocks.jsx       Decorative SVG blocks for the architecture cards
```

Every panel handles its own loading, empty and error states; `api.js` is the
only place that calls `fetch`.
