# Frontend

This folder is now a proper Vite + React frontend.

Structure:
- `src/main.jsx`: React entry point
- `src/App.jsx`: app UI and API wiring
- `src/styles.css`: frontend styling
- `index.html`: Vite HTML shell

Roles:
- `doctor`: authorized, sees raw top-k chunks after reranking
- `receptionist`: unauthorized, sees redacted top-k chunks after reranking
- `nurse`: unauthorized, sees redacted top-k chunks after reranking

## Local frontend development

In one terminal, run the backend API server from the repo root:

```powershell
$env:PYTHONPATH="src"
python -m medical_extraction.cli.run_frontend --config configs/local.yaml
```

In a second terminal, from the `frontend/` folder:

```powershell
npm install
npm run dev
```

That starts the React app on `http://127.0.0.1:5173`.
The Vite dev server proxies `/api` requests to `http://127.0.0.1:8765`.

## Build for the Python server

From the `frontend/` folder:

```powershell
npm install
npm run build
```

This writes the production frontend to `frontend/dist`.

Then run the backend server from the repo root:

```powershell
$env:PYTHONPATH="src"
python -m medical_extraction.cli.run_frontend --config configs/local.yaml
```

The Python server will serve `frontend/dist` automatically when it exists.
