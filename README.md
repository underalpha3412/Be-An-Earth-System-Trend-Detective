# Earth System Trend Detective

A rebuilt React + TypeScript dashboard for the existing Vanguard Earth Trends project. The original forecasting model and Nemotron analysis backend are retained.

## Project structure

```text
earth-system-dashboard/
├── backend/
│   ├── ai_backend.py              # Original FastAPI + Nemotron integration, with a forecast endpoint
│   ├── requirements.txt
│   ├── .env.example
│   └── model_go_v2/model_go_v2/    # Original model_go v2 package and bundled model assets
├── frontend/
│   ├── public/data/delhi-2015-2025.json  # Exact New Delhi snapshot embedded in the old dashboard
│   └── src/                        # React + TypeScript application
└── docs/
```

## Requirements

- Python 3.10+ (the included model package declares Pydantic v2 and modern NumPy/SciPy dependencies)
- Node.js 20+ and npm
- An NVIDIA API key for Nemotron insights. Historical charts and model_go forecasts do not use this key.

## Run locally

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own NVIDIA key:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
DASHBOARD_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
MODEL_GO_PROFILE=server
```

Start the API:

```bash
python -m uvicorn ai_backend:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually http://localhost:5173.

## Dashboard areas

- **Overview:** selected-city summary, annual climate trajectories, model status, AI notes.
- **Historical analysis:** parameter selection, year filter, annual trends, monthly temperature and precipitation.
- **Weather forecast:** model_go v2 output for +24h, +48h, +72h, +7d and +30d, including humidity, wind, precipitation, rain probability and supplied temperature uncertainty.
- **City comparison:** compare annual NASA POWER values across selected cities.

## Data and integration notes

- The initial New Delhi snapshot is extracted directly from the `RAW` telemetry object in the original `nasa_dashboard.html`, preserving its 2015–2025 annual and monthly values.
- When a different city is selected, the frontend requests monthly point data from the NASA POWER API for the same six original parameters and years.
- AI analysis continues to use the original `/analyze-graph` request and response contract.
- `/forecast` is a small additional route that exposes the existing `run_model_go_forecast` helper directly for the forecast page. It does not replace the original `/analyze-graph` route.
- The bundled model package is kept intact under `backend/model_go_v2/model_go_v2/`.
- No `.env` file or API secret is included.

## Important limitations to verify on your machine

1. The bundled model package can load but its runtime may depend on upstream NASA/IMD data services and local package compatibility. Confirm `/health` and `/forecast` on the machine where you run it.
2. The city comparison view requests NASA POWER data sequentially for selected cities. NASA service availability and rate limits apply.
3. The snapshot includes the values embedded in the previous dashboard. For a newly selected city, values are fetched live from NASA POWER.
4. The original AI backend invokes model_go while processing `/analyze-graph`, as it did before. If the model cannot initialize, AI graph analysis may still fail depending on the path and model availability.

## Original project files retained for reference

The original `nasa_dashboard.html`, `ai_backend.py`, and forecast scripts were inspected. The HTML monolith is replaced by the React frontend; the original backend implementation and model source package are retained.

## Vanguard welcome and automatic graph insights (v3 update)

- The app now opens on the Team Vanguard introduction. The Dashboard Guide is available beside it in the sidebar and in the workspace header.
- Historical graphs show separate Nemotron interpretation panels. The app debounces and refreshes insights automatically when the selected city, year filter, or parameter set changes. No manual Generate Insights click is required.
- While on City Comparison, selecting a city from the global search adds it to the comparison set. Remove a city with the × control.
- The FastAPI CORS configuration allows the Vite development origins `http://localhost:5173` and `http://127.0.0.1:5173` (and localhost/127.0.0.1 development ports).
- The Vanguard logo is served from `frontend/public/vanguard-logo.jpg`.

Automatic insights require the backend to be running and a valid `NVIDIA_API_KEY` in `backend/.env`. If the API is unavailable, the insight panel reports the error and preserves the last successful notes where possible.


### Weather forecast workspace updates

- The first outlook tile shows current conditions for the selected city from the Open-Meteo current-weather endpoint. The five model_go horizons follow it, forming a 3-by-2 tile layout on wide screens.
- Forecast tiles show symbol-only measurements; the Quick Key below explains precipitation, humidity, and wind symbols.
- Four compact trend charts compare temperature, precipitation, relative humidity, and wind speed across current conditions and the available model_go horizons.
- Switching workspaces returns the page to the top.
