# Implementation notes

## Existing behavior discovered in the source

- `ai_backend.py` defines `POST /analyze-graph`, calculates annual parameter statistics, calls NVIDIA Nemotron, and adds model_go forecasts to its response.
- `model_go` is imported using its Python SDK (`model_go.sdk.client.Forecaster`) and runs in-process.
- The SDK defaults to five horizons: 24, 48, 72, 168 and 720 hours.
- The old dashboard loads New Delhi 2015–2025 data from an inline `RAW` object and fetches other selected cities from NASA POWER monthly/point.
- Open-Meteo geocoding is used by the old city search. The rebuilt frontend retains that city-search provider.
- The original dashboard exposes six parameters: PRECTOTCORR, QV2M, T2M_MAX, T2M_MIN, TSOIL1 and WS10M.

## Migration choices

- Preserve the `/analyze-graph` payload fields and endpoint.
- Preserve the model package without rewriting its model code or assets.
- Add `/forecast` to separate forecasting UI requests from the AI interpretation endpoint.
- Use the exact original New Delhi snapshot as the frontend's initial offline view.
- Fetch other city histories directly from NASA POWER, as the old dashboard did.
- Replace the all-in-one HTML/Chart.js UI with a React/TypeScript UI and Recharts visualizations.
