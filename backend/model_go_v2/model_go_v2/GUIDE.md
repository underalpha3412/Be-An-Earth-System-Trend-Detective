# Absolute Beginner's Guide to `model_go` v2

Welcome to **`model_go` v2**! This guide is written so that **anyone**—even if you have never used a command line, run a Python server, or worked with artificial intelligence before—can get this weather forecasting engine running in under 5 minutes.

---

## 🌟 1. What is `model_go` v2? (In Simple Words)

Imagine having a private supercomputer meteorological service running right on your laptop:
* **No expensive subscriptions, no API keys, zero fees (₹0.00).**
* **Predicts the weather up to 30 days into the future:** Next day (`+24h`), 2 days (`+48h`), 3 days (`+72h`), 1 week (`+7d`), and 1 month (`+30d`).
* **Works for ANY city in the world:** It can specialize on any coordinate worldwide in less than a second (e.g., Delhi, Tokyo, London, New York, Sydney, Cairo, Cape Town).
* **Honest 90% Confidence Bounds:** It doesn't just say *"it will be 25°C"*. It says *"it will be 25°C with 90% certainty between 23.2°C and 26.8°C"*.
* **Dual-Zone Flood Early Warning:** Warns you if rivers or streets will flood (`GREEN`, `YELLOW`, `ORANGE`, `EXTREME_RED_ALERT`), with specialized stage calculations for River Yamuna (Delhi) and global flash-flood runoff tracking worldwide.
* **Pure Backend Engine:** It runs silently in the background and speaks through a standard Web API (REST) or Python code, so any mobile app, website, or dashboard can connect to it.

---

## 💻 2. System Requirements

* **Operating System:** Windows 10/11, macOS, or Linux.
* **Hardware:** Any standard laptop or desktop computer (runs smoothly on a regular CPU; no GPU required!).
* **RAM:** Less than 12 MB of memory used during live forecasting.
* **Python:** Version 3.10, 3.11, or 3.12 installed.
  * *If you don't have Python:* Download it for free from [python.org](https://www.python.org/downloads/). During installation on Windows, **be sure to check the box: "Add python.exe to PATH"**.

---

## 🚀 3. Quickstart: 3 Simple Steps to Run the Server

### Step 1: Open Your Terminal / Command Prompt
* **On Windows:** Press the `Windows Key`, type `cmd`, and press `Enter`.
* Navigate to the unzipped folder. For example, if you unzipped it on your `D:` drive or `Downloads`:
  ```cmd
  cd /d D:\projects\nasa_ml
  ```
  *(Replace with your actual folder path where you extracted the zip!)*

---

### Step 2: Set Up Python Virtual Environment
Run these commands one by one:

**On Windows (Command Prompt - CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**On Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Step 3: Start the Forecasting Engine!
Type this exact command:
```cmd
python -m uvicorn model_go.api.app:app --host 0.0.0.0 --port 8000
```

You will see output like this:
```text
INFO:     Started server process [14820]
INFO:     Waiting for application startup.
INFO:     ModelGoOrchestrator v2 initialized successfully [Profile: server]
INFO:     Loaded 9 specialized spatial adapters from disk
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

🎉 **Congratulations! Your weather forecasting engine is now live and running!**

---

## 🌐 4. Interactive Browser Documentation

Open your favorite web browser (Chrome, Edge, Firefox, Safari) and go to:
* **Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative Docs:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

Here you can click on any API endpoint, click **"Try it out"**, enter latitude and longitude, click **"Execute"**, and see live forecast results in your browser!

---

## 📡 5. How to Query Forecasts from the Command Line (CMD Usage)

Open a **second** Command Prompt or PowerShell window (keep the first window running the server!).

### A. Query Next-Day & 72-Hour Forecast for Delhi (Safdarjung)

**Windows CMD (curl):**
```cmd
curl -X POST "http://localhost:8000/v1/forecast/point" -H "Content-Type: application/json" -d "{\"latitude\": 28.6139, \"longitude\": 77.2090, \"station_name\": \"Delhi Safdarjung\"}"
```

**Windows PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/v1/forecast/point" -Method Post -ContentType "application/json" -Body '{"latitude": 28.6139, "longitude": 77.2090, "station_name": "Delhi Safdarjung"}'
```

---

### B. Query an International City (e.g. Tokyo, Japan)

**Windows CMD:**
```cmd
curl -X POST "http://localhost:8000/v1/forecast/point" -H "Content-Type: application/json" -d "{\"latitude\": 35.6762, \"longitude\": 139.6503, \"station_name\": \"Tokyo\"}"
```

**Windows PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/v1/forecast/point" -Method Post -ContentType "application/json" -Body '{"latitude": 35.6762, "longitude": 139.6503, "station_name": "Tokyo"}'
```

---

### C. Pre-Warm / Specialize a New City On-Demand (`POST /v1/adapt`)
Want the model to learn the microclimate of **Paris**, **New York**, or **Cairo**? Call `/v1/adapt`:

**Windows CMD:**
```cmd
curl -X POST "http://localhost:8000/v1/adapt" -H "Content-Type: application/json" -d "{\"lat\": 48.8566, \"lon\": 2.3522, \"lookback_years\": 3, \"station_name\": \"Paris\"}"
```

**What happens:** In ~0.1 seconds, `model_go` pulls historical data from NASA, trains a localized microclimate adapter, computes a custom 90% conformal margin of error, and saves a 1.5 KB file. Every subsequent forecast for Paris will now be custom-tailored in `< 15 milliseconds`!

---

### D. Check Active Flood & Disaster Alerts (`GET /v1/alerts/active`)

**Windows CMD:**
```cmd
curl -X GET "http://localhost:8000/v1/alerts/active"
```

---

### E. Check System Diagnostic & Satellite Sensor Health (`GET /v1/health`)

**Windows CMD:**
```cmd
curl -X GET "http://localhost:8000/v1/health"
```

---

## 🐍 6. How to Use `model_go` v2 in Python Code (SDK Usage)

You don't even need to run a web server if you just want to use it directly inside your Python scripts:

Create a file named `my_forecast.py`:
```python
from model_go import Forecaster

# 1. Initialize the engine
fc = Forecaster()

# 2. Get forecast for any location (e.g. Delhi, London, New York)
result = fc.predict_point(lat=28.6139, lon=77.2090, station_name="Delhi")

# 3. Print the results
h24 = result["horizons"]["+24h"]
print(f"Location: {result['target_location']['station_name']}")
print(f"Predicted Temperature (+24h): {h24['temperature_c']}°C")
print(f"90% Confidence Interval: {h24['temp_uncertainty_90pct'][0]}°C to {h24['temp_uncertainty_90pct'][1]}°C")
print(f"Rain Probability: {h24['rain_probability'] * 100:.1f}% ({h24['rain_hurdle_classification']})")
print(f"Inference Speed: {result['metadata']['inference_latency_ms']} ms")

# 4. Check alerts
for alert in result["alerts"]:
    print(f"Alert [{alert['category']} - {alert['level']}]: {alert['title']}")
```

Run it in your terminal:
```cmd
python my_forecast.py
```

Output:
```text
Location: Delhi
Predicted Temperature (+24h): 24.8°C
90% Confidence Interval: 22.97°C to 26.63°C
Rain Probability: 6.0% (NO_RAIN)
Inference Speed: 14.8 ms
Alert [YAMUNA_HYDROLOGY - GREEN]: NORMAL FLOW: River Yamuna Flowing Below Warning Level
```

---

## 📖 7. Understanding the Output Fields (Rosetta Stone)

When you receive a forecast response, here is what each field means:

| Field Name | What It Means in Simple English | Example Value |
| :--- | :--- | :--- |
| `temperature_c` | The most probable temperature at 2 meters above ground. | `24.8` (°C) |
| `temp_uncertainty_90pct` | **The 90% Conformal Range:** Statistically, there is a 90% probability that the actual temperature will fall inside this bracket `[lower, upper]`. | `[22.97, 26.63]` |
| `precipitation_expected_mm` | Expected rainfall accumulation in millimeters. | `0.0` (Dry) or `14.2` (Rain) |
| `rain_probability` | Probability of measurable rainfall between 0.0 (0%) and 1.0 (100%). | `0.12` (12% chance) |
| `rain_hurdle_classification` | Human-readable rain status: `NO_RAIN`, `ISOLATED_DRIZZLE`, `LIGHT_RAIN`, `MODERATE_MONSOON_SURGE`, `HEAVY_CONVECTIVE_EVENT`. | `NO_RAIN` |
| `dominant_engine` | The specific artificial intelligence or physics model that made this prediction. | `MODEL_ALPHA_GBDT` or `HYBRID_PHYSICS_STUDENT` |
| `adapter_status` | Whether this city has a custom-trained local adapter (`specialized`) or is using initial baseline fallback (`warming_up` / `zero_shot`). | `specialized` |
| `runoff_risk_index` | Flash-flood risk score from `0` (bone dry) to `100` (extreme catastrophic flood). | `12.5` |
| `alerts` | List of severe weather or flood advisories: `GREEN` (Normal), `YELLOW` (Watch), `ORANGE` (Warning), `EXTREME_RED_ALERT` (Emergency). | `AlertLevel.GREEN` |

---

## 🧪 8. How to Run Automated Verification Tests

To verify that all 41 unit, integration, conformal coverage, and safety tests pass on your machine:

```cmd
pytest -q
```

Expected result:
```text
41 passed, 1 warning in 10.68s
```

---

## 🐳 9. Running with Docker (Zero Installation Option)

If you have Docker installed on your machine, you don't even need to install Python!

Simply run:
```cmd
docker compose up --build
```
This automatically spins up:
1. `model_go_api`: The forecasting REST service on port 8000.
2. `redis_cache`: Fast in-memory cache on port 6379.
3. `model_go_worker`: Background synchronization worker pre-computing weather lags.

To test:
```cmd
curl http://localhost:8000/v1/health
```

---

## ❓ 10. Frequently Asked Questions (FAQ)

**Q: Does this cost any money?**  
**A:** No. `model_go` v2 is strictly **₹0.00** to operate. It uses open, publicly accessible satellite feeds from NASA POWER, GPM, and SMAP.

**Q: Can I run this completely offline without internet?**  
**A:** Yes! `model_go` v2 includes built-in circuit-breaker fallbacks and pre-packaged historical weights. If you disconnect from Wi-Fi, it automatically falls back to local physics simulations and stored climatology.

**Q: Error: `Address already in use` (Port 8000 busy)?**  
**A:** Another app is using port 8000. Start `model_go` on a different port, e.g.:
```cmd
python -m uvicorn model_go.api.app:app --host 0.0.0.0 --port 8080
```
Then use `http://localhost:8080/v1/forecast/point`.

**Q: Can I build my own website or phone app on top of this?**  
**A:** Absolutely! You can make standard `fetch()` or `axios` calls in JavaScript/React/Flutter to `http://localhost:8000/v1/forecast/point` and display the predictions in any user interface you like.

---

*Engineered with precision for global meteorology at ₹0.00 spend.*
