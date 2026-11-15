import json, sys, os

# Ensure the model package is on PYTHONPATH
model_path = r"C:/Users/Asus/My-Projects/Be-An-Earth-System-Trend-Detective/model_go_v2/model_go_v2"
if model_path not in sys.path:
    sys.path.append(model_path)

from model_go import Forecaster

# List of international cities with lat/lon (approx)
cities = [
    ("Las Vegas", 36.1699, -115.1398),
    ("Cairo", 30.0444, 31.2357),
    ("Tbilisi", 41.7151, 44.8271),
    ("Beijing", 39.9042, 116.4074),
    ("Dhaka", 23.8103, 90.4125),
    ("Tokyo", 35.6895, 139.6917),
    ("Sydney", -33.8688, 151.2093),
    ("São Paulo", -23.5505, -46.6333)
]

results = []
fc = Forecaster()

for name, lat, lon in cities:
    try:
        res = fc.predict_point(lat=lat, lon=lon, station_name=name)
        results.append({"city": name, "lat": lat, "lon": lon, "result": res})
    except Exception as e:
        results.append({"city": name, "lat": lat, "lon": lon, "error": str(e)})

# Write full results to a JSON file for later analysis
out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "batch_results.json"))
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

# Also print a compact summary to stdout (JSON) for quick inspection
print(json.dumps(results, indent=2, default=str))
