import json, sys, os
# Add model path to PYTHONPATH
model_path = r"C:/Users/Asus/My-Projects/Be-An-Earth-System-Trend-Detective/model_go_v2/model_go_v2"
if model_path not in sys.path:
    sys.path.append(model_path)

from model_go import Forecaster

fc = Forecaster()
res = fc.predict_point()
# Serialize with default=str for non‑JSON types (e.g., datetime)
print(json.dumps(res, default=str, indent=2))
