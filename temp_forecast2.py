import json, sys, os
# Add the directory containing the model_go package to PYTHONPATH
model_path = r"C:/Users/Asus/My-Projects/Be-An-Earth-System-Trend-Detective/model_go_v2/model_go_v2"
if model_path not in sys.path:
    sys.path.append(model_path)

from model_go import Forecaster

fc = Forecaster()
res = fc.predict_point()
print(json.dumps(res, indent=2))
