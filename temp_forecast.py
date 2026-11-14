import json, sys, os
# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
from model_go import Forecaster

fc = Forecaster()
res = fc.predict_point()
print(json.dumps(res, indent=2))
