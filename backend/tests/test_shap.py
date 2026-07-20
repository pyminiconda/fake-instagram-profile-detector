import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.core.prediction_engine import PredictionEngine

engine = PredictionEngine()
result = engine.predict([0.1, 0.5, 0.1, 100, 0.5, 5, 1])

print(result['shap_values'])
