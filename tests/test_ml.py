import os
import pytest
import numpy as np
import joblib
from src.ml_engine import MLEngine

def test_models_exist():
    engine = MLEngine()
    assert os.path.exists(engine.rf_path), "RandomForest model file missing."
    assert os.path.exists(engine.xgb_path), "XGBoost model file missing."
    assert os.path.exists(os.path.join(engine.models_dir, "symptoms.joblib")), "Symptoms list missing."
    assert os.path.exists(os.path.join(engine.models_dir, "disease_metadata.joblib")), "Disease metadata missing."

def test_prediction_output():
    engine = MLEngine()
    symptoms = joblib.load(os.path.join(engine.models_dir, "symptoms.joblib"))
    
    # Create vector with 0s
    vector = [0] * len(symptoms)
    
    # Try predicting
    res_xgb = engine.predict_topk(vector, k=3, model_type="xgboost")
    res_rf = engine.predict_topk(vector, k=3, model_type="random_forest")
    
    assert len(res_xgb) == 3
    assert len(res_rf) == 3
    
    for item in res_xgb:
        assert "disease" in item
        assert "confidence" in item
        assert isinstance(item["confidence"], float)
