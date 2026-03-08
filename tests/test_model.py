import os
import pickle
import numpy as np
import xgboost as xgb

MODEL_PATH = "models/best_model.pkl"

def test_model_exists():
    """Ensure the model file is present and non-empty."""
    assert os.path.exists(MODEL_PATH), "Model file not found"
    assert os.path.getsize(MODEL_PATH) > 0, "Model file is empty"

def test_model_predicts():
    """Ensure the model object can produce predictions."""
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)

    dummy = np.zeros((1, 30))

    # Case 1: scikit-learn style model (RandomForest, XGBClassifier, etc.)
    if hasattr(obj, "predict"):
        pred = obj.predict(dummy)
        assert pred.shape[0] == 1, "Unexpected prediction shape"

    # Case 2: dict with metadata + model
    elif isinstance(obj, dict) and "model" in obj:
        model = obj["model"]

        # If model itself is scikit-learn style
        if hasattr(model, "predict"):
            pred = model.predict(dummy)
            assert pred.shape[0] == 1, "Unexpected prediction shape"

        # If model is dict wrapping an XGBoost Booster
        elif isinstance(model, dict) and "booster" in model:
            booster = model["booster"]
            feature_names = [
                "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
                "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
            ]
            dmatrix = xgb.DMatrix(dummy, feature_names=feature_names)
            pred = booster.predict(dmatrix)
            assert pred.shape[0] == 1, "Unexpected booster output shape"

        else:
            raise TypeError("Unsupported model format inside dict")

    else:
        raise TypeError("Unrecognized model object type")