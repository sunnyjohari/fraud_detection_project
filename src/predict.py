#  predict.py  —  Inference interface

#  Two modes:
#    predict_single()  — one transaction dict
#    predict_batch()   — a pandas DataFrame

import os
import pickle
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from config import BEST_MODEL_PATH, MODEL_META_PATH, PREDICTORS


#  Load model once at import time 

def load_model():
    """
    Load the saved best model from disk.
    Returns (model_name, model_obj).
    """
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}. "
            "Run train.py first."
        )
    with open(BEST_MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    print(f"[predict] Loaded model: {saved['name']}")
    return saved["name"], saved["model"]


# Module-level singletons — loaded once
_MODEL_NAME, _MODEL_OBJ = None, None


def _get_model():
    global _MODEL_NAME, _MODEL_OBJ
    if _MODEL_OBJ is None:
        _MODEL_NAME, _MODEL_OBJ = load_model()
    return _MODEL_NAME, _MODEL_OBJ


#  Core inference 

def _raw_predict(model_obj, X: pd.DataFrame) -> np.ndarray:
    """
    Returns fraud probability scores (0.0 – 1.0).
    Handles sklearn, XGBoost native, LightGBM native.
    """
    if isinstance(model_obj, dict):
        booster    = model_obj["booster"]
        model_type = model_obj["type"]

        if model_type == "xgboost":
            dmatrix = xgb.DMatrix(X)
            return booster.predict(dmatrix)

        elif model_type == "lightgbm":
            return booster.predict(X.values)

    else:
        # sklearn — predict_proba[:, 1]
        return model_obj.predict_proba(X)[:, 1]


#  Public API 

def predict_single(transaction: dict, threshold: float = 0.5) -> dict:
    """
    Predict fraud probability for a single transaction.

    Parameters
    ----------
    transaction : dict
        Must contain all keys in PREDICTORS.
        Example:
          {
            "Time": 406.0, "V1": -1.36, "V2": -0.07, ..., "Amount": 149.62
          }
    threshold : float
        Decision boundary. Default 0.5.
        Lower = more sensitive (catches more fraud, more false alarms).
        Higher = more precise (fewer false alarms, misses more fraud).

    Returns
    -------
    dict with keys:
      fraud_probability : float  — model confidence 0.0–1.0
      is_fraud          : bool   — True if above threshold
      model_used        : str    — name of the model
    """
    model_name, model_obj = _get_model()

    #── Validate input keys 
    missing = [col for col in PREDICTORS if col not in transaction]
    if missing:
        raise ValueError(f"Missing features in input: {missing}")

    row = pd.DataFrame([transaction])[PREDICTORS]
    prob = float(_raw_predict(model_obj, row)[0])

    return {
        "fraud_probability": round(prob, 6),
        "is_fraud":          prob >= threshold,
        "threshold_used":    threshold,
        "model_used":        model_name,
    }


def predict_batch(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Score a DataFrame of transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns in PREDICTORS.

    Returns
    -------
    df with two new columns added:
      fraud_probability : float
      is_fraud          : bool
    """
    model_name, model_obj = _get_model()

    missing = [col for col in PREDICTORS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features in DataFrame: {missing}")

    result                    = df.copy()
    probs                     = _raw_predict(model_obj, df[PREDICTORS])
    result["fraud_probability"] = probs.round(6)
    result["is_fraud"]          = probs >= threshold

    fraud_count = result["is_fraud"].sum()
    print(f"[predict] Scored {len(df):,} transactions | "
          f"Flagged as fraud: {fraud_count:,} "
          f"({fraud_count/len(df)*100:.2f}%)")
    return result


def get_model_info() -> dict:
    """Return metadata about the currently loaded model."""
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH, "r") as f:
            return json.load(f)
    model_name, _ = _get_model()
    return {"model_name": model_name}


#  Standalone demo 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 50)
    print("  predict.py — Standalone demo")
    print("=" * 50)

    #  Demo 1: single transaction 
    print("\n[demo] Single transaction prediction")
    sample_transaction = {
        "Time": 406.0, "V1": -1.3598071336738,  "V2": -0.0727811733098497,
        "V3":  2.53634673796914, "V4":  1.37815522427443, "V5": -0.338320769942518,
        "V6":  0.462387777762292, "V7": 0.239598554061257, "V8": 0.0986979012610507,
        "V9":  0.363786969611213, "V10": 0.0907941719789316,
        "V11": -0.55159953895894, "V12": -0.617800855762348,
        "V13": -0.991389847235408, "V14": -0.311169353699879,
        "V15": 1.46817697209427,  "V16": -0.470400525259478,
        "V17": 0.207971241929242, "V18": 0.0257905801985591,
        "V19": 0.403992960255733, "V20": 0.251412098239705,
        "V21": -0.018306777944153, "V22": 0.277837575558899,
        "V23": -0.110473910188767, "V24": 0.0669280749146731,
        "V25": 0.128539358273528, "V26": -0.189114843888824,
        "V27": 0.133558376740387, "V28": -0.0210530534538215,
        "Amount": 149.62
    }

    result = predict_single(sample_transaction)
    print(f"  Fraud probability : {result['fraud_probability']:.4f}")
    print(f"  Is fraud?         : {result['is_fraud']}")
    print(f"  Model             : {result['model_used']}")

    #  Demo 2: batch scoring 
    print("\n[demo] Batch prediction on test data")
    from preprocess import load_data, add_hour_feature, split_data

    df                       = load_data()
    df                       = add_hour_feature(df)
    _, _, test_df            = split_data(df)

    scored_df = predict_batch(test_df.head(1000))
    print(scored_df[["Amount", "Class", "fraud_probability", "is_fraud"]].head(10))
