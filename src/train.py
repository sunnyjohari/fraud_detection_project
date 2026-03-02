#  train.py  —  Train all models, pick the best

#  Selection metric : ROC-AUC on validation set
#  Saves best model  : models/best_model.pkl
#  Saves metadata    : models/model_meta.json

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

from config import (
    MODEL_DIR, BEST_MODEL_PATH, MODEL_META_PATH,
    RANDOM_STATE, PREDICTORS, TARGET,
    RF_N_ESTIMATORS, RF_N_JOBS, RF_CRITERION,
    ADA_N_ESTIMATORS, ADA_LEARNING_RATE,
    XGB_PARAMS, XGB_MAX_ROUNDS, XGB_EARLY_STOP, XGB_VERBOSE,
    LGB_PARAMS, LGB_MAX_ROUNDS, LGB_EARLY_STOP, LGB_VERBOSE,
)

#  Individual model trainers

def train_random_forest(train_df, valid_df):
    """Train a RandomForestClassifier and return (model, auc_score)."""
    print("\n[train] ── RandomForest ──────────────────")
    clf = RandomForestClassifier(
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE,
        criterion=RF_CRITERION,
        n_estimators=RF_N_ESTIMATORS,
        verbose=False
    )
    clf.fit(train_df[PREDICTORS], train_df[TARGET].values)
    preds = clf.predict(valid_df[PREDICTORS])
    auc   = roc_auc_score(valid_df[TARGET].values, preds)
    print(f"[train] RandomForest  validation AUC: {auc:.4f}")
    return clf, auc


def train_adaboost(train_df, valid_df):
    """Train an AdaBoostClassifier and return (model, auc_score)."""
    print("\n[train] ── AdaBoost ──────────────────────")
    clf = AdaBoostClassifier(
        random_state=RANDOM_STATE,
        learning_rate=ADA_LEARNING_RATE,
        n_estimators=ADA_N_ESTIMATORS
    )
    clf.fit(train_df[PREDICTORS], train_df[TARGET].values)
    preds = clf.predict(valid_df[PREDICTORS])
    auc   = roc_auc_score(valid_df[TARGET].values, preds)
    print(f"[train] AdaBoost      validation AUC: {auc:.4f}")
    return clf, auc


def train_xgboost(train_df, valid_df):
    """
    Train an XGBoost model using its native API (DMatrix).
    Returns a wrapper dict so downstream code stays uniform:
      { "booster": xgb.Booster, "type": "xgboost" }
    """
    print("\n[train] ── XGBoost ───────────────────────")
    dtrain = xgb.DMatrix(train_df[PREDICTORS], train_df[TARGET].values)
    dvalid = xgb.DMatrix(valid_df[PREDICTORS], valid_df[TARGET].values)
    watchlist = [(dtrain, "train"), (dvalid, "valid")]

    booster = xgb.train(
        XGB_PARAMS,
        dtrain,
        XGB_MAX_ROUNDS,
        watchlist,
        early_stopping_rounds=XGB_EARLY_STOP,
        maximize=True,
        verbose_eval=XGB_VERBOSE
    )

    preds = booster.predict(dvalid)
    auc   = roc_auc_score(valid_df[TARGET].values, preds)
    print(f"[train] XGBoost       validation AUC: {auc:.4f}")

    model = {"booster": booster, "type": "xgboost"}
    return model, auc

# for better fraud recall
def train_lightgbm(train_df, valid_df):
    """
    Train a LightGBM model using its native API.
    Returns a wrapper dict:
      { "booster": lgb.Booster, "type": "lightgbm" }
    """
    print("\n[train] ── LightGBM ──────────────────────")
    dtrain = lgb.Dataset(
        train_df[PREDICTORS].values,
        label=train_df[TARGET].values,
        feature_name=PREDICTORS
    )
    dvalid = lgb.Dataset(
        valid_df[PREDICTORS].values,
        label=valid_df[TARGET].values,
        feature_name=PREDICTORS
    )

    booster = lgb.train(
        LGB_PARAMS,
        dtrain,
        valid_sets=[dtrain, dvalid],
        num_leaves=15,
        valid_names=["train", "valid"],
        num_boost_round=LGB_MAX_ROUNDS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=LGB_EARLY_STOP),
            lgb.log_evaluation(period=LGB_VERBOSE)
        ]
    )

    preds = booster.predict(valid_df[PREDICTORS].values)
    auc   = roc_auc_score(valid_df[TARGET].values, preds)
    print(f"[train] LightGBM      validation AUC: {auc:.4f}")

    model = {"booster": booster, "type": "lightgbm"}
    return model, auc


#  Model selection + persistence

def select_best_model(results: dict):
    """
    Given a dict of { model_name: (model_obj, auc) },
    return (best_name, best_model, best_auc).
    """
    print("\n[train] ── Model Comparison ──────────────")
    print(f"  {'Model':<20}  {'Val AUC':>8}")
    print("  " + "─" * 32)
    for name, (_, auc) in results.items():
        print(f"  {name:<20}  {auc:>8.4f}")

    best_name  = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    best_auc   = results[best_name][1]
    print(f"\n[train] ✓ Best model: {best_name}  (AUC = {best_auc:.4f})")
    return best_name, best_model, best_auc


def save_best_model(model_name: str, model_obj, auc: float):
    """
    Persist the best model to disk:
      - models/best_model.pkl  → the model object
      - models/model_meta.json → name, auc, timestamp
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump({"name": model_name, "model": model_obj}, f)

    meta = {
        "model_name":  model_name,
        "val_auc":     round(auc, 6),
        "trained_at":  datetime.utcnow().isoformat() + "Z",
        "predictors":  PREDICTORS,
        "target":      TARGET,
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[train] Model saved  → {BEST_MODEL_PATH}")
    print(f"[train] Meta saved   → {MODEL_META_PATH}")


#  Main entry point

def run_training(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """
    Train all four models, select best by validation AUC,
    save to disk, and return the best model object.
    """
    results = {}

    results["RandomForest"] = train_random_forest(train_df, valid_df)
    results["AdaBoost"]     = train_adaboost(train_df, valid_df)
    results["XGBoost"]      = train_xgboost(train_df, valid_df)
    results["LightGBM"]     = train_lightgbm(train_df, valid_df)

    best_name, best_model, best_auc = select_best_model(results)
    save_best_model(best_name, best_model, best_auc)

    return best_name, best_model, best_auc


#  Standalone run 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_data, add_hour_feature, split_data

    df                          = load_data()
    df                          = add_hour_feature(df)
    train_df, valid_df, test_df = split_data(df)

    best_name, best_model, best_auc = run_training(train_df, valid_df)
    print(f"\n[train] Done. Best: {best_name}  AUC={best_auc:.4f}")
