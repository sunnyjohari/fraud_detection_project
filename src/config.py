
#  config.py  —  Single source of truth
#  All constants live here. Change once, applies
#  everywhere across the project.

import os
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

#  Paths 
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

RAW_DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")

#  Reproducibility 
RANDOM_STATE = 2018

#  Split sizes 
TEST_SIZE  = 0.25   # 20% held-out test set
VALID_SIZE = 0.20   # 20% of remaining for validation

#  Feature definition 
TARGET = "Class"
PREDICTORS = [
    "Time", "V1",  "V2",  "V3",  "V4",  "V5",  "V6",  "V7",
    "V8",   "V9",  "V10", "V11", "V12", "V13", "V14", "V15",
    "V16",  "V17", "V18", "V19", "V20", "V21", "V22", "V23",
    "V24",  "V25", "V26", "V27", "V28", "Amount"
]

#  RandomForest 
RF_N_ESTIMATORS = params["random_forest"]["n_estimators"]
RF_N_JOBS       = 4
RF_CRITERION    = "gini"

#  XGBoost 
XGB_PARAMS = params["xgboost"]["params"]
XGB_MAX_ROUNDS  = 1000
XGB_EARLY_STOP  = 50
XGB_VERBOSE     = 50

#  LightGBM 
LGB_PARAMS = params["lightgbm"]["params"]
LGB_MAX_ROUNDS  = 1000
LGB_EARLY_STOP  = 100
LGB_VERBOSE     = 50

#  AdaBoost 
ADA_N_ESTIMATORS  = 100
ADA_LEARNING_RATE = params["adaboost"]["learning_rate"]

#  Model registry filename 
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
MODEL_META_PATH = os.path.join(MODEL_DIR, "model_meta.json")