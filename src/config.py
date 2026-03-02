#  config.py  —  Single source of truth
#  All constants live here. Change once, applies
#  everywhere across the project.

import os

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
RF_N_ESTIMATORS = 100
RF_N_JOBS       = 4
RF_CRITERION    = "gini"

#  XGBoost 
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eta":              0.039,
    "max_depth":        2,
    "subsample":        0.8,
    "colsample_bytree": 0.9,
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "verbosity":        0,
}
XGB_MAX_ROUNDS  = 1000
XGB_EARLY_STOP  = 50
XGB_VERBOSE     = 50

#  LightGBM 
LGB_PARAMS = {
    "boosting_type":    "gbdt",
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       7,
    "max_depth":        4,
    "min_child_samples":100,
    "max_bin":          100,
    "subsample":        0.9,
    "subsample_freq":   1,
    "colsample_bytree": 0.7,
    "min_child_weight": 0,
    "min_split_gain":   0,
    "nthread":          8,
    "verbose":          -1,
    "scale_pos_weight": 150,   # handles class imbalance
}
LGB_MAX_ROUNDS  = 1000
LGB_EARLY_STOP  = 100
LGB_VERBOSE     = 50

#  AdaBoost 
ADA_N_ESTIMATORS  = 100
ADA_LEARNING_RATE = 0.8

#  Model registry filename 
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
MODEL_META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
