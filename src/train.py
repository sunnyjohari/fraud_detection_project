#  evaluate.py  —  Metrics + visualisations
#
#  Works with any model type saved by train.py:
#  sklearn  (RandomForest, AdaBoost)
#  xgboost  (native Booster wrapped in dict)
#  lightgbm (native Booster wrapped in dict)

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from config import PREDICTORS, TARGET, REPORT_DIR, MODEL_META_PATH


def _predict(model_obj, X: pd.DataFrame) -> np.ndarray:
    """
    Unified predict_proba that handles all model types.
    Always returns 1-D array of fraud probability scores.
    """
    if isinstance(model_obj, dict):
        model_type = model_obj["type"]
        booster    = model_obj["booster"]

        if model_type == "xgboost":
            dmatrix = xgb.DMatrix(X)
            return booster.predict(dmatrix)             # already probabilities

        elif model_type == "lightgbm":
            return booster.predict(X.values)            # already probabilities

    else:
        # sklearn API  — predict_proba[:, 1]
        return model_obj.predict_proba(X)[:, 1]


def evaluate_on_test(model_obj, test_df: pd.DataFrame, threshold: float = 0.5):
    """
    Full evaluation on the held-out test set.
    Prints:
      - ROC-AUC
      - Average Precision (handles imbalance better than accuracy)
      - Classification report (precision, recall, F1)
      - Confusion matrix values
    Returns: dict of metric values
    """
    X    = test_df[PREDICTORS]
    y    = test_df[TARGET].values

    probs  = _predict(model_obj, X)
    preds  = (probs >= threshold).astype(int)

    auc    = roc_auc_score(y, probs)
    ap     = average_precision_score(y, probs)
    cm     = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n[evaluate] ── Test Set Results ───────────")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  Average Precision : {ap:.4f}")
    print(f"  Threshold used    : {threshold}")
    print(f"  True  Positives   : {tp}   (fraud correctly caught)")
    print(f"  False Positives   : {fp}   (legit flagged as fraud)")
    print(f"  False Negatives   : {fn}   (fraud missed ← minimize this)")
    print(f"  True  Negatives   : {tn}")
    print("\n" + classification_report(y, preds,
          target_names=["Not Fraud", "Fraud"]))

    return {
        "roc_auc": round(auc, 6),
        "avg_precision": round(ap, 6),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }


def plot_confusion_matrix(model_obj, test_df: pd.DataFrame,
                          threshold: float = 0.5, save: bool = True):
    """Heatmap of the confusion matrix on test data."""
    X     = test_df[PREDICTORS]
    y     = test_df[TARGET].values
    probs = _predict(model_obj, X)
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(y, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
        cmap="Blues",
        linewidths=0.5,
        linecolor="navy",
        ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    if save:
        _save_figure(fig, "confusion_matrix.png")
    plt.show()


def plot_roc_curve(model_obj, test_df: pd.DataFrame, save: bool = True):
    """ROC curve with AUC annotation."""
    X     = test_df[PREDICTORS]
    y     = test_df[TARGET].values
    probs = _predict(model_obj, X)
    auc   = roc_auc_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC Curve  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Fraud Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save:
        _save_figure(fig, "roc_curve.png")
    plt.show()


def plot_precision_recall(model_obj, test_df: pd.DataFrame, save: bool = True):
    """
    Precision-Recall curve.
    More informative than ROC when data is highly imbalanced (ours is 0.17%).
    """
    X     = test_df[PREDICTORS]
    y     = test_df[TARGET].values
    probs = _predict(model_obj, X)
    ap    = average_precision_score(y, probs)
    prec, rec, _ = precision_recall_curve(y, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="tomato", lw=2,
            label=f"PR Curve  (AP = {ap:.4f})")
    ax.set_xlabel("Recall  (fraction of all frauds caught)")
    ax.set_ylabel("Precision  (fraction of flags that are real fraud)")
    ax.set_title("Precision-Recall Curve — Fraud Detection")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save:
        _save_figure(fig, "precision_recall.png")
    plt.show()


def plot_feature_importance(model_obj, model_name: str,
                            top_n: int = 15, save: bool = True):
    """
    Bar chart of top-N feature importances.
    Handles sklearn, XGBoost, and LightGBM models.
    """
    if isinstance(model_obj, dict):
        booster    = model_obj["booster"]
        model_type = model_obj["type"]

        if model_type == "xgboost":
            scores = booster.get_fscore()        # dict feature → score
            imp_df = (pd.DataFrame(list(scores.items()),
                                   columns=["Feature", "Importance"])
                        .sort_values("Importance", ascending=False)
                        .head(top_n))

        elif model_type == "lightgbm":
            imp_df = pd.DataFrame({
                "Feature":    booster.feature_name(),
                "Importance": booster.feature_importance(importance_type="gain"),
            }).sort_values("Importance", ascending=False).head(top_n)

    else:
        # sklearn
        imp_df = (pd.DataFrame({
            "Feature":    PREDICTORS,
            "Importance": model_obj.feature_importances_,
        }).sort_values("Importance", ascending=False).head(top_n))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=imp_df,
                palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()

    if save:
        _save_figure(fig, f"feature_importance_{model_name.lower()}.png")
    plt.show()


def save_metrics(metrics: dict, model_name: str):
    """Append evaluation metrics to model_meta.json."""
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH, "r") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["test_metrics"] = metrics
    meta["model_name"]   = model_name

    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[evaluate] Metrics saved → {MODEL_META_PATH}")


def _save_figure(fig, filename: str):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[evaluate] Plot saved → {path}")


#  Standalone run 
if __name__ == "__main__":
    import sys, pickle
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_data, add_hour_feature, split_data
    from config import BEST_MODEL_PATH

    df                          = load_data()
    df                          = add_hour_feature(df)
    _, _, test_df               = split_data(df)

    with open(BEST_MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model_name = saved["name"]
    model_obj  = saved["model"]
    print(f"[evaluate] Loaded model: {model_name}")

    metrics = evaluate_on_test(model_obj, test_df)
    plot_confusion_matrix(model_obj, test_df)
    plot_roc_curve(model_obj, test_df)
    plot_precision_recall(model_obj, test_df)
    plot_feature_importance(model_obj, model_name)
    save_metrics(metrics, model_name)
    # Save lightweight metrics.json for DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[evaluate] Metrics saved → metrics.json")