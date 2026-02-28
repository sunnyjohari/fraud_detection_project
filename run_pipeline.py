#  Usage:
#    python run_pipeline.py --threshold 0.3

import sys
import os
import argparse

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_data, add_hour_feature, split_data
from train      import run_training
from evaluate   import (
    evaluate_on_test,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_feature_importance,
    save_metrics,
)


def main(threshold: float = 0.5):
    print("\n" + "═" * 55)
    print("  FRAUD DETECTION — Full Training Pipeline")
    print("═" * 55)

    #  Step 1: Load & preprocess 
    print("\n[pipeline] Step 1/4 — Preprocessing")
    df                          = load_data()
    df                          = add_hour_feature(df)
    train_df, valid_df, test_df = split_data(df)

    #  Step 2: Train all models, pick best 
    print("\n[pipeline] Step 2/4 — Training")
    best_name, best_model, val_auc = run_training(train_df, valid_df)

    #  Step 3: Evaluate on held-out test set 
    print("\n[pipeline] Step 3/4 — Evaluation")
    metrics = evaluate_on_test(best_model, test_df, threshold=threshold)
    save_metrics(metrics, best_name)

    #  Step 4: Generate plots 
    print("\n[pipeline] Step 4/4 — Generating reports")
    plot_confusion_matrix(best_model, test_df, threshold=threshold)
    plot_roc_curve(best_model, test_df)
    plot_precision_recall(best_model, test_df)
    plot_feature_importance(best_model, best_name)

    #  Summary 
    print("\n" + "═" * 55)
    print("  Pipeline complete")
    print(f"  Best model  : {best_name}")
    print(f"  Val AUC     : {val_auc:.4f}")
    print(f"  Test AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Avg Prec    : {metrics['avg_precision']:.4f}")
    print(f"  Frauds caught (TP) : {metrics['tp']}")
    print(f"  Frauds missed (FN) : {metrics['fn']}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for fraud classification (default: 0.5)"
    )
    args = parser.parse_args()
    main(threshold=args.threshold)
