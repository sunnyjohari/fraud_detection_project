import json
import pytest

METRICS_PATH = "metrics.json"
AUC_THRESHOLD = 0.95

def test_auc_threshold():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    auc = metrics["roc_auc"]
    assert auc > AUC_THRESHOLD, (
        f"AUC {auc:.4f} is below threshold {AUC_THRESHOLD}. "
        f"Model quality regression detected."
    )