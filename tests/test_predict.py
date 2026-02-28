# ─────────────────────────────────────────────
#  tests/test_predict.py
#  Run with: pytest tests/
# ─────────────────────────────────────────────

import sys
import os
import pytest
import pickle
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import PREDICTORS, TARGET


def make_sample_transaction():
    """A single transaction with all required feature keys."""
    np.random.seed(0)
    return {col: float(np.random.randn()) for col in PREDICTORS}


def make_sample_df(n=10):
    np.random.seed(0)
    return pd.DataFrame(
        {col: np.random.randn(n) for col in PREDICTORS}
    )


# ── We mock the model load so tests run without a trained model on disk ──

class MockSklearnModel:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])


@patch("predict.load_model", return_value=("MockRF", MockSklearnModel()))
def test_predict_single_returns_expected_keys(mock_load):
    from predict import predict_single
    result = predict_single(make_sample_transaction())
    assert "fraud_probability" in result
    assert "is_fraud" in result
    assert "model_used" in result
    assert "threshold_used" in result


@patch("predict.load_model", return_value=("MockRF", MockSklearnModel()))
def test_predict_single_probability_range(mock_load):
    from predict import predict_single
    result = predict_single(make_sample_transaction())
    assert 0.0 <= result["fraud_probability"] <= 1.0


@patch("predict.load_model", return_value=("MockRF", MockSklearnModel()))
def test_predict_single_missing_feature_raises(mock_load):
    from predict import predict_single
    bad_transaction = {"Time": 100.0}   # missing all V features and Amount
    with pytest.raises(ValueError, match="Missing features"):
        predict_single(bad_transaction)


@patch("predict.load_model", return_value=("MockRF", MockSklearnModel()))
def test_predict_batch_adds_columns(mock_load):
    from predict import predict_batch
    df     = make_sample_df()
    result = predict_batch(df)
    assert "fraud_probability" in result.columns
    assert "is_fraud" in result.columns
    assert len(result) == len(df), "Row count should not change"


@patch("predict.load_model", return_value=("MockRF", MockSklearnModel()))
def test_predict_batch_missing_columns_raises(mock_load):
    from predict import predict_batch
    bad_df = pd.DataFrame({"Time": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Missing features"):
        predict_batch(bad_df)


@patch("predict.load_model", return_value=("MockRF", MockSklearnModel()))
def test_threshold_logic(mock_load):
    from predict import predict_single
    txn = make_sample_transaction()
    # Mock returns prob=0.2, so threshold=0.1 → fraud, threshold=0.5 → not fraud
    r_low  = predict_single(txn, threshold=0.1)
    r_high = predict_single(txn, threshold=0.5)
    assert r_low["is_fraud"]  is True
    assert r_high["is_fraud"] is False
