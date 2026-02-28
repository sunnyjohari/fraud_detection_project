# ─────────────────────────────────────────────
#  tests/test_preprocess.py
#  Run with: pytest tests/
# ─────────────────────────────────────────────

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import add_hour_feature, split_data
from config import TARGET, PREDICTORS, TEST_SIZE, VALID_SIZE


def make_dummy_df(n=1000):
    """Create a minimal synthetic dataframe matching the real schema."""
    np.random.seed(42)
    data = {col: np.random.randn(n) for col in PREDICTORS}
    data["Time"]  = np.arange(n) * 3600.0
    data[TARGET]  = np.random.choice([0, 1], size=n, p=[0.998, 0.002])
    return pd.DataFrame(data)


def test_add_hour_feature():
    df   = make_dummy_df()
    out  = add_hour_feature(df)
    assert "Hour" in out.columns, "Hour column should be added"
    assert out["Hour"].iloc[0] == 0.0
    assert out["Hour"].iloc[1] == 1.0


def test_split_sizes():
    df = make_dummy_df(n=1000)
    train_df, valid_df, test_df = split_data(df)
    total = len(train_df) + len(valid_df) + len(test_df)
    assert total == len(df), "No rows should be lost in the split"


def test_split_no_target_leakage():
    """All three splits should have the TARGET column."""
    df = make_dummy_df()
    train_df, valid_df, test_df = split_data(df)
    for split, name in [(train_df, "train"), (valid_df, "valid"), (test_df, "test")]:
        assert TARGET in split.columns, f"{name} missing target column"


def test_all_predictors_present():
    df = make_dummy_df()
    train_df, valid_df, test_df = split_data(df)
    for split, name in [(train_df, "train"), (valid_df, "valid"), (test_df, "test")]:
        for col in PREDICTORS:
            assert col in split.columns, f"{name} missing predictor: {col}"
