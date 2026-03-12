#  Run with: pytest tests/

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from src.preprocess import add_hour_feature, split_data
from src import preprocess
from src.config import TARGET, PREDICTORS, TEST_SIZE, VALID_SIZE


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

# More edge test cases

def test_split_stratified_fraud_ratio():
    df = make_dummy_df(n=2000)
    train_df, valid_df, test_df = split_data(df)
    for split, name in [(train_df,"train"),(valid_df,"valid"),(test_df,"test")]:
        ratio = split["Class"].mean()
        assert ratio < 0.01, f"{name} fraud ratio {ratio:.4f} suspiciously high"

def test_hour_feature_no_mutation():
    df  = make_dummy_df()
    original_cols = list(df.columns)
    _   = add_hour_feature(df)
    assert list(df.columns) == original_cols, "add_hour_feature mutated the input df"

def test_get_X_y():
    df = make_dummy_df(n=50)
    X, y = preprocess.get_X_y(df)
    assert set(X.columns) == set(PREDICTORS)
    assert len(y) == len(df)

def test_load_data_with_missing(tmp_path):
    df = make_dummy_df(n=20)
    df.iloc[0, 0] = np.nan  # introduce missing
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    result = preprocess.load_data(path=str(csv_path))
    assert len(result) < len(df)  # dropped missing row

def test_split_data_small_df():
    df = make_dummy_df(n=50)
    with pytest.raises(ValueError):
        preprocess.split_data(df)

def test_split_data_all_one_class():
    df = make_dummy_df(n=50)
    df[TARGET] = 0  # force all non-fraud
    with pytest.raises(ValueError):
        preprocess.split_data(df)