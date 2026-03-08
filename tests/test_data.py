import pandas as pd
import pytest

DATA_PATH = "data/creditcard.csv"

def test_data_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape[1] == 31, f"Expected 31 columns, got {df.shape[1]}"
    assert len(df) > 200000, "Dataset too small — may be corrupted"

def test_no_nulls():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0, "Nulls found in dataset"

def test_class_ratio():
    df = pd.read_csv(DATA_PATH)
    fraud_pct = df["Class"].mean()
    assert 0.001 < fraud_pct < 0.005, f"Fraud ratio {fraud_pct:.4f} out of expected range"