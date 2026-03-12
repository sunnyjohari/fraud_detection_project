import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

from config import (
    RAW_DATA_PATH, TARGET, PREDICTORS,
    TEST_SIZE, VALID_SIZE, RANDOM_STATE
)


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw CSV and run basic sanity checks.
    Returns the full dataframe.
    """
    print(f"[preprocess] Loading data from: {path}")
    df = pd.read_csv(path)

    print(f"[preprocess] Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    #  Check for missing values 
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"[preprocess] WARNING: {missing} missing values found — dropping rows")
        df = df.dropna()
    else:
        print("[preprocess] No missing values ✓")

    #  Class distribution 
    fraud_count = df[TARGET].sum()
    total       = len(df)
    print(f"[preprocess] Fraud: {fraud_count:,} / {total:,} "
          f"({fraud_count/total*100:.3f}%)")

    return df


def add_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Hour-of-day from the Time column (seconds from start).
    Adds a useful temporal feature for EDA and modelling.
    """
    df = df.copy()
    df["Hour"] = df["Time"].apply(lambda x: np.floor(x / 3600))
    return df


def split_data(df: pd.DataFrame):
    """
    Stratified 60/20/20 split → train / validation / test.

    Stratified on TARGET so the rare fraud class (0.17%) is
    represented proportionally in every split.

    Returns
    -------
    train_df, valid_df, test_df  — each as a pandas DataFrame
    """

    # Guard against invalid class distributions
    if df[TARGET].nunique() < 2:
        raise ValueError("Dataset must contain at least two classes")
    if df[TARGET].value_counts().min() < 2:
        raise ValueError("Not enough samples per class to stratify split")

    # Step 1: hold out test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df[TARGET]         # keeps fraud ratio intact
    )

    # Step 2: split remaining into train + validation
    train_df, valid_df = train_test_split(
        train_val_df,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=train_val_df[TARGET]
    )

    print(f"[preprocess] Train  : {len(train_df):,} rows  "
          f"| fraud: {train_df[TARGET].sum():,}")
    print(f"[preprocess] Valid  : {len(valid_df):,} rows  "
          f"| fraud: {valid_df[TARGET].sum():,}")
    print(f"[preprocess] Test   : {len(test_df):,} rows  "
          f"| fraud: {test_df[TARGET].sum():,}")

    return train_df, valid_df, test_df


def get_X_y(df: pd.DataFrame):
    """Convenience helper — returns feature matrix X and label vector y."""
    return df[PREDICTORS], df[TARGET].values


#  Standalone run 
if __name__ == "__main__":
    df                       = load_data()
    df                       = add_hour_feature(df)
    train_df, valid_df, test_df = split_data(df)

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Save splits
    train_df.to_csv("data/processed/train.csv", index=False)
    valid_df.to_csv("data/processed/valid.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("\n[preprocess] Done. Splits ready for training.")
