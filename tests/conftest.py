import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import PREDICTORS, TARGET

@pytest.fixture
def dummy_df():
    np.random.seed(42)
    data = {col: np.random.randn(1000) for col in PREDICTORS}
    data["Time"]  = np.arange(1000) * 3600.0
    data[TARGET]  = np.random.choice([0,1], size=1000, p=[0.998, 0.002])
    return pd.DataFrame(data)

@pytest.fixture
def sample_transaction():
    np.random.seed(0)
    return {col: float(np.random.randn()) for col in PREDICTORS}