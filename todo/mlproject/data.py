"""Chargement et decoupage des donnees."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from mlproject.config import DATA_PATH, RANDOM_STATE, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
