"""Data loading and preprocessing for the credit card fraud dataset."""

from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.parquet"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset from a local parquet cache, downloading from OpenML on first use."""
    if path.exists():
        return pd.read_parquet(path)
    dataset = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    df = dataset.frame
    df["Class"] = df["Class"].astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate transactions and scale the Amount column.

    The V1-V28 features are PCA components and already standardized;
    Amount is heavily right-skewed, so a RobustScaler (median/IQR) is used.
    """
    df = df.drop_duplicates().copy()
    df["Amount"] = RobustScaler().fit_transform(df[["Amount"]])
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split keeping the fraud ratio identical in both sets."""
    X = df.drop(columns="Class")
    y = df["Class"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def resample(X_train, y_train, sampling_strategy: float = 0.1, random_state: int = 42):
    """Balance the training set with SMOTEENN.

    SMOTE oversamples the minority class up to `sampling_strategy` times the
    majority class, then Edited Nearest Neighbours removes ambiguous samples
    near the class boundary. Applied to the training set only, so the test
    set keeps the true 0.17% fraud distribution.
    """
    sampler = SMOTEENN(
        smote=SMOTE(sampling_strategy=sampling_strategy, random_state=random_state),
        enn=EditedNearestNeighbours(n_neighbors=3, n_jobs=-1),
        random_state=random_state,
    )
    return sampler.fit_resample(X_train, y_train)
