"""Train-once, load-fast model persistence shared by the demo app and the API."""

from pathlib import Path

import joblib
import torch

from .data import load_data, preprocess, resample, split
from .models import LSTMClassifier, make_knn, make_xgb, train_lstm

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


def load_split(random_state: int = RANDOM_STATE):
    """The same deterministic split the notebook and demo use."""
    return split(preprocess(load_data()), random_state=random_state)


def load_or_train(random_state: int = RANDOM_STATE, X_train=None, y_train=None):
    """Load cached models, training and caching them on first use.

    Pass (X_train, y_train) to skip re-deriving the split when the caller
    already has it; only needed when the cache is cold.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    knn_path = MODELS_DIR / "knn.joblib"
    xgb_path = MODELS_DIR / "xgb.json"
    lstm_path = MODELS_DIR / "lstm.pt"

    if knn_path.exists() and xgb_path.exists() and lstm_path.exists():
        knn = joblib.load(knn_path)
        xgb = make_xgb(random_state)
        xgb.load_model(xgb_path)
        lstm = LSTMClassifier()
        lstm.load_state_dict(torch.load(lstm_path, map_location="cpu"))
        lstm.eval()
        return knn, xgb, lstm

    print("training models (first run only, a few minutes)...")
    if X_train is None:
        X_train, _, y_train, _ = load_split(random_state)
    X_res, y_res = resample(X_train, y_train, random_state=random_state)
    knn = make_knn().fit(X_res, y_res)
    xgb = make_xgb(random_state).fit(X_res, y_res)
    lstm = train_lstm(X_res, y_res, epochs=5, seed=random_state)
    joblib.dump(knn, knn_path)
    xgb.save_model(xgb_path)
    torch.save(lstm.to("cpu").state_dict(), lstm_path)
    return knn, xgb, lstm
