"""Model definitions: KNN, XGBoost, LSTM and a soft-voting ensemble."""

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


def make_knn(n_neighbors: int = 5) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)


def make_xgb(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )


class LSTMClassifier(nn.Module):
    """Treats the 29 transaction features as a sequence of length 29.

    Each feature becomes one timestep with a single value, letting the LSTM
    learn interactions across the ordered PCA components. This mirrors the
    common sequence-model formulation for tabular fraud data.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, 1)
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :]).squeeze(-1)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_lstm(
    X_train,
    y_train,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> LSTMClassifier:
    torch.manual_seed(seed)
    device = get_device()
    model = LSTMClassifier().to(device)

    X = torch.tensor(np.asarray(X_train, dtype=np.float32))
    y = torch.tensor(np.asarray(y_train, dtype=np.float32))
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  loss={total / len(X):.4f}")
    return model


@torch.no_grad()
def lstm_predict_proba(model: LSTMClassifier, X, batch_size: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    X = torch.tensor(np.asarray(X, dtype=np.float32))
    probs = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i : i + batch_size].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def ensemble_proba(*probas: np.ndarray) -> np.ndarray:
    """Soft-voting ensemble: average of the positive-class probabilities."""
    return np.mean(probas, axis=0)
