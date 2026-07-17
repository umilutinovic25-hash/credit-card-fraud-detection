"""Real-time fraud-scoring service.

Serves the trained KNN / XGBoost / LSTM ensemble behind a REST API — the shape
a fraud model actually takes in production: the bank's payment switch POSTs the
transaction features and gets a probability + decision back in milliseconds.

Run:   uvicorn api:app --port 8000
Docs:  http://127.0.0.1:8000/docs  (interactive OpenAPI UI)

Example:
    curl -s http://127.0.0.1:8000/example?kind=fraud    # grab a real payload
    curl -s -X POST http://127.0.0.1:8000/score \
         -H 'Content-Type: application/json' -d @payload.json
"""

import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, create_model

from src.artifacts import load_or_train, load_split
from src.models import lstm_predict_proba

THRESHOLD = 0.5

print("loading data and models...")
X_train, X_test, y_train, y_test = load_split()
FEATURES = list(X_test.columns)
knn, xgb, lstm = load_or_train(X_train=X_train, y_train=y_train)
_fraud_idx = np.where(y_test.values == 1)[0]
_legit_idx = np.where(y_test.values == 0)[0]
_rng = np.random.default_rng()

# One float field per model feature (V1..V28 + Amount), so /docs shows the
# exact contract and payloads are validated before they reach the models.
Transaction = create_model("Transaction", **{f: (float, ...) for f in FEATURES})


class ModelProbabilities(BaseModel):
    knn: float
    xgboost: float
    lstm: float


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: str  # "allow" | "block"
    threshold: float
    model_probabilities: ModelProbabilities
    latency_ms: float


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    count: int
    total_latency_ms: float


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time scoring for card transactions: KNN + XGBoost + LSTM "
    "soft-voting ensemble trained on the OpenML credit card fraud dataset.",
    version="1.0.0",
)


def _score_frame(x: pd.DataFrame) -> list[ScoreResponse]:
    start = time.perf_counter()
    p_knn = knn.predict_proba(x)[:, 1]
    p_xgb = xgb.predict_proba(x)[:, 1]
    p_lstm = lstm_predict_proba(lstm, x)
    p_ens = (p_knn + p_xgb + p_lstm) / 3
    latency = (time.perf_counter() - start) * 1000 / len(x)
    return [
        ScoreResponse(
            fraud_probability=round(float(pe), 6),
            decision="block" if pe >= THRESHOLD else "allow",
            threshold=THRESHOLD,
            model_probabilities=ModelProbabilities(
                knn=round(float(pk), 6), xgboost=round(float(px), 6), lstm=round(float(pl), 6)
            ),
            latency_ms=round(latency, 2),
        )
        for pk, px, pl, pe in zip(p_knn, p_xgb, p_lstm, p_ens)
    ]


@app.get("/health")
def health():
    return {"status": "ok", "models": ["knn", "xgboost", "lstm"], "features": len(FEATURES)}


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction):
    """Score one transaction; the payment switch would call this per swipe."""
    x = pd.DataFrame([[getattr(tx, f) for f in FEATURES]], columns=FEATURES)
    return _score_frame(x)[0]


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(txs: list[Transaction]):
    """Score up to 10k transactions in one call (offline re-scoring, backfills)."""
    if not txs:
        raise HTTPException(400, "empty batch")
    if len(txs) > 10_000:
        raise HTTPException(400, "batch limited to 10000 transactions")
    start = time.perf_counter()
    x = pd.DataFrame([[getattr(t, f) for f in FEATURES] for t in txs], columns=FEATURES)
    results = _score_frame(x)
    return BatchScoreResponse(
        results=results,
        count=len(results),
        total_latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.get("/example")
def example(kind: str = Query("any", pattern="^(any|legit|fraud)$")):
    """A real held-out transaction as a ready-to-POST payload (for demos/tests)."""
    if kind == "fraud":
        i = int(_rng.choice(_fraud_idx))
    elif kind == "legit":
        i = int(_rng.choice(_legit_idx))
    else:
        i = int(_rng.integers(len(X_test)))
    return {
        "payload": {f: round(float(v), 6) for f, v in X_test.iloc[i].items()},
        "true_label": "fraud" if int(y_test.iloc[i]) else "legit",
        "test_index": i,
    }
