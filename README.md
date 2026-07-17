# Credit Card Fraud Detection

End-to-end fraud detection on a **highly imbalanced dataset (≈0.17% fraud)** —
taken further than a notebook:

> **F1 0.82** (tuned ensemble, held-out test set with the true fraud distribution) ·
> **FastAPI scoring service, measured p50 latency 3.5 ms** with pytest contract tests ·
> **interactive demo** that simulates a day of bank transactions with a live
> decision-threshold slider · an honest [limitations section](#limitations--what-production-would-need)
> covering exactly what separates this from a production system.

Everyone has trained a model on this dataset; this repo also **serves** it,
**tests** it, **measures** it and knows its own limits.

## Highlights

- **Imbalanced classification done right**: accuracy is meaningless at 0.17% fraud,
  so everything is optimized around precision, recall, F1 and PR-AUC.
- **SMOTEENN resampling** — SMOTE oversampling of the minority class combined with
  Edited Nearest Neighbours cleaning of the noisy class boundary, applied to the
  training set only.
- **Three architecturally different voters**: distance-based (KNN), gradient-boosted
  trees (XGBoost — the industry standard for tabular fraud) and a recurrent network
  (LSTM, PyTorch) whose errors are decorrelated from the tree model's.
- **Soft-voting ensemble** that averages predicted probabilities for robustness.
- **Threshold tuning** on the precision–recall curve — the operating point is
  treated as a business decision, not a fixed 0.5.

## Results (held-out test set, true 0.17% fraud distribution)

| model    | precision | recall | F1     | ROC-AUC | PR-AUC |
|----------|-----------|--------|--------|---------|--------|
| KNN      | 0.401     | 0.832  | 0.541  | 0.920   | 0.551  |
| XGBoost  | 0.740     | 0.779  | 0.759  | 0.971   | 0.783  |
| LSTM     | 0.672     | 0.821  | 0.739  | 0.966   | 0.679  |
| **Ensemble** | **0.778** | **0.811** | **0.794** | 0.968 | 0.779 |

Threshold tuning on the ensemble's precision–recall curve raises F1 to **0.82**.
Full analysis with PR curves and confusion matrices:
[`notebooks/fraud_detection.ipynb`](notebooks/fraud_detection.ipynb)

## Real-time scoring API

```bash
uvicorn api:app --port 8000     # interactive docs at http://127.0.0.1:8000/docs
```

The production shape of a fraud model: the payment switch POSTs transaction
features, the service answers with a probability and an allow/block decision.

```bash
# grab a real held-out transaction as a payload, then score it
curl -s "http://127.0.0.1:8000/example?kind=fraud" | jq .payload > tx.json
curl -s -X POST http://127.0.0.1:8000/score \
     -H 'Content-Type: application/json' -d @tx.json
# {"fraud_probability":0.9994,"decision":"block","threshold":0.5,
#  "model_probabilities":{"knn":1.0,"xgboost":0.9997,"lstm":0.9984},"latency_ms":2.9}
```

- **Measured end-to-end latency** (local, 200 sequential requests, full
  KNN+XGBoost+LSTM ensemble): **p50 3.5 ms, p95 3.8 ms** — comfortably inside a
  real-time authorization budget (~100 ms).
- `POST /score/batch` scores up to 10k transactions per call (backfills,
  offline re-scoring); `GET /example` serves real test payloads for demos.
- Payloads are validated against the exact 29-feature contract (pydantic),
  and the OpenAPI schema doubles as documentation.
- Contract tests in [`tests/test_api.py`](tests/test_api.py) (`pytest`).

## Interactive demo

```bash
python app.py   # http://127.0.0.1:7860
```

A Gradio app with two experiences:

- **Single-transaction check** — load a real held-out transaction (or edit any
  feature value) and watch the three models vote, then reveal the ground truth.
- **"A day at the bank" simulation** — stream thousands of unseen transactions
  through the ensemble and see caught frauds, missed frauds and false alarms,
  with a live decision-threshold slider showing the precision–recall trade-off.

Models are trained and cached on first launch (a few minutes), then start instantly.

## Project structure

```
├── api.py                      # FastAPI real-time scoring service
├── app.py                      # interactive Gradio demo
├── notebooks/
│   └── fraud_detection.ipynb   # full analysis, executed with outputs
├── src/
│   ├── data.py                 # loading, dedup, scaling, split, SMOTEENN
│   ├── models.py               # KNN, XGBoost, LSTM (PyTorch), ensemble
│   ├── evaluate.py             # metrics, PR curves, confusion matrices, threshold tuning
│   └── artifacts.py            # train-once/load-fast model persistence
├── tests/
│   └── test_api.py             # API contract tests
├── data/                       # dataset cache (auto-downloaded, not committed)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb
```

The dataset (284,807 transactions) is downloaded automatically from
[OpenML](https://www.openml.org/d/1597) on first run and cached in `data/`.

## Limitations — what production would need

This project covers the full ML lifecycle honestly, on what public data allows.
A real bank deployment would additionally require:

- **Feature engineering as the main value driver.** V1–V28 are pre-made PCA
  components. Production systems live off behavioral signals a bank computes
  itself: transaction velocity (count/amount per hour), deviation from the
  cardholder's habits, device fingerprint, merchant risk, geo-impossibility.
- **Serving at scale.** The measured 3.5 ms is a local single-process number.
  Production means thousands of requests/sec against a feature store with fresh
  aggregates — and KNN would be the first model cut (it scans the training set
  per query); XGBoost would carry the load.
- **Concept drift.** Fraudsters adapt; a model trained once decays in months.
  Production needs monitoring, scheduled retraining and champion/challenger
  evaluation — none of which a static dataset can exercise.
- **Delayed, selective labels.** Banks learn about missed fraud weeks later
  (chargebacks) and never learn the truth about transactions they blocked. This
  dataset's clean labels sidestep a hard research problem.
- **Tiered decisions.** Real systems don't just allow/block — they can step up
  authentication (3-D Secure), and the cost of an error scales with the amount.
- **Explainability & rules.** Declines must be explainable to customers and
  regulators; an ML score always runs alongside a human-maintained rule engine.
- **About the LSTM**: here it serves as an architecturally different third voter
  over a fixed feature vector. A production sequence model would instead consume
  the *cardholder's transaction history* — that is where recurrence actually
  earns its keep.

## Dataset

[Credit Card Fraud Detection](https://www.openml.org/d/1597) — transactions made by
European cardholders in September 2013. Features V1–V28 are anonymized PCA
components plus the transaction `Amount`; 492 of 284,807 transactions are fraud.
