# Credit Card Fraud Detection

End-to-end fraud detection system that identifies rare fraudulent transactions in a
**highly imbalanced dataset (≈0.17% fraud)**. Three models — KNN, XGBoost and an
LSTM — are combined in a soft-voting ensemble, with SMOTEENN resampling, data
deduplication and robust feature scaling to handle the extreme class imbalance.

## Highlights

- **Imbalanced classification done right**: accuracy is meaningless at 0.17% fraud,
  so everything is optimized around precision, recall, F1 and PR-AUC.
- **SMOTEENN resampling** — SMOTE oversampling of the minority class combined with
  Edited Nearest Neighbours cleaning of the noisy class boundary, applied to the
  training set only.
- **Three complementary models**: distance-based (KNN), gradient-boosted trees
  (XGBoost) and a recurrent network (LSTM, PyTorch, runs on Apple-Silicon GPU).
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

## Project structure

```
├── notebooks/
│   └── fraud_detection.ipynb   # full analysis, executed with outputs
├── src/
│   ├── data.py                 # loading, dedup, scaling, split, SMOTEENN
│   ├── models.py               # KNN, XGBoost, LSTM (PyTorch), ensemble
│   └── evaluate.py             # metrics, PR curves, confusion matrices, threshold tuning
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

## Dataset

[Credit Card Fraud Detection](https://www.openml.org/d/1597) — transactions made by
European cardholders in September 2013. Features V1–V28 are anonymized PCA
components plus the transaction `Amount`; 492 of 284,807 transactions are fraud.
