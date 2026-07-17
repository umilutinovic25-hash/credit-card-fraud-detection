"""API contract tests. Run with: pytest

Uses the cached models (trains them on first ever run), so the first
invocation on a fresh clone takes a few minutes; afterwards seconds.
"""

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_example_and_score_roundtrip():
    payload = client.get("/example", params={"kind": "fraud"}).json()["payload"]
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"allow", "block"}
    assert set(body["model_probabilities"]) == {"knn", "xgboost", "lstm"}
    assert body["latency_ms"] > 0


def test_known_fraud_scores_high():
    # across a handful of real frauds, the ensemble should flag most of them
    decisions = []
    for _ in range(5):
        payload = client.get("/example", params={"kind": "fraud"}).json()["payload"]
        decisions.append(client.post("/score", json=payload).json()["decision"])
    assert decisions.count("block") >= 3


def test_validation_rejects_incomplete_payload():
    r = client.post("/score", json={"Amount": 1.0})
    assert r.status_code == 422


def test_batch():
    payloads = [client.get("/example").json()["payload"] for _ in range(3)]
    r = client.post("/score/batch", json=payloads)
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_batch_rejects_empty():
    assert client.post("/score/batch", json=[]).status_code == 400
