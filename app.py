"""Interactive fraud-detection demo.

Two experiences on top of the trained KNN / XGBoost / LSTM ensemble:
  1. single-transaction check — load a real test transaction (or edit it) and
     watch the three models vote;
  2. "a day at the bank" simulation — stream thousands of unseen transactions
     through the ensemble and see what gets caught, missed and falsely flagged,
     with an adjustable decision threshold.

Run:  python app.py   ->  http://127.0.0.1:7860
"""

import time

import gradio as gr
import numpy as np
import pandas as pd

from src.artifacts import MODELS_DIR, RANDOM_STATE, load_or_train, load_split
from src.models import ensemble_proba, lstm_predict_proba

DEFAULT_THRESHOLD = 0.5

print("loading data...")
X_train, X_test, y_train, y_test = load_split()
FEATURES = list(X_test.columns)
y_test_arr = y_test.values

knn, xgb, lstm = load_or_train(X_train=X_train, y_train=y_train)

# Score the whole held-out test set once at startup: the simulation tab then
# answers threshold changes instantly, and we get an honest throughput number.
PROBAS_PATH = MODELS_DIR / "test_ensemble_probas.npy"
if PROBAS_PATH.exists():
    ENS_PROBAS = np.load(PROBAS_PATH)
    SCAN_RATE = None
else:
    print("scoring full test set once (a few seconds)...")
    t0 = time.time()
    p_knn_all = knn.predict_proba(X_test)[:, 1]
    p_xgb_all = xgb.predict_proba(X_test)[:, 1]
    p_lstm_all = lstm_predict_proba(lstm, X_test)
    elapsed = time.time() - t0
    ENS_PROBAS = ensemble_proba(p_knn_all, p_xgb_all, p_lstm_all)
    np.save(PROBAS_PATH, ENS_PROBAS)
    SCAN_RATE = len(X_test) / elapsed
    print(f"scored {len(X_test):,} tx in {elapsed:.1f}s ({SCAN_RATE:,.0f} tx/s)")

rng = np.random.default_rng()
legit_idx = np.where(y_test_arr == 0)[0]
fraud_idx = np.where(y_test_arr == 1)[0]

EXPLAINER = """
### How does this work in the real world?

**1. You swipe your card** (or pay online) → **2.** the bank's system instantly assembles
the transaction data (amount, time, location, merchant, deviation from your habits...) →
**3.** the model returns a fraud probability within milliseconds → **4.** the system decides:
allow ✅ / ask for an SMS confirmation 📲 / block and call the cardholder 🚨.

Nobody types anything by hand — the model is an invisible customs officer that **every**
transaction passes through. This demo simulates exactly that: the "random transaction"
button means "a new transaction just arrived in the system".

### Why do the columns have strange names (V1–V28)?

The dataset contains **real transactions** by European cardholders, so before publishing
it the bank ran everything through a mathematical transformation (PCA) that protects
privacy. V1–V28 are "encrypted" combinations of the real fields — models work well on
them, but only the bank holds the key. That's why this demo loads a real transaction you
can then modify, instead of offering an "amount + location" form.

### Who makes the call?

Three different models vote independently and their probabilities are averaged (an
**ensemble**):
- **KNN** — "who does this transaction resemble?" (compares against thousands of known cases)
- **XGBoost** — a forest of decision trees, the strongest single model
- **LSTM** — a neural network that reads the features as a sequence

They are trained on a set where fraud was synthetically amplified (SMOTEENN), because the
raw data has just 1 fraud per 578 transactions — too few to learn from. **The test
transactions you see here were never shown to the models**, and their fraud ratio is the
real one (0.17%).

### What do the simulation numbers mean?

- **Caught fraud** — the model raised an alert, and it really was fraud ✔
- **Missed fraud** — slipped under the radar (a direct cost to the bank)
- **False alarms** — a legitimate purchase blocked (an annoyed customer at the register)

The **decision threshold** is a business decision, not a mathematical one: a stricter
customs officer (low threshold) catches more fraud but hassles more innocent customers —
move the slider and watch the balance shift.
"""


# ---------- tab 1: single transaction ----------

def sample_transaction(kind: str):
    if kind == "fraud":
        i = int(rng.choice(fraud_idx))
    elif kind == "legit":
        i = int(rng.choice(legit_idx))
    else:
        i = int(rng.integers(len(X_test)))
    row = X_test.iloc[[i]].round(4)
    true = int(y_test_arr[i])
    return (
        row,
        true,
        f"📥 Transaction **#{i}** just arrived from the test set — the models have never "
        "seen it, and you learn the truth only after the check.",
        "",           # clear previous verdict
        None,         # clear previous label
        pd.DataFrame(),  # clear previous votes
    )


def predict(table: pd.DataFrame, true_label: int):
    x = table.iloc[[0]][FEATURES].astype(float)

    p_knn = float(knn.predict_proba(x)[0, 1])
    p_xgb = float(xgb.predict_proba(x)[0, 1])
    p_lstm = float(lstm_predict_proba(lstm, x)[0])
    p_ens = float(np.mean([p_knn, p_xgb, p_lstm]))

    verdict_label = {"🚨 FRAUD": p_ens, "✅ LEGITIMATE": 1 - p_ens}

    votes = pd.DataFrame(
        {
            "model": ["KNN", "XGBoost", "LSTM", "🏛 ENSEMBLE (average)"],
            "P(fraud)": [f"{p:.1%}" for p in (p_knn, p_xgb, p_lstm, p_ens)],
            "vote": ["🚨 fraud" if p >= DEFAULT_THRESHOLD else "✅ legit" for p in (p_knn, p_xgb, p_lstm, p_ens)],
        }
    )

    decision = ("🚨 **FRAUD** — the transaction gets blocked" if p_ens >= DEFAULT_THRESHOLD
                else "✅ **LEGITIMATE** — the transaction goes through")
    truth = {
        1: "🚨 this really **was** fraud",
        0: "✅ this really **was** a legitimate purchase",
        -1: "✍️ values were edited by hand — the truth is unknown",
    }[true_label]
    correct = ""
    if true_label != -1:
        hit = (p_ens >= DEFAULT_THRESHOLD) == bool(true_label)
        correct = "→ the model got it **right** 🎯" if hit else "→ the model got it **wrong** 💥"

    summary = f"### Verdict: {decision}\n\n**Ground truth:** {truth} {correct}"
    return summary, verdict_label, votes


# ---------- tab 2: day-at-the-bank simulation ----------

def simulate(n: int, threshold: float):
    n = int(min(n, len(X_test)))
    idx = rng.choice(len(X_test), size=n, replace=False)
    return _sim_stats(idx.tolist(), threshold)


def restat(idx: list, threshold: float):
    if not idx:
        return "Run the transactions with the button above first. 👆", pd.DataFrame(), pd.DataFrame(), idx
    return _sim_stats(idx, threshold)


def _sim_stats(idx: list, threshold: float):
    idx_arr = np.asarray(idx)
    probs = ENS_PROBAS[idx_arr]
    truth = y_test_arr[idx_arr]
    flagged = probs >= threshold

    n = len(idx_arr)
    n_fraud = int(truth.sum())
    caught = int((flagged & (truth == 1)).sum())
    missed = n_fraud - caught
    false_alarms = int((flagged & (truth == 0)).sum())
    n_alerts = int(flagged.sum())

    precision = caught / n_alerts if n_alerts else 0.0
    recall = caught / n_fraud if n_fraud else 0.0

    lines = [
        f"## 🏦 {n:,} transactions just passed through the system",
        "",
        f"| | |",
        f"|---|---|",
        f"| 💳 Total transactions | **{n:,}** |",
        f"| 🕵️ Real frauds among them | **{n_fraud}** ({n_fraud / n:.2%}) |",
        f"| 🔔 Alerts raised | **{n_alerts}** |",
        f"| ✔ Frauds caught | **{caught} / {n_fraud}** |",
        f"| 👻 Frauds missed | **{missed}** |",
        f"| 😤 False alarms (innocents blocked) | **{false_alarms}** |",
        "",
    ]
    if n_fraud == 0:
        lines.append("Not a single fraud in today's sample — that's what 0.17% looks like! "
                     "Increase the number of transactions or run another day.")
    else:
        lines.append(
            f"Out of every 100 alerts, **{precision:.0%}** are real fraud (precision); "
            f"the model caught **{recall:.0%}** of all fraud (recall)."
        )
    if missed == 0 and false_alarms <= 2 and n_fraud > 0:
        lines.append("\n🏆 A great day: no fraud got through, and almost no innocent customer was bothered.")

    def tx_table(mask, limit=15):
        sel = idx_arr[mask]
        if len(sel) == 0:
            return pd.DataFrame({"—": ["no such transactions"]})
        sel_probs = ENS_PROBAS[sel]
        order = np.argsort(sel_probs)[::-1][:limit]
        return pd.DataFrame(
            {
                "transaction": [f"#{i}" for i in sel[order]],
                "P(fraud)": [f"{p:.1%}" for p in sel_probs[order]],
                "truth": ["🚨 fraud" if y_test_arr[i] else "✅ legitimate" for i in sel[order]],
            }
        )

    alerts_df = tx_table(flagged)
    missed_df = tx_table((~flagged) & (truth == 1))
    return "\n".join(lines), alerts_df, missed_df, idx


# ---------- UI ----------

with gr.Blocks(title="Fraud Detection Demo") as demo:
    gr.Markdown(
        "# 💳 Credit Card Fraud Detection\n"
        "A KNN + XGBoost + LSTM ensemble trained on 284,807 real transactions (0.17% fraud). "
        "Everything you see here are **real held-out test transactions** the models have never seen."
    )
    with gr.Accordion("ℹ️ How does this work? (click)", open=False):
        gr.Markdown(EXPLAINER)

    with gr.Tabs():
        with gr.Tab("🔍 Single-transaction check"):
            gr.Markdown(
                "Load a real transaction, then click **Check**. You can also edit any value in "
                "the table — e.g. load a fraud, push `V14` and `V17` toward zero, and watch the "
                "fraud probability drop."
            )
            true_state = gr.State(-1)
            with gr.Row():
                btn_any = gr.Button("🎲 Random transaction")
                btn_legit = gr.Button("✅ Random legitimate")
                btn_fraud = gr.Button("🚨 Random fraud")
            info = gr.Markdown("")
            table = gr.Dataframe(
                value=X_test.iloc[[0]].round(4),
                headers=FEATURES,
                label="Transaction data (V1–V28 = anonymized signals, Amount = scaled amount)",
                interactive=True,
            )
            btn_predict = gr.Button("🔍 Check the transaction", variant="primary", size="lg")
            summary = gr.Markdown("")
            with gr.Row():
                verdict = gr.Label(label="Ensemble verdict")
                votes = gr.Dataframe(label="How each model voted", interactive=False)

            outs = [table, true_state, info, summary, verdict, votes]
            btn_any.click(lambda: sample_transaction("any"), outputs=outs)
            btn_legit.click(lambda: sample_transaction("legit"), outputs=outs)
            btn_fraud.click(lambda: sample_transaction("fraud"), outputs=outs)
            table.input(lambda: (-1, "✍️ Values edited by hand — the ground truth is no longer known."),
                        outputs=[true_state, info])
            btn_predict.click(predict, inputs=[table, true_state], outputs=[summary, verdict, votes])

        with gr.Tab("🏦 A day at the bank"):
            gr.Markdown(
                "Send a wave of random transactions through the system and see the day's balance "
                "sheet: how much fraud was caught, how much slipped under the radar, and how many "
                "innocent customers were bothered. Then move the **decision threshold** — the "
                "balance recomputes instantly."
            )
            sim_state = gr.State([])
            with gr.Row():
                n_slider = gr.Slider(500, 50000, value=5000, step=500, label="Transactions per day")
                thr_slider = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                       label="Decision threshold (low = stricter officer, high = laxer)")
            btn_sim = gr.Button("▶️ Run the transactions", variant="primary", size="lg")
            sim_summary = gr.Markdown("")
            with gr.Row():
                alerts_df = gr.Dataframe(label="🔔 Alerts raised (top 15 by suspicion)", interactive=False)
                missed_df = gr.Dataframe(label="👻 Missed frauds", interactive=False)

            btn_sim.click(simulate, inputs=[n_slider, thr_slider],
                          outputs=[sim_summary, alerts_df, missed_df, sim_state])
            thr_slider.release(restat, inputs=[sim_state, thr_slider],
                               outputs=[sim_summary, alerts_df, missed_df, sim_state])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=False,
        theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate"),
    )
