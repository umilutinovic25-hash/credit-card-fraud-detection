"""Interactive fraud-detection demo.

Two experiences on top of the trained KNN / XGBoost / LSTM ensemble:
  1. single-transaction check — load a real test transaction (or edit it) and
     watch the three models vote;
  2. "a day at the bank" simulation — stream thousands of unseen transactions
     through the ensemble and see what gets caught, missed and falsely flagged,
     with an adjustable decision threshold.

Run:  python app.py
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
### Kako ovo radi u stvarnom svetu?

**1. Prevučeš karticu** (ili platiš online) → **2.** bankin sistem u istom trenutku sastavi
podatke o transakciji (iznos, vreme, lokacija, prodavac, odstupanje od tvojih navika...) →
**3.** model vrati verovatnoću prevare za par milisekundi → **4.** sistem odluči:
propusti ✅ / traži SMS potvrdu 📲 / blokiraj i zovi vlasnika 🚨.

Niko ništa ne kuca ručno — model je nevidljivi carinik kroz koga prođe **svaka** transakcija.
Ovaj demo simulira baš to: dugme „nasumična transakcija" = „stigla je nova transakcija u sistem".

### Zašto kolone imaju čudna imena (V1–V28)?

Dataset sadrži **prave transakcije** evropskih korisnika kartica, pa je banka pre objavljivanja
sve provukla kroz matematičku transformaciju (PCA) koja štiti privatnost. V1–V28 su „šifrovane"
kombinacije pravih podataka — model na njima odlično radi, ali ključ za dešifrovanje ima samo banka.
Zato se ovde učitava prava transakcija koju onda možeš da menjaš, umesto unosa „iznos + lokacija".

### Ko presuđuje?

Tri različita modela glasaju nezavisno, pa se glasovi uprosečuju (**ensemble**):
- **KNN** — „na koga ova transakcija liči?" (poredi sa hiljadama poznatih slučajeva)
- **XGBoost** — šuma odlučujućih stabala, najjači pojedinačni model
- **LSTM** — neuronska mreža koja čita feature-e kao sekvencu

Trenirani su na skupu gde je prevara veštački „pojačana" (SMOTEENN), jer u sirovim podacima
na 578 transakcija dođe samo 1 prevara — premalo da se od nje uči. **Test transakcije koje ovde
vidiš modeli nikad nisu videli**, i u njima je odnos prevara realan (0,17%).

### Šta znače brojke u simulaciji?

- **Uhvaćene prevare** — model digao alarm, i stvarno jeste prevara ✔
- **Propuštene prevare** — prošlo ispod radara (direktan trošak banke)
- **Lažni alarmi** — legitimna kupovina blokirana (nervozan korisnik na kasi)

**Prag odluke** je poslovna odluka, ne matematička: strožiji carinik (nizak prag) hvata više
prevara ali maltretira više nevinih kupaca — pomeri klizač i gledaj kako se vaga pomera.
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
        f"📥 Stigla transakcija **#{i}** iz test skupa — modeli je nikad nisu videli, "
        "a istinu saznaješ tek posle provere.",
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

    verdict_label = {"🚨 PREVARA": p_ens, "✅ LEGITIMNA": 1 - p_ens}

    votes = pd.DataFrame(
        {
            "model": ["KNN", "XGBoost", "LSTM", "🏛 ENSEMBLE (prosek)"],
            "P(prevara)": [f"{p:.1%}" for p in (p_knn, p_xgb, p_lstm, p_ens)],
            "glas": ["🚨 prevara" if p >= DEFAULT_THRESHOLD else "✅ legit" for p in (p_knn, p_xgb, p_lstm, p_ens)],
        }
    )

    decision = "🚨 **PREVARA** — transakcija se blokira" if p_ens >= DEFAULT_THRESHOLD else "✅ **LEGITIMNA** — transakcija prolazi"
    truth = {
        1: "🚨 ovo **jeste** bila prevara",
        0: "✅ ovo **jeste** bila legitimna kupovina",
        -1: "✍️ vrednosti su ručno izmenjene — istina nije poznata",
    }[true_label]
    correct = ""
    if true_label != -1:
        hit = (p_ens >= DEFAULT_THRESHOLD) == bool(true_label)
        correct = "→ model je **pogodio** 🎯" if hit else "→ model je **promašio** 💥"

    summary = f"### Presuda: {decision}\n\n**Istina:** {truth} {correct}"
    return summary, verdict_label, votes


# ---------- tab 2: day-at-the-bank simulation ----------

def simulate(n: int, threshold: float):
    n = int(min(n, len(X_test)))
    idx = rng.choice(len(X_test), size=n, replace=False)
    return _sim_stats(idx.tolist(), threshold)


def restat(idx: list, threshold: float):
    if not idx:
        return "Prvo pusti transakcije dugmetom iznad. 👆", pd.DataFrame(), pd.DataFrame(), idx
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
        f"## 🏦 Prošlo je {n:,} transakcija kroz sistem",
        "",
        f"| | |",
        f"|---|---|",
        f"| 💳 Ukupno transakcija | **{n:,}** |",
        f"| 🕵️ Stvarnih prevara među njima | **{n_fraud}** ({n_fraud / n:.2%}) |",
        f"| 🔔 Alarma podignuto | **{n_alerts}** |",
        f"| ✔ Uhvaćene prevare | **{caught} / {n_fraud}** |",
        f"| 👻 Propuštene prevare | **{missed}** |",
        f"| 😤 Lažni alarmi (blokirani nevini) | **{false_alarms}** |",
        "",
    ]
    if n_fraud == 0:
        lines.append("Danas nije bilo nijedne prevare u uzorku — tako izgleda 0,17%! "
                     "Povećaj broj transakcija ili pusti novi dan.")
    else:
        lines.append(
            f"Od svakih 100 alarma, **{precision:.0%}** su stvarne prevare (preciznost); "
            f"model je uhvatio **{recall:.0%}** svih prevara (odziv)."
        )
    if missed == 0 and false_alarms <= 2 and n_fraud > 0:
        lines.append("\n🏆 Odličan dan: nijedna prevara nije prošla, a skoro niko nevin nije uznemiren.")

    def tx_table(mask, limit=15):
        sel = idx_arr[mask]
        if len(sel) == 0:
            return pd.DataFrame({"—": ["nema takvih transakcija"]})
        sel_probs = ENS_PROBAS[sel]
        order = np.argsort(sel_probs)[::-1][:limit]
        return pd.DataFrame(
            {
                "transakcija": [f"#{i}" for i in sel[order]],
                "P(prevara)": [f"{p:.1%}" for p in sel_probs[order]],
                "istina": ["🚨 prevara" if y_test_arr[i] else "✅ legitimna" for i in sel[order]],
            }
        )

    alerts_df = tx_table(flagged)
    missed_df = tx_table((~flagged) & (truth == 1))
    return "\n".join(lines), alerts_df, missed_df, idx


# ---------- UI ----------

theme = gr.themes.Soft(primary_hue="red", neutral_hue="slate")

with gr.Blocks(title="Fraud Detection Demo", theme=theme) as demo:
    gr.Markdown(
        "# 💳 Credit Card Fraud Detection\n"
        "KNN + XGBoost + LSTM ensemble treniran na 284.807 pravih transakcija (0,17% prevara). "
        "Sve što vidiš ovde su **prave transakcije iz test skupa** koje modeli nikad nisu videli."
    )
    with gr.Accordion("ℹ️ Kako ovo radi? (klikni)", open=False):
        gr.Markdown(EXPLAINER)

    with gr.Tabs():
        with gr.Tab("🔍 Pojedinačna provera"):
            gr.Markdown(
                "Učitaj pravu transakciju, pa klikni **Proveri**. Možeš i ručno izmeniti bilo koju "
                "vrednost u tabeli — npr. učitaj prevaru pa gurni `V14` i `V17` ka nuli i gledaj "
                "kako verovatnoća prevare pada."
            )
            true_state = gr.State(-1)
            with gr.Row():
                btn_any = gr.Button("🎲 Nasumična transakcija")
                btn_legit = gr.Button("✅ Nasumična legitimna")
                btn_fraud = gr.Button("🚨 Nasumična prevara")
            info = gr.Markdown("")
            table = gr.Dataframe(
                value=X_test.iloc[[0]].round(4),
                headers=FEATURES,
                label="Podaci o transakciji (V1–V28 = anonimizovani signali, Amount = skaliran iznos)",
                interactive=True,
            )
            btn_predict = gr.Button("🔍 Proveri transakciju", variant="primary", size="lg")
            summary = gr.Markdown("")
            with gr.Row():
                verdict = gr.Label(label="Ensemble presuda")
                votes = gr.Dataframe(label="Kako je ko glasao", interactive=False)

            outs = [table, true_state, info, summary, verdict, votes]
            btn_any.click(lambda: sample_transaction("any"), outputs=outs)
            btn_legit.click(lambda: sample_transaction("legit"), outputs=outs)
            btn_fraud.click(lambda: sample_transaction("fraud"), outputs=outs)
            table.input(lambda: (-1, "✍️ Vrednosti izmenjene ručno — istina više nije poznata."), outputs=[true_state, info])
            btn_predict.click(predict, inputs=[table, true_state], outputs=[summary, verdict, votes])

        with gr.Tab("🏦 Simulacija dana u banci"):
            gr.Markdown(
                "Pusti talas nasumičnih transakcija kroz sistem i vidi bilans dana: koliko je prevara "
                "uhvaćeno, koliko je prošlo ispod radara i koliko je nevinih kupaca uznemireno. "
                "Zatim pomeraj **prag odluke** — bilans se preračunava odmah."
            )
            sim_state = gr.State([])
            with gr.Row():
                n_slider = gr.Slider(500, 50000, value=5000, step=500, label="Broj transakcija u danu")
                thr_slider = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                       label="Prag odluke (nisko = strožiji carinik, visoko = blaži)")
            btn_sim = gr.Button("▶️ Pusti transakcije", variant="primary", size="lg")
            sim_summary = gr.Markdown("")
            with gr.Row():
                alerts_df = gr.Dataframe(label="🔔 Podignuti alarmi (top 15 po sumnjivosti)", interactive=False)
                missed_df = gr.Dataframe(label="👻 Propuštene prevare", interactive=False)

            btn_sim.click(simulate, inputs=[n_slider, thr_slider],
                          outputs=[sim_summary, alerts_df, missed_df, sim_state])
            thr_slider.release(restat, inputs=[sim_state, thr_slider],
                               outputs=[sim_summary, alerts_df, missed_df, sim_state])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
