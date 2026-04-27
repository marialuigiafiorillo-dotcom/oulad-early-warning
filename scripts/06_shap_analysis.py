import sys
import os

# evita errore "No module named src"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

print("Caricamento dati...")

# carica dataset
data = pd.read_csv("data/processed/modeling_dataset_early.csv")

# separa feature e target
X = data.drop(columns=["failed"])
y = data["failed"]

print("Caricamento modello...")

# carica modello
model = joblib.load("results/models/boosting_model.pkl")

print("Preparazione dati...")

# usa solo variabili numeriche 
X_numeric = X.select_dtypes(include=["number"])

print("Calcolo SHAP...")

# crea explainer
explainer = shap.Explainer(model.predict, X_numeric)

# calcolo shap values
shap_values = explainer(X_numeric)

print("Calcolo predizioni...")

# probabilità di fallimento (classe 1)
y_pred = model.predict_proba(X)[:, 1]

# aggiungi al dataset
data["pred"] = y_pred

print("Selezione studenti...")

# studente più a rischio
i = data["pred"].idxmax()

# studente meno a rischio
j = data["pred"].idxmin()

print(f"Studente rischio alto: {i} - Probabilità: {data.loc[i, 'pred']:.3f}")
print(f"Studente rischio basso: {j} - Probabilità: {data.loc[j, 'pred']:.3f}")

# crea cartella risultati se non esiste
os.makedirs("results", exist_ok=True)


# SUMMARY PLOT
print("Salvataggio summary plot...")

shap.summary_plot(shap_values, X_numeric, show=False)
plt.savefig("results/shap_summary.png", bbox_inches='tight')
plt.clf()

# WATERFALL - RISCHIO ALTO
print("Salvataggio waterfall (rischio alto)...")

shap.plots.waterfall(shap_values[i], show=False)
plt.savefig("results/waterfall_rischio.png", bbox_inches='tight')
plt.clf()

# WATERFALL - RISCHIO BASSO
print("Salvataggio waterfall (rischio basso)...")

shap.plots.waterfall(shap_values[j], show=False)
plt.savefig("results/waterfall_ok.png", bbox_inches='tight')
plt.clf()


# DEPENDENCE PLOT - SCORE
print("Salvataggio dependence plot (score)...")

shap.dependence_plot(
    "first_assessment_score_norm",
    shap_values.values,
    X_numeric,
    show=False
)

plt.savefig("results/dependence_score.png", bbox_inches='tight')
plt.clf()


# DEPENDENCE PLOT - CLICKS
print("Salvataggio dependence plot (clicks)...")

shap.dependence_plot(
    "total_clicks",
    shap_values.values,
    X_numeric,
    show=False
)

plt.savefig("results/dependence_clicks.png", bbox_inches='tight')
plt.clf()

print("Tutti i grafici salvati in /results/")
print("Fine")