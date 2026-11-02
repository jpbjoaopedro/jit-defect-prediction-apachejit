"""Etapa 7: otimismo da validação aleatória e sensibilidade da janela de latência W."""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MOD = ["RL", "RF", "LGBM"]

# --- (a) 10-fold aleatório x walk-forward (W = 90) ---
cv = pd.concat([pd.read_csv(f) for f in glob.glob("resultados/cv/*.csv")])
wf = pd.read_csv("resultados/metricas_w90.csv")
a = cv.groupby(["projeto", "modelo"])[["auc_roc", "auc_pr"]].mean()
b = wf[wf.modelo.isin(MOD)].groupby(["projeto", "modelo"])[["auc_roc", "auc_pr"]].mean()
comp = a.join(b, lsuffix="_cv", rsuffix="_wf")
comp["dif_auc_roc"] = comp["auc_roc_cv"] - comp["auc_roc_wf"]
comp["dif_auc_pr"] = comp["auc_pr_cv"] - comp["auc_pr_wf"]
comp.round(3).to_csv("resultados/etapa7_cv_vs_walkforward.csv")
print(comp.groupby("modelo")[["auc_roc_cv", "auc_roc_wf", "dif_auc_roc", "auc_pr_cv", "auc_pr_wf", "dif_auc_pr"]].mean().round(3))
print("faixa da diferença de AUC-ROC:", comp["dif_auc_roc"].min().round(3), "a", comp["dif_auc_roc"].max().round(3))
melhor = comp.reset_index().groupby("projeto").apply(
    lambda g: pd.Series({"melhor_cv": g.loc[g.auc_pr_cv.idxmax(), "modelo"],
                         "melhor_wf": g.loc[g.auc_pr_wf.idxmax(), "modelo"]}))
melhor.to_csv("resultados/etapa7_melhor_modelo.csv")
print(melhor.value_counts())

fig, ax = plt.subplots(figsize=(6, 5))
for m, mk in zip(MOD, "os^"):
    s = comp.xs(m, level="modelo")
    ax.scatter(s["auc_pr_wf"], s["auc_pr_cv"], marker=mk, label=m)
ax.plot([0.3, 1], [0.3, 1], "k--", lw=1)
ax.set(xlabel="AUC-PR no walk-forward (W = 90)", ylabel="AUC-PR na validação cruzada 10-fold",
       title="Validação aleatória x temporal (um ponto por projeto)")
ax.legend(); fig.tight_layout(); fig.savefig("resultados/figuras/fig_cv_vs_walkforward.png", dpi=200); plt.close(fig)

# --- (b) sensibilidade de W (regressão logística) ---
ws = []
for w in (30, 90, 180, 365):
    r = pd.concat([pd.read_csv(f) for f in glob.glob(f"resultados/metricas/*_w{w}.csv")])
    r = r[r.modelo == "RL"]
    r["prev_treino"] = r["pos_treino"] / r["n_treino"]
    ws.append(r)
ws = pd.concat(ws)
sens = ws.groupby("w_dias")[["auc_roc", "auc_pr", "mcc", "brier", "recall20_dens", "prev_treino"]].mean()
sens["janelas"] = ws.groupby("w_dias").size()
sens.round(3).to_csv("resultados/etapa7_sensibilidade_w.csv")
print(sens.round(3))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for col, lab in [("auc_pr", "AUC-PR"), ("mcc", "MCC"), ("brier", "Brier")]:
    ax[0].plot(sens.index, sens[col], marker="o", label=lab)
ax[0].set(xlabel="W (dias)", title="Regressão logística por janela de latência", xticks=sens.index)
ax[0].legend()
ax[1].plot(sens.index, sens["prev_treino"] * 100, marker="o", color="#c0504d")
ax[1].set(xlabel="W (dias)", ylabel="% de indutores no treino", title="Indutores visíveis no treino", xticks=sens.index)
fig.tight_layout(); fig.savefig("resultados/figuras/fig_sensibilidade_w.png", dpi=200); plt.close(fig)
