"""Etapa 6: consolida as métricas do walk-forward (W = 90), testes estatísticos e figuras."""
import glob
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from metricas import comparar

r = pd.concat([pd.read_csv(f) for f in glob.glob("resultados/metricas/*_w90.csv")])
assert not r.duplicated(["projeto", "inicio_teste", "modelo"]).any()
assert (r.groupby(["projeto", "inicio_teste"]).size() == 5).all(), "janela incompleta no CSV"
r.to_csv("resultados/metricas_w90.csv", index=False)
SUP = ["RL", "RF", "LGBM", "LApredict"]

# esforço: supervisionados ordenados por densidade de risco; "Tamanho" já é um ranking por tamanho
for m in ["recall20", "pci20", "ifa", "popt"]:
    r[m + "_rank"] = r[m + "_dens"].where(r["modelo"] != "Tamanho", r[m])

METR = ["auc_roc", "auc_pr", "precisao", "revocacao", "f1", "mcc", "brier",
        "recall20_rank", "pci20_rank", "ifa_rank", "popt_rank", "recall20", "popt"]
geral = r.groupby("modelo")[METR].agg(["mean", "std"]).round(3)
geral.to_csv("resultados/etapa6_resumo_modelos.csv")
print(r.groupby("modelo")[METR].mean().round(3).T.to_string())

proj = r.pivot_table(index="projeto", columns="modelo", values="auc_pr").round(3)
proj["prevalencia_teste"] = r.groupby("projeto").apply(
    lambda g: (g["pos_teste"] / g["n_teste"]).mean()).round(3)
proj.to_csv("resultados/etapa6_auc_pr_por_projeto.csv")

# Wilcoxon pareado (mesmas janelas) + delta de Cliff
def par(metrica, a, b):
    x = r[r.modelo == a].set_index(["projeto", "inicio_teste"])[metrica]
    y = r[r.modelo == b].set_index(["projeto", "inicio_teste"])[metrica].reindex(x.index)
    return {"metrica": metrica, "a": a, "b": b, "media_a": x.mean(), "media_b": y.mean(),
            **comparar(x.values, y.values)}

testes = [par(m, a, b) for m in ["auc_pr", "auc_roc", "mcc", "brier"]
          for a, b in itertools.combinations(SUP, 2)]
testes += [par(m, a, "Tamanho") for m in ["recall20_rank", "popt_rank", "ifa_rank", "pci20_rank"]
           for a in SUP]
t = pd.DataFrame(testes).round(4)
t.to_csv("resultados/etapa6_testes.csv", index=False)
print(t.to_string(index=False))

# Figuras
ordem = ["RL", "RF", "LGBM", "LApredict", "Tamanho"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].boxplot([r.loc[r.modelo == m, "auc_pr"] for m in ordem])
ax[0].set_xticks(range(1, 6), ordem)
ax[0].set(title="AUC-PR por janela de teste", ylabel="AUC-PR")
ax[1].boxplot([r.loc[r.modelo == m, "recall20_rank"] for m in ordem])
ax[1].set_xticks(range(1, 6), ordem)
ax[1].set(title="Recall@20% do esforço (ranking por densidade)", ylabel="Recall@20%")
fig.tight_layout(); fig.savefig("resultados/figuras/fig_modelos_aucpr_recall20.png", dpi=200); plt.close(fig)
