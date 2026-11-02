"""Etapa 3: correlação entre atributos e remoção de redundantes por projeto."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dados import carregar, anexar_descoberta, periodo_estudo
from atributos import ATRIBUTOS, transformar, remover_correlacionados

os.makedirs("resultados/figuras", exist_ok=True)
est = periodo_estudo(anexar_descoberta(carregar()))
X = transformar(est[ATRIBUTOS])

rho = X.corr(method="spearman")
rho.round(2).to_csv("resultados/etapa3_spearman.csv")
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(ATRIBUTOS)), ATRIBUTOS, rotation=45, ha="right")
ax.set_yticks(range(len(ATRIBUTOS)), ATRIBUTOS)
for i in range(len(ATRIBUTOS)):
    for j in range(len(ATRIBUTOS)):
        ax.text(j, i, f"{rho.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(rho.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, fraction=0.046)
ax.set_title("Correlação de Spearman entre os atributos (2010 em diante)")
fig.tight_layout(); fig.savefig("resultados/figuras/fig_correlacao_spearman.png", dpi=200); plt.close(fig)

sel = pd.read_csv("resultados/etapa2_projetos.csv").query("selecionado")["projeto"]
linhas = []
for proj in sel:
    g = X[est["project"] == "apache/" + proj]
    mantidos = remover_correlacionados(g)
    linhas.append({"projeto": proj, "mantidos": " ".join(mantidos),
                   "removidos": " ".join(a for a in ATRIBUTOS if a not in mantidos)})
tab = pd.DataFrame(linhas)
tab.to_csv("resultados/etapa3_atributos_por_projeto.csv", index=False)
print(tab.to_string(index=False))
print("\npares com |rho| >= 0.7:")
for i, a in enumerate(ATRIBUTOS):
    for b in ATRIBUTOS[i + 1:]:
        if abs(rho.loc[a, b]) >= 0.7:
            print(f"  {a} x {b}: {rho.loc[a, b]:.2f}")
