"""Etapa 2: exploração, latência de descoberta e seleção dos projetos."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dados import carregar, anexar_descoberta, periodo_estudo
from janelas import janelas

FIG = "resultados/figuras"
os.makedirs(FIG, exist_ok=True)

todos = anexar_descoberta(carregar())
todos["ano"] = pd.to_datetime(todos["author_date"], unit="s").dt.year
est = periodo_estudo(todos)
nome = lambda p: p.replace("apache/", "")

# --- Figura: taxa de indutores e latência por ano (justifica o recorte 2010-2018) ---
por_ano = todos.groupby("ano").agg(taxa=("buggy", "mean"))
por_ano["latencia_mediana"] = todos[todos["buggy"] == 1].groupby("ano")["latencia_dias"].median()
por_ano.round(3).to_csv("resultados/etapa2_por_ano.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(por_ano.index, por_ano["taxa"] * 100, color="#4a6fa5")
ax[0].set(title="Commits indutores por ano", xlabel="Ano de autoria", ylabel="% de indutores")
ax[1].bar(por_ano.index, por_ano["latencia_mediana"], color="#c0504d")
ax[1].set(title="Latência mediana de descoberta por ano", xlabel="Ano de autoria", ylabel="Dias (escala log)", yscale="log")
for a in ax:
    a.axvspan(2009.5, 2018.5, color="green", alpha=0.08)  # período usado nas janelas de teste
    a.set_xticks(range(2003, 2020, 2))
fig.tight_layout(); fig.savefig(f"{FIG}/fig_taxa_latencia_por_ano.png", dpi=200); plt.close(fig)

# --- Latência por projeto (2010-2017, sem os cortes da coleta) ---
ind = est[(est["buggy"] == 1) & (est["ano"] <= 2017)]
ordem = ind.groupby("project")["latencia_dias"].median().sort_values().index
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.boxplot([ind.loc[ind["project"] == p, "latencia_dias"].clip(lower=0.5) for p in ordem],
           showfliers=False)
ax.set_xticks(range(1, len(ordem) + 1), [nome(p) for p in ordem], rotation=35, ha="right")
ax.set_yscale("log")
ax.axhline(90, color="red", ls="--", lw=1, label="90 dias (Cabral et al., 2019)")
ax.set(ylabel="Dias até a correção (escala log)", title="Latência de descoberta por projeto (commits de 2010 a 2017)")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_latencia_por_projeto.png", dpi=200); plt.close(fig)

# --- Tabela de projetos e seleção ---
linhas = []
for p, g in est.groupby("project"):
    datas = pd.to_datetime(g["author_date"], unit="s")
    lat = ind.loc[ind["project"] == p, "latencia_dias"]
    n_jan = sum(1 for _ in janelas(g))
    linhas.append({
        "projeto": nome(p), "commits": len(g), "indutores": int(g["buggy"].sum()),
        "taxa": round(g["buggy"].mean(), 3), "inicio": datas.min().date(), "fim": datas.max().date(),
        "latencia_mediana": round(lat.median(), 1), "pct_ate_90d": round((lat <= 90).mean() * 100, 1),
        "janelas_teste": n_jan,
    })
tab = pd.DataFrame(linhas).sort_values("commits", ascending=False)
tab["selecionado"] = (tab["commits"] >= 1000) & (tab["janelas_teste"] >= 2)
tab.to_csv("resultados/etapa2_projetos.csv", index=False)
print(tab.to_string(index=False))
print("\nselecionados:", tab["selecionado"].sum(), "de", len(tab))
print(f"latência 2010-2017: mediana {ind['latencia_dias'].median():.1f} dias, "
      f"{(ind['latencia_dias'] <= 90).mean() * 100:.1f}% em até 90 dias")
