"""Etapa 8: interpretação (SHAP e permutação) na última janela de cada projeto.

Uso: python src/etapa8.py <projeto>   -> resultados/interpretacao/<projeto>.csv
     python src/etapa8.py --resumo    -> tabelas e figuras de estabilidade
"""
import ast
import glob
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from dados import carregar, anexar_descoberta, periodo_estudo
from janelas import janelas
from atributos import ATRIBUTOS, transformar, remover_correlacionados
from modelos import modelos
from interpretar import shap_global, permutacao

warnings.filterwarnings("ignore")
OUT = "resultados/interpretacao"
os.makedirs(OUT, exist_ok=True)


def projeto(proj):
    df = periodo_estudo(anexar_descoberta(carregar()))
    g = df[df["project"] == "apache/" + proj]
    T, tr, te = list(janelas(g))[-1]                      # última janela de teste
    X_tr, X_te = transformar(tr[ATRIBUTOS]), transformar(te[ATRIBUTOS])
    cols = remover_correlacionados(X_tr)
    met = pd.read_csv("resultados/metricas_w90.csv")
    linhas = []
    for nome, base in modelos().items():
        prm = met[(met.projeto == proj) & (met.modelo == nome)].sort_values("inicio_teste").iloc[-1]["params"]
        m = clone(base).set_params(**ast.literal_eval(prm)).fit(X_tr[cols], tr["y"].values)
        perm = permutacao(m, X_te[cols], te["y"].values)
        linhas += [{"projeto": proj, "modelo": nome, "metodo": "permutacao", "atributo": a, "importancia": v}
                   for a, v in perm.items()]
        if nome in ("RF", "LGBM"):
            imp, vals, amostra = shap_global(m, X_te[cols], n=500)
            linhas += [{"projeto": proj, "modelo": nome, "metodo": "shap", "atributo": a, "importancia": v}
                       for a, v in imp.items()]
            # direção do efeito: correlação entre o valor do atributo e o seu valor SHAP
            for k, a in enumerate(amostra.columns):
                c = np.corrcoef(amostra[a], vals[:, k])[0, 1] if amostra[a].std() > 0 else np.nan
                linhas.append({"projeto": proj, "modelo": nome, "metodo": "shap_direcao", "atributo": a, "importancia": c})
            if nome == "RF" and proj == "hadoop":
                shap.summary_plot(vals, amostra, show=False, plot_size=(7, 4.5))
                plt.title("SHAP do Random Forest, Hadoop (última janela)")
                plt.tight_layout(); plt.savefig("resultados/figuras/fig_shap_hadoop_rf.png", dpi=200); plt.close()
    pd.DataFrame(linhas).to_csv(f"{OUT}/{proj}.csv", index=False)
    print(proj, "ok, teste a partir de", pd.Timestamp(T, unit="s").date())


def resumo():
    r = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/*.csv")])
    imp = r[r.metodo.isin(["shap", "permutacao"])].copy()
    imp["rank"] = imp.groupby(["projeto", "modelo", "metodo"])["importancia"].rank(ascending=False)
    top = imp[imp["rank"] <= 3]
    est = (top.groupby(["modelo", "metodo", "atributo"])["projeto"].nunique()
              .rename("projetos_no_top3").reset_index()
              .sort_values(["modelo", "metodo", "projetos_no_top3"], ascending=[True, True, False]))
    est.to_csv("resultados/etapa8_estabilidade_top3.csv", index=False)
    print(est.groupby(["modelo", "metodo"]).head(4).to_string(index=False))
    dire = (r[r.metodo == "shap_direcao"].groupby(["modelo", "atributo"])["importancia"]
              .agg(["mean", lambda s: (s > 0).mean()]).round(2))
    dire.columns = ["corr_media", "frac_projetos_positiva"]
    dire.to_csv("resultados/etapa8_direcao_shap.csv")
    print(dire.to_string())

    # mapa de calor: ranking do atributo (permutação, RL) por projeto
    rl = imp[(imp.modelo == "RL") & (imp.metodo == "permutacao")].pivot(index="atributo", columns="projeto", values="rank")
    rl = rl.loc[rl.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(rl, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(rl.shape[1]), rl.columns, rotation=40, ha="right")
    ax.set_yticks(range(rl.shape[0]), rl.index)
    for i in range(rl.shape[0]):
        for j in range(rl.shape[1]):
            if not np.isnan(rl.iloc[i, j]):
                ax.text(j, i, int(rl.iloc[i, j]), ha="center", va="center", fontsize=7,
                        color="black" if rl.iloc[i, j] <= 3 else "white")
    fig.colorbar(im, label="posição no ranking (1 = mais importante)")
    ax.set_title("Importância por permutação na regressão logística (vazio = atributo removido)")
    fig.tight_layout(); fig.savefig("resultados/figuras/fig_ranking_atributos_rl.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    resumo() if sys.argv[1] == "--resumo" else projeto(sys.argv[1])
