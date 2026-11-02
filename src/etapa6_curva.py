"""Etapa 6b: curva de esforço na última janela de um projeto (figura de exemplo para o texto).

Uso: python src/etapa6_curva.py [projeto]   (padrão: hadoop)
"""
import ast
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from dados import carregar, anexar_descoberta, periodo_estudo
from janelas import janelas
from atributos import ATRIBUTOS, transformar, remover_correlacionados
from modelos import modelos, escore_tamanho
from metricas import esforco

warnings.filterwarnings("ignore")  # mesmos avisos repetidos do experimento.py


def curva(y, escore, linhas):
    """Pontos (fração do esforço, fração dos indutores achados) revisando do maior escore ao menor."""
    ordem = np.argsort(-np.asarray(escore), kind="stable")
    x = np.concatenate([[0], np.cumsum(linhas[ordem]) / linhas.sum()])
    v = np.concatenate([[0], np.cumsum(y[ordem]) / y.sum()])
    return x, v


def main(proj="hadoop"):
    df = periodo_estudo(anexar_descoberta(carregar()))
    T, tr, te = list(janelas(df[df["project"] == "apache/" + proj]))[-1]
    X_tr, X_te = transformar(tr[ATRIBUTOS]), transformar(te[ATRIBUTOS])
    cols = remover_correlacionados(X_tr)
    met = pd.read_csv("resultados/metricas_w90.csv")
    params = met[(met.projeto == proj) & (met.modelo == "RL")].sort_values("inicio_teste").iloc[-1]["params"]
    rl = clone(modelos()["RL"]).set_params(**ast.literal_eval(params)).fit(X_tr[cols], tr["y"].values)
    p = rl.predict_proba(X_te[cols])[:, 1]

    y = te["y"].values
    linhas = (te["la"] + te["ld"] + 1).values.astype(float)
    ordenacoes = {
        "Ótima (indutores menores primeiro)": y * 1e9 - linhas,
        "RL, densidade (risco por linha)": p / linhas,
        "Tamanho (menor primeiro)": escore_tamanho(te).values,
        "RL, probabilidade": p,
    }
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for rotulo, escore in ordenacoes.items():
        x, v = curva(y, escore, linhas)
        r20 = esforco(y, escore, te["la"].values, te["ld"].values)["recall20"]
        ax.plot(x, v, label=f"{rotulo} (Recall@20% = {r20:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Aleatória")
    ax.axvline(0.2, color="grey", ls="--", lw=1)
    ax.set(xlabel="Fração das linhas modificadas revisadas", ylabel="Fração dos commits indutores encontrados",
           title=f"Curva de esforço, {proj.capitalize()} (teste a partir de {pd.Timestamp(T, unit='s').date()})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"resultados/figuras/fig_curva_esforco_{proj}.png", dpi=200)
    plt.close(fig)
    print(proj, "janela", pd.Timestamp(T, unit="s").date(), "commits no teste:", len(te), "indutores:", int(y.sum()))


if __name__ == "__main__":
    main(*sys.argv[1:])
