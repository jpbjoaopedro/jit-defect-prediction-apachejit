"""Etapa 7a: validação cruzada aleatória (10-fold), para medir o otimismo frente ao walk-forward.

Uso: python src/etapa7_cv.py <projeto>   -> resultados/cv/<projeto>.csv
"""
import os
import sys
import warnings
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from dados import carregar, anexar_descoberta, periodo_estudo
from atributos import ATRIBUTOS, transformar, remover_correlacionados
from modelos import modelos, SEED

warnings.filterwarnings("ignore")  # mesmos avisos repetidos do experimento.py
proj = sys.argv[1]
df = periodo_estudo(anexar_descoberta(carregar()))
# Mesmo período do walk-forward (até 2018), porém com o rótulo final e partições sem ordem temporal.
# A diferença entre as duas avaliações é o otimismo que se quer medir.
g = df[(df["project"] == "apache/" + proj) & (df["author_date"] < pd.Timestamp("2019-01-01").timestamp())]
X, y = transformar(g[ATRIBUTOS]), g["buggy"].values
linhas = []
for k, (i_tr, i_te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y)):
    cols = remover_correlacionados(X.iloc[i_tr])
    for nome, base in modelos().items():
        p = clone(base).fit(X.iloc[i_tr][cols], y[i_tr]).predict_proba(X.iloc[i_te][cols])[:, 1]
        linhas.append({"projeto": proj, "fold": k, "modelo": nome,
                       "auc_roc": roc_auc_score(y[i_te], p), "auc_pr": average_precision_score(y[i_te], p)})
os.makedirs("resultados/cv", exist_ok=True)
pd.DataFrame(linhas).to_csv(f"resultados/cv/{proj}.csv", index=False)
print(proj, pd.DataFrame(linhas).groupby("modelo")[["auc_roc", "auc_pr"]].mean().round(3).to_dict())
