"""Etapa 9a: treina o modelo usado pela ferramenta e grava em modelos/modelo_rl.joblib.

A regressão logística foi a melhor no walk-forward (AUC-PR e calibração). Aqui ela é treinada uma
vez só, com os 12 projetos juntos, para servir a repositórios que não estão no ApacheJIT.
O rótulo segue a mesma regra de latência do experimento, com corte em T = 01/01/2019 e W = 90.
"""
import os
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from dados import carregar, anexar_descoberta, periodo_estudo
from janelas import rotular_em
from atributos import ATRIBUTOS, transformar, remover_correlacionados
from modelos import modelos
from experimento import ajustar, GRADE, FRACAO_VALIDACAO

CORTE = pd.Timestamp("2019-01-01").timestamp()
W_DIAS = 90
ARQUIVO = "modelos/modelo_rl.joblib"


def treinar():
    df = periodo_estudo(anexar_descoberta(carregar()))
    selecionados = pd.read_csv("resultados/etapa2_projetos.csv").query("selecionado")["projeto"]
    df = df[df["project"].isin("apache/" + selecionados)]
    treino = rotular_em(df, CORTE, W_DIAS).sort_values(["author_date", "commit_id"], kind="mergesort")
    X = transformar(treino[ATRIBUTOS])
    y = treino["y"].values
    cols = remover_correlacionados(X)
    corte = int(len(treino) * (1 - FRACAO_VALIDACAO))
    params, limiar = ajustar(modelos()["RL"], GRADE["RL"], X.iloc[:corte][cols], y[:corte],
                             X.iloc[corte:][cols], y[corte:])
    modelo = clone(modelos()["RL"]).set_params(**params).fit(X[cols], y)
    artefato = {
        "modelo": modelo, "atributos": cols, "limiar": float(limiar), "params": params,
        "w_dias": W_DIAS, "treinado_ate": "2019-01-01", "projetos": sorted(selecionados),
        "n_treino": len(treino), "indutores_treino": int(y.sum()),
        "versao_sklearn": sklearn.__version__,
    }
    os.makedirs(os.path.dirname(ARQUIVO), exist_ok=True)
    joblib.dump(artefato, ARQUIVO)
    coef = modelo[-1].coef_[0]
    print(f"{len(treino)} commits, {int(y.sum())} indutores, params={params}, limiar={limiar:.2f}")
    print(pd.Series(coef, index=cols).sort_values(key=np.abs, ascending=False).round(3).to_string())


if __name__ == "__main__":
    treinar()
