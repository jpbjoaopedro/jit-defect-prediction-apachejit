"""Etapas 4 a 6: walk-forward com latência, modelos, linhas de base e métricas.

Uso: python src/experimento.py <projeto> [w_dias] [modelos separados por vírgula]
Grava resultados/metricas/<projeto>_w<w_dias>.csv (uma linha por janela e modelo).
Retoma de onde parou: janelas já gravadas no CSV são puladas.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from dados import carregar, anexar_descoberta, periodo_estudo
from janelas import janelas
from atributos import ATRIBUTOS, transformar, remover_correlacionados
from modelos import modelos, escore_tamanho, lapredict
from metricas import limiar_por_mcc, estatisticas, esforco

# Os avisos se repetem a cada janela e enterram o progresso no terminal.
# A convergência da regressão logística foi conferida numa rodada sem este filtro.
warnings.filterwarnings("ignore")
FRACAO_VALIDACAO = 0.2  # parte mais recente do treino, usada só para escolher hiperparâmetros e limiar
GRADE = {"RL": [{"logisticregression__C": c} for c in (0.1, 1, 10)],
         "RF": [{"min_samples_leaf": m} for m in (1, 5, 10)],
         "LGBM": [{"num_leaves": n} for n in (15, 31, 63)]}


def ajustar(base, grade, X_fit, y_fit, X_val, y_val):
    """Escolhe hiperparâmetros pela AUC-PR na validação temporal; devolve (params, limiar por MCC)."""
    if y_val.sum() == 0 or y_fit.sum() == 0:
        # Sem indutores não há o que comparar: fica o valor do meio da grade e o limiar neutro.
        return grade[len(grade) // 2], 0.5
    melhor, melhor_ap, melhor_p = None, -1, None
    for params in grade:
        p = clone(base).set_params(**params).fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, p)
        if ap > melhor_ap:
            melhor, melhor_ap, melhor_p = params, ap, p
    return melhor, limiar_por_mcc(y_val, melhor_p)


def avaliar(y, p, limiar, teste):
    """Métricas de classificação e de esforço, com o ranking pela probabilidade e pela densidade."""
    linha = estatisticas(y, p, limiar)
    la, ld = teste["la"].values, teste["ld"].values
    linha.update(esforco(y, p, la, ld))
    # Ordenar só pela probabilidade põe os commits grandes na frente e gasta o esforço neles;
    # a densidade (risco por linha) segue os modelos sensíveis ao esforço de Kamei et al. (2013).
    linha.update({k + "_dens": v for k, v in esforco(y, p / (la + ld + 1), la, ld).items()})
    return linha


def avaliar_lapredict(X_tr, X_te, y_tr, corte, teste):
    """Linha de base LApredict: limiar escolhido na validação, como nos outros modelos."""
    la_tr, la_te = X_tr[["la"]], X_te[["la"]]
    y_fit, y_val = y_tr[:corte], y_tr[corte:]
    limiar = 0.5
    if y_fit.sum() and y_val.sum():
        p_val = lapredict().fit(la_tr.iloc[:corte], y_fit).predict_proba(la_tr.iloc[corte:])[:, 1]
        limiar = limiar_por_mcc(y_val, p_val)
    p = lapredict().fit(la_tr, y_tr).predict_proba(la_te)[:, 1]
    return avaliar(teste["y"].values, p, limiar, teste)


def avaliar_tamanho(teste):
    """Linha de base por tamanho: é um ranking, não uma probabilidade, então não tem limiar nem Brier."""
    y, s = teste["y"].values, escore_tamanho(teste).values
    linha = {"auc_roc": roc_auc_score(y, s), "auc_pr": average_precision_score(y, s)}
    linha.update(esforco(y, s, teste["la"].values, teste["ld"].values))
    return linha


def rodar(projeto, w_dias=90, so=None):
    df = periodo_estudo(anexar_descoberta(carregar()))
    g = df[df["project"] == "apache/" + projeto]
    os.makedirs("resultados/metricas", exist_ok=True)
    arq = f"resultados/metricas/{projeto}_w{w_dias}.csv"
    feitas = set(pd.read_csv(arq)["inicio_teste"].astype(str)) if os.path.exists(arq) else set()
    for T, tr, te in janelas(g, w_dias=w_dias):
        if str(pd.Timestamp(T, unit="s").date()) in feitas:
            continue
        saida = []
        X_tr, X_te = transformar(tr[ATRIBUTOS]), transformar(te[ATRIBUTOS])
        cols = remover_correlacionados(X_tr)
        corte = int(len(tr) * (1 - FRACAO_VALIDACAO))  # o treino já vem em ordem de data
        X_fit, X_val = X_tr.iloc[:corte][cols], X_tr.iloc[corte:][cols]
        y_tr, y_te = tr["y"].values, te["y"].values
        y_fit, y_val = y_tr[:corte], y_tr[corte:]
        base_info = {"projeto": projeto, "w_dias": w_dias,
                     "inicio_teste": pd.Timestamp(T, unit="s").date(),
                     "n_treino": len(tr), "pos_treino": int(y_tr.sum()),
                     "n_teste": len(te), "pos_teste": int(y_te.sum()),
                     "atributos": " ".join(cols)}

        for nome, base in modelos().items():
            if so and nome not in so:
                continue
            params, limiar = ajustar(base, GRADE[nome], X_fit, y_fit, X_val, y_val)
            m = clone(base).set_params(**params).fit(X_tr[cols], y_tr)
            p = m.predict_proba(X_te[cols])[:, 1]
            saida.append({**base_info, "modelo": nome, "params": str(params),
                          **avaliar(y_te, p, limiar, te)})

        saida.append({**base_info, "modelo": "LApredict", **avaliar_lapredict(X_tr, X_te, y_tr, corte, te)})
        saida.append({**base_info, "modelo": "Tamanho", **avaliar_tamanho(te)})
        # Grava janela a janela: uma rodada longa interrompida retoma de onde parou.
        pd.DataFrame(saida).to_csv(arq, mode="a", header=not os.path.exists(arq), index=False)
    return pd.read_csv(arq)


if __name__ == "__main__":
    proj = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    so = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    r = rodar(proj, w, so)
    print(proj, f"w={w}", "janelas:", r["inicio_teste"].nunique())
    print(r.groupby("modelo")[["auc_roc", "auc_pr", "mcc", "brier", "recall20", "popt", "ifa"]].mean().round(3))
