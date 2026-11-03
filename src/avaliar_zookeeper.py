"""Etapa 9c: desempenho do modelo da ferramenta num projeto que ele não viu (ZooKeeper).

Uso: python src/avaliar_zookeeper.py   -> resultados/etapa9_zookeeper.csv

Usa os atributos do próprio ApacheJIT, para separar a qualidade do modelo da qualidade da
extração (essa já medida em validar_ferramenta.py). Mesmo recorte do estudo: 2010 a 2018.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from dados import carregar, anexar_descoberta, periodo_estudo
from atributos import ATRIBUTOS, transformar
from metricas import esforco
from modelos import escore_tamanho

FIM_TESTE = pd.Timestamp("2019-01-01").timestamp()


def main():
    df = periodo_estudo(anexar_descoberta(carregar()))
    zk = df[(df["project"] == "apache/zookeeper") & (df["author_date"] < FIM_TESTE)]
    artefato = joblib.load("modelos/modelo_rl.joblib")
    X = transformar(zk[ATRIBUTOS])[artefato["atributos"]]
    y = zk["buggy"].values
    p = artefato["modelo"].predict_proba(X)[:, 1]
    la, ld = zk["la"].values, zk["ld"].values
    linhas = []
    for nome, escore in (("RL agrupada", p), ("Tamanho", escore_tamanho(zk).values)):
        linha = {"modelo": nome, "commits": len(zk), "indutores": int(y.sum()),
                 "auc_roc": roc_auc_score(y, escore), "auc_pr": average_precision_score(y, escore)}
        if nome == "RL agrupada":
            linha.update(mcc=matthews_corrcoef(y, p >= artefato["limiar"]), brier=brier_score_loss(y, p))
            linha.update(esforco(y, p / (la + ld + 1), la, ld))
        else:
            linha.update(esforco(y, escore, la, ld))
        linhas.append(linha)
    tabela = pd.DataFrame(linhas).round(3)
    tabela.to_csv("resultados/etapa9_zookeeper.csv", index=False)
    print(tabela.to_string(index=False))
    print("prevalência:", round(y.mean(), 3))


if __name__ == "__main__":
    main()
