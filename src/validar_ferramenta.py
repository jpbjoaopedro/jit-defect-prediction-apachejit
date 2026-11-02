"""Etapa 9b: validação funcional da ferramenta em 50 commits do ZooKeeper (fora do treino).

Uso: python src/validar_ferramenta.py <clone do apache/zookeeper>

Compara, atributo por atributo, o que a ferramenta calcula com o que está no ApacheJIT.
As definições de nd e age foram acertadas antes, em outros 20 commits do ZooKeeper (semente 7),
para que a amostra de validação não fosse usada para ajustar o código.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from atributos import ATRIBUTOS
from extrair import extrair
from jit import carregar_modelo, explicar, grafico

N, SEMENTE, TOLERANCIA = 50, 42, 0.01


def comparar(calculado, apachejit):
    linhas = []
    for a in ATRIBUTOS:
        x, y = calculado[a].astype(float), apachejit[a].astype(float)
        dif = (x - y).abs()
        linhas.append({"atributo": a,
                       "iguais": (dif < 1e-6).mean(),
                       "ate_1pct": ((dif < 1e-6) | (dif <= TOLERANCIA * y.abs())).mean(),
                       "dif_absoluta_mediana": dif.median()})
    return pd.DataFrame(linhas)


def main(repo):
    df = pd.read_csv("data/apachejit_total.csv")
    amostra = df[df["project"] == "apache/zookeeper"].sample(N, random_state=SEMENTE).set_index("commit_id")
    metricas = extrair(repo, amostra.index)
    faltando = set(amostra.index) - set(metricas)
    assert not faltando, f"commits não encontrados no clone: {faltando}"
    calculado = pd.DataFrame.from_dict(metricas, orient="index").loc[amostra.index]

    tabela = comparar(calculado, amostra)
    tabela.round(3).to_csv("resultados/etapa9_validacao.csv", index=False)
    lado_a_lado = calculado.add_suffix("_ferramenta").join(amostra[ATRIBUTOS].add_suffix("_apachejit"))
    lado_a_lado.to_csv("resultados/etapa9_atributos_zookeeper.csv")
    print(tabela.round(2).to_string(index=False))

    artefato = carregar_modelo("modelos/modelo_rl.joblib")
    risco = pd.Series({h: explicar(artefato, calculado.loc[h].to_dict())[0] for h in amostra.index})
    y = amostra["buggy"].astype(int)
    print(f"\n{int(y.sum())} indutores na amostra; risco médio {risco[y == 1].mean():.2f} nos indutores "
          f"e {risco[y == 0].mean():.2f} nos limpos; AUC-ROC {roc_auc_score(y, risco):.2f} (n = {N}, só indicativo)")

    # Dois exemplos de explicação: o indutor com maior risco e o limpo com maior risco (alarme falso).
    for rotulo, h in (("indutor", risco[y == 1].idxmax()), ("alarme_falso", risco[y == 0].idxmax())):
        p, contribuicoes, valores = explicar(artefato, calculado.loc[h].to_dict())
        grafico(f"resultados/figuras/fig_waterfall_{rotulo}.png", artefato["modelo"][-1], contribuicoes,
                valores, f"ZooKeeper {h[:10]} ({rotulo.replace('_', ' ')}): risco {p:.0%}")
        print(f"{rotulo}: {h} risco {p:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
