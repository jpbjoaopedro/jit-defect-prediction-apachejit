"""Ferramenta de linha de comando: risco de um commit introduzir defeito e os fatores que pesaram.

Uso: python src/jit.py --repo <caminho do repositório Git> --commit <hash> [--grafico arquivo.png]

A explicação vem da própria regressão logística: a contribuição de cada atributo é
coeficiente x valor padronizado, em log-odds, comparado a um commit médio do treino.
Para um modelo linear isso coincide com o valor SHAP de cada atributo.
"""
import argparse
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from atributos import transformar
from extrair import extrair

NOMES = {
    "fix": "commit de correção", "la": "linhas adicionadas", "ld": "linhas removidas",
    "nf": "arquivos modificados", "nd": "diretórios modificados", "ns": "subsistemas modificados",
    "ent": "espalhamento da mudança (entropia)", "ndev": "desenvolvedores que já mexeram nos arquivos",
    "age": "dias desde a última mudança nos arquivos", "nuc": "mudanças anteriores nos arquivos",
    "aexp": "experiência do autor (commits)", "arexp": "experiência recente do autor",
    "asexp": "experiência do autor nos subsistemas",
}


def carregar_modelo(caminho):
    artefato = joblib.load(caminho)
    if artefato["versao_sklearn"] != sklearn.__version__:
        print(f"Aviso: o modelo foi treinado com scikit-learn {artefato['versao_sklearn']} e aqui há "
              f"{sklearn.__version__}. Rode 'python src/modelo_final.py' para treiná-lo de novo.",
              file=sys.stderr)
    return artefato


def explicar(artefato, metricas):
    """Devolve (probabilidade, contribuições em log-odds por atributo, valores originais)."""
    cols = artefato["atributos"]
    bruto = pd.DataFrame([metricas])
    X = transformar(bruto)[cols]
    escalonador, rl = artefato["modelo"][0], artefato["modelo"][-1]
    contribuicoes = pd.Series(rl.coef_[0] * escalonador.transform(X)[0], index=cols)
    p = artefato["modelo"].predict_proba(X)[0, 1]
    return p, contribuicoes, bruto[cols].iloc[0]


def relatorio(hash_, p, limiar, contribuicoes, valores, n=3):
    linhas = [f"Commit {hash_[:10]}",
              f"Risco estimado de introduzir defeito: {p:.0%} "
              f"({'ACIMA' if p >= limiar else 'abaixo'} do limiar de revisão, {limiar:.0%})",
              f"Fatores que mais pesaram (comparado a um commit médio do treino):"]
    for atributo in contribuicoes.abs().sort_values(ascending=False).index[:n]:
        efeito = "aumenta" if contribuicoes[atributo] > 0 else "reduz"
        linhas.append(f"  - {NOMES[atributo]} = {valores[atributo]:g}: {efeito} o risco "
                      f"({contribuicoes[atributo]:+.2f} log-odds)")
    return "\n".join(linhas)


def grafico(arquivo, rl, contribuicoes, valores, titulo):
    import shap  # só quem pede o gráfico precisa do shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    explicacao = shap.Explanation(values=contribuicoes.values, base_values=rl.intercept_[0],
                                  data=valores.values, feature_names=list(contribuicoes.index))
    shap.plots.waterfall(explicacao, show=False)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=200)
    plt.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--modelo", default="modelos/modelo_rl.joblib")
    ap.add_argument("--grafico")
    args = ap.parse_args(argv)

    artefato = carregar_modelo(args.modelo)
    metricas = extrair(args.repo, [args.commit])
    if args.commit not in metricas:
        sys.exit(f"Commit {args.commit} não encontrado em {args.repo} (use o hash completo).")
    p, contribuicoes, valores = explicar(artefato, metricas[args.commit])
    print(relatorio(args.commit, p, artefato["limiar"], contribuicoes, valores))
    if args.grafico:
        grafico(args.grafico, artefato["modelo"][-1], contribuicoes, valores,
                f"Commit {args.commit[:10]}: risco {p:.0%}")


if __name__ == "__main__":
    main()
