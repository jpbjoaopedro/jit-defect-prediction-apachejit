import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ATRIBUTOS = ["fix", "la", "ld", "nf", "nd", "ns", "ent", "ndev", "age",
             "nuc", "aexp", "arexp", "asexp"]
LOG = ["la", "ld", "nf", "nd", "ns", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]

def transformar(X):
    X = X.copy()
    X[LOG] = np.log1p(X[LOG].clip(lower=0))
    return X

def remover_correlacionados(X, limiar_rho=0.7, limiar_vif=5.0):
    """Devolve as colunas que sobram após tirar redundâncias, inspirado no AutoSpearman.

    Passo 1: enquanto algum par tiver |rho de Spearman| >= limiar_rho, sai um dos dois.
    Passo 2: enquanto o maior VIF for >= limiar_vif, sai o atributo com esse VIF.
    """
    cols = list(X.columns)
    while True:
        rho = X[cols].corr(method="spearman").abs().fillna(0)   # atributo constante na janela dá NaN
        np.fill_diagonal(rho.values, 0)
        if rho.values.max() < limiar_rho:
            break
        i, j = np.unravel_index(rho.values.argmax(), rho.shape)
        a, b = cols[i], cols[j]
        # Do par, sai quem mais se sobrepõe ao resto: o outro carrega a informação com menos redundância.
        sai = a if rho[a].mean() >= rho[b].mean() else b
        cols.remove(sai)
    while len(cols) > 2:
        vifs = {}
        for c in cols:
            outros = [o for o in cols if o != c]
            r2 = LinearRegression().fit(X[outros], X[c]).score(X[outros], X[c])
            vifs[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
        pior = max(vifs, key=vifs.get)
        if vifs[pior] < limiar_vif:
            break
        cols.remove(pior)
    return cols