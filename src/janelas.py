import pandas as pd

DIA = 86400
MES = 30.44 * DIA

def janelas(df, w_dias=90, teste_meses=6, min_hist_meses=6):
    """Gera (treino, teste) por projeto, sem vazamento temporal."""
    t_ini, t_fim = df["author_date"].min(), df["author_date"].max()
    T = t_ini + min_hist_meses * MES + w_dias * DIA
    W = w_dias * DIA
    while T + teste_meses * MES <= t_fim + 1:
        fim = T + teste_meses * MES
        hist = df[df["author_date"] < T]
        descoberto = hist["discovery_date"].notna() & (hist["discovery_date"] < T)
        antigo = hist["author_date"] < T - W
        # entra no treino: commit antigo (rótulo conhecido em T) ou indutor já descoberto
        treino = hist[antigo | descoberto].copy()
        treino["y"] = descoberto[antigo | descoberto].astype(int)
        teste = df[(df["author_date"] >= T) & (df["author_date"] < fim)].copy()
        teste["y"] = teste["buggy"]          # avaliação usa o rótulo final
        if treino["y"].sum() >= 10 and teste["y"].sum() >= 5:
            yield T, treino, teste
        T = fim