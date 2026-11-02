import pandas as pd

DIA = 86400
MES = 30.44 * DIA

def janelas(df, w_dias=90, teste_meses=6, min_hist_meses=6, teste_ate="2019-01-01"):
    """Gera (treino, teste) por projeto, sem vazamento temporal.
    teste_ate: nenhuma janela de teste passa desta data (defeitos de 2019 ainda não descobertos)."""
    t_ini = df["author_date"].min()
    t_fim = min(df["author_date"].max(), pd.Timestamp(teste_ate).timestamp())
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
        # checagens de vazamento: se falharem, todos os resultados ficam inválidos
        assert treino["author_date"].max() < T
        assert not ((treino["y"] == 1) & (treino["discovery_date"] >= T)).any()
        assert not set(treino["commit_id"]) & set(teste["commit_id"])
        if treino["y"].sum() >= 10 and teste["y"].sum() >= 5 and (teste["y"] == 0).sum() >= 5:
            yield T, treino, teste
        T = fim