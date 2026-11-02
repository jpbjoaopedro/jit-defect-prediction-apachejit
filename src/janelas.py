import pandas as pd
from dados import DIA

MES = 30.44 * DIA  # duração média do mês no calendário gregoriano
# Com menos exemplos, o treino quase não vê indutores e AUC e MCC no teste ficam indefinidos ou instáveis.
MIN_INDUTORES_TREINO = 10
MIN_POR_CLASSE_TESTE = 5

def rotular_em(df, T, w_dias):
    """Commits anteriores a T com o rótulo que se teria em T (regra de latência de Cabral et al., 2019)."""
    hist = df[df["author_date"] < T]
    descoberto = hist["discovery_date"].notna() & (hist["discovery_date"] < T)
    antigo = hist["author_date"] < T - w_dias * DIA
    # Commit recente sem correção ainda pode virar indutor: fica fora até completar W dias.
    treino = hist[antigo | descoberto].copy()
    treino["y"] = descoberto[antigo | descoberto].astype(int)
    return treino


def janelas(df, w_dias=90, teste_meses=6, min_hist_meses=6, teste_ate="2019-01-01"):
    """Gera (T, treino, teste) em janelas sucessivas: o treino acumula tudo antes de T.

    Cada commit de treino recebe o rótulo que se teria em T (regra de latência de Cabral et al., 2019).
    teste_ate: nenhuma janela de teste passa desta data, porque os defeitos de 2019 ainda não
    tinham sido descobertos quando o ApacheJIT foi coletado.
    """
    t_ini = df["author_date"].min()
    t_fim = min(df["author_date"].max(), pd.Timestamp(teste_ate).timestamp())
    T = t_ini + min_hist_meses * MES + w_dias * DIA
    while T + teste_meses * MES <= t_fim + 1:
        fim = T + teste_meses * MES
        treino = rotular_em(df, T, w_dias)
        teste = df[(df["author_date"] >= T) & (df["author_date"] < fim)].copy()
        teste["y"] = teste["buggy"]  # o teste é julgado com o rótulo final, o que de fato aconteceu
        # Se alguma destas falhar, houve vazamento do futuro e nenhum resultado do estudo vale.
        assert treino["author_date"].max() < T
        assert not ((treino["y"] == 1) & (treino["discovery_date"] >= T)).any()
        assert not set(treino["commit_id"]) & set(teste["commit_id"])
        positivos_teste = teste["y"].sum()
        negativos_teste = len(teste) - positivos_teste
        if (treino["y"].sum() >= MIN_INDUTORES_TREINO
                and min(positivos_teste, negativos_teste) >= MIN_POR_CLASSE_TESTE):
            yield T, treino, teste
        T = fim