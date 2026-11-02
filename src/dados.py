import glob
import pandas as pd

DIA = 86400  # segundos; as datas do ApacheJIT são timestamps Unix

# O ApacheJIT só usou relatos de defeito abertos de 2010 em diante; commits anteriores têm rótulo incompleto.
INICIO_ESTUDO = pd.Timestamp("2010-01-01").timestamp()

def carregar(caminho="data/apachejit_total.csv"):
    """Lê o ApacheJIT, junta os três rótulos do Hadoop (mesmo repositório) e ordena por data."""
    df = pd.read_csv(caminho)
    df["project"] = df["project"].replace({
        "apache/hadoop-hdfs": "apache/hadoop",
        "apache/hadoop-mapreduce": "apache/hadoop",
    })
    df["fix"] = df["fix"].astype(int)
    df["buggy"] = df["buggy"].astype(int)
    # 1.155 commits têm a mesma data. Sem desempate pelo hash, a ordem (e as janelas) mudaria entre versões do pandas.
    return df.sort_values(["author_date", "commit_id"], kind="mergesort").reset_index(drop=True)

def anexar_descoberta(df, pasta="data/links"):
    """Acrescenta discovery_date (primeira correção ligada ao commit indutor) e latencia_dias."""
    links = pd.concat([pd.read_csv(f) for f in glob.glob(f"{pasta}/commit_links_*.csv")])
    desc = links.groupby("bug_hash")["fix_date"].min().rename("discovery_date")
    df = df.merge(desc, left_on="commit_id", right_index=True, how="left")
    df.loc[df["buggy"] == 0, "discovery_date"] = float("nan")
    df["latencia_dias"] = (df["discovery_date"] - df["author_date"]) / DIA
    return df

def periodo_estudo(df):
    """Recorte do estudo: commits a partir de 2010, sem latência negativa (2 casos inconsistentes)."""
    ok = (df["author_date"] >= INICIO_ESTUDO) & ~(df["latencia_dias"] < 0)
    return df[ok].reset_index(drop=True)
