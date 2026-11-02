import glob
import pandas as pd

def carregar(caminho="data/apachejit_total.csv"):
    df = pd.read_csv(caminho)
    df["project"] = df["project"].replace({
        "apache/hadoop-hdfs": "apache/hadoop",
        "apache/hadoop-mapreduce": "apache/hadoop",
    })
    df["fix"] = df["fix"].astype(int)
    df["buggy"] = df["buggy"].astype(int)
    return df.sort_values("author_date").reset_index(drop=True)

def anexar_descoberta(df, pasta="data/links"):
    """Data de descoberta = primeira correção ligada ao commit indutor."""
    links = pd.concat([pd.read_csv(f) for f in glob.glob(f"{pasta}/commit_links_*.csv")])
    desc = links.groupby("bug_hash")["fix_date"].min().rename("discovery_date")
    df = df.merge(desc, left_on="commit_id", right_index=True, how="left")
    df.loc[df["buggy"] == 0, "discovery_date"] = float("nan")
    df["latencia_dias"] = (df["discovery_date"] - df["author_date"]) / 86400
    return df