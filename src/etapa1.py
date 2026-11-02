import pandas as pd
from dados import carregar, anexar_descoberta

df = carregar()

# conferência contra o artigo do ApacheJIT
assert len(df) == 106_674, len(df)
assert df["buggy"].sum() == 28_239, df["buggy"].sum()
assert df.isna().sum().sum() == 0
datas = pd.to_datetime(df["author_date"], unit="s")
print(f"taxa de indutores: {df['buggy'].mean():.4f}")
print("período:", datas.min(), "a", datas.max())

por_proj = df.groupby("project").agg(commits=("buggy", "size"), taxa=("buggy", "mean"))
print(por_proj.sort_values("commits", ascending=False))
por_proj.to_csv("resultados/etapa1_projetos.csv")

# latência de descoberta (dados dos commit_links)
df = anexar_descoberta(df)
ind = df[df["buggy"] == 1]
print("indutores sem data de descoberta:", ind["discovery_date"].isna().sum())
print("latências negativas:", (ind["latencia_dias"] < 0).sum())
lat = ind.groupby("project")["latencia_dias"].median().sort_values()
print("mediana da latência (dias) por projeto:\n", lat.round(1))
lat.to_csv("resultados/etapa1_latencia.csv")